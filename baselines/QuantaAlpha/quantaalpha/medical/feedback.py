"""Feedback step for medical QuantaAlpha experiments."""

from __future__ import annotations

import json
import os

from quantaalpha.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.medical.costeer import ingest_costeer_experiment
from quantaalpha.medical.embedding_rag import MedicalEmbeddingRAG
from quantaalpha.medical.experiment import MedicalFactorExperiment
from quantaalpha.medical.task_config import get_task_config


def _metric_improved(metric: str, current: float, previous: float, task: str) -> bool:
    direction = get_task_config(task).metric_direction.get(metric, "max")
    if direction == "min":
        return current < previous
    return current > previous


def _metric_not_worse(metric: str, current: float, previous: float, task: str) -> bool:
    direction = get_task_config(task).metric_direction.get(metric, "max")
    if direction == "min":
        return current <= previous
    return current >= previous


def _best_prior_metric(trace: Trace, metric: str, task: str) -> float | None:
    best = None
    for _, exp, _ in trace.hist:
        scores = ((getattr(exp, "result", {}) or {}).get("scores", {}) or {})
        value = scores.get(metric)
        if isinstance(value, (int, float)):
            if best is None or _metric_improved(metric, float(value), float(best), task):
                best = value
    return best


def _as_metric_map(scores: dict, metrics: tuple[str, ...]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for metric in metrics:
        value = scores.get(metric)
        values[metric] = float(value) if isinstance(value, (int, float)) else None
    return values


def _primary_delta_map(result: dict, metrics: tuple[str, ...]) -> dict[str, float | None]:
    baseline_deltas = result.get("baseline_deltas", {}) or {}
    primary_model = result.get("primary_model", "")
    candidate_keys = []
    if primary_model.startswith("combined_"):
        candidate_keys.append(primary_model)
    candidate_keys.extend(
        key
        for key in sorted(baseline_deltas)
        if key.startswith("combined_") and key not in candidate_keys
    )
    for key in candidate_keys:
        test_deltas = (baseline_deltas.get(key, {}) or {}).get("test", {}) or {}
        if test_deltas:
            return _as_metric_map(test_deltas, metrics)
    return {metric: None for metric in metrics}


class MedicalPyHealthFeedback(HypothesisExperiment2Feedback):
    def generate_feedback(
        self,
        exp: MedicalFactorExperiment,
        hypothesis: Hypothesis,
        trace: Trace,
    ) -> HypothesisFeedback:
        result = getattr(exp, "result", {}) or {}
        scores = result.get("scores", {})
        all_model_scores = result.get("all_model_scores", {})
        baseline_deltas = result.get("baseline_deltas", {})
        factor_summary = result.get("factor_summary", [])
        task_config = get_task_config(result.get("task"))
        primary_metric = task_config.primary_metric
        report_metrics = task_config.report_metrics
        prior = _best_prior_metric(trace, primary_metric, task_config.name)
        current_primary = scores.get(primary_metric, 0.0)
        current_report_metrics = _as_metric_map(scores, report_metrics)
        previous_best_report_metrics = {
            metric: _best_prior_metric(trace, metric, task_config.name)
            for metric in report_metrics
        }
        baseline_delta_report_metrics = _primary_delta_map(result, report_metrics)
        metric_comparison = {}
        improved_metrics = []
        worse_metrics = []
        for metric in report_metrics:
            current = current_report_metrics.get(metric)
            previous = previous_best_report_metrics.get(metric)
            if current is None or previous is None:
                improved = None
                not_worse = None
            else:
                improved = _metric_improved(
                    metric,
                    float(current),
                    float(previous),
                    task_config.name,
                )
                not_worse = _metric_not_worse(
                    metric,
                    float(current),
                    float(previous),
                    task_config.name,
                )
                if improved:
                    improved_metrics.append(metric)
                if not not_worse:
                    worse_metrics.append(metric)
            metric_comparison[metric] = {
                "direction": task_config.metric_direction.get(metric, "max"),
                "current": current,
                "previous_best": previous,
                "combined_minus_baseline_test_delta": baseline_delta_report_metrics.get(metric),
                "improved_vs_previous_best": improved,
                "not_worse_vs_previous_best": not_worse,
            }
        decision = (
            prior is None
            or (
                isinstance(current_primary, (int, float))
                and _metric_not_worse(
                    primary_metric,
                    float(current_primary),
                    float(prior),
                    task_config.name,
                )
                and len(worse_metrics) <= max(1, len(report_metrics) // 3)
            )
        )

        system_prompt = (
            "You are reviewing a clinical symbolic-factor experiment. Return "
            "only JSON with observations, hypothesis_evaluation, "
            "new_hypothesis, reason, and decision."
        )
        user_prompt = f"""
Scenario:
{self.scen.get_scenario_all_desc()}

Hypothesis:
{hypothesis}

Scores:
{json.dumps(scores, indent=2, sort_keys=True)}

All tabular model scores, if available:
{json.dumps(all_model_scores, indent=2, sort_keys=True)}

Combined-minus-baseline deltas, if available:
{json.dumps(baseline_deltas, indent=2, sort_keys=True)}

Primary evaluator/model:
{result.get("evaluator", "pyhealth")} / {result.get("primary_model", "default")}

Primary task metric:
{primary_metric}

Report metrics for this task:
{json.dumps(report_metrics, indent=2)}

Task metric guidance:
{task_config.metric_guidance}

Previous best {primary_metric}:
{prior}

Report metric comparison:
{json.dumps(metric_comparison, indent=2, sort_keys=True)}

Factor activity summary:
{json.dumps(factor_summary, indent=2, sort_keys=True)}

When evaluator is tabular, judge success by all report metrics together and the
task metric guidance. Prefer Pareto-style improvements over baseline_logistic:
higher is better for AUROC/PRAUC/F1/accuracy, lower is better for loss. Do not
over-reward a tiny primary-metric gain if PRAUC, F1, or loss clearly regress.
Also consider factor standalone strength/activity. Penalize zero-activity,
ultra-rare, leakage-prone, or redundant factors.

Return JSON:
{{
  "observations": "...",
  "hypothesis_evaluation": "...",
  "new_hypothesis": "...",
  "reason": "...",
  "decision": true
}}
"""
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                reasoning_flag=False,
                max_retry=2,
                max_tokens=int(os.environ.get("MEDICAL_FEEDBACK_MAX_TOKENS", "1200")),
                temperature=float(os.environ.get("MEDICAL_FEEDBACK_TEMPERATURE", "0.2")),
            )
            data = robust_json_parse(response)
            feedback = HypothesisFeedback(
                observations=data.get("observations", ""),
                hypothesis_evaluation=data.get("hypothesis_evaluation", ""),
                new_hypothesis=data.get("new_hypothesis", ""),
                reason=data.get("reason", ""),
                decision=bool(data.get("decision", decision)),
            )
            ingest_costeer_experiment(exp, feedback)
            MedicalEmbeddingRAG().ingest_experiment(exp, feedback)
            return feedback
        except Exception as exc:
            feedback = HypothesisFeedback(
                observations=(
                    f"PyHealth scores: {json.dumps(scores, sort_keys=True)}. "
                    f"Report metric comparison: "
                    f"{json.dumps(metric_comparison, sort_keys=True)}. "
                    f"Factor summary path: {result.get('factor_summary_path')}"
                ),
                hypothesis_evaluation=(
                    f"Current {primary_metric}={current_primary}; previous best={prior}; "
                    f"improved report metrics={improved_metrics}; "
                    f"worse report metrics={worse_metrics}."
                ),
                new_hypothesis=(
                    "Refine the next factor set toward active, non-redundant "
                    "signals with direct clinical severity meaning."
                ),
                reason=f"LLM feedback failed, used deterministic fallback: {exc}",
                decision=decision,
            )
            ingest_costeer_experiment(exp, feedback)
            MedicalEmbeddingRAG().ingest_experiment(exp, feedback)
            return feedback
