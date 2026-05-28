"""AlphaAgent-style medical proposal components.

These classes keep QuantaAlpha's original loop contract but adapt the factor
construction interface to the safe medical DSL and eICU tabular evaluator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, StrictUndefined

from quantaalpha.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Trace,
)
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.log import logger
from quantaalpha.medical.dsl import (
    NUMERIC_AGGREGATIONS,
    NUMERIC_OPERATORS,
    NUMERIC_TEMPORAL_OPS,
    NUMERIC_TRANSFORMS,
    TEMPORAL_OPS,
    normalize_factors,
)
from quantaalpha.medical.embedding_rag import MedicalEmbeddingRAG
from quantaalpha.medical.experiment import MedicalFactorExperiment, MedicalFactorTask
from quantaalpha.medical.proposal import _factor_formula_text
from quantaalpha.medical.safe_python import (
    SAFE_PYTHON_OPERATION,
    SafePythonFactorError,
    code_hash,
    validate_safe_python_code,
)
from quantaalpha.medical.source_profile import numeric_sources_for_profile, numeric_sources_text, source_profile
from quantaalpha.medical.task_config import get_task_config, observation_end_hours
from quantaalpha.medical.vocab_profile import build_or_load_vocab_profile


DEFAULT_HISTORY_LIMIT = 6
MIN_HISTORY_LIMIT = 1


def _task_phrase() -> str:
    return get_task_config().description


def _primary_metric_phrase() -> str:
    task_config = get_task_config()
    report_metrics = ", ".join(task_config.report_metrics)
    direction_notes = [
        f"{metric} {task_config.metric_direction.get(metric, 'max')}"
        for metric in task_config.report_metrics
    ]
    return (
        f"combined-minus-baseline report metrics ({report_metrics}); primary "
        f"ranking metric is {task_config.primary_metric}, but prefer Pareto-style "
        f"improvement across {', '.join(direction_notes)}"
    )


def _clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _history_text(trace: Trace, limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    if not trace.hist:
        return "No previous hypothesis and feedback available since it is the first round."
    blocks = []
    compact = str_to_bool(os.environ.get("MEDICAL_ALPHA_COMPACT_HISTORY", "1"))
    factor_def_limit = int(os.environ.get("MEDICAL_ALPHA_HISTORY_FACTOR_LIMIT", "8"))
    activity_limit = int(os.environ.get("MEDICAL_ALPHA_HISTORY_ACTIVITY_LIMIT", "10"))
    text_limit = int(os.environ.get("MEDICAL_ALPHA_HISTORY_TEXT_LIMIT", "900"))
    for idx, (hypothesis, exp, feedback) in enumerate(trace.hist[-limit:], start=1):
        result = getattr(exp, "result", {}) or {}
        scores = result.get("scores", {})
        all_scores = result.get("all_model_scores", {})
        deltas = result.get("baseline_deltas", {})
        factor_summary = result.get("factor_summary", [])
        task_config = get_task_config(result.get("task"))
        report_scores = {
            metric: scores.get(metric)
            for metric in task_config.report_metrics
            if isinstance(scores.get(metric), (int, float))
        }
        primary_model = result.get("primary_model", "")
        primary_deltas = {}
        if primary_model:
            primary_deltas = ((deltas.get(primary_model, {}) or {}).get("test", {}) or {})
        compact_activity = [
            {
                "name": item.get("name"),
                "operation": item.get("operation"),
                "nonzero_rate": item.get("nonzero_rate"),
                "mean": item.get("mean"),
                "std": item.get("std"),
            }
            for item in factor_summary[:activity_limit]
        ]
        factor_defs = [
            task.to_factor_dict()
            for task in getattr(exp, "sub_tasks", [])[:factor_def_limit]
            if hasattr(task, "to_factor_dict")
        ]
        if compact:
            blocks.append(
                "\n".join(
                    [
                        f"Hypothesis {idx}: {_clip_text(hypothesis, text_limit)}",
                        f"Report scores: {json.dumps(report_scores, sort_keys=True)}",
                        f"Primary model: {primary_model}",
                        "Primary combined-minus-baseline test deltas: "
                        f"{json.dumps(primary_deltas, sort_keys=True)}",
                        f"Evaluated factor count: {result.get('evaluated_factor_count')}",
                        f"Factor definitions sample: {json.dumps(factor_defs, sort_keys=True)}",
                        f"Factor activity sample: {json.dumps(compact_activity, sort_keys=True)}",
                        f"Observations: {_clip_text(feedback.observations, text_limit)}",
                        "Feedback for hypothesis: "
                        f"{_clip_text(feedback.hypothesis_evaluation, text_limit)}",
                        f"New feedback/context: {_clip_text(feedback.new_hypothesis, text_limit)}",
                        f"Reasoning: {_clip_text(feedback.reason, text_limit)}",
                        f"Did this replace best result? {feedback.decision}",
                    ]
                )
            )
        else:
            blocks.append(
                "\n".join(
                    [
                        f"Hypothesis {idx}: {hypothesis}",
                        f"Primary scores: {json.dumps(scores, sort_keys=True)}",
                        f"All model scores: {json.dumps(all_scores, sort_keys=True)}",
                        f"Combined-minus-baseline deltas: {json.dumps(deltas, sort_keys=True)}",
                        f"Factor definitions: {json.dumps(factor_defs, sort_keys=True)}",
                        f"Factor activity: {json.dumps(compact_activity, sort_keys=True)}",
                        f"Observations: {feedback.observations}",
                        f"Feedback for hypothesis: {feedback.hypothesis_evaluation}",
                        f"New feedback/context: {feedback.new_hypothesis}",
                        f"Reasoning: {feedback.reason}",
                        f"Did this replace best result? {feedback.decision}",
                    ]
                )
            )
    return "\n\n".join(blocks)


def _medical_function_lib_description() -> str:
    horizon = observation_end_hours()
    profile = source_profile()
    numeric_sources = numeric_sources_text(profile)
    numeric_enabled = bool(numeric_sources_for_profile(profile))
    safe_python_enabled = str_to_bool(os.environ.get("MEDICAL_SAFE_PYTHON_FACTORS", "0"))
    horizon_text = (
        f"\nObservation horizon guard: generated factors must not use events after "
        f"{horizon:g} hours from ICU admission for this task. Keep every "
        "window_end_hours, early/late window endpoint, and windows_hours endpoint "
        "within this horizon."
        if horizon is not None
        else ""
    )
    description = """
Only the following safe medical factor DSL operations are allowed:

- log_count: {"operation": "log_count", "sources": ["conditions"|"procedures"|"drugs"]}
  Computes log(1 + number of observed strings in one source).
- count_ratio: {"operation": "count_ratio", "numerator_source": "...", "denominator_source": "..."}
  Computes count(numerator_source) / max(count(denominator_source), 1).
- keyword_any: needs sources and keywords. Returns 1 if any source item contains any keyword.
- keyword_count: needs sources and keywords. Counts source items containing any keyword.
- keyword_density: needs sources and keywords. keyword_count / max(number of source items, 1).
- temporal_keyword_count: needs sources, keywords, window_start_hours, and window_end_hours.
  Counts keyword hits whose eICU event offset is within [start, end) hours after ICU admission.
- temporal_keyword_density: same temporal window, but returns keyword hits / max(number of source events in the window, 1).
- first_keyword_offset: returns log(1 + first matching event offset in hours), or 0 if absent.
- early_late_keyword_delta: needs early_window_hours and late_window_hours. Returns late keyword density minus early keyword density.
- keyword_persistence: needs windows_hours such as [[0,24],[24,72],[72,168]]. Returns fraction of windows with at least one keyword hit.
- numeric_window_mean / numeric_window_min / numeric_window_max / numeric_window_std / numeric_window_last / numeric_window_count / numeric_window_slope:
  needs numeric_source plus window_start_hours/window_end_hours. Computes a safe
  window statistic; slope is least-squares value change per hour.
- numeric_early_late_delta: needs numeric_source plus early_window_hours and late_window_hours. Returns late mean minus early mean.
- numeric_abnormal_fraction: needs numeric_source plus abnormal_low and/or abnormal_high. Returns fraction of window values outside the threshold.
- numeric_persistence: needs numeric_source, windows_hours, and either threshold or abnormal_low/abnormal_high. Returns fraction of windows with abnormal/extreme values.
- numeric_source_interaction: needs numeric_source, secondary_numeric_source, aggregation, secondary_aggregation, operator, and a window.
  Allowed aggregations: mean, min, max, std, last, count, slope.
  Allowed operators: add, sub, mul, ratio, max, min. Ratio returns 0 for near-zero denominator.
  Allowed transform for numeric outputs: identity, log1p, abs, neg, sqrt_abs.
- keyword_gated_numeric: needs sources, keywords, numeric_source, aggregation, and a window.
  It returns the numeric window statistic only when at least one keyword event
  appears in the same window; otherwise it returns 0. Use this for context-gated
  factors such as renal-failure-gated fluid balance or respiratory-failure-gated
  oxygenation instability.
- safe_python: enabled only when MEDICAL_SAFE_PYTHON_FACTORS=1. It needs a
  `code` string defining exactly `compute(sample)`. The function may read only
  sample["conditions"], sample["procedures"], sample["drugs"] and their
  *_offsets arrays. It cannot import modules, read files, access labels,
  patient_id/visit_id, or call unsafe functions. Prefer compact logic using
  helper functions: get_texts, events, contains_any, count_keywords,
  density_keywords, first_offset_hours, persistence, safe_div, log1p.

Valid sources are exactly: conditions, procedures, drugs.
Temporal offsets are available for eICU diagnosis, medication, and physical exam/procedure events. Use only current-stay offsets, typically nonnegative hours after ICU admission. Do not use unit discharge time or target labels.
Valid numeric_source values are: {{ numeric_sources }}.
Numeric operations are {{ numeric_status }} for the active source profile.
Safe Python factors are {{ safe_python_status }}.
Keyword matching is lower-case substring matching against observed eICU strings.
Prefer observed eICU vocabulary terms and lexical variants. Avoid terms absent from the vocabulary profile.
Avoid factors that duplicate baseline count/ratio features unless they add clinical specificity.
Target useful nonzero rates: roughly 1%-50% for specific binary factors; broader burden/count factors may be more common if they add distinct clinical meaning.
Prefer temporal factors when they express onset, early severity, persistence, or escalation, e.g. 0-24h burden, 24-72h burden, 0-72h persistence, or late-minus-early change.
""".strip()
    description = description.replace("{{ numeric_sources }}", numeric_sources)
    description = description.replace(
        "{{ numeric_status }}",
        "enabled" if numeric_enabled else "disabled; do not generate numeric operations",
    )
    description = description.replace(
        "{{ safe_python_status }}",
        "enabled; generate at most half safe_python factors and keep code compact"
        if safe_python_enabled
        else "disabled; do not generate safe_python operations",
    )
    return f"{description}{horizon_text}"


def _factor_output_format() -> str:
    max_factors = int(os.environ.get("MEDICAL_ALPHA_MAX_FACTORS", "12"))
    return """
Return JSON only, with this schema:
{
  "factors": [
    {
      "name": "short_snake_case",
      "description": "human-readable description",
      "rationale": "why this factor may improve the target eICU prediction task",
      "operation": "keyword_count",
      "sources": ["conditions"],
      "numeric_source": "",
      "secondary_numeric_source": "",
      "aggregation": "mean",
      "secondary_aggregation": "mean",
      "operator": "",
      "transform": "identity",
      "keywords": ["sepsis", "septic shock"],
      "code": "",
      "numerator_source": "",
      "denominator_source": "",
      "window_start_hours": 0,
      "window_end_hours": 24,
      "early_window_hours": [0, 24],
      "late_window_hours": [24, 72],
      "windows_hours": [[0, 24], [24, 72], [72, 168]],
      "abnormal_low": null,
      "abnormal_high": null,
      "threshold": null
    }
  ]
}

Normally generate 2 to 3 independent factors per round. If the target hypothesis
explicitly requests a clean-slate, exact-name, ablation, or full factor-list
experiment, generate the complete requested set instead, up to {{ max_factors }}
factors. Do not reference other generated factors. Each factor should be simple,
clinically interpretable, vocabulary-grounded, and non-redundant.
For operation=safe_python, put the complete function source in `code`, for
example: "def compute(sample):\n    dx = get_texts(sample, 'conditions')\n    return float(count_keywords(dx, ['j96', '518.81']) > 0)".
Do not include markdown fences.
""".replace("{{ max_factors }}", str(max_factors)).strip()


def _hypothesis_output_format() -> str:
    return """
Return JSON only, with this schema:
{
  "hypothesis": "single clear actionable hypothesis",
  "concise_knowledge": "transferable medical/data knowledge learned so far",
  "concise_observation": "observation from previous rounds or eICU vocabulary",
  "concise_justification": "why this should improve delta over baseline",
  "concise_specification": "scope, constraints, expected factor activity"
}
""".strip()


def _safe_json_parse(response: str) -> dict[str, Any]:
    data = robust_json_parse(response)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM response.")
    return data


class MedicalAlphaHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str,
    ) -> None:
        super().__init__(
            hypothesis=hypothesis,
            reason="",
            concise_reason=concise_specification,
            concise_observation=concise_observation,
            concise_justification=concise_justification,
            concise_knowledge=concise_knowledge,
        )
        self.concise_specification = concise_specification

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                Concise Specification: {self.concise_specification}
                """


class MedicalAlphaHypothesisGen(HypothesisGen):
    """Original AlphaAgent-style hypothesis generation for medical factors."""

    def __init__(self, scen, potential_direction: str | None = None) -> None:
        super().__init__(scen)
        self.potential_direction = potential_direction
        self.targets = "medical symbolic factors"
        self.embedding_rag = MedicalEmbeddingRAG()

    def _prepare_context(self, trace: Trace, history_limit: int) -> dict[str, Any]:
        if len(trace.hist) > 0:
            hypothesis_and_feedback = _history_text(trace, history_limit)
        elif self.potential_direction:
            hypothesis_and_feedback = (
                "It is the first round. Transform the user direction into a "
                f"formal, actionable hypothesis: {self.potential_direction}"
            )
        else:
            hypothesis_and_feedback = (
                "No previous hypothesis and feedback available since it is the first round."
            )
        rag_query = "\n".join(
            [
                _task_phrase(),
                self.potential_direction or "",
                hypothesis_and_feedback[-4000:],
            ]
        )
        retrieved_factor_knowledge = self.embedding_rag.query(trace, rag_query)
        return {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "retrieved_factor_knowledge": retrieved_factor_knowledge,
            "hypothesis_output_format": _hypothesis_output_format(),
            "hypothesis_specification": (
                f"Generate hypotheses for {_task_phrase()} tabular factor mining. "
                f"Optimize for {_primary_metric_phrase()}, not only absolute combined score. "
                "Use previous factor activity, baseline deltas, and retrieved factor memories "
                "to refine, mutate, or abandon directions."
            ),
        }

    def gen(self, trace: Trace) -> MedicalAlphaHypothesis:
        history_limit = int(os.environ.get("MEDICAL_ALPHA_HISTORY_LIMIT", str(DEFAULT_HISTORY_LIMIT)))
        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                context = self._prepare_context(trace, history_limit)
                system_prompt = f"""
The user is working on generating new hypotheses for {self.targets} in a data-driven research and development process.
The targets are used in the following scenario:
{self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")}

The user has already proposed hypotheses and evaluated them. Check whether a similar hypothesis exists. If one exists and the evidence supports it, refine it; otherwise generate an improved version.

Additional hypothesis specification:
{context["hypothesis_specification"]}

Please generate output using this JSON format:
{context["hypothesis_output_format"]}
"""
                user_prompt = f"""
Former hypotheses and corresponding feedback:
{context["hypothesis_and_feedback"]}

Embedding-retrieved factor memories, including successful, mixed, and failed
factor attempts:
{context["retrieved_factor_knowledge"] or "None"}

Also generate the reasoning and distilled knowledge in the context of the target eICU prediction task, not generic medical knowledge.
"""
                response = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    reasoning_flag=str_to_bool(os.environ.get("MEDICAL_ALPHA_REASONING", "1")),
                    max_tokens=int(os.environ.get("MEDICAL_ALPHA_HYPOTHESIS_MAX_TOKENS", "1800")),
                    temperature=float(os.environ.get("MEDICAL_ALPHA_TEMPERATURE", "0.4")),
                )
                data = _safe_json_parse(response)
                return MedicalAlphaHypothesis(
                    hypothesis=data.get("hypothesis", ""),
                    concise_observation=data.get("concise_observation", ""),
                    concise_justification=data.get("concise_justification", ""),
                    concise_knowledge=data.get("concise_knowledge", ""),
                    concise_specification=data.get("concise_specification", data.get("concise_reason", "")),
                )
            except Exception as exc:
                if history_limit > MIN_HISTORY_LIMIT and "context" in str(exc).lower():
                    history_limit -= 1
                    logger.warning(
                        f"Medical alpha hypothesis prompt too long; retrying with history_limit={history_limit}"
                    )
                    continue
                raise
        raise RuntimeError("Failed to generate medical alpha hypothesis.")


@dataclass
class _RegulatorResult:
    passed: bool
    feedback: str
    factors: list[dict[str, Any]]


class MedicalDSLRegulator:
    """Small medical equivalent of QuantaAlpha's expression regulator."""

    def __init__(self, trace: Trace | None = None) -> None:
        self.trace = trace
        self._profile = build_or_load_vocab_profile()
        self._observed_values = {
            source: [
                item["text"].lower()
                for item in self._profile.get("sources", {}).get(source, {}).get("top_values", [])
            ]
            for source in ("conditions", "procedures", "drugs")
        }
        self._observed_tokens = {
            source: {
                item["token"].lower()
                for item in self._profile.get("sources", {}).get(source, {}).get("top_tokens", [])
            }
            for source in ("conditions", "procedures", "drugs")
        }

    def _previous_signatures(self) -> set[tuple[Any, ...]]:
        signatures = set()
        if self.trace is None:
            return signatures
        for _, exp, _ in self.trace.hist:
            for task in getattr(exp, "sub_tasks", []):
                if hasattr(task, "to_factor_dict"):
                    factor = task.to_factor_dict()
                    signatures.add(self._signature(factor))
        return signatures

    def _signature(self, factor: dict[str, Any]) -> tuple[Any, ...]:
        temporal_part: tuple[Any, ...]
        if factor.get("operation") in TEMPORAL_OPS:
            temporal_part = (
                factor.get("window_start_hours", ""),
                factor.get("window_end_hours", ""),
                tuple(factor.get("early_window_hours", [])),
                tuple(factor.get("late_window_hours", [])),
                tuple(tuple(item) for item in factor.get("windows_hours", [])),
            )
        else:
            temporal_part = ()
        numeric_part: tuple[Any, ...]
        if factor.get("operation") in NUMERIC_TEMPORAL_OPS:
            numeric_part = (
                factor.get("numeric_source", ""),
                factor.get("secondary_numeric_source", ""),
                factor.get("aggregation", ""),
                factor.get("secondary_aggregation", ""),
                factor.get("operator", ""),
                factor.get("transform", ""),
                factor.get("window_start_hours", ""),
                factor.get("window_end_hours", ""),
                tuple(factor.get("early_window_hours", [])),
                tuple(factor.get("late_window_hours", [])),
                tuple(tuple(item) for item in factor.get("windows_hours", [])),
                factor.get("abnormal_low", ""),
                factor.get("abnormal_high", ""),
                factor.get("threshold", ""),
            )
        else:
            numeric_part = ()
        return (
            factor.get("operation"),
            tuple(factor.get("sources", [])),
            factor.get("numeric_source", ""),
            tuple(sorted(factor.get("keywords", []))),
            factor.get("numerator_source", ""),
            factor.get("denominator_source", ""),
            code_hash(factor.get("code", "")) if factor.get("operation") == SAFE_PYTHON_OPERATION else "",
            *temporal_part,
            *numeric_part,
        )

    def _keyword_coverage(self, factor: dict[str, Any]) -> tuple[list[str], list[str]]:
        covered = []
        uncovered = []
        for keyword in factor.get("keywords", []):
            keyword_l = str(keyword).lower()
            keyword_tokens = {tok for tok in keyword_l.replace("|", " ").replace("/", " ").split() if tok}
            found = False
            for source in factor.get("sources", []):
                values = self._observed_values.get(source, [])
                tokens = self._observed_tokens.get(source, set())
                if any(keyword_l in value for value in values) or keyword_tokens.intersection(tokens):
                    found = True
                    break
            if found:
                covered.append(keyword)
            else:
                uncovered.append(keyword)
        return covered, uncovered

    def _window_endpoint_violations(self, factor: dict[str, Any]) -> list[float]:
        horizon = observation_end_hours()
        if horizon is None:
            return []
        endpoints: list[float] = []
        op = factor.get("operation")
        check_keys = []
        check_window_lists = []
        if op in {
            "temporal_keyword_count",
            "temporal_keyword_density",
            "first_keyword_offset",
            "numeric_window_mean",
            "numeric_window_min",
            "numeric_window_max",
            "numeric_window_std",
            "numeric_window_last",
            "numeric_window_count",
            "numeric_window_slope",
            "numeric_source_interaction",
            "keyword_gated_numeric",
            "numeric_abnormal_fraction",
        }:
            check_keys.extend(("window_start_hours", "window_end_hours"))
        if op in {"early_late_keyword_delta", "numeric_early_late_delta"}:
            check_window_lists.extend(("early_window_hours", "late_window_hours"))
        if op in {"keyword_persistence", "numeric_persistence"}:
            check_window_lists.append("windows_hours")
        for key in check_keys:
            try:
                endpoints.append(float(factor.get(key, 0.0)))
            except (TypeError, ValueError):
                pass
        for key in ("early_window_hours", "late_window_hours"):
            if key not in check_window_lists:
                continue
            for value in factor.get(key, []) or []:
                try:
                    endpoints.append(float(value))
                except (TypeError, ValueError):
                    pass
        if "windows_hours" in check_window_lists:
            for window in factor.get("windows_hours", []) or []:
                for value in window:
                    try:
                        endpoints.append(float(value))
                    except (TypeError, ValueError):
                        pass
        return sorted(value for value in endpoints if value > horizon)

    def evaluate(self, payload: dict[str, Any]) -> _RegulatorResult:
        feedback_items = []
        try:
            factors = normalize_factors(payload)
        except Exception as exc:
            return _RegulatorResult(False, f"DSL validation failed: {exc}", [])

        min_factors = int(os.environ.get("MEDICAL_ALPHA_MIN_FACTORS", "3"))
        max_factors = int(os.environ.get("MEDICAL_ALPHA_MAX_FACTORS", "12"))
        if len(factors) < min_factors:
            feedback_items.append(
                f"Too few factors: {len(factors)} generated; generate at least {min_factors} "
                "independent factors unless explicitly disabled."
            )
        if len(factors) > max_factors:
            feedback_items.append(
                f"Too many factors: {len(factors)} generated; keep at most {max_factors}."
            )

        previous = self._previous_signatures()
        names = set()
        signatures = set()
        for factor in factors:
            name = factor["name"]
            signature = self._signature(factor)
            if name in names:
                feedback_items.append(f"Duplicate factor name in this batch: {name}")
            names.add(name)
            if signature in signatures or signature in previous:
                feedback_items.append(
                    f"Duplicate or previously evaluated factor structure: {name}"
                )
            signatures.add(signature)

            if factor["operation"] == SAFE_PYTHON_OPERATION:
                if not str_to_bool(os.environ.get("MEDICAL_SAFE_PYTHON_FACTORS", "0")):
                    feedback_items.append(
                        f"{name}: safe_python is disabled; set MEDICAL_SAFE_PYTHON_FACTORS=1 to allow it."
                    )
                try:
                    validate_safe_python_code(factor.get("code", ""))
                except SafePythonFactorError as exc:
                    feedback_items.append(f"{name}: safe_python validation failed: {exc}")

            if (
                factor["operation"].startswith("keyword")
                or factor["operation"] in TEMPORAL_OPS
                or factor["operation"] == "keyword_gated_numeric"
            ):
                covered, uncovered = self._keyword_coverage(factor)
                if not covered:
                    feedback_items.append(
                        f"{name}: none of its keywords appear in observed eICU vocabulary."
                    )
                elif uncovered:
                    feedback_items.append(
                        f"{name}: some keywords may be absent from observed vocabulary: {uncovered[:6]}"
                    )
                if len(factor["keywords"]) > 10:
                    feedback_items.append(
                        f"{name}: too many keywords; use a compact vocabulary-grounded set."
                    )

            max_definition_len = 2600 if factor["operation"] == SAFE_PYTHON_OPERATION else 900
            if len(json.dumps(factor, sort_keys=True)) > max_definition_len:
                feedback_items.append(
                    f"{name}: factor definition is too long; simplify it."
                )
            if factor["operation"] in TEMPORAL_OPS:
                violations = self._window_endpoint_violations(factor)
                if violations:
                    horizon = observation_end_hours()
                    feedback_items.append(
                        f"{name}: window endpoint(s) exceed the configured "
                        f"observation horizon {horizon:g}h: {violations[:6]}."
                    )
                if factor.get("window_end_hours", 24.0) > 336:
                    feedback_items.append(
                        f"{name}: temporal windows should usually stay within the first 14 ICU days."
                    )
                if factor["operation"] == "early_late_keyword_delta":
                    early = factor.get("early_window_hours", [0, 24])
                    late = factor.get("late_window_hours", [24, 72])
                    if early[1] > late[0]:
                        feedback_items.append(
                            f"{name}: early_window_hours should end before late_window_hours starts."
                        )
            if factor["operation"] in NUMERIC_TEMPORAL_OPS:
                allowed_numeric_sources = numeric_sources_for_profile()
                if not allowed_numeric_sources:
                    feedback_items.append(
                        f"{name}: numeric operations are disabled for "
                        f"MEDICAL_SOURCE_PROFILE={source_profile()}."
                    )
                violations = self._window_endpoint_violations(factor)
                if violations:
                    horizon = observation_end_hours()
                    feedback_items.append(
                        f"{name}: numeric window endpoint(s) exceed the configured "
                        f"observation horizon {horizon:g}h: {violations[:6]}."
                    )
                if factor.get("numeric_source", "") == "":
                    feedback_items.append(f"{name}: numeric_source is required.")
                elif allowed_numeric_sources and factor.get("numeric_source", "") not in allowed_numeric_sources:
                    feedback_items.append(
                        f"{name}: numeric_source is not available in "
                        f"MEDICAL_SOURCE_PROFILE={source_profile()}."
                    )
                if factor.get("transform", "identity") not in NUMERIC_TRANSFORMS:
                    feedback_items.append(
                        f"{name}: transform must be one of {sorted(NUMERIC_TRANSFORMS)}."
                    )
                if factor["operation"] == "keyword_gated_numeric" and (
                    not factor.get("sources") or not factor.get("keywords")
                ):
                    feedback_items.append(
                        f"{name}: keyword_gated_numeric needs sources and keywords for the gate."
                    )
                if factor["operation"] == "numeric_source_interaction":
                    if factor.get("secondary_numeric_source", "") == "":
                        feedback_items.append(
                            f"{name}: secondary_numeric_source is required for numeric_source_interaction."
                        )
                    elif allowed_numeric_sources and factor.get("secondary_numeric_source", "") not in allowed_numeric_sources:
                        feedback_items.append(
                            f"{name}: secondary_numeric_source is not available in "
                            f"MEDICAL_SOURCE_PROFILE={source_profile()}."
                        )
                    if factor.get("aggregation", "mean") not in NUMERIC_AGGREGATIONS:
                        feedback_items.append(
                            f"{name}: aggregation must be one of {sorted(NUMERIC_AGGREGATIONS)}."
                        )
                    if factor.get("secondary_aggregation", "mean") not in NUMERIC_AGGREGATIONS:
                        feedback_items.append(
                            f"{name}: secondary_aggregation must be one of {sorted(NUMERIC_AGGREGATIONS)}."
                        )
                    if factor.get("operator", "") not in NUMERIC_OPERATORS:
                        feedback_items.append(
                            f"{name}: operator must be one of {sorted(NUMERIC_OPERATORS)}."
                        )
                if factor.get("window_end_hours", 24.0) > 336:
                    feedback_items.append(
                        f"{name}: numeric temporal windows should usually stay within the first 14 ICU days."
                    )
                if factor["operation"] == "numeric_abnormal_fraction" and (
                    factor.get("abnormal_low") is None
                    and factor.get("abnormal_high") is None
                ):
                    feedback_items.append(
                        f"{name}: numeric_abnormal_fraction needs abnormal_low and/or abnormal_high."
                    )
                if factor["operation"] == "numeric_persistence" and (
                    factor.get("threshold") is None
                    and factor.get("abnormal_low") is None
                    and factor.get("abnormal_high") is None
                ):
                    feedback_items.append(
                        f"{name}: numeric_persistence needs threshold or abnormal bounds."
                    )

        return _RegulatorResult(
            passed=not feedback_items,
            feedback="\n".join(feedback_items) or "Medical DSL regulator passed.",
            factors=factors,
        )


class MedicalAlphaHypothesis2FactorExpression(Hypothesis2Experiment[MedicalFactorExperiment]):
    """AlphaAgent-style factor construction with regulator and LLM critic."""

    def __init__(self, *args, consistency_enabled: bool = False, **kwargs) -> None:
        self.consistency_enabled = consistency_enabled
        self.targets = "medical symbolic factors"
        self.embedding_rag = MedicalEmbeddingRAG()

    def _prepare_context(
        self,
        hypothesis: Hypothesis,
        trace: Trace,
        history_limit: int,
        regulator_feedback: str | None = None,
    ) -> dict[str, Any]:
        experiment_list = [item[1] for item in trace.hist]
        factor_list = []
        for experiment in experiment_list:
            for task in getattr(experiment, "sub_tasks", []):
                if hasattr(task, "to_factor_dict"):
                    factor_list.append(task.to_factor_dict())
        return {
            "target_hypothesis": str(hypothesis),
            "scenario": trace.scen.get_scenario_all_desc(),
            "hypothesis_and_feedback": _history_text(trace, history_limit),
            "retrieved_factor_knowledge": self.embedding_rag.query(trace, str(hypothesis)),
            "function_lib_description": _medical_function_lib_description(),
            "experiment_output_format": _factor_output_format(),
            "target_list": factor_list[-30:],
            "regulator_feedback": regulator_feedback,
            "primary_metric_phrase": _primary_metric_phrase(),
        }

    def _request_factors(self, context: dict[str, Any]) -> dict[str, Any]:
        system_prompt = f"""
The user is trying to generate new {self.targets} based on a hypothesis.
The scenario is:
{context["scenario"]}

You will receive:
1. The target hypothesis.
2. Former hypotheses, evaluations, feedback, factor activity, and baseline deltas.
3. Former proposed factors to avoid duplicating.
4. Medical DSL constraints and observed eICU vocabulary.
5. Optional regulator feedback from an invalid previous attempt.

Follow QuantaAlpha's factor-construction discipline:
- Generate a small independent batch of factors.
- If the target hypothesis explicitly asks for a clean-slate, exact-name,
  ablation, or full factor-list experiment, generate that complete factor set.
- Prioritize simplicity, novelty, and robustness.
- Avoid repeating former failed factors.
- Treat retrieved memories with Memory status=failed as negative examples:
  do not repeat their operation/source/window/keyword pattern unless the
  new factor explicitly fixes the listed error tags.
- Treat Memory status=success as reusable design evidence, but still avoid
  exact duplicate factors.
- Optimize for incremental {context["primary_metric_phrase"]}.
- Avoid zero-activity and ultra-rare terms.

Allowed function library:
{context["function_lib_description"]}

Output format:
{context["experiment_output_format"]}
"""
        user_prompt = f"""
Target hypothesis:
{context["target_hypothesis"]}

Former hypothesis and feedback:
{context["hypothesis_and_feedback"]}

Embedding-retrieved factor memories, including successful, mixed, and failed
factor attempts:
{context["retrieved_factor_knowledge"] or "None"}

Former proposed factors:
{json.dumps(context["target_list"], indent=2, sort_keys=True)}

Regulator feedback to fix, if any:
{context.get("regulator_feedback") or "None"}

Generate the new medical symbolic factors in JSON format.
"""
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=str_to_bool(os.environ.get("MEDICAL_ALPHA_REASONING", "1")),
            max_tokens=int(os.environ.get("MEDICAL_ALPHA_FACTOR_MAX_TOKENS", "3200")),
            temperature=float(os.environ.get("MEDICAL_ALPHA_TEMPERATURE", "0.4")),
        )
        return _safe_json_parse(response)

    def _critic_refine(
        self,
        payload: dict[str, Any],
        hypothesis: Hypothesis,
        trace: Trace,
        regulator_feedback: str,
    ) -> dict[str, Any]:
        if not str_to_bool(os.environ.get("MEDICAL_ALPHA_FACTOR_CRITIC", "1")):
            return payload
        system_prompt = f"""
You are the QuantaAlpha factor quality critic for medical symbolic factors.
Review the candidate factors before execution. Return JSON only.

Scenario:
{trace.scen.get_scenario_all_desc()}

Medical DSL:
{_medical_function_lib_description()}

You must check:
- consistency with the target hypothesis,
- observed eICU vocabulary grounding,
- redundancy with former factors,
- expected nonzero activity,
- expected incremental gain over baseline features,
- simplicity and interpretability.

Return the same factor JSON schema with corrected factors. Normally keep only 2-3
strong new factors. If the target hypothesis asks for a clean-slate, exact-name,
ablation, or full factor-list experiment, preserve the complete requested set.
"""
        user_prompt = f"""
Target hypothesis:
{hypothesis}

History:
{_history_text(trace, int(os.environ.get("MEDICAL_ALPHA_HISTORY_LIMIT", str(DEFAULT_HISTORY_LIMIT))))}

Candidate factors:
{json.dumps(payload, indent=2, sort_keys=True)}

Deterministic regulator feedback:
{regulator_feedback}

Refine the factor set. If the regulator passed, still improve vocabulary grounding and reduce redundancy.
"""
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=str_to_bool(os.environ.get("MEDICAL_ALPHA_REASONING", "1")),
            max_tokens=int(os.environ.get("MEDICAL_ALPHA_CRITIC_MAX_TOKENS", "3200")),
            temperature=float(os.environ.get("MEDICAL_ALPHA_CRITIC_TEMPERATURE", "0.2")),
        )
        return _safe_json_parse(response)

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> MedicalFactorExperiment:
        history_limit = int(os.environ.get("MEDICAL_ALPHA_HISTORY_LIMIT", str(DEFAULT_HISTORY_LIMIT)))
        max_attempts = int(os.environ.get("MEDICAL_ALPHA_REPAIR_ATTEMPTS", "2"))
        regulator = MedicalDSLRegulator(trace=trace)
        regulator_feedback = None
        last_payload: dict[str, Any] | None = None
        result = _RegulatorResult(False, "No attempt yet.", [])

        for attempt in range(max_attempts + 1):
            context = self._prepare_context(
                hypothesis,
                trace,
                history_limit,
                regulator_feedback=regulator_feedback,
            )
            payload = self._request_factors(context)
            last_payload = payload
            result = regulator.evaluate(payload)
            logger.info(f"Medical DSL regulator attempt {attempt}: {result.feedback}")
            if result.passed:
                critic_payload = self._critic_refine(
                    payload,
                    hypothesis,
                    trace,
                    result.feedback,
                )
                critic_result = regulator.evaluate(critic_payload)
                logger.info(f"Medical critic regulator result: {critic_result.feedback}")
                if critic_result.passed:
                    result = critic_result
                    last_payload = critic_payload
                break
            self.embedding_rag.ingest_regulator_failure(
                hypothesis=hypothesis,
                payload=payload,
                regulator_feedback=result.feedback,
            )
            regulator_feedback = result.feedback

        if not result.factors:
            self.embedding_rag.ingest_regulator_failure(
                hypothesis=hypothesis,
                payload=last_payload,
                regulator_feedback=result.feedback,
            )
            based_experiments = [item[1] for item in trace.hist if item[2]]
            if based_experiments and str_to_bool(
                os.environ.get("MEDICAL_ALPHA_EMPTY_BATCH_FALLBACK", "1")
            ):
                exp = MedicalFactorExperiment([])
                exp.target_hypothesis = str(hypothesis)
                exp.based_experiments = based_experiments
                logger.warning(
                    "Medical alpha factor construction returned no factors; "
                    "falling back to previously accepted factor pool. "
                    f"Last payload={last_payload}; regulator={result.feedback}"
                )
                return exp
            raise ValueError(
                "Medical alpha factor construction failed. "
                f"Last payload={last_payload}; regulator={result.feedback}"
            )

        tasks = []
        for factor in result.factors:
            tasks.append(
                MedicalFactorTask(
                    factor_name=factor["name"],
                    factor_description=factor["description"],
                    factor_formulation=_factor_formula_text(factor),
                    operation=factor["operation"],
                    sources=factor["sources"],
                    numeric_source=factor.get("numeric_source", ""),
                    secondary_numeric_source=factor.get("secondary_numeric_source", ""),
                    aggregation=factor.get("aggregation", "mean"),
                    secondary_aggregation=factor.get("secondary_aggregation", "mean"),
                    operator=factor.get("operator", ""),
                    transform=factor.get("transform", "identity"),
                    keywords=factor["keywords"],
                    rationale=factor["rationale"],
                    numerator_source=factor["numerator_source"],
                    denominator_source=factor["denominator_source"],
                    window_start_hours=factor.get("window_start_hours", 0.0),
                    window_end_hours=factor.get("window_end_hours", 24.0),
                    early_window_hours=factor.get("early_window_hours", [0.0, 24.0]),
                    late_window_hours=factor.get("late_window_hours", [24.0, 72.0]),
                    windows_hours=factor.get(
                        "windows_hours",
                        [[0.0, 24.0], [24.0, 72.0], [72.0, 168.0]],
                    ),
                    abnormal_low=factor.get("abnormal_low"),
                    abnormal_high=factor.get("abnormal_high"),
                    threshold=factor.get("threshold"),
                    code=factor.get("code", ""),
                    raw_factor=factor,
                )
            )
        exp = MedicalFactorExperiment(tasks)
        exp.target_hypothesis = str(hypothesis)
        exp.based_experiments = [
            item[1] for item in trace.hist if item[2]
        ]
        logger.info(f"Constructed {len(tasks)} alpha-style medical factor tasks")
        return exp
