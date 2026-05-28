"""LLM proposal components for the medical QuantaAlpha workflow."""

from __future__ import annotations

import json
import os
from typing import Any

from quantaalpha.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Trace,
)
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.log import logger
from quantaalpha.medical.dsl import normalize_factors
from quantaalpha.medical.experiment import MedicalFactorExperiment, MedicalFactorTask


def _history_text(trace: Trace, limit: int = 4) -> str:
    if not trace.hist:
        return "No previous medical factor rounds have been evaluated."
    blocks = []
    for idx, (hyp, exp, feedback) in enumerate(trace.hist[-limit:], start=1):
        metrics = getattr(exp, "result", {}) or {}
        baseline_deltas = metrics.get("baseline_deltas", {})
        factor_summary = metrics.get("factor_summary", [])
        compact_activity = [
            {
                "name": item.get("name"),
                "nonzero_rate": item.get("nonzero_rate"),
                "std": item.get("std"),
            }
            for item in factor_summary[:12]
        ]
        blocks.append(
            f"Round {idx}\n"
            f"Hypothesis: {getattr(hyp, 'hypothesis', str(hyp))}\n"
            f"Metrics: {json.dumps(metrics.get('scores', metrics), sort_keys=True)}\n"
            f"Baseline deltas: {json.dumps(baseline_deltas, sort_keys=True)}\n"
            f"Factor activity: {json.dumps(compact_activity, sort_keys=True)}\n"
            f"Feedback: {feedback}"
        )
    return "\n\n".join(blocks)


class MedicalHypothesisGen(HypothesisGen):
    def __init__(self, scen, potential_direction: str | None = None) -> None:
        super().__init__(scen)
        self.potential_direction = potential_direction or (
            "Find clinically meaningful factors for the target eICU prediction task."
        )

    def gen(self, trace: Trace) -> Hypothesis:
        system_prompt = (
            "You are a clinical informatics researcher. Propose one concise, "
            "testable hypothesis for symbolic factors that may improve the "
            "target eICU prediction task. Return only JSON."
        )
        user_prompt = f"""
Scenario:
{self.scen.get_scenario_all_desc()}

User direction:
{self.potential_direction}

Previous rounds:
{_history_text(trace)}

Important constraints:
- Prefer keyword factors whose keywords appear in the observed eICU vocabulary
  profile above.
- Avoid zero-activity or extremely rare keywords unless they are paired with
  broader observed lexical variants.
- Propose factors expected to have useful nonzero rates, roughly 1% to 50% for
  binary/specific keyword factors, while still remaining clinically meaningful.

Return JSON:
{{
  "hypothesis": "...",
  "reason": "...",
  "concise_reason": "...",
  "concise_observation": "...",
  "concise_justification": "...",
  "concise_knowledge": "..."
}}
"""
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=False,
            max_retry=2,
            max_tokens=int(os.environ.get("MEDICAL_HYPOTHESIS_MAX_TOKENS", "1200")),
            temperature=float(os.environ.get("MEDICAL_HYPOTHESIS_TEMPERATURE", "0.2")),
        )
        data = robust_json_parse(response)
        return Hypothesis(
            hypothesis=data.get("hypothesis", ""),
            reason=data.get("reason", ""),
            concise_reason=data.get("concise_reason", ""),
            concise_observation=data.get("concise_observation", ""),
            concise_justification=data.get("concise_justification", ""),
            concise_knowledge=data.get("concise_knowledge", ""),
        )


class MedicalHypothesis2FactorExpression(Hypothesis2Experiment[MedicalFactorExperiment]):
    def __init__(self, *args, consistency_enabled: bool = False, **kwargs) -> None:
        self.consistency_enabled = consistency_enabled

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> MedicalFactorExperiment:
        system_prompt = (
            "You are designing safe, interpretable clinical symbolic factors. "
            "Return only valid JSON. Do not write Python code."
        )
        user_prompt = f"""
Scenario:
{trace.scen.get_scenario_all_desc()}

Target hypothesis:
{hypothesis}

Previous rounds:
{_history_text(trace)}

Use only this DSL:
- log_count: needs sources = one of ["conditions", "procedures", "drugs"]
- count_ratio: needs numerator_source and denominator_source
- keyword_any: needs sources and keywords
- keyword_count: needs sources and keywords
- keyword_density: needs sources and keywords
- temporal_keyword_count: needs sources, keywords, window_start_hours, window_end_hours
- temporal_keyword_density: same temporal window, but hits / window event count
- first_keyword_offset: first matching event offset in hours, log-scaled by evaluator
- early_late_keyword_delta: needs early_window_hours and late_window_hours
- keyword_persistence: needs windows_hours such as [[0,24],[24,72],[72,168]]
- numeric_window_mean/min/max: needs numeric_source plus window_start_hours/window_end_hours
- numeric_window_std/last/count/slope: safe numeric window statistics
- numeric_early_late_delta: needs numeric_source plus early_window_hours/late_window_hours
- numeric_abnormal_fraction: needs numeric_source plus abnormal_low and/or abnormal_high
- numeric_persistence: needs numeric_source, windows_hours, and threshold or abnormal bounds
- numeric_source_interaction: combines two numeric sources with whitelisted
  aggregations (mean/min/max/std/last/count/slope), operators (add/sub/mul/ratio/max/min),
  and optional transform (identity/log1p/abs/neg/sqrt_abs)
- keyword_gated_numeric: returns a numeric window statistic only when source
  keywords are present in the same window

Return exactly this JSON shape:
{{
  "factors": [
    {{
      "name": "short_snake_case",
      "description": "human-readable factor description",
      "rationale": "why it might relate to the target eICU prediction task",
      "operation": "keyword_any",
      "sources": ["conditions", "drugs"],
      "numeric_source": "",
      "secondary_numeric_source": "",
      "aggregation": "mean",
      "secondary_aggregation": "mean",
      "operator": "",
      "transform": "identity",
      "keywords": ["sepsis", "infection"],
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
    }}
  ]
}}

Generate 6 to 10 diverse factors. Avoid repeating factors already evaluated.
Prefer temporal symbolic factors when onset, early severity, persistence, or
late escalation is clinically meaningful.
Prefer clinically meaningful concepts such as infection burden, respiratory
failure, shock/vasopressors, renal failure, cardiac disease, neurologic disease,
metabolic derangement, medication burden, and procedure/exam complexity.
Use the observed eICU vocabulary profile in the scenario. Keywords must be
lexical variants that plausibly occur in eICU strings; avoid zero-match terms
from previous rounds and avoid ultra-rare device/procedure phrases unless the
profile shows they occur.
"""
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=False,
            max_retry=2,
            max_tokens=int(os.environ.get("MEDICAL_FACTOR_MAX_TOKENS", "2400")),
            temperature=float(os.environ.get("MEDICAL_FACTOR_TEMPERATURE", "0.2")),
        )
        payload = robust_json_parse(response)
        factors = normalize_factors(payload)
        tasks = []
        for factor in factors:
            formulation = _factor_formula_text(factor)
            tasks.append(
                MedicalFactorTask(
                    factor_name=factor["name"],
                    factor_description=factor["description"],
                    factor_formulation=formulation,
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
        logger.info(f"Constructed {len(tasks)} medical factor tasks")
        return exp


def _factor_formula_text(factor: dict[str, Any]) -> str:
    op = factor["operation"]
    if op == "safe_python":
        return factor.get("code", "").strip()[:500]
    if op == "log_count":
        return f"log(1 + count({factor['sources'][0]}))"
    if op == "count_ratio":
        return (
            f"count({factor['numerator_source']}) / "
            f"max(count({factor['denominator_source']}), 1)"
        )
    source = " U ".join(factor["sources"])
    keywords = "/".join(factor["keywords"])
    if op == "keyword_any":
        return f"1[contains({source}, {keywords})]"
    if op == "keyword_count":
        return f"count_items_containing({source}, {keywords})"
    if op == "keyword_density":
        return f"count_items_containing({source}, {keywords}) / max(count({source}), 1)"
    if op == "temporal_keyword_count":
        return (
            f"count_items_containing({source}, {keywords}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}))"
        )
    if op == "temporal_keyword_density":
        return (
            f"density_items_containing({source}, {keywords}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}))"
        )
    if op == "first_keyword_offset":
        return f"log1p(first_offset_hours({source}, {keywords}))"
    if op == "early_late_keyword_delta":
        return (
            f"density({factor.get('late_window_hours', [24, 72])}) - "
            f"density({factor.get('early_window_hours', [0, 24])}) for {keywords}"
        )
    if op == "keyword_persistence":
        return f"fraction_active_windows({source}, {keywords}, {factor.get('windows_hours')})"
    if op == "numeric_window_mean":
        return (
            f"mean({factor.get('numeric_source')}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}])"
        )
    if op == "numeric_window_min":
        return (
            f"min({factor.get('numeric_source')}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}])"
        )
    if op == "numeric_window_max":
        return (
            f"max({factor.get('numeric_source')}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}])"
        )
    if op in {"numeric_window_std", "numeric_window_last", "numeric_window_count", "numeric_window_slope"}:
        stat = op.replace("numeric_window_", "")
        return (
            f"{stat}({factor.get('numeric_source')}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}], "
            f"transform={factor.get('transform', 'identity')})"
        )
    if op == "numeric_source_interaction":
        return (
            f"{factor.get('operator')}("
            f"{factor.get('aggregation', 'mean')}({factor.get('numeric_source')}), "
            f"{factor.get('secondary_aggregation', 'mean')}({factor.get('secondary_numeric_source')}), "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}], "
            f"transform={factor.get('transform', 'identity')})"
        )
    if op == "keyword_gated_numeric":
        return (
            f"if temporal_contains({' U '.join(factor.get('sources', []))}, "
            f"{'/'.join(factor.get('keywords', []))}, "
            f"hours=[{factor.get('window_start_hours', 0)}, {factor.get('window_end_hours', 24)}]) "
            f"then {factor.get('aggregation', 'mean')}({factor.get('numeric_source')}) else 0"
        )
    if op == "numeric_early_late_delta":
        return (
            f"mean({factor.get('numeric_source')}, {factor.get('late_window_hours', [24, 72])}) - "
            f"mean({factor.get('numeric_source')}, {factor.get('early_window_hours', [0, 24])})"
        )
    if op == "numeric_abnormal_fraction":
        return (
            f"abnormal_fraction({factor.get('numeric_source')}, "
            f"low={factor.get('abnormal_low')}, high={factor.get('abnormal_high')})"
        )
    if op == "numeric_persistence":
        return (
            f"numeric_persistence({factor.get('numeric_source')}, "
            f"windows={factor.get('windows_hours')}, threshold={factor.get('threshold')})"
        )
    return op
