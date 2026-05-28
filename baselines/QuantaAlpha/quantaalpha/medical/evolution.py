"""Medical adaptation of QuantaAlpha's trajectory evolution controller."""

from __future__ import annotations

from typing import Any

from quantaalpha.pipeline.evolution.controller import EvolutionConfig, EvolutionController
from quantaalpha.pipeline.evolution.trajectory import RoundPhase, StrategyTrajectory
from quantaalpha.medical.task_config import get_task_config


class MedicalEvolutionController(EvolutionController):
    """Use the original controller with medical metric extraction."""

    def _extract_metrics(self, result: Any) -> dict[str, float | None]:
        task_config = get_task_config((result or {}).get("task") if isinstance(result, dict) else None)
        metrics: dict[str, float | None] = {
            "IC": None,
            "ICIR": None,
            "RankIC": None,
            "RankICIR": None,
            "annualized_return": None,
            "information_ratio": None,
            "max_drawdown": None,
        }
        if not isinstance(result, dict):
            return metrics
        scores = result.get("scores", {}) or {}
        for metric in task_config.report_metrics:
            value = scores.get(metric)
            if isinstance(value, (int, float)):
                metrics[metric] = float(value)
        primary = task_config.primary_metric
        primary_value = scores.get(primary)
        if isinstance(primary_value, (int, float)):
            # Original trajectory sorting uses RankIC as primary. Map the task
            # primary metric there so parent selection remains unchanged.
            mapped = float(primary_value)
            if task_config.metric_direction.get(primary) == "min":
                mapped = -mapped
            metrics["RankIC"] = mapped
            metrics["medical_primary"] = mapped
        return metrics

    def create_trajectory_from_loop_result(
        self,
        task: dict[str, Any],
        hypothesis: Any,
        experiment: Any,
        feedback: Any,
    ) -> StrategyTrajectory:
        trajectory = super().create_trajectory_from_loop_result(
            task,
            hypothesis,
            experiment,
            feedback,
        )
        result = getattr(experiment, "result", {}) or {}
        trajectory.extra_info.update(
            {
                "medical_task": result.get("task"),
                "primary_metric": result.get("primary_metric"),
                "primary_model": result.get("primary_model"),
                "factor_hash": result.get("factor_hash"),
                "evaluated_factor_count": result.get("evaluated_factor_count"),
                "sample_size": result.get("sample_size"),
                "scores": result.get("scores", {}),
                "best_summary_paths": result.get("best_summary_paths", {}),
            }
        )
        return trajectory


def default_medical_direction_suffixes(task_name: str) -> list[str]:
    if task_name == "readmission":
        return [
            "Original direction A: explore respiratory instability, FiO2/PEEP support, and oxygenation burden within 0-48h.",
            "Original direction B: explore renal dysfunction, urine/output balance, dialysis, and fluid-balance instability within 0-48h.",
            "Original direction C: explore hemodynamic shock, MAP hypotension, tachycardia, sepsis/infection burden, and persistent instability within 0-48h.",
        ]
    if task_name == "mortality":
        return [
            "Original direction A: explore early shock, MAP hypotension, tachycardia, and vasopressor/procedure severity within 0-48h.",
            "Original direction B: explore respiratory failure, FiO2/PEEP support, hypoxemia, and ventilation intensity within 0-48h.",
            "Original direction C: explore renal failure, dialysis, fluid balance, low output, and multi-organ dysfunction within 0-48h.",
        ]
    return [
        "Original direction A: explore respiratory and oxygenation burden.",
        "Original direction B: explore renal/fluid burden.",
        "Original direction C: explore hemodynamic and infection burden.",
    ]


__all__ = [
    "EvolutionConfig",
    "MedicalEvolutionController",
    "RoundPhase",
    "default_medical_direction_suffixes",
]
