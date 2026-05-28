"""Medical factor task and experiment objects."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from quantaalpha.core.experiment import Experiment, Task, Workspace


class MedicalFactorTask(Task):
    """A single safe medical symbolic factor."""

    def __init__(
        self,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        operation: str,
        sources: list[str] | None = None,
        numeric_source: str = "",
        secondary_numeric_source: str = "",
        aggregation: str = "mean",
        secondary_aggregation: str = "mean",
        operator: str = "",
        transform: str = "identity",
        keywords: list[str] | None = None,
        rationale: str = "",
        numerator_source: str = "",
        denominator_source: str = "",
        window_start_hours: float = 0.0,
        window_end_hours: float = 24.0,
        early_window_hours: list[float] | None = None,
        late_window_hours: list[float] | None = None,
        windows_hours: list[list[float]] | None = None,
        abnormal_low: float | None = None,
        abnormal_high: float | None = None,
        threshold: float | None = None,
        code: str = "",
        raw_factor: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=factor_name)
        self.factor_name = factor_name
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_expression = factor_formulation
        self.operation = operation
        self.sources = sources or []
        self.numeric_source = numeric_source
        self.secondary_numeric_source = secondary_numeric_source
        self.aggregation = aggregation
        self.secondary_aggregation = secondary_aggregation
        self.operator = operator
        self.transform = transform
        self.keywords = keywords or []
        self.rationale = rationale
        self.numerator_source = numerator_source
        self.denominator_source = denominator_source
        self.window_start_hours = window_start_hours
        self.window_end_hours = window_end_hours
        self.early_window_hours = early_window_hours or [0.0, 24.0]
        self.late_window_hours = late_window_hours or [24.0, 72.0]
        self.windows_hours = windows_hours or [[0.0, 24.0], [24.0, 72.0], [72.0, 168.0]]
        self.abnormal_low = abnormal_low
        self.abnormal_high = abnormal_high
        self.threshold = threshold
        self.code = code
        self.raw_factor = raw_factor or {}

    def to_factor_dict(self) -> dict[str, Any]:
        return {
            "name": self.factor_name,
            "description": self.factor_description,
            "rationale": self.rationale,
            "formulation": self.factor_formulation,
            "operation": self.operation,
            "sources": self.sources,
            "numeric_source": self.numeric_source,
            "secondary_numeric_source": getattr(self, "secondary_numeric_source", ""),
            "aggregation": getattr(self, "aggregation", "mean"),
            "secondary_aggregation": getattr(self, "secondary_aggregation", "mean"),
            "operator": getattr(self, "operator", ""),
            "transform": getattr(self, "transform", "identity"),
            "keywords": self.keywords,
            "numerator_source": self.numerator_source,
            "denominator_source": self.denominator_source,
            "window_start_hours": self.window_start_hours,
            "window_end_hours": self.window_end_hours,
            "early_window_hours": self.early_window_hours,
            "late_window_hours": self.late_window_hours,
            "windows_hours": self.windows_hours,
            "abnormal_low": self.abnormal_low,
            "abnormal_high": self.abnormal_high,
            "threshold": self.threshold,
            "code": self.code,
        }

    def get_task_information(self) -> str:
        return (
            f"factor_name: {self.factor_name}\n"
            f"description: {self.factor_description}\n"
            f"formulation: {self.factor_formulation}\n"
            f"operation: {self.operation}\n"
            f"sources: {self.sources}\n"
            f"numeric_source: {self.numeric_source}\n"
            f"secondary_numeric_source: {getattr(self, 'secondary_numeric_source', '')}\n"
            f"aggregation: {getattr(self, 'aggregation', 'mean')}\n"
            f"secondary_aggregation: {getattr(self, 'secondary_aggregation', 'mean')}\n"
            f"operator: {getattr(self, 'operator', '')}\n"
            f"transform: {getattr(self, 'transform', 'identity')}\n"
            f"keywords: {self.keywords}\n"
            f"rationale: {self.rationale}\n"
            f"window_start_hours: {self.window_start_hours}\n"
            f"window_end_hours: {self.window_end_hours}\n"
            f"early_window_hours: {self.early_window_hours}\n"
            f"late_window_hours: {self.late_window_hours}\n"
            f"windows_hours: {self.windows_hours}\n"
            f"abnormal_low: {self.abnormal_low}\n"
            f"abnormal_high: {self.abnormal_high}\n"
            f"threshold: {self.threshold}\n"
            f"code: {self.code}"
        )

    def __repr__(self) -> str:
        return f"<MedicalFactorTask[{self.factor_name}]>"


class MedicalFactorWorkspace(Workspace[MedicalFactorTask]):
    """Lightweight workspace holding a validated factor definition."""

    def __init__(self, target_task: MedicalFactorTask | None = None) -> None:
        super().__init__(target_task=target_task)
        self.factor = target_task.to_factor_dict() if target_task else {}

    def execute(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return self.factor

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        name = self.target_task.factor_name if self.target_task else "empty"
        return f"<MedicalFactorWorkspace[{name}]>"


class MedicalFactorExperiment(Experiment[MedicalFactorTask, MedicalFactorWorkspace, MedicalFactorWorkspace]):
    """A group of candidate medical symbolic factors plus PyHealth results."""

    def factor_dicts(self) -> list[dict[str, Any]]:
        return [task.to_factor_dict() for task in self.sub_tasks]
