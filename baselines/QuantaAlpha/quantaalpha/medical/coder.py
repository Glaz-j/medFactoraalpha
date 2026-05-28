"""Medical factor coder/parser step."""

from __future__ import annotations

from quantaalpha.core.developer import Developer
from quantaalpha.medical.dsl import normalize_factors
from quantaalpha.medical.experiment import (
    MedicalFactorExperiment,
    MedicalFactorWorkspace,
)


class MedicalFactorParser(Developer[MedicalFactorExperiment]):
    """Validate DSL factors and attach lightweight workspaces."""

    def develop(self, exp: MedicalFactorExperiment) -> MedicalFactorExperiment:
        factors = normalize_factors({"factors": [task.to_factor_dict() for task in exp.sub_tasks]})
        for task, factor in zip(exp.sub_tasks, factors, strict=True):
            task.raw_factor = factor
        exp.sub_workspace_list = [
            MedicalFactorWorkspace(target_task=task) for task in exp.sub_tasks
        ]
        return exp

