"""PyHealth/eICU scenario description for QuantaAlpha."""

from __future__ import annotations

from quantaalpha.core.scenario import Scenario
from quantaalpha.medical.source_profile import (
    EXPANDED_V2,
    PYHEALTH_STANDARD,
    numeric_sources_text,
    source_profile,
)
from quantaalpha.medical.task_config import get_task_config, observation_end_hours
from quantaalpha.medical.vocab_profile import vocab_profile_prompt


class PyHealthEICULOSScenario(Scenario):
    def __init__(self, use_local: bool = True) -> None:
        self.use_local = use_local

    @property
    def task_name(self) -> str:
        return get_task_config().name

    @property
    def task_description(self) -> str:
        return get_task_config().description

    @property
    def background(self) -> str:
        return (
            "We are mining clinically interpretable symbolic factors for ICU "
            f"{self.task_description} using PyHealth-style tabular evaluation."
        )

    def get_source_data_desc(self, task=None) -> str:
        horizon = observation_end_hours(self.task_name)
        horizon_text = (
            f" For this task, factors must stay within the configured observation "
            f"horizon of 0-{horizon:g} hours after ICU admission."
            if horizon is not None
            else ""
        )
        profile = source_profile()
        if profile == PYHEALTH_STANDARD:
            if self.task_name == "los":
                field_text = (
                    "conditions=diagnosis.diagnosisstring, "
                    "procedures=physicalExam.physicalexamvalue, drugs=medication.drugname"
                )
            else:
                field_text = (
                    "conditions=diagnosis.icd9code, "
                    "procedures=physicalExam.physicalexampath, drugs=medication.drugname"
                )
            desc = (
                "MEDICAL_SOURCE_PROFILE=pyhealth_standard. Available current-stay "
                "features match the PyHealth eICU task source tables: diagnosis, "
                f"physicalExam, and medication only ({field_text}). The tabular "
                "runner also exposes offsets from these same tables, so safe "
                "temporal text factors may use early, late, persistence, and "
                "first-occurrence timing within the current ICU stay. Numeric "
                "temporal sources are disabled in this profile. Do not use future "
                "information, discharge time, mortality/readmission/LOS labels, or "
                "other target leakage when defining factors. For binary "
                "mortality/readmission tasks, prefer early windows such as 0-24h "
                f"or 0-48h unless explicitly allowed later.{horizon_text}"
            )
        elif profile == EXPANDED_V2:
            desc = (
                "MEDICAL_SOURCE_PROFILE=expanded_v2. Available current-stay features "
                "are conditions (diagnosis strings or codes), procedures (physical "
                "exam/procedure/intervention strings or codes), and drugs "
                "(medication strings or codes). These text sources merge "
                "diagnosis/admissionDx, physicalExam/treatment/respiratoryCare, "
                "and medication/infusionDrug/admissionDrug. The tabular runner "
                "also exposes eICU current-stay event offsets in minutes for these "
                "events. Numeric temporal sources are also available from "
                "vitalPeriodic, vitalAperiodic, intakeOutput, lab, nurseCharting, "
                "infusionDrug, selected respiratoryCharting rows, and APACHE APS "
                f"physiology variables: {numeric_sources_text(profile)}. Do not use "
                "future information, discharge time, mortality/readmission/LOS "
                "labels, or other target leakage when defining factors. For binary "
                "mortality/readmission tasks, prefer early windows such as 0-24h "
                f"or 0-48h unless explicitly allowed later.{horizon_text}"
            )
        else:
            raise ValueError(f"Unsupported source profile: {profile}")
        profile = vocab_profile_prompt()
        if profile:
            desc = f"{desc}\n\n{profile}"
        return desc

    @property
    def interface(self) -> str:
        return (
            "Factors must be expressed in a safe JSON DSL. Valid operations are "
            "log_count, count_ratio, keyword_any, keyword_count, and "
            "keyword_density plus temporal_keyword_count, temporal_keyword_density, "
            "first_keyword_offset, early_late_keyword_delta, and keyword_persistence. "
            "Numeric temporal operations are numeric_window_mean, numeric_window_min, "
            "numeric_window_max, numeric_window_std, numeric_window_last, "
            "numeric_window_count, numeric_window_slope, numeric_early_late_delta, "
            "numeric_abnormal_fraction, numeric_persistence, and "
            "numeric_source_interaction. The keyword_gated_numeric op returns a "
            "numeric statistic only when source keywords are present in the same "
            "window. The safe_python op is available only when explicitly enabled "
            "and must define a restricted compute(sample) function over current "
            "conditions/procedures/drugs and offsets. The interaction op combines two numeric sources with "
            "whitelisted aggregations/operators/transforms only. "
            "Valid text sources are conditions, procedures, drugs. Numeric sources "
            "are available only when listed in the source data description."
        )

    @property
    def output_format(self) -> str:
        return (
            "Return JSON with a `factors` list. Each factor has name, "
            "description, rationale, operation, sources, keywords, and optional "
            "numerator_source/denominator_source. Temporal factors may also include "
            "window_start_hours/window_end_hours, early_window_hours/late_window_hours, "
            "or windows_hours. Numeric factors should include numeric_source and may "
            "include abnormal_low, abnormal_high, threshold, transform, or for "
            "numeric_source_interaction secondary_numeric_source, aggregation, "
            "secondary_aggregation, and operator. Gated numeric factors should "
            "include sources, keywords, numeric_source, aggregation, and window "
            "bounds. Safe Python factors should set operation=safe_python and put "
            "the complete restricted compute(sample) function source in code."
        )

    @property
    def simulator(self) -> str:
        return (
            "The PyHealth runner materializes factors into numeric tensor "
            "features and trains a GRU-based PyHealth model. The tabular runner "
            "materializes an explicit factor table, evaluates baseline, "
            "factors-only, and baseline+factors feature sets with lightweight "
            "models, and reports accuracy, balanced_accuracy, f1_macro, "
            "f1_weighted, cohen_kappa, and loss. Binary tasks also report AUROC "
            "and PRAUC."
        )

    @property
    def rich_style_description(self) -> str:
        return self.get_scenario_all_desc()

    @property
    def experiment_setting(self) -> str:
        return get_task_config(self.task_name).experiment_setting

    def get_scenario_all_desc(
        self,
        task=None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        return "\n\n".join(
            [
                f"Background: {self.background}",
                f"Source data: {self.source_data}",
                f"Interface: {self.interface}",
                f"Output format: {self.output_format}",
                f"Simulator: {self.simulator}",
                f"Experiment setting: {self.experiment_setting}",
            ]
        )
