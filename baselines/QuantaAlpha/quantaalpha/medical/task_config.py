"""Shared task configuration for eICU medical factor workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MedicalTaskConfig:
    name: str
    slug: str
    label_key: str
    labels: np.ndarray
    primary_metric: str
    report_metrics: tuple[str, ...]
    best_metrics: tuple[str, ...]
    metric_direction: dict[str, str]
    default_observation_end_hours: float | None
    description: str
    experiment_setting: str
    metric_guidance: str


METRIC_DIRECTION = {
    "accuracy": "max",
    "auroc": "max",
    "balanced_accuracy": "max",
    "cohen_kappa": "max",
    "f1_macro": "max",
    "f1_weighted": "max",
    "loss": "min",
    "prauc": "max",
}


TASK_ALIASES = {
    "los": "los",
    "length_of_stay": "los",
    "length-of-stay": "los",
    "mortality": "mortality",
    "hospital_mortality": "mortality",
    "icu_mortality": "mortality",
    "readmission": "readmission",
}


TASK_CONFIGS = {
    "los": MedicalTaskConfig(
        name="los",
        slug="eicu_los",
        label_key="los",
        labels=np.arange(10, dtype=np.int64),
        primary_metric="f1_macro",
        report_metrics=("accuracy", "f1_macro", "loss"),
        best_metrics=("f1_macro", "balanced_accuracy", "loss"),
        metric_direction=METRIC_DIRECTION,
        default_observation_end_hours=None,
        description=(
            "eICU ICU length-of-stay prediction, a multiclass prediction of the "
            "current ICU stay LOS bucket"
        ),
        experiment_setting=(
            "Dataset: eICU; Task: LengthOfStayPredictioneICU; primary metric: f1_macro."
        ),
        metric_guidance=(
            "Judge success by the report metric set: improve f1_macro and accuracy "
            "where possible while reducing loss. Treat f1_macro as the primary "
            "ranking metric, but do not ignore a large loss regression."
        ),
    ),
    "mortality": MedicalTaskConfig(
        name="mortality",
        slug="eicu_mortality",
        label_key="mortality",
        labels=np.asarray([0, 1], dtype=np.int64),
        primary_metric="auroc",
        report_metrics=("auroc", "prauc", "f1_macro", "accuracy", "loss"),
        best_metrics=("auroc", "prauc", "f1_macro", "loss"),
        metric_direction=METRIC_DIRECTION,
        default_observation_end_hours=48.0,
        description=(
            "eICU mortality prediction, a binary prediction of whether the next "
            "hospital visit/stay discharge status is expired"
        ),
        experiment_setting=(
            "Dataset: eICU; Task: MortalityPredictionEICU; primary metrics: AUROC, PRAUC, f1_macro."
        ),
        metric_guidance=(
            "Judge success by the report metric set: improve AUROC, PRAUC, "
            "f1_macro, and accuracy where possible while reducing loss. Treat "
            "AUROC as the primary ranking metric, but prefer Pareto-style gains "
            "over baseline_logistic and avoid factors that trade away PRAUC or "
            "loss for a tiny AUROC change."
        ),
    ),
    "readmission": MedicalTaskConfig(
        name="readmission",
        slug="eicu_readmission",
        label_key="readmission",
        labels=np.asarray([0, 1], dtype=np.int64),
        primary_metric="auroc",
        report_metrics=("auroc", "prauc", "f1_macro", "accuracy", "loss"),
        best_metrics=("auroc", "prauc", "f1_macro", "loss"),
        metric_direction=METRIC_DIRECTION,
        default_observation_end_hours=48.0,
        description=(
            "eICU readmission prediction, a binary prediction of whether the next "
            "observed ICU stay belongs to the same hospital system stay"
        ),
        experiment_setting=(
            "Dataset: eICU; Task: readmission; primary metrics: AUROC, PRAUC, f1_macro."
        ),
        metric_guidance=(
            "Judge success by the report metric set: improve AUROC, PRAUC, "
            "f1_macro, and accuracy where possible while reducing loss. Treat "
            "AUROC as the primary ranking metric, but prefer Pareto-style gains "
            "over baseline_logistic and avoid factors that trade away PRAUC or "
            "loss for a tiny AUROC change."
        ),
    ),
}


def get_task_name(value: str | None = None) -> str:
    task = (value or os.environ.get("MEDICAL_EICU_TASK", "los")).strip().lower()
    try:
        return TASK_ALIASES[task]
    except KeyError as exc:
        raise ValueError(
            "MEDICAL_EICU_TASK must be one of: los, mortality, readmission"
        ) from exc


def get_task_config(value: str | None = None) -> MedicalTaskConfig:
    return TASK_CONFIGS[get_task_name(value)]


def observation_end_hours(task: str | None = None) -> float | None:
    config = get_task_config(task)
    raw = os.environ.get("MEDICAL_OBSERVATION_END_HOURS")
    if raw is None or raw.strip() == "":
        return config.default_observation_end_hours
    value = float(raw)
    return value if value > 0 else None
