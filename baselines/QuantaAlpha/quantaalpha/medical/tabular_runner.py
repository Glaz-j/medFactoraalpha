"""Qlib-like tabular backtest runner for medical factor experiments.

This runner evaluates LLM-discovered medical factors as an explicit factor
table, closer to QuantaAlpha's original Qlib style than the PyHealth GRU
runner. It trains lightweight supervised models on:

- baseline count/ratio features,
- symbolic factors only,
- baseline + symbolic factors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import litdata
import numpy as np
import polars as pl
import torch

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import LengthOfStayPredictioneICU
from quantaalpha.core.developer import Developer
from quantaalpha.medical.dsl import (
    DEFAULT_TEMPORAL_WINDOWS,
    NUMERIC_TEMPORAL_OPS,
    TEMPORAL_OPS,
    VALID_NUMERIC_SOURCES,
    apply_numeric_operator,
    apply_numeric_transform,
    compute_factor,
    flatten_text,
)
from quantaalpha.medical.experiment import MedicalFactorExperiment
from quantaalpha.medical.safe_python import SAFE_PYTHON_OPERATION, code_hash, compute_safe_python_factor
from quantaalpha.medical.source_profile import (
    EXPANDED_V2,
    PYHEALTH_STANDARD,
    numeric_sources_for_profile,
    source_profile,
)
from quantaalpha.medical.task_config import (
    get_task_config,
    get_task_name,
    observation_end_hours,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "quantaalpha_medical_workflow"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "pyhealth_runs"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "pyhealth_cache"

def _medical_eicu_task() -> str:
    return get_task_name()


def _task_label_key(task: str) -> str:
    return get_task_config(task).label_key


def _task_cache_slug(task: str) -> str:
    return get_task_config(task).slug


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def require_eicu_files(root: Path) -> None:
    required = ["patient.csv", "diagnosis.csv", "medication.csv", "physicalExam.csv"]
    missing = [
        name
        for name in required
        if not (root / name).exists() and not (root / f"{name}.gz").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required eICU files under {root}: {', '.join(missing)}"
        )


def factor_hash(factors: list[dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(factors, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()[:8]


def _factor_signature(factor: dict[str, Any]) -> tuple[Any, ...]:
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
        factor.get("name"),
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


NON_FACTOR_TEXT_NAMES = {
    "abnormal_high",
    "abnormal_low",
    "accuracy",
    "balanced_accuracy",
    "baseline",
    "baseline_logistic",
    "cohen_kappa",
    "combined_logistic",
    "count_ratio",
    "drug_log_count",
    "f1_macro",
    "f1_weighted",
    "factor_logistic",
    "factors_logistic",
    "aggregation",
    "keyword_any",
    "keyword_count",
    "keyword_density",
    "keyword_gated_numeric",
    "keyword_persistence",
    "log_count",
    "numeric_abnormal_fraction",
    "numeric_early_late_delta",
    "numeric_persistence",
    "numeric_source_interaction",
    "numeric_window_max",
    "numeric_window_mean",
    "numeric_window_min",
    "numeric_window_std",
    "numeric_window_last",
    "numeric_window_count",
    "numeric_window_slope",
    "operator",
    "procedure_log_count",
    "procedure_to_condition_ratio",
    "procedure_to_drug_ratio",
    "drug_to_condition_ratio",
    "resp_fio2",
    "resp_peep",
    "secondary_aggregation",
    "secondary_numeric_source",
    "temporal_keyword_count",
    "temporal_keyword_density",
    "test_f1_macro",
    "val_f1_macro",
    "vital_sao2",
    "transform",
    "window_end_hours",
    "window_start_hours",
    "windows_hours",
}


def _is_non_factor_text_name(name: str) -> bool:
    if name in NON_FACTOR_TEXT_NAMES:
        return True
    if name.endswith("_logistic") or name.endswith("_sgd") or name.endswith("_gbdt"):
        return True
    if name.startswith("baseline_") or name.startswith("combined_"):
        return True
    if name.startswith("phenotype_"):
        return True
    return False


def _include_based_policy() -> str:
    legacy_value = os.environ.get("MEDICAL_TABULAR_INCLUDE_BASED_FACTORS")
    if legacy_value is not None:
        return "all" if str_to_bool(legacy_value) else "none"
    return os.environ.get("MEDICAL_TABULAR_BASED_FACTOR_POLICY", "auto").strip().lower()


def _factor_names_from_text(text: str) -> set[str]:
    return {
        name
        for name in re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", text.lower())
        if not _is_non_factor_text_name(name)
    }


def _requested_factor_names(text: str) -> set[str]:
    if not str_to_bool(os.environ.get("MEDICAL_TABULAR_ENABLE_REQUESTED_FILTER", "0")):
        return set()
    text_l = text.lower()
    if not any(
        marker in text_l
        for marker in (
            "exact-factor-list",
            "allowlist",
        )
    ):
        return set()
    return _factor_names_from_text(text_l)


def _denied_factor_names(text: str) -> set[str]:
    text_l = text.lower()
    names = _factor_names_from_text(text_l)
    denied: set[str] = set()
    deny_markers = (
        "exclude",
        "remove",
        "without",
        "assert absence",
        "prohibited",
        "deny",
        "no legacy",
        "no broad",
        "no saturated",
    )
    for name in names:
        start = text_l.find(name)
        while start != -1:
            window = text_l[max(0, start - 100) : start]
            if any(marker in window for marker in deny_markers):
                denied.add(name)
                break
            start = text_l.find(name, start + 1)
    if "no cardiovascular" in text_l or ("no legacy" in text_l and "cardiovascular" in text_l):
        denied.update(name for name in names if "cardiovascular" in name or "cardiac" in name)
    if (
        "no supportive" in text_l
        or "supportive-medication" in text_l
        or ("no legacy" in text_l and "supportive" in text_l)
    ):
        denied.update(name for name in names if "supportive" in name)
    extra = {
        item.strip().lower()
        for item in os.environ.get("MEDICAL_TABULAR_DENY_FACTOR_NAMES", "").split(",")
        if item.strip()
    }
    denied.update(extra)
    return denied


def _factor_selection_policy(exp: MedicalFactorExperiment) -> dict[str, Any]:
    hypothesis = str(getattr(exp, "target_hypothesis", ""))
    requested = _requested_factor_names(hypothesis)
    denied = _denied_factor_names(hypothesis)
    allow_env = {
        item.strip().lower()
        for item in os.environ.get("MEDICAL_TABULAR_ALLOW_FACTOR_NAMES", "").split(",")
        if item.strip()
    }
    if allow_env:
        requested = allow_env
    return {
        "based_policy": _include_based_policy(),
        "requested_factor_names": requested,
        "denied_factor_names": denied,
        "hypothesis": hypothesis,
    }


def _collect_based_factors(exp: MedicalFactorExperiment) -> list[dict[str, Any]]:
    if _include_based_policy() == "none":
        return []
    based_factors: list[dict[str, Any]] = []
    for based_exp in getattr(exp, "based_experiments", []) or []:
        if not hasattr(based_exp, "factor_dicts"):
            continue
        based_factors.extend(based_exp.factor_dicts())
    return based_factors


def _merge_factor_sets(
    based_factors: list[dict[str, Any]],
    new_factors: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    max_factors = int(os.environ.get("MEDICAL_TABULAR_MAX_EVAL_FACTORS", "24"))
    requested = set(policy.get("requested_factor_names", set()))
    denied = set(policy.get("denied_factor_names", set()))
    candidate_names = {
        str(factor.get("name", "")).lower()
        for factor in [*based_factors, *new_factors]
    }
    active_requested = requested.intersection(candidate_names)
    min_requested_matches = int(
        os.environ.get("MEDICAL_TABULAR_REQUESTED_MIN_MATCHES", "3")
    )
    apply_requested_filter = len(active_requested) >= min_requested_matches
    policy["active_requested_factor_names"] = active_requested
    policy["requested_filter_applied"] = apply_requested_filter

    candidates = [*based_factors, *new_factors]

    def select_candidates(ignore_denied: bool = False, ignore_requested: bool = False):
        selected: list[dict[str, Any]] = []
        for factor in candidates:
            name = str(factor.get("name", "")).lower()
            if not ignore_denied and name in denied:
                continue
            if (
                apply_requested_filter
                and not ignore_requested
                and name not in active_requested
            ):
                continue
            selected.append(factor)
        return selected

    filtered = select_candidates()
    policy["selection_fallback"] = ""
    if not filtered and not str_to_bool(os.environ.get("MEDICAL_TABULAR_STRICT_SELECTION", "0")):
        filtered = list(new_factors) or select_candidates(
            ignore_denied=True,
            ignore_requested=True,
        )
        policy["selection_fallback"] = "empty_selection_fell_back_to_current_batch"

    merged: list[dict[str, Any]] = []
    signature_to_index: dict[tuple[Any, ...], int] = {}
    name_to_index: dict[str, int] = {}
    for factor in filtered:
        name = str(factor.get("name", "")).lower()
        signature = _factor_signature(factor)
        if signature in signature_to_index:
            continue
        if name in name_to_index:
            idx = name_to_index[name]
            old_signature = _factor_signature(merged[idx])
            signature_to_index.pop(old_signature, None)
            merged[idx] = factor
            signature_to_index[signature] = idx
            continue
        name_to_index[name] = len(merged)
        signature_to_index[signature] = len(merged)
        merged.append(factor)
        if len(merged) >= max_factors:
            break
    if not merged and filtered:
        merged = filtered[:max_factors]
    return merged


class LengthOfStayPredictioneICUWithTabularFactors(LengthOfStayPredictioneICU):
    task_name = "LengthOfStayPredictioneICUWithTabularFactors"
    input_schema = {
        **LengthOfStayPredictioneICU.input_schema,
        "symbolic_factors": "tensor",
    }
    output_schema = LengthOfStayPredictioneICU.output_schema

    def __init__(self, factors: list[dict[str, Any]]):
        self.factors = factors
        self.task_name = f"{self.task_name}_{factor_hash(factors)}"

    def __call__(self, patient):
        samples = super().__call__(patient)
        for sample in samples:
            sample["symbolic_factors"] = [
                compute_factor(sample, factor) for factor in self.factors
            ]
        return samples


def summarize_factor_matrix(
    factor_matrix: np.ndarray,
    factors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for factor_idx, factor in enumerate(factors):
        vals = factor_matrix[:, factor_idx]
        mean = float(np.mean(vals)) if len(vals) else 0.0
        std = float(np.std(vals)) if len(vals) else 0.0
        nonzero = float(np.mean(np.abs(vals) > 1e-12)) if len(vals) else 0.0
        summary.append(
            {
                "name": factor["name"],
                "operation": factor["operation"],
                "mean": mean,
                "std": std,
                "nonzero_rate": nonzero,
            }
        )
    return summary


BASE_FEATURE_NAMES = [
    "condition_log_count",
    "procedure_log_count",
    "drug_log_count",
    "procedure_to_condition_ratio",
    "drug_to_condition_ratio",
    "procedure_to_drug_ratio",
]


def _base_features(sample: dict[str, Any]) -> list[float]:
    n_conditions = len(flatten_text(sample.get("conditions", [])))
    n_procedures = len(flatten_text(sample.get("procedures", [])))
    n_drugs = len(flatten_text(sample.get("drugs", [])))
    return _base_features_from_counts(n_conditions, n_procedures, n_drugs)


def _base_features_from_counts(
    n_conditions: int,
    n_procedures: int,
    n_drugs: int,
) -> list[float]:
    return [
        math.log1p(n_conditions),
        math.log1p(n_procedures),
        math.log1p(n_drugs),
        n_procedures / max(n_conditions, 1),
        n_drugs / max(n_conditions, 1),
        n_procedures / max(n_drugs, 1),
    ]


def _materialize_raw_dataset(dataset, label_key: str):
    base_rows = []
    factor_rows = []
    labels = []
    patient_ids = []
    total = len(dataset)
    print(f"Materializing raw tabular task dataset: n={total}", flush=True)
    for idx in range(total):
        sample = dataset[idx]
        base_rows.append(_base_features(sample))
        factor_rows.append([float(value) for value in sample["symbolic_factors"]])
        value = sample[label_key]
        labels.append(int(value.item()) if torch.is_tensor(value) else int(value))
        patient_ids.append(str(sample.get("patient_id", idx)))
        if (idx + 1) % 10000 == 0 or idx + 1 == total:
            print(
                f"  materialized {idx + 1}/{total} samples",
                flush=True,
            )
    return (
        np.asarray(base_rows, dtype=np.float32),
        np.asarray(factor_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(patient_ids, dtype=object),
    )


def _eicu_csv_path(root: Path, filename: str) -> Path:
    path = root / filename
    if path.exists():
        return path
    gz_path = root / f"{filename}.gz"
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Missing eICU table: {path} or {gz_path}")


def _scan_eicu_csv(root: Path, filename: str) -> pl.LazyFrame:
    return pl.scan_csv(
        str(_eicu_csv_path(root, filename)),
        infer_schema_length=10000,
        ignore_errors=True,
    )


def _group_text_table(
    root: Path,
    filename: str,
    source_col: str,
    output_col: str,
) -> pl.LazyFrame:
    return _group_text_rows(_text_rows_table(root, filename, source_col, output_col), output_col)


def _group_text_offset_table(
    root: Path,
    filename: str,
    source_col: str,
    offset_expr: pl.Expr,
    output_col: str,
) -> pl.LazyFrame:
    return _group_text_offset_rows(
        _text_offset_rows_table(root, filename, source_col, offset_expr, output_col),
        output_col,
    )


def _text_rows_table(
    root: Path,
    filename: str,
    source_col: str,
    output_col: str,
) -> pl.LazyFrame:
    return (
        _scan_eicu_csv(root, filename)
        .select(
            pl.col("patientunitstayid").cast(pl.Utf8).alias("visit_id"),
            pl.col(source_col).cast(pl.Utf8).alias(output_col),
        )
        .filter(
            pl.col("visit_id").is_not_null()
            & pl.col(output_col).is_not_null()
            & (pl.col(output_col).str.len_chars() > 0)
        )
    )


def _text_offset_rows_table(
    root: Path,
    filename: str,
    source_col: str,
    offset_expr: pl.Expr,
    output_col: str,
) -> pl.LazyFrame:
    offset_col = f"{output_col}_offsets"
    return (
        _scan_eicu_csv(root, filename)
        .select(
            pl.col("patientunitstayid").cast(pl.Utf8).alias("visit_id"),
            pl.col(source_col).cast(pl.Utf8).alias(output_col),
            offset_expr.cast(pl.Int64, strict=False).alias(offset_col),
        )
        .filter(
            pl.col("visit_id").is_not_null()
            & pl.col(output_col).is_not_null()
            & (pl.col(output_col).str.len_chars() > 0)
            & pl.col(offset_col).is_not_null()
        )
    )


def _group_text_rows(frame: pl.LazyFrame, output_col: str) -> pl.LazyFrame:
    return frame.group_by("visit_id").agg(pl.col(output_col))


def _group_text_offset_rows(frame: pl.LazyFrame, output_col: str) -> pl.LazyFrame:
    offset_col = f"{output_col}_offsets"
    return (
        frame
        .sort(["visit_id", offset_col])
        .group_by("visit_id")
        .agg(pl.col(output_col), pl.col(offset_col))
    )


def _concat_text_tables(tables: list[pl.LazyFrame], output_col: str) -> pl.LazyFrame:
    return _group_text_rows(pl.concat(tables, how="vertical_relaxed"), output_col)


def _concat_text_offset_tables(tables: list[pl.LazyFrame], output_col: str) -> pl.LazyFrame:
    return _group_text_offset_rows(pl.concat(tables, how="vertical_relaxed"), output_col)


def _group_numeric_table(
    root: Path,
    filename: str,
    offset_expr: pl.Expr,
    value_expr: pl.Expr,
    source_name: str,
    filter_expr: pl.Expr | None = None,
) -> pl.LazyFrame:
    value_col = f"{source_name}_values"
    offset_col = f"{source_name}_offsets"
    frame = _scan_eicu_csv(root, filename)
    if filter_expr is not None:
        frame = frame.filter(filter_expr)
    return (
        frame.select(
            pl.col("patientunitstayid").cast(pl.Utf8).alias("visit_id"),
            offset_expr.cast(pl.Int64, strict=False).alias(offset_col),
            value_expr.cast(pl.Float64, strict=False).alias(value_col),
        )
        .filter(
            pl.col("visit_id").is_not_null()
            & pl.col(offset_col).is_not_null()
            & pl.col(value_col).is_not_null()
            & pl.col(value_col).is_finite()
        )
        .sort(["visit_id", offset_col])
        .group_by("visit_id")
        .agg(pl.col(value_col), pl.col(offset_col))
    )


def _group_lab_numeric_table(
    root: Path,
    lab_pattern: str,
    source_name: str,
) -> pl.LazyFrame:
    lab_name = pl.col("labname").cast(pl.Utf8).str.to_lowercase()
    return _group_numeric_table(
        root,
        "lab.csv",
        pl.col("labresultoffset"),
        pl.col("labresult"),
        source_name,
        lab_name.str.contains(lab_pattern),
    )


def _group_nurse_numeric_table(
    root: Path,
    chart_pattern: str,
    source_name: str,
) -> pl.LazyFrame:
    chart_name = (
        pl.concat_str(
            [
                pl.col("nursingchartcelltypevallabel").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("nursingchartcelltypevalname").cast(pl.Utf8),
            ],
            ignore_nulls=True,
        )
        .str.to_lowercase()
    )
    return _group_numeric_table(
        root,
        "nurseCharting.csv",
        pl.coalesce([pl.col("nursingchartentryoffset"), pl.col("nursingchartoffset")]),
        pl.col("nursingchartvalue"),
        source_name,
        chart_name.str.contains(chart_pattern),
    )


def _group_infusion_numeric_table(
    root: Path,
    drug_pattern: str,
    source_name: str,
) -> pl.LazyFrame:
    drug_name = pl.col("drugname").cast(pl.Utf8).str.to_lowercase()
    return _group_numeric_table(
        root,
        "infusionDrug.csv",
        pl.col("infusionoffset"),
        pl.coalesce([pl.col("drugrate"), pl.col("infusionrate"), pl.col("drugamount")]),
        source_name,
        drug_name.str.contains(drug_pattern),
    )


def _numeric_temporal_tables(root: Path) -> list[pl.LazyFrame]:
    vital_specs = {
        "vital_sao2": "sao2",
        "vital_heartrate": "heartrate",
        "vital_respiration": "respiration",
        "vital_temperature": "temperature",
        "vital_systemicsystolic": "systemicsystolic",
        "vital_systemicdiastolic": "systemicdiastolic",
        "vital_systemicmean": "systemicmean",
    }
    tables: list[pl.LazyFrame] = [
        _group_numeric_table(
            root,
            "vitalPeriodic.csv",
            pl.col("observationoffset"),
            pl.col(column),
            source_name,
        )
        for source_name, column in vital_specs.items()
    ]
    vital_aperiodic_specs = {
        "vital_noninvasivesystolic": "noninvasivesystolic",
        "vital_noninvasivediastolic": "noninvasivediastolic",
        "vital_noninvasivemean": "noninvasivemean",
        "vital_cardiacoutput": "cardiacoutput",
    }
    tables.extend(
        _group_numeric_table(
            root,
            "vitalAperiodic.csv",
            pl.col("observationoffset"),
            pl.col(column),
            source_name,
        )
        for source_name, column in vital_aperiodic_specs.items()
    )
    io_specs = {
        "io_intake_total": "intaketotal",
        "io_output_total": "outputtotal",
        "io_net_total": "nettotal",
        "io_dialysis_total": "dialysistotal",
    }
    tables.extend(
        _group_numeric_table(
            root,
            "intakeOutput.csv",
            pl.coalesce([pl.col("intakeoutputentryoffset"), pl.col("intakeoutputoffset")]),
            pl.col(column),
            source_name,
        )
        for source_name, column in io_specs.items()
    )
    resp_label = pl.col("respchartvaluelabel").cast(pl.Utf8).str.to_lowercase()
    resp_type = pl.col("respcharttypecat").cast(pl.Utf8).str.to_lowercase()
    tables.extend(
        [
            _group_numeric_table(
                root,
                "respiratoryCharting.csv",
                pl.coalesce([pl.col("respchartentryoffset"), pl.col("respchartoffset")]),
                pl.col("respchartvalue"),
                "resp_fio2",
                resp_label.str.contains("fio2|fi02|fraction inspired oxygen")
                | resp_type.str.contains("fio2|fi02"),
            ),
            _group_numeric_table(
                root,
                "respiratoryCharting.csv",
                pl.coalesce([pl.col("respchartentryoffset"), pl.col("respchartoffset")]),
                pl.col("respchartvalue"),
                "resp_peep",
                resp_label.str.contains("peep")
                | resp_type.str.contains("peep"),
            ),
        ]
    )
    lab_specs = {
        "lab_albumin": r"(^|\\b)albumin($|\\b)",
        "lab_bilirubin": r"bilirubin",
        "lab_bun": r"(^|\\b)(bun|blood urea nitrogen)($|\\b)",
        "lab_creatinine": r"creatinine",
        "lab_glucose": r"glucose",
        "lab_hematocrit": r"hematocrit|hct",
        "lab_hemoglobin": r"hemoglobin|hgb",
        "lab_lactate": r"lactate|lactic",
        "lab_platelets": r"platelet",
        "lab_potassium": r"potassium",
        "lab_sodium": r"sodium",
        "lab_wbc": r"(^|\\b)(wbc|white blood)($|\\b)",
        "lab_ph": r"(^|\\b)ph($|\\b)",
        "lab_pao2": r"pao2|pa o2|po2",
        "lab_paco2": r"paco2|pa co2|pco2",
    }
    tables.extend(
        _group_lab_numeric_table(root, pattern, source_name)
        for source_name, pattern in lab_specs.items()
    )
    nurse_specs = {
        "nurse_gcs_total": r"glasgow|gcs",
        "nurse_pain_score": r"pain",
        "nurse_rass": r"rass|richmond",
        "nurse_spo2": r"o2 saturation|spo2|sao2",
        "nurse_bedside_glucose": r"glucose",
    }
    tables.extend(
        _group_nurse_numeric_table(root, pattern, source_name)
        for source_name, pattern in nurse_specs.items()
    )
    infusion_specs = {
        "infusion_vasopressor_rate": r"norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine|dobutamine",
        "infusion_norepinephrine_rate": r"norepinephrine",
        "infusion_propofol_rate": r"propofol",
        "infusion_insulin_rate": r"insulin",
        "infusion_milrinone_rate": r"milrinone",
    }
    tables.extend(
        _group_infusion_numeric_table(root, pattern, source_name)
        for source_name, pattern in infusion_specs.items()
    )
    apache_specs = {
        "apache_urine": "urine",
        "apache_wbc": "wbc",
        "apache_temperature": "temperature",
        "apache_respiratoryrate": "respiratoryrate",
        "apache_sodium": "sodium",
        "apache_heartrate": "heartrate",
        "apache_meanbp": "meanbp",
        "apache_ph": "ph",
        "apache_hematocrit": "hematocrit",
        "apache_creatinine": "creatinine",
        "apache_albumin": "albumin",
        "apache_pao2": "pao2",
        "apache_pco2": "pco2",
        "apache_bun": "bun",
        "apache_glucose": "glucose",
        "apache_bilirubin": "bilirubin",
        "apache_fio2": "fio2",
    }
    tables.extend(
        _group_numeric_table(
            root,
            "apacheApsVar.csv",
            pl.lit(0),
            pl.col(column),
            source_name,
        )
        for source_name, column in apache_specs.items()
    )
    return tables


def _build_or_load_base_sample_frame(
    eicu_root: Path,
    cache_dir: Path,
    dev: bool,
    task: str,
) -> pl.DataFrame:
    profile = source_profile()
    temporal_source = str_to_bool(os.environ.get("MEDICAL_TEMPORAL_SOURCE", "1"))
    numeric_temporal_source = str_to_bool(
        os.environ.get("MEDICAL_NUMERIC_TEMPORAL_SOURCE", "1")
    )
    if profile == PYHEALTH_STANDARD:
        numeric_temporal_source = False
    cache_profile = profile
    temporal_suffix = (
        f"_{cache_profile}_temporal_numeric_v2"
        if temporal_source and numeric_temporal_source
        else f"_{cache_profile}_temporal"
        if temporal_source
        else f"_{cache_profile}"
    )
    if task == "los":
        cache_name = (
            f"tabular_base_samples_dev{temporal_suffix}.parquet"
            if dev
            else f"tabular_base_samples_full{temporal_suffix}.parquet"
        )
    else:
        cache_name = (
            f"tabular_base_samples_{task}_dev{temporal_suffix}.parquet"
            if dev
            else f"tabular_base_samples_{task}_full{temporal_suffix}.parquet"
        )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        print(f"Using cached direct tabular base samples: {cache_path}", flush=True)
        return pl.read_parquet(cache_path)

    print(f"Building direct tabular base samples: {cache_path}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    patient_raw = _scan_eicu_csv(eicu_root, "patient.csv").select(
        pl.col("patientunitstayid").cast(pl.Utf8).alias("visit_id"),
        pl.col("uniquepid").cast(pl.Utf8).alias("patient_id"),
        pl.col("unitdischargeoffset")
        .cast(pl.Int64, strict=False)
        .alias("unitdischargeoffset"),
        pl.col("hospitaldischargestatus")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .alias("hospitaldischargestatus"),
        pl.col("unitdischargestatus")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .alias("unitdischargestatus"),
        pl.col("patienthealthsystemstayid")
        .cast(pl.Int64, strict=False)
        .alias("patienthealthsystemstayid"),
        pl.col("unitvisitnumber")
        .cast(pl.Int64, strict=False)
        .alias("unitvisitnumber"),
        pl.col("age").cast(pl.Utf8).alias("age"),
    )
    if task == "los":
        patient = patient_raw.filter(
            pl.col("visit_id").is_not_null()
            & pl.col("patient_id").is_not_null()
            & pl.col("unitdischargeoffset").is_not_null()
        )
    elif task == "mortality":
        if profile == PYHEALTH_STANDARD:
            patient = (
                patient_raw.filter(
                    pl.col("visit_id").is_not_null()
                    & pl.col("patient_id").is_not_null()
                    & pl.col("patienthealthsystemstayid").is_not_null()
                )
                .sort(["patient_id", "patienthealthsystemstayid", "unitvisitnumber"])
                .with_columns(
                    pl.col("hospitaldischargestatus")
                    .shift(-1)
                    .over("patient_id")
                    .alias("next_hospitaldischargestatus")
                )
                .with_columns(
                    pl.when(pl.col("next_hospitaldischargestatus") == "expired")
                    .then(1)
                    .otherwise(0)
                    .cast(pl.Int64)
                    .alias("mortality")
                )
                .filter(pl.col("next_hospitaldischargestatus").is_not_null())
            )
        else:
            status_field = os.environ.get(
                "MEDICAL_MORTALITY_STATUS_FIELD",
                "hospitaldischargestatus",
            ).strip()
            if status_field not in {"hospitaldischargestatus", "unitdischargestatus"}:
                raise ValueError(
                    "MEDICAL_MORTALITY_STATUS_FIELD must be hospitaldischargestatus "
                    "or unitdischargestatus"
                )
            patient = (
                patient_raw.with_columns(
                    pl.when(pl.col(status_field) == "expired")
                    .then(1)
                    .when(pl.col(status_field) == "alive")
                    .then(0)
                    .otherwise(None)
                    .cast(pl.Int64)
                    .alias("mortality")
                )
                .filter(
                    pl.col("visit_id").is_not_null()
                    & pl.col("patient_id").is_not_null()
                    & pl.col("mortality").is_not_null()
                )
            )
    elif task == "readmission":
        patient = (
            patient_raw.filter(
                pl.col("visit_id").is_not_null()
                & pl.col("patient_id").is_not_null()
                & pl.col("patienthealthsystemstayid").is_not_null()
            )
            .sort(["patient_id", "patienthealthsystemstayid", "unitvisitnumber"])
            .with_columns(
                pl.col("patienthealthsystemstayid")
                .shift(-1)
                .over("patient_id")
                .alias("next_patienthealthsystemstayid")
            )
            .with_columns(
                (
                    pl.col("patienthealthsystemstayid")
                    == pl.col("next_patienthealthsystemstayid")
                )
                .cast(pl.Int64)
                .alias("readmission")
            )
            .filter(pl.col("next_patienthealthsystemstayid").is_not_null())
        )
        if profile == PYHEALTH_STANDARD:
            age_int = pl.col("age").cast(pl.Int64, strict=False)
            patient = patient.filter(age_int.is_null() | (age_int >= 18))
    else:
        raise ValueError(f"Unsupported eICU task: {task}")
    if dev:
        dev_patients = patient.select("patient_id").unique(maintain_order=True).limit(1000)
        patient = patient.join(dev_patients, on="patient_id", how="inner")

    if profile == PYHEALTH_STANDARD:
        if task == "los":
            condition_col = "diagnosisstring"
            procedure_col = "physicalexamvalue"
        else:
            condition_col = "icd9code"
            procedure_col = "physicalexampath"
        if temporal_source:
            diagnosis = _group_text_offset_table(
                eicu_root,
                "diagnosis.csv",
                condition_col,
                pl.col("diagnosisoffset"),
                "conditions",
            )
            physical_exam = _group_text_offset_table(
                eicu_root,
                "physicalExam.csv",
                procedure_col,
                pl.col("physicalexamoffset"),
                "procedures",
            )
            medication = _group_text_offset_table(
                eicu_root,
                "medication.csv",
                "drugname",
                pl.coalesce([pl.col("drugstartoffset"), pl.col("drugorderoffset")]),
                "drugs",
            )
        else:
            diagnosis = _group_text_table(
                eicu_root,
                "diagnosis.csv",
                condition_col,
                "conditions",
            )
            physical_exam = _group_text_table(
                eicu_root,
                "physicalExam.csv",
                procedure_col,
                "procedures",
            )
            medication = _group_text_table(
                eicu_root,
                "medication.csv",
                "drugname",
                "drugs",
            )
    elif profile == EXPANDED_V2 and temporal_source:
        diagnosis = _concat_text_offset_tables(
            [
                _text_offset_rows_table(
                    eicu_root,
                    "diagnosis.csv",
                    "diagnosisstring",
                    pl.col("diagnosisoffset"),
                    "conditions",
                ),
                _text_offset_rows_table(
                    eicu_root,
                    "admissionDx.csv",
                    "admitdxname",
                    pl.col("admitdxenteredoffset"),
                    "conditions",
                ),
            ],
            "conditions",
        )
        physical_exam = _concat_text_offset_tables(
            [
                _text_offset_rows_table(
                    eicu_root,
                    "physicalExam.csv",
                    "physicalexamvalue",
                    pl.col("physicalexamoffset"),
                    "procedures",
                ),
                _text_offset_rows_table(
                    eicu_root,
                    "treatment.csv",
                    "treatmentstring",
                    pl.col("treatmentoffset"),
                    "procedures",
                ),
                _text_offset_rows_table(
                    eicu_root,
                    "respiratoryCare.csv",
                    "airwaytype",
                    pl.col("respcarestatusoffset"),
                    "procedures",
                ),
            ],
            "procedures",
        )
        medication = _concat_text_offset_tables(
            [
                _text_offset_rows_table(
                    eicu_root,
                    "medication.csv",
                    "drugname",
                    pl.coalesce([pl.col("drugstartoffset"), pl.col("drugorderoffset")]),
                    "drugs",
                ),
                _text_offset_rows_table(
                    eicu_root,
                    "infusionDrug.csv",
                    "drugname",
                    pl.col("infusionoffset"),
                    "drugs",
                ),
                _text_offset_rows_table(
                    eicu_root,
                    "admissionDrug.csv",
                    "drugname",
                    pl.coalesce([pl.col("drugenteredoffset"), pl.col("drugoffset")]),
                    "drugs",
                ),
            ],
            "drugs",
        )
    elif profile == EXPANDED_V2:
        diagnosis = _concat_text_tables(
            [
                _text_rows_table(eicu_root, "diagnosis.csv", "diagnosisstring", "conditions"),
                _text_rows_table(eicu_root, "admissionDx.csv", "admitdxname", "conditions"),
            ],
            "conditions",
        )
        physical_exam = _concat_text_tables(
            [
                _text_rows_table(eicu_root, "physicalExam.csv", "physicalexamvalue", "procedures"),
                _text_rows_table(eicu_root, "treatment.csv", "treatmentstring", "procedures"),
                _text_rows_table(eicu_root, "respiratoryCare.csv", "airwaytype", "procedures"),
            ],
            "procedures",
        )
        medication = _concat_text_tables(
            [
                _text_rows_table(eicu_root, "medication.csv", "drugname", "drugs"),
                _text_rows_table(eicu_root, "infusionDrug.csv", "drugname", "drugs"),
                _text_rows_table(eicu_root, "admissionDrug.csv", "drugname", "drugs"),
            ],
            "drugs",
        )
    else:
        raise ValueError(f"Unsupported source profile: {profile}")
    if task == "los":
        los_days = pl.col("unitdischargeoffset") // (60 * 24)
        task_label = (
            pl.when(los_days < 1)
            .then(0)
            .when(los_days <= 7)
            .then(los_days)
            .when(los_days <= 14)
            .then(8)
            .otherwise(9)
            .cast(pl.Int64)
            .alias("los")
        )
    else:
        task_label = pl.col(_task_label_key(task)).cast(pl.Int64)

    frame_lf = (
        patient.join(diagnosis, on="visit_id", how="inner")
        .join(physical_exam, on="visit_id", how="inner")
        .join(medication, on="visit_id", how="inner")
        .with_columns(task_label)
    )
    if temporal_source and numeric_temporal_source:
        for numeric_table in _numeric_temporal_tables(eicu_root):
            frame_lf = frame_lf.join(numeric_table, on="visit_id", how="left")
    available_cols = set(frame_lf.collect_schema().names())
    selected_cols = [
        col
        for col in (
            "visit_id",
            "patient_id",
            "conditions",
            "conditions_offsets",
            "procedures",
            "procedures_offsets",
            "drugs",
            "drugs_offsets",
            _task_label_key(task),
            *[
                numeric_col
                for source_name in sorted(numeric_sources_for_profile(profile))
                for numeric_col in (
                    f"{source_name}_values",
                    f"{source_name}_offsets",
                )
            ],
        )
        if col in available_cols
    ]
    frame = frame_lf.select(selected_cols).collect()
    frame.write_parquet(cache_path)
    print(f"Cached direct tabular base samples: {cache_path}", flush=True)
    return frame


def _sample_text_cache(sample: dict[str, Any]) -> dict[str, Any]:
    texts_by_source = {
        source: flatten_text(sample.get(source, []))
        for source in ("conditions", "procedures", "drugs")
    }
    offsets_by_source: dict[str, list[float | None]] = {}
    for source, texts in texts_by_source.items():
        raw_offsets = sample.get(f"{source}_offsets", [])
        if not isinstance(raw_offsets, (list, tuple)):
            raw_offsets = []
        offsets: list[float | None] = []
        for idx in range(len(texts)):
            raw_offset = raw_offsets[idx] if idx < len(raw_offsets) else None
            try:
                offsets.append(float(raw_offset) if raw_offset is not None else None)
            except (TypeError, ValueError):
                offsets.append(None)
        offsets_by_source[source] = offsets
    numeric_by_source: dict[str, dict[str, list[float | None] | list[float]]] = {}
    for source in VALID_NUMERIC_SOURCES:
        raw_values = sample.get(f"{source}_values", [])
        raw_offsets = sample.get(f"{source}_offsets", [])
        if not isinstance(raw_values, (list, tuple)):
            raw_values = []
        if not isinstance(raw_offsets, (list, tuple)):
            raw_offsets = []
        values: list[float] = []
        offsets: list[float | None] = []
        for idx, raw_value in enumerate(raw_values):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            raw_offset = raw_offsets[idx] if idx < len(raw_offsets) else None
            try:
                offset = float(raw_offset) if raw_offset is not None else None
            except (TypeError, ValueError):
                offset = None
            values.append(value)
            offsets.append(offset)
        numeric_by_source[source] = {"values": values, "offsets": offsets}
    return {
        "texts": texts_by_source,
        "lowered": {
            source: [text.lower() for text in texts]
            for source, texts in texts_by_source.items()
        },
        "offsets": offsets_by_source,
        "numeric": numeric_by_source,
        "counts": {source: len(texts) for source, texts in texts_by_source.items()},
    }


def _cached_keyword_hits(
    cache: dict[str, Any],
    sources: list[str],
    keywords: list[str],
) -> tuple[int, int]:
    hits = 0
    total = 0
    lowered = cache["lowered"]
    for source in sources:
        texts = lowered[source]
        total += len(texts)
        for text in texts:
            if any(keyword in text for keyword in keywords):
                hits += 1
    return hits, total


def _cached_temporal_keyword_hits(
    cache: dict[str, Any],
    sources: list[str],
    keywords: list[str],
    start_hours: float,
    end_hours: float,
) -> tuple[int, int]:
    hits = 0
    total = 0
    lowered = cache["lowered"]
    offsets = cache["offsets"]
    for source in sources:
        texts = lowered[source]
        source_offsets = offsets[source]
        for idx, text in enumerate(texts):
            offset = source_offsets[idx] if idx < len(source_offsets) else None
            if offset is None:
                continue
            offset_hours = offset / 60.0
            if offset_hours < start_hours or offset_hours >= end_hours:
                continue
            total += 1
            if any(keyword in text for keyword in keywords):
                hits += 1
    return hits, total


def _temporal_density(
    cache: dict[str, Any],
    factor: dict[str, Any],
    window: list[float],
) -> float:
    hits, total = _cached_temporal_keyword_hits(
        cache,
        factor["sources"],
        factor["keywords"],
        float(window[0]),
        float(window[1]),
    )
    return hits / max(total, 1)


def _cached_numeric_window_values(
    cache: dict[str, Any],
    numeric_source: str,
    start_hours: float,
    end_hours: float,
) -> list[float]:
    numeric = cache["numeric"].get(numeric_source, {"values": [], "offsets": []})
    values = numeric["values"]
    offsets = numeric["offsets"]
    selected: list[float] = []
    for idx, value in enumerate(values):
        offset = offsets[idx] if idx < len(offsets) else None
        if offset is None:
            continue
        offset_hours = offset / 60.0
        if offset_hours >= start_hours and offset_hours < end_hours:
            selected.append(float(value))
    return selected


def _cached_numeric_window_events(
    cache: dict[str, Any],
    numeric_source: str,
    start_hours: float,
    end_hours: float,
) -> list[tuple[float, float]]:
    numeric = cache["numeric"].get(numeric_source, {"values": [], "offsets": []})
    values = numeric["values"]
    offsets = numeric["offsets"]
    selected: list[tuple[float, float]] = []
    for idx, value in enumerate(values):
        offset = offsets[idx] if idx < len(offsets) else None
        if offset is None:
            continue
        offset_hours = offset / 60.0
        if offset_hours >= start_hours and offset_hours < end_hours:
            selected.append((float(value), float(offset_hours)))
    return selected


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _last(events: list[tuple[float, float]]) -> float:
    if not events:
        return 0.0
    return max(events, key=lambda item: item[1])[0]


def _slope(events: list[tuple[float, float]]) -> float:
    if len(events) < 2:
        return 0.0
    xs = [offset_hours for _, offset_hours in events]
    ys = [value for value, _ in events]
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    return float(
        sum((x - x_mean) * (y - y_mean) for y, x in events) / denom
    )


def _aggregate_cached_numeric(
    cache: dict[str, Any],
    numeric_source: str,
    start_hours: float,
    end_hours: float,
    aggregation: str,
) -> float:
    events = _cached_numeric_window_events(cache, numeric_source, start_hours, end_hours)
    values = [value for value, _ in events]
    if aggregation == "mean":
        return _mean(values)
    if aggregation == "min":
        return min(values) if values else 0.0
    if aggregation == "max":
        return max(values) if values else 0.0
    if aggregation == "std":
        return _std(values)
    if aggregation == "last":
        return _last(events)
    if aggregation == "count":
        return float(len(values))
    if aggregation == "slope":
        return _slope(events)
    return _mean(values)


def _numeric_abnormal_fraction(
    values: list[float],
    low: float | None,
    high: float | None,
) -> float:
    if not values or (low is None and high is None):
        return 0.0
    abnormal = 0
    for value in values:
        if (low is not None and value < low) or (high is not None and value > high):
            abnormal += 1
    return abnormal / len(values)


def _compute_factor_cached(cache: dict[str, Any], factor: dict[str, Any]) -> float:
    op = factor["operation"]
    if op == SAFE_PYTHON_OPERATION:
        sample = {
            "conditions": cache["texts"]["conditions"],
            "procedures": cache["texts"]["procedures"],
            "drugs": cache["texts"]["drugs"],
            "conditions_offsets": cache["offsets"]["conditions"],
            "procedures_offsets": cache["offsets"]["procedures"],
            "drugs_offsets": cache["offsets"]["drugs"],
        }
        return compute_safe_python_factor(sample, factor)
    if op in NUMERIC_TEMPORAL_OPS:
        numeric_source = factor["numeric_source"]
        values = _cached_numeric_window_values(
            cache,
            numeric_source,
            float(factor.get("window_start_hours", 0.0)),
            float(factor.get("window_end_hours", 24.0)),
        )
        if op == "numeric_window_mean":
            return apply_numeric_transform(_mean(values), factor.get("transform"))
        if op == "numeric_window_min":
            return apply_numeric_transform(min(values) if values else 0.0, factor.get("transform"))
        if op == "numeric_window_max":
            return apply_numeric_transform(max(values) if values else 0.0, factor.get("transform"))
        if op == "numeric_window_std":
            return apply_numeric_transform(_std(values), factor.get("transform"))
        if op == "numeric_window_last":
            events = _cached_numeric_window_events(
                cache,
                numeric_source,
                float(factor.get("window_start_hours", 0.0)),
                float(factor.get("window_end_hours", 24.0)),
            )
            return apply_numeric_transform(_last(events), factor.get("transform"))
        if op == "numeric_window_count":
            return apply_numeric_transform(float(len(values)), factor.get("transform"))
        if op == "numeric_window_slope":
            events = _cached_numeric_window_events(
                cache,
                numeric_source,
                float(factor.get("window_start_hours", 0.0)),
                float(factor.get("window_end_hours", 24.0)),
            )
            return apply_numeric_transform(_slope(events), factor.get("transform"))
        if op == "numeric_source_interaction":
            start = float(factor.get("window_start_hours", 0.0))
            end = float(factor.get("window_end_hours", 24.0))
            left = _aggregate_cached_numeric(
                cache,
                numeric_source,
                start,
                end,
                factor.get("aggregation", "mean"),
            )
            right = _aggregate_cached_numeric(
                cache,
                factor.get("secondary_numeric_source", ""),
                start,
                end,
                factor.get("secondary_aggregation", "mean"),
            )
            value = apply_numeric_operator(left, right, factor.get("operator", ""))
            return apply_numeric_transform(value, factor.get("transform"))
        if op == "keyword_gated_numeric":
            start = float(factor.get("window_start_hours", 0.0))
            end = float(factor.get("window_end_hours", 24.0))
            hits, _ = _cached_temporal_keyword_hits(
                cache,
                factor["sources"],
                factor["keywords"],
                start,
                end,
            )
            if hits <= 0:
                return 0.0
            value = _aggregate_cached_numeric(
                cache,
                numeric_source,
                start,
                end,
                factor.get("aggregation", "mean"),
            )
            return apply_numeric_transform(value, factor.get("transform"))
        if op == "numeric_abnormal_fraction":
            return apply_numeric_transform(
                _numeric_abnormal_fraction(
                    values,
                    factor.get("abnormal_low"),
                    factor.get("abnormal_high"),
                ),
                factor.get("transform"),
            )
        if op == "numeric_early_late_delta":
            early = _mean(
                _cached_numeric_window_values(
                    cache,
                    numeric_source,
                    float(factor.get("early_window_hours", [0.0, 24.0])[0]),
                    float(factor.get("early_window_hours", [0.0, 24.0])[1]),
                )
            )
            late = _mean(
                _cached_numeric_window_values(
                    cache,
                    numeric_source,
                    float(factor.get("late_window_hours", [24.0, 72.0])[0]),
                    float(factor.get("late_window_hours", [24.0, 72.0])[1]),
                )
            )
            return apply_numeric_transform(late - early, factor.get("transform"))
        if op == "numeric_persistence":
            threshold = factor.get("threshold")
            low = factor.get("abnormal_low")
            high = factor.get("abnormal_high")
            active = 0
            windows = factor.get("windows_hours", DEFAULT_TEMPORAL_WINDOWS)
            for window in windows:
                window_values = _cached_numeric_window_values(
                    cache,
                    numeric_source,
                    float(window[0]),
                    float(window[1]),
                )
                if threshold is not None:
                    active += int(any(abs(value) >= threshold for value in window_values))
                else:
                    active += int(
                        _numeric_abnormal_fraction(window_values, low, high) > 0
                    )
            return apply_numeric_transform(
                active / max(len(windows), 1),
                factor.get("transform"),
            )

    counts = cache["counts"]
    if op == "log_count":
        return math.log1p(counts[factor["sources"][0]])
    if op == "count_ratio":
        numerator = counts[factor["numerator_source"]]
        denominator = counts[factor["denominator_source"]]
        return numerator / max(denominator, 1)

    hits, total = _cached_keyword_hits(
        cache,
        factor["sources"],
        factor["keywords"],
    )
    if op == "keyword_any":
        return float(hits > 0)
    if op == "keyword_count":
        return float(hits)
    if op == "keyword_density":
        return hits / max(total, 1)
    if op == "temporal_keyword_count":
        hits, _ = _cached_temporal_keyword_hits(
            cache,
            factor["sources"],
            factor["keywords"],
            float(factor.get("window_start_hours", 0.0)),
            float(factor.get("window_end_hours", 24.0)),
        )
        return float(hits)
    if op == "temporal_keyword_density":
        hits, total = _cached_temporal_keyword_hits(
            cache,
            factor["sources"],
            factor["keywords"],
            float(factor.get("window_start_hours", 0.0)),
            float(factor.get("window_end_hours", 24.0)),
        )
        return hits / max(total, 1)
    if op == "first_keyword_offset":
        first_offset: float | None = None
        for source in factor["sources"]:
            texts = cache["lowered"][source]
            offsets = cache["offsets"][source]
            for idx, text in enumerate(texts):
                offset = offsets[idx] if idx < len(offsets) else None
                if offset is None or offset < 0:
                    continue
                if any(keyword in text for keyword in factor["keywords"]):
                    first_offset = offset if first_offset is None else min(first_offset, offset)
        return math.log1p(first_offset / 60.0) if first_offset is not None else 0.0
    if op == "early_late_keyword_delta":
        early = _temporal_density(
            cache,
            factor,
            factor.get("early_window_hours", [0.0, 24.0]),
        )
        late = _temporal_density(
            cache,
            factor,
            factor.get("late_window_hours", [24.0, 72.0]),
        )
        return late - early
    if op == "keyword_persistence":
        windows = factor.get("windows_hours", DEFAULT_TEMPORAL_WINDOWS)
        active = 0
        for window in windows:
            hits, _ = _cached_temporal_keyword_hits(
                cache,
                factor["sources"],
                factor["keywords"],
                float(window[0]),
                float(window[1]),
            )
            active += int(hits > 0)
        return active / max(len(windows), 1)
    raise ValueError(f"Unsupported operation: {op}")


def _materialize_sample_row(
    sample: dict[str, Any],
    factors: list[dict[str, Any]],
    label_key: str,
) -> tuple[list[float], list[float], int, str]:
    cache = _sample_text_cache(sample)
    counts = cache["counts"]
    base_row = _base_features_from_counts(
        counts["conditions"],
        counts["procedures"],
        counts["drugs"],
    )
    factor_row = [_compute_factor_cached(cache, factor) for factor in factors]
    return base_row, factor_row, int(sample[label_key]), str(sample["patient_id"])


def _materialize_rows_chunk(args):
    rows, factors, label_key = args
    base_rows = []
    factor_rows = []
    labels = []
    patient_ids = []
    for sample in rows:
        base_row, factor_row, label, patient_id = _materialize_sample_row(
            sample,
            factors,
            label_key,
        )
        base_rows.append(base_row)
        factor_rows.append(factor_row)
        labels.append(label)
        patient_ids.append(patient_id)
    return base_rows, factor_rows, labels, patient_ids


def _chunk_rows(rows: list[dict[str, Any]], chunk_size: int):
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _materialize_base_sample_frame(
    frame: pl.DataFrame,
    factors: list[dict[str, Any]],
    label_key: str,
):
    base_rows = []
    factor_rows = []
    labels = []
    patient_ids = []
    rows = frame.to_dicts()
    total = len(rows)
    print(f"Materializing direct tabular factor matrix: n={total}", flush=True)
    workers = int(
        os.environ.get(
            "MEDICAL_TABULAR_FACTOR_WORKERS",
            os.environ.get("NUM_WORKERS", "1"),
        )
    )
    if workers > 1 and total >= 5000:
        chunk_size = int(os.environ.get("MEDICAL_TABULAR_FACTOR_CHUNK_SIZE", "5000"))
        chunks = [(chunk, factors, label_key) for chunk in _chunk_rows(rows, chunk_size)]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk_result in pool.map(_materialize_rows_chunk, chunks):
                chunk_base, chunk_factors, chunk_labels, chunk_patient_ids = chunk_result
                base_rows.extend(chunk_base)
                factor_rows.extend(chunk_factors)
                labels.extend(chunk_labels)
                patient_ids.extend(chunk_patient_ids)
                completed += len(chunk_labels)
                print(
                    f"  materialized {completed}/{total} samples",
                    flush=True,
                )
    else:
        for idx, sample in enumerate(rows):
            base_row, factor_row, label, patient_id = _materialize_sample_row(
                sample,
                factors,
                label_key,
            )
            base_rows.append(base_row)
            factor_rows.append(factor_row)
            labels.append(label)
            patient_ids.append(patient_id)
            if (idx + 1) % 10000 == 0 or idx + 1 == total:
                print(f"  materialized {idx + 1}/{total} samples", flush=True)
    return (
        np.asarray(base_rows, dtype=np.float32),
        np.asarray(factor_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(patient_ids, dtype=object),
    )


def _select_matrix(base_x, factor_x, use_base: bool, use_factors: bool):
    if use_base and use_factors:
        return np.concatenate([base_x, factor_x], axis=1)
    if use_base:
        return base_x
    if use_factors:
        return factor_x
    raise ValueError("At least one of use_base or use_factors must be true.")


def _split_indices_by_patient(
    patient_ids: np.ndarray,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    rng = np.random.default_rng(seed)
    patient_to_indices: dict[str, list[int]] = {}
    for idx, patient_id in enumerate(patient_ids):
        patient_to_indices.setdefault(str(patient_id), []).append(idx)

    unique_patients = np.asarray(list(patient_to_indices), dtype=object)
    rng.shuffle(unique_patients)
    n = len(unique_patients)
    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))
    groups = {
        "train": unique_patients[:n_train],
        "val": unique_patients[n_train:n_val],
        "test": unique_patients[n_val:],
    }
    return {
        name: np.asarray(
            [idx for patient_id in ids for idx in patient_to_indices[str(patient_id)]],
            dtype=np.int64,
        )
        for name, ids in groups.items()
    }


def _task_cache_dir(base_dataset, task) -> Path:
    task_params = json.dumps(
        {
            **vars(task),
            "input_schema": task.input_schema,
            "output_schema": task.output_schema,
        },
        sort_keys=True,
        default=str,
    )
    return (
        Path(base_dataset.cache_dir)
        / "tasks"
        / f"{task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
    )


def _load_or_build_raw_task_dataset(base_dataset, task, num_workers: int):
    cache_dir = _task_cache_dir(base_dataset, task)
    task_df_path = cache_dir / "task_df.ld"
    task_df_path.mkdir(parents=True, exist_ok=True)
    if not (task_df_path / "index.json").exists():
        print(f"Building raw tabular task cache: {task_df_path}", flush=True)
        base_dataset._task_transform(task, task_df_path, num_workers)
    else:
        print(f"Using cached raw tabular task cache: {task_df_path}", flush=True)
    return litdata.StreamingDataset(
        str(task_df_path),
        transform=lambda x: pickle.loads(x["sample"]),
    )


def _classification_scores(model, x, y, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    pred = model.predict(x)
    scores = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)
            scores["loss"] = float(log_loss(y, proba, labels=labels))
            if len(labels) == 2 and proba.shape[1] >= 2:
                positive_proba = proba[:, 1]
                scores["auroc"] = float(roc_auc_score(y, positive_proba))
                scores["prauc"] = float(average_precision_score(y, positive_proba))
        except Exception:
            pass
    return scores


def _fit_logistic(x_train, y_train, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    max_iter = int(os.environ.get("MEDICAL_LOGISTIC_MAX_ITER", "1000"))
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), clf).fit(x_train, y_train)


def _fit_sgd(x_train, y_train, seed: int):
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    max_iter = int(os.environ.get("MEDICAL_SGD_MAX_ITER", "1000"))
    clf = SGDClassifier(
        loss="log_loss",
        max_iter=max_iter,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), clf).fit(x_train, y_train)


def _fit_gbdt(x_train, y_train, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    max_iter = int(os.environ.get("MEDICAL_GBDT_MAX_ITER", "120"))
    learning_rate = float(os.environ.get("MEDICAL_GBDT_LEARNING_RATE", "0.05"))
    max_leaf_nodes = int(os.environ.get("MEDICAL_GBDT_MAX_LEAF_NODES", "31"))
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        early_stopping=True,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    return clf.fit(x_train, y_train, sample_weight=sample_weight)


def _csv_env(name: str, default: str) -> list[str]:
    return [
        item.strip().lower()
        for item in os.environ.get(name, default).split(",")
        if item.strip()
    ]


def _ensure_baseline_feature_set(feature_sets: list[str]) -> list[str]:
    if not str_to_bool(os.environ.get("MEDICAL_TABULAR_ALWAYS_BASELINE", "1")):
        return feature_sets
    if "baseline" in feature_sets:
        return feature_sets
    return ["baseline", *feature_sets]


def _fit_model(model_name: str, x_train, y_train, seed: int):
    if model_name == "sgd":
        return _fit_sgd(x_train, y_train, seed)
    if model_name == "logistic":
        return _fit_logistic(x_train, y_train, seed)
    if model_name == "gbdt":
        return _fit_gbdt(x_train, y_train, seed)
    raise ValueError(f"Unsupported MEDICAL_TABULAR_MODELS item: {model_name}")


def _logistic_importance(model, feature_names: list[str]) -> list[dict[str, float | str]]:
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return []
    weights = np.mean(np.abs(coef), axis=0)
    order = np.argsort(-weights)
    return [
        {"feature": feature_names[idx], "importance": float(weights[idx])}
        for idx in order
    ]


def _permutation_importance(model, x_val, y_val, feature_names: list[str]):
    if not str_to_bool(os.environ.get("MEDICAL_TABULAR_PERMUTATION_IMPORTANCE", "0")):
        return []
    from sklearn.inspection import permutation_importance

    repeats = int(os.environ.get("MEDICAL_TABULAR_PERMUTATION_REPEATS", "3"))
    result = permutation_importance(
        model,
        x_val,
        y_val,
        n_repeats=repeats,
        random_state=0,
        scoring="f1_macro",
    )
    order = np.argsort(-result.importances_mean)
    return [
        {
            "feature": feature_names[idx],
            "importance": float(result.importances_mean[idx]),
            "std": float(result.importances_std[idx]),
        }
        for idx in order
    ]


def _baseline_deltas(all_scores: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    metrics = [
        "accuracy",
        "auroc",
        "balanced_accuracy",
        "cohen_kappa",
        "f1_macro",
        "f1_weighted",
        "loss",
        "prauc",
    ]
    for key, scores in all_scores.items():
        if not key.startswith("combined_"):
            continue
        model_name = key.removeprefix("combined_")
        baseline_key = f"baseline_{model_name}"
        if baseline_key not in all_scores:
            continue
        deltas[key] = {}
        for split in ("val", "test"):
            deltas[key][split] = {}
            combined_split = scores.get(split, {})
            baseline_split = all_scores[baseline_key].get(split, {})
            for metric in metrics:
                if metric in combined_split and metric in baseline_split:
                    deltas[key][split][metric] = float(
                        combined_split[metric] - baseline_split[metric]
                    )
    return deltas


def _format_score_line(scores: dict[str, Any], metrics: tuple[str, ...]) -> str:
    parts = []
    for metric in metrics:
        value = scores.get(metric)
        if isinstance(value, (int, float)):
            parts.append(f"test_{metric}={float(value):.4f}")
    return ", ".join(parts)


def _metric_improved(metric: str, current: float, previous: float) -> bool:
    direction = get_task_config().metric_direction.get(metric, "max")
    if direction == "min":
        return current < previous
    return current > previous


def _update_best_artifacts(
    output_root: Path,
    result: dict[str, Any],
    factors: list[dict[str, Any]],
) -> dict[str, str]:
    task_config = get_task_config(result.get("task"))
    scores = result.get("scores", {}) or {}
    updated_paths: dict[str, str] = {}
    for metric in task_config.best_metrics:
        value = scores.get(metric)
        if not isinstance(value, (int, float)):
            continue
        path = output_root / f"best_workflow_summary_by_{metric}.json"
        previous_value = None
        if path.exists():
            try:
                previous = json.loads(path.read_text())
                previous_value = (
                    (previous.get("result", {}) or {})
                    .get("scores", {})
                    .get(metric)
                )
            except Exception:
                previous_value = None
        if previous_value is not None and not _metric_improved(
            metric,
            float(value),
            float(previous_value),
        ):
            continue
        summary = {
            "direction": os.environ.get("MEDICAL_DIRECTION", ""),
            "evaluator": result.get("evaluator", "tabular"),
            "workflow_flavor": os.environ.get("MEDICAL_WORKFLOW_FLAVOR", "alpha"),
            "task": task_config.name,
            "best_metric": metric,
            "best_metric_direction": task_config.metric_direction.get(metric, "max"),
            "best_metric_value": float(value),
            "previous_best_metric_value": previous_value,
            "result": result,
            "factors": factors,
        }
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        updated_paths[metric] = str(path)

    primary_path = output_root / "best_workflow_summary_primary.json"
    primary_metric = task_config.primary_metric
    metric_path = output_root / f"best_workflow_summary_by_{primary_metric}.json"
    if metric_path.exists():
        primary_path.write_text(metric_path.read_text())
        updated_paths["primary"] = str(primary_path)
    return updated_paths


class MedicalTabularFactorRunner(Developer[MedicalFactorExperiment]):
    """Evaluate medical factors with Qlib-like tabular light models."""

    def develop(
        self,
        exp: MedicalFactorExperiment,
        use_local: bool = True,
    ) -> MedicalFactorExperiment:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        new_factors = exp.factor_dicts()
        selection_policy = _factor_selection_policy(exp)
        based_factors = _collect_based_factors(exp)
        factors = _merge_factor_sets(based_factors, new_factors, selection_policy)
        if not factors:
            if str_to_bool(os.environ.get("MEDICAL_TABULAR_STRICT_SELECTION", "0")):
                raise ValueError(
                    "No factors left after applying tabular factor selection policy. "
                    f"requested={sorted(selection_policy['requested_factor_names'])}; "
                    f"denied={sorted(selection_policy['denied_factor_names'])}"
                )
            factors = new_factors or based_factors
            selection_policy["selection_fallback"] = "develop_empty_selection_fallback"
        fhash = factor_hash(factors)

        output_root = Path(os.environ.get("MEDICAL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
        output_root.mkdir(parents=True, exist_ok=True)
        factor_path = output_root / f"workflow_factors_{fhash}.json"
        factor_path.write_text(
            json.dumps(
                {
                    "based_factor_count": len(based_factors),
                    "new_factor_count": len(new_factors),
                    "evaluated_factor_count": len(factors),
                    "selection_policy": {
                        "based_policy": selection_policy["based_policy"],
                        "requested_factor_names": sorted(
                            selection_policy["requested_factor_names"]
                        ),
                        "active_requested_factor_names": sorted(
                            selection_policy.get("active_requested_factor_names", set())
                        ),
                        "requested_filter_applied": bool(
                            selection_policy.get("requested_filter_applied", False)
                        ),
                        "selection_fallback": selection_policy.get(
                            "selection_fallback", ""
                        ),
                        "denied_factor_names": sorted(
                            selection_policy["denied_factor_names"]
                        ),
                    },
                    "factors": factors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        eicu_root = Path(os.environ.get("EICU_ROOT", DEFAULT_EICU_ROOT)).expanduser()
        require_eicu_files(eicu_root)

        dev = str_to_bool(os.environ.get("EICU_DEV", "1"))
        batch_size = int(os.environ.get("BATCH_SIZE", os.environ.get("MEDICAL_BATCH_SIZE", "64")))
        num_workers = int(os.environ.get("NUM_WORKERS", os.environ.get("MEDICAL_NUM_WORKERS", "4")))
        seed = int(os.environ.get("SEED", "0"))
        task_name = _medical_eicu_task()
        task_config = get_task_config(task_name)
        label_key = task_config.label_key
        task_slug = task_config.slug

        default_cache_name = (
            f"{task_slug}_quantaalpha_tabular_dev"
            if dev
            else f"{task_slug}_quantaalpha_tabular_full"
        )
        cache_dir = Path(
            os.environ.get(
                "PYHEALTH_LOCAL_CACHE_DIR",
                DEFAULT_CACHE_ROOT / default_cache_name,
            )
        )
        run_root = Path(os.environ.get("PYHEALTH_LOCAL_OUTPUT_DIR", DEFAULT_RUN_ROOT))
        exp_name = (
            f"{task_slug}_quantaalpha_tabular_full_{fhash}"
            if not dev
            else f"{task_slug}_quantaalpha_tabular_dev_{fhash}"
        )
        run_dir = run_root / exp_name
        run_dir.mkdir(parents=True, exist_ok=True)

        use_direct_source = str_to_bool(
            os.environ.get("MEDICAL_TABULAR_DIRECT_SOURCE", "1")
        )
        if use_direct_source:
            stage_start = time.perf_counter()
            base_frame = _build_or_load_base_sample_frame(
                eicu_root,
                cache_dir,
                dev,
                task_name,
            )
            if len(base_frame) == 0:
                raise RuntimeError("Direct tabular sample table produced zero samples.")
            timings["base_sample_cache_sec"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            base_matrix, factor_matrix, labels_array, patient_ids = (
                _materialize_base_sample_frame(base_frame, factors, label_key)
            )
            timings["materialize_sec"] = time.perf_counter() - stage_start
        else:
            if task_name != "los":
                raise ValueError(
                    "The PyHealth task-cache path currently supports only LOS. "
                    "Use MEDICAL_TABULAR_DIRECT_SOURCE=1 for mortality/readmission."
                )
            stage_start = time.perf_counter()
            base_dataset = eICUDataset(
                root=str(eicu_root),
                tables=["diagnosis", "medication", "physicalexam"],
                cache_dir=str(cache_dir),
                num_workers=num_workers,
                dev=dev,
            )
            base_dataset.stats()
            timings["dataset_init_sec"] = time.perf_counter() - stage_start

            task = LengthOfStayPredictioneICUWithTabularFactors(factors)
            stage_start = time.perf_counter()
            raw_dataset = _load_or_build_raw_task_dataset(
                base_dataset,
                task,
                num_workers,
            )
            if len(raw_dataset) == 0:
                raise RuntimeError("Task produced zero samples.")
            timings["raw_task_cache_sec"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            base_matrix, factor_matrix, labels_array, patient_ids = (
                _materialize_raw_dataset(
                    raw_dataset,
                    label_key,
                )
            )
            timings["materialize_sec"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        split_indices = _split_indices_by_patient(patient_ids, seed=seed)
        timings["split_sec"] = time.perf_counter() - stage_start
        summary = summarize_factor_matrix(factor_matrix, factors)
        summary_path = output_root / f"workflow_factor_summary_{fhash}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
        train_base, train_factors, y_train = (
            base_matrix[train_idx],
            factor_matrix[train_idx],
            labels_array[train_idx],
        )
        val_base, val_factors, y_val = (
            base_matrix[val_idx],
            factor_matrix[val_idx],
            labels_array[val_idx],
        )
        test_base, test_factors, y_test = (
            base_matrix[test_idx],
            factor_matrix[test_idx],
            labels_array[test_idx],
        )

        factor_names = [factor["name"] for factor in factors]
        feature_set_configs = {
            "baseline": (True, False, BASE_FEATURE_NAMES),
            "factors": (False, True, factor_names),
            "combined": (True, True, BASE_FEATURE_NAMES + factor_names),
        }
        selected_feature_sets = _csv_env(
            "MEDICAL_TABULAR_FEATURE_SETS",
            "baseline,factors,combined",
        )
        selected_feature_sets = _ensure_baseline_feature_set(selected_feature_sets)
        unknown_feature_sets = set(selected_feature_sets) - set(feature_set_configs)
        if unknown_feature_sets:
            raise ValueError(
                "Unsupported MEDICAL_TABULAR_FEATURE_SETS item(s): "
                f"{sorted(unknown_feature_sets)}"
            )
        selected_models = _csv_env("MEDICAL_TABULAR_MODELS", "logistic")
        unknown_models = set(selected_models) - {"sgd", "logistic", "gbdt"}
        if unknown_models:
            raise ValueError(
                f"Unsupported MEDICAL_TABULAR_MODELS item(s): {sorted(unknown_models)}"
            )

        labels = task_config.labels
        all_scores: dict[str, Any] = {}
        importances: dict[str, Any] = {}
        model_fit_start = time.perf_counter()

        for feature_set in selected_feature_sets:
            use_base, use_factors, feature_names = feature_set_configs[feature_set]
            x_train = _select_matrix(train_base, train_factors, use_base, use_factors)
            x_val = _select_matrix(val_base, val_factors, use_base, use_factors)
            x_test = _select_matrix(test_base, test_factors, use_base, use_factors)

            for model_name in selected_models:
                model_key = f"{feature_set}_{model_name}"
                print(
                    f"Fitting tabular model {model_key}: "
                    f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}",
                    flush=True,
                )
                model = _fit_model(model_name, x_train, y_train, seed)
                all_scores[model_key] = {
                    "val": _classification_scores(model, x_val, y_val, labels),
                    "test": _classification_scores(model, x_test, y_test, labels),
                }
                if model_name in {"sgd", "logistic"}:
                    importances[model_key] = _logistic_importance(model, feature_names)
                else:
                    importances[model_key] = _permutation_importance(
                        model, x_val, y_val, feature_names
                    )
                print(
                    f"Finished {model_key}: "
                    f"{_format_score_line(all_scores[model_key]['test'], task_config.report_metrics)}",
                    flush=True,
                )
        timings["model_fit_and_score_sec"] = time.perf_counter() - model_fit_start
        baseline_deltas = _baseline_deltas(all_scores)

        primary_key = os.environ.get("MEDICAL_TABULAR_PRIMARY_MODEL", "combined_logistic")
        if primary_key not in all_scores:
            for candidate in (
                "combined_logistic",
                "factors_logistic",
                "combined_sgd",
                "factors_sgd",
                sorted(all_scores)[0],
            ):
                if candidate in all_scores:
                    primary_key = candidate
                    break
        scores = all_scores[primary_key]["test"]

        scores_path = run_dir / "final_test_scores.json"
        scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
        all_scores_path = run_dir / "all_model_scores.json"
        all_scores_path.write_text(json.dumps(all_scores, indent=2, sort_keys=True) + "\n")
        baseline_deltas_path = run_dir / "baseline_deltas.json"
        baseline_deltas_path.write_text(
            json.dumps(baseline_deltas, indent=2, sort_keys=True) + "\n"
        )
        importance_path = run_dir / "feature_importance.json"
        importance_path.write_text(
            json.dumps(importances, indent=2, sort_keys=True) + "\n"
        )

        exp.result = {
            "scores": scores,
            "all_model_scores": all_scores,
            "baseline_deltas": baseline_deltas,
            "primary_model": primary_key,
            "factor_hash": fhash,
            "factor_path": str(factor_path),
            "factor_summary_path": str(summary_path),
            "scores_path": str(scores_path),
            "all_scores_path": str(all_scores_path),
            "baseline_deltas_path": str(baseline_deltas_path),
            "feature_importance_path": str(importance_path),
            "run_dir": str(run_dir),
            "cache_dir": str(cache_dir),
            "sample_size": int(len(labels_array)),
            "task": task_name,
            "source_profile": source_profile(),
            "label_key": label_key,
            "primary_metric": task_config.primary_metric,
            "report_metrics": list(task_config.report_metrics),
            "best_metrics": list(task_config.best_metrics),
            "observation_end_hours": observation_end_hours(task_name),
            "based_factor_count": int(len(based_factors)),
            "new_factor_count": int(len(new_factors)),
            "evaluated_factor_count": int(len(factors)),
            "evaluated_factor_names": [factor["name"] for factor in factors],
            "selection_policy": {
                "based_policy": selection_policy["based_policy"],
                "requested_factor_names": sorted(
                    selection_policy["requested_factor_names"]
                ),
                "active_requested_factor_names": sorted(
                    selection_policy.get("active_requested_factor_names", set())
                ),
                "requested_filter_applied": bool(
                    selection_policy.get("requested_filter_applied", False)
                ),
                "selection_fallback": selection_policy.get("selection_fallback", ""),
                "denied_factor_names": sorted(
                    selection_policy["denied_factor_names"]
                ),
            },
            "label_counts": {
                int(label): int(count)
                for label, count in sorted(Counter(labels_array.tolist()).items())
            },
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "batch_size": batch_size,
            "timings_sec": {
                **{k: round(v, 3) for k, v in timings.items()},
                "total_sec": round(time.perf_counter() - total_start, 3),
            },
            "factor_summary": summary,
            "evaluator": "tabular",
            "feature_sets": selected_feature_sets,
            "models": sorted(all_scores),
            "selected_models": selected_models,
        }
        exp.result["best_summary_paths"] = _update_best_artifacts(
            output_root,
            exp.result,
            factors,
        )
        exp.sub_results = {
            k: float(v) for k, v in scores.items() if isinstance(v, (int, float))
        }
        return exp
