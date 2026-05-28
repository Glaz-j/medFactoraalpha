"""Observed eICU vocabulary profiling for medical factor prompts."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from quantaalpha.medical.source_profile import PYHEALTH_STANDARD, source_profile
from quantaalpha.medical.task_config import get_task_config

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "pyhealth_cache"

SOURCE_TO_BASE_COLUMN = {
    "conditions": "conditions",
    "procedures": "procedures",
    "drugs": "drugs",
}
SOURCE_TO_CSV = {
    "conditions": ("diagnosis.csv", "diagnosisstring"),
    "procedures": ("physicalExam.csv", "physicalexamvalue"),
    "drugs": ("medication.csv", "drugname"),
}
STOPWORDS = {
    "able",
    "and",
    "are",
    "assessment",
    "care",
    "for",
    "from",
    "has",
    "left",
    "not",
    "normal",
    "note",
    "notes",
    "other",
    "patient",
    "performed",
    "right",
    "status",
    "the",
    "with",
}


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _eicu_csv_path(root: Path, filename: str) -> Path:
    path = root / filename
    if path.exists():
        return path
    gz_path = root / f"{filename}.gz"
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Missing eICU table: {path} or {gz_path}")


def _normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_/-]{2,}", text)
        if token not in STOPWORDS
    ]


def _source_to_csv(task: str, profile: str) -> dict[str, tuple[str, str]]:
    if profile == PYHEALTH_STANDARD and task != "los":
        return {
            "conditions": ("diagnosis.csv", "icd9code"),
            "procedures": ("physicalExam.csv", "physicalexampath"),
            "drugs": ("medication.csv", "drugname"),
        }
    return SOURCE_TO_CSV


def _base_sample_path(dev: bool, task: str, profile: str) -> Path:
    task_slug = get_task_config(task).slug
    cache_name = (
        f"{task_slug}_quantaalpha_tabular_dev"
        if dev
        else f"{task_slug}_quantaalpha_tabular_full"
    )
    temporal_source = str_to_bool(os.environ.get("MEDICAL_TEMPORAL_SOURCE", "1"))
    numeric_temporal_source = str_to_bool(
        os.environ.get("MEDICAL_NUMERIC_TEMPORAL_SOURCE", "1")
    )
    if profile == PYHEALTH_STANDARD:
        numeric_temporal_source = False
    temporal_suffix = (
        f"_{profile}_temporal_numeric_v2"
        if temporal_source and numeric_temporal_source
        else f"_{profile}_temporal"
        if temporal_source
        else f"_{profile}"
    )
    if task == "los":
        sample_name = (
            f"tabular_base_samples_dev{temporal_suffix}.parquet"
            if dev
            else f"tabular_base_samples_full{temporal_suffix}.parquet"
        )
    else:
        sample_name = (
            f"tabular_base_samples_{task}_dev{temporal_suffix}.parquet"
            if dev
            else f"tabular_base_samples_{task}_full{temporal_suffix}.parquet"
        )
    return DEFAULT_CACHE_ROOT / cache_name / sample_name


def _top_values_from_base_sample(
    base_sample_path: Path,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    column = SOURCE_TO_BASE_COLUMN[source]
    frame = (
        pl.scan_parquet(str(base_sample_path))
        .select(pl.col(column).explode().cast(pl.Utf8).alias("value"))
        .with_columns(pl.col("value").str.to_lowercase().str.strip_chars())
        .filter(pl.col("value").is_not_null() & (pl.col("value").str.len_chars() > 0))
        .group_by("value")
        .len()
        .sort("len", descending=True)
        .limit(limit)
        .collect()
    )
    return [
        {"text": row["value"], "count": int(row["len"])}
        for row in frame.to_dicts()
    ]


def _top_values_from_csv(
    eicu_root: Path,
    source: str,
    limit: int,
    csv_sources: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    filename, column = (csv_sources or SOURCE_TO_CSV)[source]
    frame = (
        pl.scan_csv(
            str(_eicu_csv_path(eicu_root, filename)),
            infer_schema_length=10000,
            ignore_errors=True,
        )
        .select(pl.col(column).cast(pl.Utf8).alias("value"))
        .with_columns(pl.col("value").str.to_lowercase().str.strip_chars())
        .filter(pl.col("value").is_not_null() & (pl.col("value").str.len_chars() > 0))
        .group_by("value")
        .len()
        .sort("len", descending=True)
        .limit(limit)
        .collect()
    )
    return [
        {"text": row["value"], "count": int(row["len"])}
        for row in frame.to_dicts()
    ]


def _token_profile(top_values: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in top_values:
        counter.update({token: int(item["count"]) for token in _tokens(item["text"])})
    return [
        {"token": token, "count": int(count)}
        for token, count in counter.most_common(limit)
    ]


def build_or_load_vocab_profile(
    dev: bool | None = None,
    force: bool = False,
) -> dict[str, Any]:
    dev = str_to_bool(os.environ.get("EICU_DEV", "1")) if dev is None else dev
    task = get_task_config().name
    profile_name = source_profile()
    profile_root = Path(
        os.environ.get(
            "MEDICAL_VOCAB_PROFILE_DIR",
            PROJECT_ROOT / "results" / "quantaalpha_medical_workflow",
        )
    )
    profile_root.mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / (
        f"eicu_vocab_profile_{task}_{profile_name}_dev.json"
        if dev
        else f"eicu_vocab_profile_{task}_{profile_name}_full.json"
    )
    if profile_path.exists() and not force:
        return json.loads(profile_path.read_text())

    eicu_root = Path(os.environ.get("EICU_ROOT", DEFAULT_EICU_ROOT)).expanduser()
    value_limit = int(os.environ.get("MEDICAL_VOCAB_PROFILE_VALUE_LIMIT", "120"))
    token_limit = int(os.environ.get("MEDICAL_VOCAB_PROFILE_TOKEN_LIMIT", "80"))
    base_path = Path(
        os.environ.get(
            "MEDICAL_VOCAB_BASE_SAMPLE_PATH",
            _base_sample_path(dev, task, profile_name),
        )
    )
    csv_sources = _source_to_csv(task, profile_name)
    source_data: dict[str, Any] = {}
    source_mode = "base_sample" if base_path.exists() else "csv"

    for source in SOURCE_TO_BASE_COLUMN:
        if base_path.exists():
            top_values = _top_values_from_base_sample(base_path, source, value_limit)
        else:
            top_values = _top_values_from_csv(eicu_root, source, value_limit, csv_sources)
        source_data[source] = {
            "top_values": top_values,
            "top_tokens": _token_profile(top_values, token_limit),
        }

    profile = {
        "dataset": "eicu",
        "dev": dev,
        "task": task,
        "source_profile": profile_name,
        "csv_sources": {
            source: {"table": filename, "column": column}
            for source, (filename, column) in csv_sources.items()
        },
        "source_mode": source_mode,
        "base_sample_path": str(base_path) if base_path.exists() else "",
        "profile_path": str(profile_path),
        "sources": source_data,
    }
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def vocab_profile_prompt() -> str:
    if not str_to_bool(os.environ.get("MEDICAL_VOCAB_PROFILE", "1")):
        return ""
    try:
        profile = build_or_load_vocab_profile(
            force=str_to_bool(os.environ.get("MEDICAL_VOCAB_PROFILE_FORCE", "0"))
        )
    except Exception as exc:
        return f"Observed eICU vocabulary profile unavailable: {exc}"

    value_n = int(os.environ.get("MEDICAL_VOCAB_PROMPT_VALUES", "25"))
    token_n = int(os.environ.get("MEDICAL_VOCAB_PROMPT_TOKENS", "35"))
    blocks = [
        "Observed eICU vocabulary profile:",
        (
            "Use these observed strings/tokens when proposing keyword factors. "
            "Avoid clinically plausible keywords that are absent from this profile, "
            "unless you pair them with observed lexical variants."
        ),
    ]
    for source in ("conditions", "procedures", "drugs"):
        data = profile["sources"][source]
        values = ", ".join(
            f"{item['text']} ({item['count']})"
            for item in data["top_values"][:value_n]
        )
        tokens = ", ".join(
            f"{item['token']} ({item['count']})"
            for item in data["top_tokens"][:token_n]
        )
        blocks.append(f"{source} top observed values: {values}")
        blocks.append(f"{source} top observed tokens: {tokens}")
    return "\n".join(blocks)
