"""Safe symbolic factor DSL for medical PyHealth tasks."""

from __future__ import annotations

import math
import re
from typing import Any

from quantaalpha.medical.safe_python import SAFE_PYTHON_OPERATION, compute_safe_python_factor


VALID_FEATURES = {"conditions", "procedures", "drugs"}
VALID_NUMERIC_SOURCES = {
    "vital_sao2",
    "vital_heartrate",
    "vital_respiration",
    "vital_temperature",
    "vital_systemicsystolic",
    "vital_systemicdiastolic",
    "vital_systemicmean",
    "vital_noninvasivesystolic",
    "vital_noninvasivediastolic",
    "vital_noninvasivemean",
    "vital_cardiacoutput",
    "io_intake_total",
    "io_output_total",
    "io_net_total",
    "io_dialysis_total",
    "resp_fio2",
    "resp_peep",
    "lab_albumin",
    "lab_bilirubin",
    "lab_bun",
    "lab_creatinine",
    "lab_glucose",
    "lab_hematocrit",
    "lab_hemoglobin",
    "lab_lactate",
    "lab_platelets",
    "lab_potassium",
    "lab_sodium",
    "lab_wbc",
    "lab_ph",
    "lab_pao2",
    "lab_paco2",
    "nurse_gcs_total",
    "nurse_pain_score",
    "nurse_rass",
    "nurse_spo2",
    "nurse_bedside_glucose",
    "infusion_vasopressor_rate",
    "infusion_norepinephrine_rate",
    "infusion_propofol_rate",
    "infusion_insulin_rate",
    "infusion_milrinone_rate",
    "apache_urine",
    "apache_wbc",
    "apache_temperature",
    "apache_respiratoryrate",
    "apache_sodium",
    "apache_heartrate",
    "apache_meanbp",
    "apache_ph",
    "apache_hematocrit",
    "apache_creatinine",
    "apache_albumin",
    "apache_pao2",
    "apache_pco2",
    "apache_bun",
    "apache_glucose",
    "apache_bilirubin",
    "apache_fio2",
}
VALID_OPS = {
    "log_count",
    "count_ratio",
    "keyword_any",
    "keyword_count",
    "keyword_density",
    "temporal_keyword_count",
    "temporal_keyword_density",
    "first_keyword_offset",
    "early_late_keyword_delta",
    "keyword_persistence",
    "numeric_window_mean",
    "numeric_window_min",
    "numeric_window_max",
    "numeric_window_std",
    "numeric_window_last",
    "numeric_window_count",
    "numeric_window_slope",
    "numeric_early_late_delta",
    "numeric_abnormal_fraction",
    "numeric_persistence",
    "numeric_source_interaction",
    "keyword_gated_numeric",
    "safe_python",
}

TEMPORAL_OPS = {
    "temporal_keyword_count",
    "temporal_keyword_density",
    "first_keyword_offset",
    "early_late_keyword_delta",
    "keyword_persistence",
}

NUMERIC_TEMPORAL_OPS = {
    "numeric_window_mean",
    "numeric_window_min",
    "numeric_window_max",
    "numeric_window_std",
    "numeric_window_last",
    "numeric_window_count",
    "numeric_window_slope",
    "numeric_early_late_delta",
    "numeric_abnormal_fraction",
    "numeric_persistence",
    "numeric_source_interaction",
    "keyword_gated_numeric",
}

DEFAULT_TEMPORAL_WINDOWS = [[0.0, 24.0], [24.0, 72.0], [72.0, 168.0]]
NUMERIC_AGGREGATIONS = {"mean", "min", "max", "std", "last", "count", "slope"}
NUMERIC_OPERATORS = {"add", "sub", "mul", "ratio", "max", "min"}
NUMERIC_TRANSFORMS = {"identity", "log1p", "abs", "neg", "sqrt_abs"}


def safe_name(name: str, fallback: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return name[:64] or fallback


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_text(value: Any) -> list[str]:
    texts = []
    if isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(flatten_text(item))
    elif value is not None:
        texts.append(str(value))
    return texts


def as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_window(value: Any, default: list[float]) -> list[float]:
    items = as_list(value)
    if len(items) < 2:
        return default
    fallback_start = default[0] if len(default) >= 2 else 0.0
    fallback_end = default[1] if len(default) >= 2 else fallback_start + 24.0
    start = as_float(items[0], fallback_start)
    end = as_float(items[1], fallback_end)
    if end <= start:
        return default
    return [start, end]


def normalize_windows(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return DEFAULT_TEMPORAL_WINDOWS
    windows = []
    for item in value[:8]:
        window = normalize_window(item, [])
        if len(window) == 2:
            windows.append(window)
    return windows or DEFAULT_TEMPORAL_WINDOWS


def normalize_factors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_factors = payload.get("factors", [])
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("LLM response must contain a non-empty `factors` list.")

    normalized = []
    for idx, raw in enumerate(raw_factors, start=1):
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("operation", "")).strip().lower()
        if op not in VALID_OPS:
            continue

        numeric_source = str(raw.get("numeric_source", "")).strip().lower()
        secondary_numeric_source = str(
            raw.get("secondary_numeric_source", "")
        ).strip().lower()
        aggregation = str(raw.get("aggregation", "mean")).strip().lower() or "mean"
        secondary_aggregation = str(
            raw.get("secondary_aggregation", aggregation)
        ).strip().lower() or aggregation
        operator = str(raw.get("operator", "")).strip().lower()
        transform = str(raw.get("transform", "identity")).strip().lower() or "identity"
        sources = raw.get("sources", raw.get("source", []))
        sources = [str(item).strip() for item in as_list(sources)]
        sources = [item for item in sources if item in VALID_FEATURES]

        numerator_source = str(raw.get("numerator_source", "")).strip()
        denominator_source = str(raw.get("denominator_source", "")).strip()
        code = str(raw.get("code", ""))
        if op == "safe_python":
            if not code.strip():
                continue
        elif op in NUMERIC_TEMPORAL_OPS:
            if numeric_source not in VALID_NUMERIC_SOURCES:
                continue
            if transform not in NUMERIC_TRANSFORMS:
                transform = "identity"
            if op in {"numeric_source_interaction", "keyword_gated_numeric"}:
                if aggregation not in NUMERIC_AGGREGATIONS:
                    aggregation = "mean"
            if op == "numeric_source_interaction":
                if secondary_numeric_source not in VALID_NUMERIC_SOURCES:
                    continue
                if secondary_aggregation not in NUMERIC_AGGREGATIONS:
                    secondary_aggregation = aggregation
                if operator not in NUMERIC_OPERATORS:
                    continue
        elif op == "count_ratio":
            if numerator_source not in VALID_FEATURES or denominator_source not in VALID_FEATURES:
                continue
        elif not sources:
            continue

        keywords = [
            str(item).strip().lower()
            for item in as_list(raw.get("keywords", []))
            if str(item).strip()
        ][:16]
        if (op.startswith("keyword") or op in TEMPORAL_OPS) and not keywords:
            continue
        if op == "keyword_gated_numeric" and (not sources or not keywords):
            continue

        window_start_hours = as_float(raw.get("window_start_hours"), 0.0)
        window_end_hours = as_float(raw.get("window_end_hours"), 24.0)
        if window_end_hours <= window_start_hours:
            window_start_hours, window_end_hours = 0.0, 24.0
        early_window_hours = normalize_window(
            raw.get("early_window_hours"),
            [0.0, 24.0],
        )
        late_window_hours = normalize_window(
            raw.get("late_window_hours"),
            [24.0, 72.0],
        )
        windows_hours = normalize_windows(raw.get("windows_hours"))
        abnormal_low = raw.get("abnormal_low")
        abnormal_high = raw.get("abnormal_high")
        abnormal_low = None if abnormal_low in (None, "") else as_float(abnormal_low, math.nan)
        abnormal_high = None if abnormal_high in (None, "") else as_float(abnormal_high, math.nan)
        if isinstance(abnormal_low, float) and math.isnan(abnormal_low):
            abnormal_low = None
        if isinstance(abnormal_high, float) and math.isnan(abnormal_high):
            abnormal_high = None
        threshold = raw.get("threshold")
        threshold = None if threshold in (None, "") else as_float(threshold, math.nan)
        if isinstance(threshold, float) and math.isnan(threshold):
            threshold = None

        normalized.append(
            {
                "name": safe_name(str(raw.get("name", "")), f"factor_{idx}"),
                "description": str(raw.get("description", ""))[:500],
                "rationale": str(raw.get("rationale", ""))[:500],
                "operation": op,
                "sources": sources,
                "numeric_source": numeric_source,
                "secondary_numeric_source": secondary_numeric_source,
                "aggregation": aggregation,
                "secondary_aggregation": secondary_aggregation,
                "operator": operator,
                "transform": transform,
                "keywords": keywords,
                "numerator_source": numerator_source,
                "denominator_source": denominator_source,
                "window_start_hours": window_start_hours,
                "window_end_hours": window_end_hours,
                "early_window_hours": early_window_hours,
                "late_window_hours": late_window_hours,
                "windows_hours": windows_hours,
                "abnormal_low": abnormal_low,
                "abnormal_high": abnormal_high,
                "threshold": threshold,
                "code": code,
            }
        )

    if not normalized:
        raise ValueError("No valid factors survived DSL validation.")
    return normalized[:12]


def text_for_sources(sample: dict[str, Any], sources: list[str]) -> list[str]:
    texts: list[str] = []
    for source in sources:
        texts.extend(flatten_text(sample.get(source, [])))
    return texts


def keyword_hits(texts: list[str], keywords: list[str]) -> int:
    lowered = [text.lower() for text in texts]
    hits = 0
    for text in lowered:
        if any(keyword in text for keyword in keywords):
            hits += 1
    return hits


def offset_values(sample: dict[str, Any], source: str) -> list[float | None]:
    offsets = sample.get(f"{source}_offsets", [])
    if not isinstance(offsets, (list, tuple)):
        return []
    values: list[float | None] = []
    for offset in offsets:
        try:
            if offset is None:
                values.append(None)
            else:
                values.append(float(offset))
        except (TypeError, ValueError):
            values.append(None)
    return values


def temporal_events_for_sources(
    sample: dict[str, Any],
    sources: list[str],
) -> list[tuple[str, float | None]]:
    events: list[tuple[str, float | None]] = []
    for source in sources:
        texts = flatten_text(sample.get(source, []))
        offsets = offset_values(sample, source)
        for idx, text in enumerate(texts):
            offset = offsets[idx] if idx < len(offsets) else None
            events.append((text, offset))
    return events


def event_in_window(offset_min: float | None, start_hours: float, end_hours: float) -> bool:
    if offset_min is None:
        return False
    offset_hours = offset_min / 60.0
    return offset_hours >= start_hours and offset_hours < end_hours


def temporal_keyword_hits(
    events: list[tuple[str, float | None]],
    keywords: list[str],
    start_hours: float,
    end_hours: float,
) -> tuple[int, int]:
    hits = 0
    total = 0
    for text, offset in events:
        if not event_in_window(offset, start_hours, end_hours):
            continue
        total += 1
        text_l = text.lower()
        if any(keyword in text_l for keyword in keywords):
            hits += 1
    return hits, total


def temporal_keyword_density(
    events: list[tuple[str, float | None]],
    keywords: list[str],
    window: list[float],
) -> float:
    hits, total = temporal_keyword_hits(events, keywords, window[0], window[1])
    return hits / max(total, 1)


def numeric_events_for_source(
    sample: dict[str, Any],
    numeric_source: str,
) -> list[tuple[float, float | None]]:
    values = sample.get(f"{numeric_source}_values", [])
    offsets = sample.get(f"{numeric_source}_offsets", [])
    if not isinstance(values, (list, tuple)):
        values = []
    if not isinstance(offsets, (list, tuple)):
        offsets = []
    events: list[tuple[float, float | None]] = []
    for idx, raw_value in enumerate(values):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        raw_offset = offsets[idx] if idx < len(offsets) else None
        try:
            offset = float(raw_offset) if raw_offset is not None else None
        except (TypeError, ValueError):
            offset = None
        events.append((value, offset))
    return events


def numeric_window_values(
    events: list[tuple[float, float | None]],
    window: list[float],
) -> list[float]:
    return [
        value
        for value, offset in events
        if event_in_window(offset, window[0], window[1])
    ]


def numeric_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def numeric_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = numeric_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def numeric_last(events: list[tuple[float, float | None]], window: list[float]) -> float:
    selected = [
        (offset, value)
        for value, offset in events
        if event_in_window(offset, window[0], window[1])
    ]
    if not selected:
        return 0.0
    selected.sort(key=lambda item: item[0] if item[0] is not None else -math.inf)
    return selected[-1][1]


def numeric_slope(events: list[tuple[float, float | None]], window: list[float]) -> float:
    selected = [
        (float(offset) / 60.0, value)
        for value, offset in events
        if event_in_window(offset, window[0], window[1])
    ]
    if len(selected) < 2:
        return 0.0
    xs = [item[0] for item in selected]
    ys = [item[1] for item in selected]
    x_mean = numeric_mean(xs)
    y_mean = numeric_mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in selected) / denom


def aggregate_numeric_values(
    events: list[tuple[float, float | None]],
    window: list[float],
    aggregation: str,
) -> float:
    values = numeric_window_values(events, window)
    if aggregation == "mean":
        return numeric_mean(values)
    if aggregation == "min":
        return min(values) if values else 0.0
    if aggregation == "max":
        return max(values) if values else 0.0
    if aggregation == "std":
        return numeric_std(values)
    if aggregation == "last":
        return numeric_last(events, window)
    if aggregation == "count":
        return float(len(values))
    if aggregation == "slope":
        return numeric_slope(events, window)
    return numeric_mean(values)


def apply_numeric_operator(left: float, right: float, operator: str) -> float:
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    if operator == "ratio":
        return 0.0 if abs(right) < 1e-6 else left / right
    if operator == "max":
        return max(left, right)
    if operator == "min":
        return min(left, right)
    return 0.0


def apply_numeric_transform(value: float, transform: str | None) -> float:
    if not math.isfinite(value):
        return 0.0
    transform = transform or "identity"
    if transform == "identity":
        return value
    if transform == "log1p":
        return math.copysign(math.log1p(abs(value)), value)
    if transform == "abs":
        return abs(value)
    if transform == "neg":
        return -value
    if transform == "sqrt_abs":
        return math.sqrt(abs(value))
    return value


def numeric_abnormal_fraction(values: list[float], low: float | None, high: float | None) -> float:
    if not values or (low is None and high is None):
        return 0.0
    abnormal = 0
    for value in values:
        if (low is not None and value < low) or (high is not None and value > high):
            abnormal += 1
    return abnormal / len(values)


def compute_factor(sample: dict[str, Any], factor: dict[str, Any]) -> float:
    op = factor["operation"]
    if op == SAFE_PYTHON_OPERATION:
        return compute_safe_python_factor(sample, factor)
    if op in NUMERIC_TEMPORAL_OPS:
        events = numeric_events_for_source(sample, factor["numeric_source"])
        window = [
            factor.get("window_start_hours", 0.0),
            factor.get("window_end_hours", 24.0),
        ]
        values = numeric_window_values(events, window)
        if op == "numeric_window_mean":
            return apply_numeric_transform(numeric_mean(values), factor.get("transform"))
        if op == "numeric_window_min":
            return apply_numeric_transform(min(values) if values else 0.0, factor.get("transform"))
        if op == "numeric_window_max":
            return apply_numeric_transform(max(values) if values else 0.0, factor.get("transform"))
        if op == "numeric_window_std":
            return apply_numeric_transform(numeric_std(values), factor.get("transform"))
        if op == "numeric_window_last":
            return apply_numeric_transform(numeric_last(events, window), factor.get("transform"))
        if op == "numeric_window_count":
            return apply_numeric_transform(float(len(values)), factor.get("transform"))
        if op == "numeric_window_slope":
            return apply_numeric_transform(numeric_slope(events, window), factor.get("transform"))
        if op == "numeric_source_interaction":
            left = aggregate_numeric_values(
                events,
                window,
                factor.get("aggregation", "mean"),
            )
            right_events = numeric_events_for_source(
                sample,
                factor.get("secondary_numeric_source", ""),
            )
            right = aggregate_numeric_values(
                right_events,
                window,
                factor.get("secondary_aggregation", "mean"),
            )
            value = apply_numeric_operator(left, right, factor.get("operator", ""))
            return apply_numeric_transform(value, factor.get("transform"))
        if op == "keyword_gated_numeric":
            gate_events = temporal_events_for_sources(sample, factor.get("sources", []))
            hits, _ = temporal_keyword_hits(
                gate_events,
                factor.get("keywords", []),
                window[0],
                window[1],
            )
            if hits <= 0:
                return 0.0
            value = aggregate_numeric_values(
                events,
                window,
                factor.get("aggregation", "mean"),
            )
            return apply_numeric_transform(value, factor.get("transform"))
        if op == "numeric_abnormal_fraction":
            return apply_numeric_transform(
                numeric_abnormal_fraction(
                    values,
                    factor.get("abnormal_low"),
                    factor.get("abnormal_high"),
                ),
                factor.get("transform"),
            )
        if op == "numeric_early_late_delta":
            early = numeric_mean(
                numeric_window_values(events, factor.get("early_window_hours", [0.0, 24.0]))
            )
            late = numeric_mean(
                numeric_window_values(events, factor.get("late_window_hours", [24.0, 72.0]))
            )
            return apply_numeric_transform(late - early, factor.get("transform"))
        if op == "numeric_persistence":
            threshold = factor.get("threshold")
            low = factor.get("abnormal_low")
            high = factor.get("abnormal_high")
            active = 0
            windows = factor.get("windows_hours", DEFAULT_TEMPORAL_WINDOWS)
            for item in windows:
                vals = numeric_window_values(events, item)
                if threshold is not None:
                    active += int(any(abs(value) >= threshold for value in vals))
                else:
                    active += int(numeric_abnormal_fraction(vals, low, high) > 0)
            return apply_numeric_transform(
                active / max(len(windows), 1),
                factor.get("transform"),
            )

    if op == "log_count":
        source = factor["sources"][0]
        return math.log1p(len(flatten_text(sample.get(source, []))))
    if op == "count_ratio":
        numerator = len(flatten_text(sample.get(factor["numerator_source"], [])))
        denominator = len(flatten_text(sample.get(factor["denominator_source"], [])))
        return numerator / max(denominator, 1)

    texts = text_for_sources(sample, factor["sources"])
    hits = keyword_hits(texts, factor["keywords"])
    if op == "keyword_any":
        return float(hits > 0)
    if op == "keyword_count":
        return float(hits)
    if op == "keyword_density":
        return hits / max(len(texts), 1)
    if op in TEMPORAL_OPS:
        events = temporal_events_for_sources(sample, factor["sources"])
        if op == "temporal_keyword_count":
            hits, _ = temporal_keyword_hits(
                events,
                factor["keywords"],
                factor["window_start_hours"],
                factor["window_end_hours"],
            )
            return float(hits)
        if op == "temporal_keyword_density":
            hits, total = temporal_keyword_hits(
                events,
                factor["keywords"],
                factor["window_start_hours"],
                factor["window_end_hours"],
            )
            return hits / max(total, 1)
        if op == "first_keyword_offset":
            first_offset: float | None = None
            for text, offset in events:
                if offset is None or offset < 0:
                    continue
                if any(keyword in text.lower() for keyword in factor["keywords"]):
                    first_offset = offset if first_offset is None else min(first_offset, offset)
            return math.log1p(first_offset / 60.0) if first_offset is not None else 0.0
        if op == "early_late_keyword_delta":
            early = temporal_keyword_density(
                events,
                factor["keywords"],
                factor["early_window_hours"],
            )
            late = temporal_keyword_density(
                events,
                factor["keywords"],
                factor["late_window_hours"],
            )
            return late - early
        if op == "keyword_persistence":
            active = 0
            windows = factor.get("windows_hours", DEFAULT_TEMPORAL_WINDOWS)
            for window in windows:
                hits, _ = temporal_keyword_hits(events, factor["keywords"], window[0], window[1])
                active += int(hits > 0)
            return active / max(len(windows), 1)
    raise ValueError(f"Unsupported operation: {op}")
