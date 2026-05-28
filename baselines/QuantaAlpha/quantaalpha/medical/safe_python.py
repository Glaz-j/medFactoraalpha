"""Restricted Python factor execution for medical tabular samples."""

from __future__ import annotations

import ast
import hashlib
import math
import os
from functools import lru_cache
from typing import Any


SAFE_PYTHON_OPERATION = "safe_python"
ALLOWED_SAMPLE_KEYS = {
    "conditions",
    "procedures",
    "drugs",
    "conditions_offsets",
    "procedures_offsets",
    "drugs_offsets",
}
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
ALLOWED_METHODS = {
    "count",
    "endswith",
    "find",
    "get",
    "index",
    "join",
    "lower",
    "replace",
    "split",
    "startswith",
    "strip",
}
ALLOWED_NAMES = set(ALLOWED_BUILTINS) | {
    "contains_any",
    "count_keywords",
    "density_keywords",
    "events",
    "first_offset_hours",
    "get_texts",
    "in_window",
    "isfinite",
    "log1p",
    "persistence",
    "safe_div",
}
FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
    "os",
    "pathlib",
    "subprocess",
    "sys",
}


class SafePythonFactorError(ValueError):
    """Raised when a safe Python factor fails validation or execution."""


def _flatten_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(_flatten_text(item))
    elif value is not None:
        texts.append(str(value))
    return texts


def _offsets(sample: dict[str, Any], source: str) -> list[float | None]:
    raw_offsets = sample.get(f"{source}_offsets", [])
    if not isinstance(raw_offsets, (list, tuple)):
        return []
    values: list[float | None] = []
    for raw in raw_offsets:
        try:
            values.append(float(raw) / 60.0 if raw is not None else None)
        except (TypeError, ValueError):
            values.append(None)
    return values


def sanitize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Expose only non-label, current-stay fields to generated factor code."""
    sanitized: dict[str, Any] = {}
    for source in ("conditions", "procedures", "drugs"):
        sanitized[source] = _flatten_text(sample.get(source, []))
        sanitized[f"{source}_offsets"] = _offsets(sample, source)
    return sanitized


def get_texts(sample: dict[str, Any], source: str) -> list[str]:
    if source not in {"conditions", "procedures", "drugs"}:
        return []
    return [str(item).lower() for item in sample.get(source, [])]


def events(
    sample: dict[str, Any],
    source: str,
    start_hours: float | None = None,
    end_hours: float | None = None,
) -> list[tuple[str, float | None]]:
    texts = get_texts(sample, source)
    offsets = sample.get(f"{source}_offsets", [])
    if not isinstance(offsets, (list, tuple)):
        offsets = []
    result: list[tuple[str, float | None]] = []
    for idx, text in enumerate(texts):
        raw_offset = offsets[idx] if idx < len(offsets) else None
        try:
            offset_hours = float(raw_offset) if raw_offset is not None else None
        except (TypeError, ValueError):
            offset_hours = None
        if start_hours is not None and end_hours is not None:
            if not in_window(offset_hours, float(start_hours), float(end_hours)):
                continue
        result.append((text, offset_hours))
    return result


def contains_any(text: Any, keywords: list[str] | tuple[str, ...]) -> bool:
    if isinstance(text, (list, tuple)):
        return any(contains_any(item[0] if isinstance(item, tuple) and item else item, keywords) for item in text)
    text_l = str(text).lower()
    return any(str(keyword).lower() in text_l for keyword in keywords)


def count_keywords(texts: list[Any], keywords: list[str] | tuple[str, ...]) -> int:
    return sum(
        1
        for text in texts
        if contains_any(text[0] if isinstance(text, tuple) and text else text, keywords)
    )


def density_keywords(texts: list[Any], keywords: list[str] | tuple[str, ...]) -> float:
    return safe_div(count_keywords(texts, keywords), len(texts))


def in_window(offset_hours: float | None, start_hours: float, end_hours: float) -> bool:
    return (
        offset_hours is not None
        and offset_hours >= float(start_hours)
        and offset_hours < float(end_hours)
    )


def first_offset_hours(
    event_list: list[tuple[str, float | None]],
    keywords: list[str] | tuple[str, ...],
    start_hours: float = 0.0,
    end_hours: float = 48.0,
) -> float:
    candidates = [
        float(offset)
        for text, offset in event_list
        if in_window(offset, start_hours, end_hours) and contains_any(text, keywords)
    ]
    return min(candidates) if candidates else 0.0


def persistence(
    event_or_sample: list[tuple[str, float | None]] | dict[str, Any],
    source_or_keywords: str | list[str] | tuple[str, ...],
    keywords_or_windows: list[str] | tuple[str, ...] | list[list[float]] | tuple[tuple[float, float], ...],
    windows: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
) -> float:
    if isinstance(event_or_sample, dict):
        if not isinstance(source_or_keywords, str):
            return 0.0
        event_list = events(event_or_sample, source_or_keywords)
        keywords = keywords_or_windows  # type: ignore[assignment]
        active_windows = windows
    else:
        event_list = event_or_sample
        keywords = source_or_keywords  # type: ignore[assignment]
        active_windows = keywords_or_windows  # type: ignore[assignment]
    if not isinstance(keywords, (list, tuple)):
        return 0.0
    if active_windows is None:
        return 0.0
    windows = active_windows
    if not windows:
        return 0.0
    active = 0
    for start, end in windows:
        active += int(
            any(
                in_window(offset, float(start), float(end)) and contains_any(text, keywords)
                for text, offset in event_list
            )
        )
    return safe_div(active, len(windows))


def safe_div(left: float, right: float) -> float:
    try:
        denom = float(right)
        return 0.0 if abs(denom) < 1e-12 else float(left) / denom
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def log1p(value: float) -> float:
    try:
        return math.log1p(max(float(value), 0.0))
    except (TypeError, ValueError):
        return 0.0


def isfinite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


SAFE_GLOBALS = {
    "__builtins__": ALLOWED_BUILTINS,
    "contains_any": contains_any,
    "count_keywords": count_keywords,
    "density_keywords": density_keywords,
    "events": events,
    "first_offset_hours": first_offset_hours,
    "get_texts": get_texts,
    "in_window": in_window,
    "isfinite": isfinite,
    "log1p": log1p,
    "persistence": persistence,
    "safe_div": safe_div,
}


class _SafePythonValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_count = 0
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > 260:
            raise SafePythonFactorError("safe_python code is too complex.")
        if isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.ClassDef,
                ast.Delete,
                ast.Global,
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.Nonlocal,
                ast.Raise,
                ast.Try,
                ast.While,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise SafePythonFactorError(f"Forbidden Python construct: {type(node).__name__}")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        if node.name != "compute":
            raise SafePythonFactorError("safe_python code must define only compute(sample).")
        if len(node.args.args) != 1 or node.args.args[0].arg != "sample":
            raise SafePythonFactorError("compute must take exactly one argument named sample.")
        if node.decorator_list or node.returns or node.args.vararg or node.args.kwarg:
            raise SafePythonFactorError("compute must not use decorators, annotations, *args, or **kwargs.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in ALLOWED_METHODS:
            raise SafePythonFactorError(f"Forbidden attribute or method: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id in FORBIDDEN_NAMES:
            raise SafePythonFactorError(f"Forbidden name: {node.id}")
        if isinstance(node.ctx, ast.Load) and node.id not in ALLOWED_NAMES and node.id != "sample":
            # Local variables are allowed once assigned; this is checked more
            # precisely by Python compilation. We keep the explicit forbidden
            # list above as the security boundary.
            pass

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_NAMES:
                raise SafePythonFactorError(f"Function call is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("_") or node.func.attr not in ALLOWED_METHODS:
                raise SafePythonFactorError(f"Method call is not allowed: {node.func.attr}")
        else:
            raise SafePythonFactorError("Only direct function and safe method calls are allowed.")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "sample":
            key = None
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                key = node.slice.value
            if key not in ALLOWED_SAMPLE_KEYS:
                raise SafePythonFactorError(f"sample key is not allowed: {key!r}")
        self.generic_visit(node)


def validate_safe_python_code(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise SafePythonFactorError("safe_python code must be a non-empty string.")
    if len(code) > 2400:
        raise SafePythonFactorError("safe_python code is too long.")
    tree = ast.parse(code, mode="exec")
    validator = _SafePythonValidator()
    validator.visit(tree)
    if validator.function_count != 1:
        raise SafePythonFactorError("safe_python code must define exactly one compute(sample) function.")


def code_hash(code: str) -> str:
    return hashlib.md5(code.encode(), usedforsecurity=False).hexdigest()[:12]


@lru_cache(maxsize=256)
def _compile_compute(code: str):
    validate_safe_python_code(code)
    local_ns: dict[str, Any] = {}
    exec(compile(code, f"<medical_safe_python_{code_hash(code)}>", "exec"), SAFE_GLOBALS, local_ns)
    compute = local_ns.get("compute")
    if not callable(compute):
        raise SafePythonFactorError("safe_python code did not define callable compute(sample).")
    return compute


def compute_safe_python_factor(sample: dict[str, Any], factor: dict[str, Any]) -> float:
    try:
        compute = _compile_compute(str(factor.get("code", "")))
    except Exception as exc:  # noqa: BLE001
        if str(os.environ.get("MEDICAL_SAFE_PYTHON_STRICT_RUNTIME", "0")).lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            raise SafePythonFactorError(f"safe_python validation failed: {exc}") from exc
        return 0.0
    safe_sample = sanitize_sample(sample)
    try:
        value = compute(safe_sample)
        value_f = float(value)
    except Exception as exc:  # noqa: BLE001
        if str(os.environ.get("MEDICAL_SAFE_PYTHON_STRICT_RUNTIME", "0")).lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            raise SafePythonFactorError(f"safe_python execution failed: {exc}") from exc
        return 0.0
    if not math.isfinite(value_f):
        return 0.0
    return value_f
