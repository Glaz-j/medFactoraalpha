"""CoSTEER-style task evolution for safe medical DSL factors.

This module mirrors QuantaAlpha's CoSTEER contract at the task level, but keeps
the medical implementation surface as a deterministic DSL instead of free-form
Python code. The goal is to retain CoSTEER's success/failure memory, retry
limits, duplicate filtering, and task feedback loops without weakening the
medical leakage guardrails.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from quantaalpha.core.developer import Developer
from quantaalpha.log import logger
from quantaalpha.medical.dsl import NUMERIC_TEMPORAL_OPS, TEMPORAL_OPS, normalize_factors
from quantaalpha.medical.embedding_rag import (
    _activity_error_tags,
    _performance_error_tags,
    str_to_bool,
)
from quantaalpha.medical.experiment import MedicalFactorExperiment, MedicalFactorWorkspace
from quantaalpha.medical.safe_python import SAFE_PYTHON_OPERATION, code_hash
from quantaalpha.medical.task_config import get_task_config


def _output_root() -> Path:
    return Path(
        os.environ.get(
            "MEDICAL_OUTPUT_ROOT",
            Path.cwd() / "results" / "quantaalpha_medical_workflow",
        )
    )


def _knowledge_path() -> Path:
    return Path(
        os.environ.get(
            "MEDICAL_DSL_COSTEER_LOG",
            _output_root() / "medical_dsl_costeer_knowledge.jsonl",
        )
    )


def _normal_list(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def factor_task_signature(factor: dict[str, Any], *, include_name: bool = False) -> tuple[Any, ...]:
    """Stable DSL task signature used like CoSTEER's task information key."""
    name_part = (factor.get("name", ""),) if include_name else ()
    temporal_part: tuple[Any, ...]
    if factor.get("operation") in TEMPORAL_OPS:
        temporal_part = (
            factor.get("window_start_hours", ""),
            factor.get("window_end_hours", ""),
            tuple(_normal_list(factor.get("early_window_hours"))),
            tuple(_normal_list(factor.get("late_window_hours"))),
            tuple(tuple(_normal_list(item)) for item in _normal_list(factor.get("windows_hours"))),
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
            tuple(_normal_list(factor.get("early_window_hours"))),
            tuple(_normal_list(factor.get("late_window_hours"))),
            tuple(tuple(_normal_list(item)) for item in _normal_list(factor.get("windows_hours"))),
            factor.get("abnormal_low", ""),
            factor.get("abnormal_high", ""),
            factor.get("threshold", ""),
        )
    else:
        numeric_part = ()
    return (
        *name_part,
        factor.get("operation", ""),
        tuple(sorted(_normal_list(factor.get("sources")))),
        factor.get("numeric_source", ""),
        tuple(sorted(str(item).lower() for item in _normal_list(factor.get("keywords")))),
        factor.get("numerator_source", ""),
        factor.get("denominator_source", ""),
        code_hash(factor.get("code", "")) if factor.get("operation") == SAFE_PYTHON_OPERATION else "",
        *temporal_part,
        *numeric_part,
    )


def _signature_key(signature: tuple[Any, ...]) -> str:
    return json.dumps(signature, sort_keys=True, default=str)


class MedicalDSLCoSTEERKnowledge:
    """JSONL-backed memory following CoSTEER's success/failed task split."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _knowledge_path()
        self.records = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        try:
            with self.path.open() as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning(f"Failed to read medical DSL-CoSTEER memory: {exc}")
        return rows

    def append(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.extend(records)

    def task_counts(self, task_name: str) -> dict[str, Counter]:
        counts = {"success": Counter(), "failed": Counter(), "mixed": Counter()}
        for record in self.records:
            if record.get("task") != task_name:
                continue
            status = record.get("memory_status", "failed")
            if status not in counts:
                status = "failed"
            counts[status][record.get("signature", "")] += 1
        return counts

    def failed_task_info_set(self, task_name: str, fail_limit: int) -> set[str]:
        counts = self.task_counts(task_name)
        successes = counts["success"]
        return {
            signature
            for signature, count in counts["failed"].items()
            if count >= fail_limit and successes.get(signature, 0) == 0
        }


class MedicalDSLCoSTEER(Developer[MedicalFactorExperiment]):
    """CoSTEER-like developer for medical DSL factors.

    It does not synthesize code. Instead it maps factor tasks to validated DSL
    workspaces, filters repeated failures, and records task-level feedback that
    later proposal/RAG stages can use.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled = str_to_bool(os.environ.get("MEDICAL_DSL_COSTEER", "1"))
        self.fail_task_trial_limit = int(
            os.environ.get("MEDICAL_DSL_COSTEER_FAIL_TASK_TRIAL_LIMIT", "20")
        )
        self.knowledge = MedicalDSLCoSTEERKnowledge()

    def develop(self, exp: MedicalFactorExperiment) -> MedicalFactorExperiment:
        if not self.enabled:
            return self._attach_workspaces(exp, exp.sub_tasks)

        task_name = get_task_config().name
        failed_set = self.knowledge.failed_task_info_set(task_name, self.fail_task_trial_limit)
        factors = normalize_factors({"factors": [task.to_factor_dict() for task in exp.sub_tasks]})
        kept_tasks = []
        seen = set()
        skipped = []

        task_by_name = {task.factor_name: task for task in exp.sub_tasks}
        for factor in factors:
            signature = _signature_key(factor_task_signature(factor))
            if signature in seen:
                skipped.append(
                    {
                        "factor": factor,
                        "signature": signature,
                        "reason": "duplicate_in_current_batch",
                    }
                )
                continue
            if signature in failed_set:
                skipped.append(
                    {
                        "factor": factor,
                        "signature": signature,
                        "reason": "failed_task_trial_limit",
                    }
                )
                continue
            task = task_by_name.get(factor.get("name"))
            if task is None:
                continue
            task.raw_factor = factor
            kept_tasks.append(task)
            seen.add(signature)

        if skipped:
            self.knowledge.append(
                [
                    {
                        "task": task_name,
                        "memory_status": "failed",
                        "stage": "coder",
                        "error_tags": [item["reason"]],
                        "signature": item["signature"],
                        "factor": item["factor"],
                    }
                    for item in skipped
                ]
            )

        if not kept_tasks and str_to_bool(os.environ.get("MEDICAL_DSL_COSTEER_EMPTY_FALLBACK", "1")):
            logger.warning(
                "Medical DSL-CoSTEER filtered all tasks; falling back to original "
                "candidate batch so the workflow can produce evaluator feedback."
            )
            kept_tasks = list(exp.sub_tasks)

        exp.sub_tasks = kept_tasks
        exp.sub_workspace_list = [
            MedicalFactorWorkspace(target_task=task) for task in exp.sub_tasks
        ]
        exp.costeer_report = {
            "enabled": True,
            "failed_task_trial_limit": self.fail_task_trial_limit,
            "kept_factor_names": [task.factor_name for task in kept_tasks],
            "skipped": skipped,
        }
        logger.info(
            "Medical DSL-CoSTEER kept "
            f"{len(kept_tasks)}/{len(factors)} factors; skipped={len(skipped)}"
        )
        return exp

    @staticmethod
    def _attach_workspaces(
        exp: MedicalFactorExperiment,
        tasks: list[Any],
    ) -> MedicalFactorExperiment:
        exp.sub_tasks = list(tasks)
        exp.sub_workspace_list = [
            MedicalFactorWorkspace(target_task=task) for task in exp.sub_tasks
        ]
        return exp


def ingest_costeer_experiment(exp: MedicalFactorExperiment, feedback: Any) -> int:
    """Persist post-evaluation task feedback into DSL-CoSTEER memory."""
    result = getattr(exp, "result", {}) or {}
    if not result:
        return 0
    task_name = result.get("task") or get_task_config().name
    activity_by_name = {
        item.get("name"): item
        for item in result.get("factor_summary", []) or []
        if item.get("name")
    }
    perf_tags = _performance_error_tags(task_name, result.get("baseline_deltas", {}) or {})
    decision = bool(getattr(feedback, "decision", False))
    records = []
    for task in getattr(exp, "sub_tasks", []) or []:
        if not hasattr(task, "to_factor_dict"):
            continue
        factor = task.to_factor_dict()
        signature = _signature_key(factor_task_signature(factor))
        activity = activity_by_name.get(factor.get("name"), {})
        error_tags = sorted(set(_activity_error_tags(activity) + perf_tags))
        severe = {
            "zero_activity",
            "constant_factor",
            "primary_metric_regression",
        }
        if decision and not severe.intersection(error_tags):
            status = "success"
        elif decision:
            status = "mixed"
        else:
            status = "failed"
        records.append(
            {
                "task": task_name,
                "memory_status": status,
                "stage": "feedback",
                "error_tags": error_tags,
                "signature": signature,
                "factor": factor,
                "activity": activity,
                "scores": result.get("scores", {}),
                "baseline_deltas": result.get("baseline_deltas", {}),
                "feedback_decision": decision,
            }
        )
    MedicalDSLCoSTEERKnowledge().append(records)
    return len(records)
