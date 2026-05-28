"""Embedding-backed retrieval for medical QuantaAlpha factor memory."""

from __future__ import annotations

import json
import os
import re
import ast
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from quantaalpha.coder.knowledge.vector_base import Document, PDVectorBase
from quantaalpha.log import logger
from quantaalpha.medical.local_embedding import cosine_scores, local_embed
from quantaalpha.medical.safe_python import SAFE_PYTHON_OPERATION, code_hash
from quantaalpha.medical.source_profile import source_profile
from quantaalpha.medical.task_config import get_task_config


_VECTOR_EMBEDDING_DISABLED = False


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _output_root() -> Path:
    return Path(
        os.environ.get(
            "MEDICAL_OUTPUT_ROOT",
            Path.cwd() / "results" / "quantaalpha_medical_workflow",
        )
    )


def _rag_path() -> Path:
    profile = source_profile()
    return Path(
        os.environ.get(
            "MEDICAL_EMBEDDING_RAG_PATH",
            _output_root() / f"medical_embedding_rag_{profile}.pkl",
        )
    )


def _knowledge_log_path() -> Path:
    profile = source_profile()
    return Path(
        os.environ.get(
            "MEDICAL_FACTOR_KNOWLEDGE_LOG",
            _output_root() / f"medical_factor_knowledge_{profile}.jsonl",
        )
    )


def _metric_subset(scores: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    return {
        metric: scores.get(metric)
        for metric in metrics
        if isinstance(scores.get(metric), (int, float))
    }


def _clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _normal_list(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _factor_pattern_signature(factor: dict[str, Any]) -> tuple[Any, ...]:
    return (
        factor.get("operation", ""),
        tuple(sorted(_normal_list(factor.get("sources")))),
        factor.get("numeric_source", ""),
        factor.get("secondary_numeric_source", ""),
        factor.get("aggregation", ""),
        factor.get("secondary_aggregation", ""),
        factor.get("operator", ""),
        factor.get("transform", ""),
        tuple(sorted(str(item).lower() for item in _normal_list(factor.get("keywords")))),
        factor.get("numerator_source", ""),
        factor.get("denominator_source", ""),
        factor.get("window_start_hours", ""),
        factor.get("window_end_hours", ""),
        tuple(_normal_list(factor.get("early_window_hours"))),
        tuple(_normal_list(factor.get("late_window_hours"))),
        tuple(tuple(_normal_list(item)) for item in _normal_list(factor.get("windows_hours"))),
        factor.get("abnormal_low", ""),
        factor.get("abnormal_high", ""),
        factor.get("threshold", ""),
        code_hash(factor.get("code", "")) if factor.get("operation") == SAFE_PYTHON_OPERATION else "",
    )


def _line_value(content: str, key: str) -> str:
    prefix = f"{key}:"
    for line in content.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _parsed_line_value(content: str, key: str, default: Any) -> Any:
    raw = _line_value(content, key)
    if not raw:
        return default
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except Exception:  # noqa: BLE001
            continue
    return default


def _factor_from_content(content: str) -> dict[str, Any]:
    """Best-effort recovery for vector rows created before JSONL metadata existed."""
    factor: dict[str, Any] = {
        "name": _line_value(content, "Factor"),
        "operation": _line_value(content, "Operation"),
        "numeric_source": _line_value(content, "Numeric source"),
        "secondary_numeric_source": _line_value(content, "Secondary numeric source"),
        "aggregation": _line_value(content, "Aggregation"),
        "secondary_aggregation": _line_value(content, "Secondary aggregation"),
        "operator": _line_value(content, "Operator"),
        "transform": _line_value(content, "Transform"),
    }
    for key, field in [
        ("Sources", "sources"),
        ("Keywords", "keywords"),
        ("Early window", "early_window_hours"),
        ("Late window", "late_window_hours"),
        ("Windows", "windows_hours"),
    ]:
        raw = _line_value(content, key)
        if raw:
            try:
                factor[field] = json.loads(raw.replace("'", '"'))
            except Exception:  # noqa: BLE001
                factor[field] = raw
    window = _line_value(content, "Window")
    if " to " in window:
        start, end = window.split(" to ", 1)
        factor["window_start_hours"] = start.strip()
        factor["window_end_hours"] = end.replace("hours", "").strip()
    return factor


def _memory_status_from_doc(doc: Document, record: dict[str, Any] | None = None) -> str:
    if record and record.get("memory_status"):
        return str(record["memory_status"])
    content_status = _line_value(str(getattr(doc, "content", "")), "Memory status")
    if content_status:
        return content_status
    label = str(getattr(doc, "label", ""))
    if "failure" in label:
        return "failed"
    if "seed" in label:
        return "success"
    return "mixed"


def _component_tags(text: str) -> list[str]:
    text = text.lower()
    patterns = {
        "respiratory": r"respir|fio2|peep|vent|oxygen|spo2|sao2",
        "hemodynamic": r"shock|map|blood pressure|hypotension|pressor|heart rate|tachy",
        "renal": r"renal|kidney|creatinine|dialysis|urine|bun",
        "fluid": r"fluid|intake|output|balance|net",
        "infection": r"sepsis|infect|antibiotic|culture|wbc|lactate",
        "neurologic": r"gcs|neuro|coma|sedat|delir",
        "utilization": r"procedure|medication|drug|chart|treatment|count",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def _metric_gain_for_record(record: dict[str, Any], task_name: str) -> float:
    task_config = get_task_config(task_name or None)
    primary = task_config.primary_metric
    deltas = _first_combined_test_delta(record.get("baseline_deltas") or {})
    delta = deltas.get(primary)
    if isinstance(delta, (int, float)):
        value = float(delta)
        return -value if task_config.metric_direction.get(primary) == "min" else value
    scores = record.get("scores") or {}
    score = scores.get(primary)
    if isinstance(score, (int, float)):
        value = float(score)
        return -value if task_config.metric_direction.get(primary) == "min" else value
    return 0.0


def _factor_by_name(exp: Any) -> dict[str, dict[str, Any]]:
    factors = {}
    for task in getattr(exp, "sub_tasks", []) or []:
        if hasattr(task, "to_factor_dict"):
            factor = task.to_factor_dict()
            factors[factor.get("name", "")] = factor
    return factors


def _first_combined_test_delta(deltas: dict[str, Any]) -> dict[str, Any]:
    for model_key, model_deltas in (deltas or {}).items():
        if model_key.startswith("combined_"):
            return (model_deltas or {}).get("test", {}) or {}
    return {}


def _regulator_error_tags(feedback: str) -> list[str]:
    text = feedback.lower()
    tag_patterns = [
        ("dsl_validation_failed", r"dsl validation failed|no valid factors"),
        ("empty_factor_batch", r"non-empty `factors`|factors` list|no factors"),
        ("too_few_factors", r"too few factors"),
        ("too_many_factors", r"too many factors"),
        ("duplicate_factor", r"duplicate|previously evaluated"),
        ("invalid_or_absent_keyword", r"none of its keywords|keywords may be absent|observed vocabulary"),
        ("too_many_keywords", r"too many keywords"),
        ("definition_too_long", r"definition is too long"),
        ("horizon_violation", r"observation horizon|endpoint\\(s\\) exceed"),
        ("invalid_window_order", r"early_window_hours should end before"),
        ("invalid_numeric_source", r"numeric_source is required"),
        ("missing_numeric_threshold", r"needs abnormal_low|needs threshold"),
    ]
    tags = [tag for tag, pattern in tag_patterns if re.search(pattern, text)]
    return tags or ["regulator_failed"]


def _activity_error_tags(activity: dict[str, Any]) -> list[str]:
    tags = []
    nonzero = activity.get("nonzero_rate")
    std = activity.get("std")
    try:
        nonzero_f = float(nonzero)
        if nonzero_f <= float(os.environ.get("MEDICAL_FACTOR_ZERO_ACTIVITY_RATE", "0.0001")):
            tags.append("zero_activity")
        elif nonzero_f < float(os.environ.get("MEDICAL_FACTOR_ULTRA_RARE_RATE", "0.005")):
            tags.append("ultra_rare")
        elif nonzero_f > float(os.environ.get("MEDICAL_FACTOR_OVER_DENSE_RATE", "0.995")):
            tags.append("overly_dense")
    except (TypeError, ValueError):
        pass
    try:
        if abs(float(std)) <= 1e-12:
            tags.append("constant_factor")
    except (TypeError, ValueError):
        pass
    return tags


def _performance_error_tags(task_name: str, deltas: dict[str, Any]) -> list[str]:
    task_config = get_task_config(task_name)
    test_deltas = _first_combined_test_delta(deltas)
    tags = []
    for metric in task_config.report_metrics:
        value = test_deltas.get(metric)
        if not isinstance(value, (int, float)):
            continue
        direction = task_config.metric_direction.get(metric, "max")
        if direction == "min":
            regressed = float(value) > float(os.environ.get("MEDICAL_FACTOR_LOSS_REGRESSION_EPS", "0.0"))
        else:
            regressed = float(value) < -float(os.environ.get("MEDICAL_FACTOR_METRIC_REGRESSION_EPS", "0.0"))
        if regressed:
            tags.append(f"{metric}_regression")
    primary = task_config.primary_metric
    if f"{primary}_regression" in tags:
        tags.append("primary_metric_regression")
    return tags


def _memory_status(decision: bool, error_tags: list[str]) -> str:
    severe_tags = {
        "regulator_failed",
        "dsl_validation_failed",
        "empty_factor_batch",
        "horizon_violation",
        "zero_activity",
        "primary_metric_regression",
    }
    if decision and not severe_tags.intersection(error_tags):
        return "success"
    if decision:
        return "mixed"
    return "failed"


def _document_content(
    *,
    task_name: str,
    factor: dict[str, Any],
    activity: dict[str, Any],
    scores: dict[str, Any],
    deltas: dict[str, Any],
    feedback: Any,
    decision: bool,
    memory_status: str,
    error_tags: list[str],
    regulator_feedback: str = "",
    source_profile_name: str | None = None,
) -> str:
    report_metrics = list(get_task_config(task_name).report_metrics)
    test_deltas = _first_combined_test_delta(deltas)
    profile = source_profile_name or source_profile()
    return "\n".join(
        [
            f"Memory status: {memory_status}",
            f"Task: {task_name}",
            f"Source profile: {profile}",
            f"Factor: {factor.get('name')}",
            f"Description: {factor.get('description')}",
            f"Rationale: {factor.get('rationale')}",
            f"Operation: {factor.get('operation')}",
            f"Sources: {factor.get('sources')}",
            f"Numeric source: {factor.get('numeric_source')}",
            f"Secondary numeric source: {factor.get('secondary_numeric_source')}",
            f"Aggregation: {factor.get('aggregation')}",
            f"Secondary aggregation: {factor.get('secondary_aggregation')}",
            f"Operator: {factor.get('operator')}",
            f"Transform: {factor.get('transform')}",
            f"Keywords: {factor.get('keywords')}",
            f"Formula: {factor.get('formulation')}",
            f"Code: {factor.get('code', '')[:900]}",
            f"Window: {factor.get('window_start_hours')} to {factor.get('window_end_hours')} hours",
            f"Early window: {factor.get('early_window_hours')}",
            f"Late window: {factor.get('late_window_hours')}",
            f"Windows: {factor.get('windows_hours')}",
            f"Activity: {json.dumps(activity, sort_keys=True)}",
            f"Scores: {json.dumps(_metric_subset(scores, report_metrics), sort_keys=True)}",
            f"Combined-minus-baseline test delta: {json.dumps(_metric_subset(test_deltas, report_metrics), sort_keys=True)}",
            f"Error tags: {error_tags}",
            f"Regulator feedback: {regulator_feedback}",
            f"Accepted by feedback: {decision}",
            f"Feedback observations: {getattr(feedback, 'observations', '')}",
            f"Feedback evaluation: {getattr(feedback, 'hypothesis_evaluation', '')}",
            f"Feedback next hypothesis: {getattr(feedback, 'new_hypothesis', '')}",
        ]
    )


def _append_jsonl(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path = _knowledge_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _disable_vector_embeddings(reason: Exception) -> None:
    global _VECTOR_EMBEDDING_DISABLED
    _VECTOR_EMBEDDING_DISABLED = True
    logger.warning(
        "Medical embedding RAG vector store disabled for this process; "
        f"continuing with JSONL knowledge only. Reason: {reason}"
    )


class MedicalEmbeddingRAG:
    """Small adapter over QuantaAlpha's existing pandas vector base."""

    def __init__(self) -> None:
        self.enabled = str_to_bool(os.environ.get("MEDICAL_EMBEDDING_RAG", "0"))
        self.backend = os.environ.get("MEDICAL_EMBEDDING_BACKEND", "api").strip().lower()
        self.top_k = int(os.environ.get("MEDICAL_EMBEDDING_RAG_TOPK", "5"))
        self.threshold = float(os.environ.get("MEDICAL_EMBEDDING_RAG_THRESHOLD", "0.25"))
        self.style = os.environ.get("MEDICAL_RAG_STYLE", "costeer").strip().lower()
        self.success_limit = int(os.environ.get("MEDICAL_RAG_SUCCESS_LIMIT", "2"))
        self.failure_limit = int(os.environ.get("MEDICAL_RAG_FAILURE_LIMIT", "1"))
        self.error_limit = int(os.environ.get("MEDICAL_RAG_ERROR_LIMIT", "1"))
        self.max_chars_per_item = int(os.environ.get("MEDICAL_RAG_MAX_CHARS_PER_ITEM", "800"))
        self.candidate_limit = int(os.environ.get("MEDICAL_RAG_CANDIDATE_LIMIT", "50"))
        self.block_dup_signatures = str_to_bool(
            os.environ.get("MEDICAL_RAG_BLOCK_DUP_SIGNATURES", "1")
        )
        self.path = _rag_path()
        self.knowledge_log_path = _knowledge_log_path()
        self.vector_base: PDVectorBase | None = None
        self._seeded = False
        if self.enabled and not _VECTOR_EMBEDDING_DISABLED:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_base = PDVectorBase(self.path)

    def _known_ids(self) -> set[str]:
        if self.vector_base is None or not hasattr(self.vector_base, "vector_df"):
            return set()
        if "id" not in self.vector_base.vector_df:
            return set()
        return set(self.vector_base.vector_df["id"].astype(str).tolist())

    def _use_local_embedding(self) -> bool:
        return self.backend in {"local", "hf", "huggingface", "transformers", "qwen"}

    def _prepare_documents(self, docs: list[Document]) -> list[Document]:
        if not docs or not self._use_local_embedding():
            return docs
        embeddings = local_embed([doc.content for doc in docs], is_query=False)
        for doc, embedding in zip(docs, embeddings):
            doc.embedding = embedding
        return docs

    def _search_local(self, query_text: str, top_k: int) -> tuple[list[Document], list[float]]:
        if self.vector_base is None or self.vector_base.vector_df.empty:
            return [], []
        query_embedding = local_embed([query_text], is_query=True)[0]
        rows = self.vector_base.vector_df
        doc_embeddings = rows["embedding"].tolist()
        scores = cosine_scores(query_embedding, doc_embeddings)
        order = np.argsort(-scores)
        docs: list[Document] = []
        kept_scores: list[float] = []
        for idx in order:
            score = float(scores[idx])
            if score < self.threshold:
                continue
            docs.append(Document().from_dict(rows.iloc[int(idx)].to_dict()))
            kept_scores.append(score)
            if len(docs) >= top_k:
                break
        return docs, kept_scores

    def ingest_experiment(self, exp: Any, feedback: Any) -> int:
        result = getattr(exp, "result", {}) or {}
        if not result:
            return 0
        task_name = result.get("task") or get_task_config().name
        profile = result.get("source_profile") or source_profile()
        scores = result.get("scores", {}) or {}
        deltas = result.get("baseline_deltas", {}) or {}
        factor_hash = result.get("factor_hash", "unknown")
        factor_defs = _factor_by_name(exp)
        if not factor_defs:
            return 0
        activity_by_name = {
            item.get("name"): item
            for item in result.get("factor_summary", []) or []
            if item.get("name")
        }
        known_ids = self._known_ids()
        new_docs = []
        ledger_records = []
        decision = bool(getattr(feedback, "decision", False))
        perf_tags = _performance_error_tags(task_name, deltas)
        for name, factor in factor_defs.items():
            identity = f"medical-rag:{profile}:{task_name}:{factor_hash}:{name}"
            if identity in known_ids:
                continue
            activity = activity_by_name.get(name, {})
            error_tags = sorted(set(_activity_error_tags(activity) + perf_tags))
            memory_status = _memory_status(decision, error_tags)
            content = _document_content(
                task_name=task_name,
                factor=factor,
                activity=activity,
                scores=scores,
                deltas=deltas,
                feedback=feedback,
                decision=decision,
                memory_status=memory_status,
                error_tags=error_tags,
                source_profile_name=profile,
            )
            new_docs.append(Document(content=content, label=f"medical_factor:{task_name}", identity=identity))
            ledger_records.append(
                {
                    "id": identity,
                    "task": task_name,
                    "source_profile": profile,
                    "memory_status": memory_status,
                    "error_tags": error_tags,
                    "factor": factor,
                    "activity": activity,
                    "scores": scores,
                    "baseline_deltas": deltas,
                    "feedback_decision": decision,
                    "feedback": {
                        "observations": getattr(feedback, "observations", ""),
                        "hypothesis_evaluation": getattr(feedback, "hypothesis_evaluation", ""),
                        "new_hypothesis": getattr(feedback, "new_hypothesis", ""),
                        "reason": getattr(feedback, "reason", ""),
                    },
                }
            )
            known_ids.add(identity)
        _append_jsonl(ledger_records)
        if not self.enabled or self.vector_base is None or not new_docs:
            return len(ledger_records)
        try:
            self.vector_base.add(self._prepare_documents(new_docs))
            self.vector_base.dump()
            logger.info(f"Medical embedding RAG added {len(new_docs)} current experiment memories")
        except Exception as exc:  # noqa: BLE001
            _disable_vector_embeddings(exc)
            self.vector_base = None
        return len(ledger_records)

    def ingest_trace(self, trace: Any) -> int:
        if not self.enabled:
            return 0
        known_ids = self._known_ids()
        new_docs = []
        ledger_records = []
        for _, exp, feedback in getattr(trace, "hist", []) or []:
            result = getattr(exp, "result", {}) or {}
            task_name = result.get("task") or get_task_config().name
            profile = result.get("source_profile") or source_profile()
            scores = result.get("scores", {}) or {}
            deltas = result.get("baseline_deltas", {}) or {}
            factor_hash = result.get("factor_hash", "unknown")
            factor_defs = _factor_by_name(exp)
            activity_by_name = {
                item.get("name"): item
                for item in result.get("factor_summary", []) or []
                if item.get("name")
            }
            decision = bool(getattr(feedback, "decision", False))
            perf_tags = _performance_error_tags(task_name, deltas)
            for name, factor in factor_defs.items():
                identity = f"medical-rag:{profile}:{task_name}:{factor_hash}:{name}"
                if identity in known_ids:
                    continue
                activity = activity_by_name.get(name, {})
                error_tags = sorted(set(_activity_error_tags(activity) + perf_tags))
                memory_status = _memory_status(decision, error_tags)
                content = _document_content(
                    task_name=task_name,
                    factor=factor,
                    activity=activity,
                    scores=scores,
                    deltas=deltas,
                    feedback=feedback,
                    decision=decision,
                    memory_status=memory_status,
                    error_tags=error_tags,
                    source_profile_name=profile,
                )
                new_docs.append(Document(content=content, label=f"medical_factor:{task_name}", identity=identity))
                ledger_records.append(
                    {
                        "id": identity,
                        "task": task_name,
                        "source_profile": profile,
                        "memory_status": memory_status,
                        "error_tags": error_tags,
                        "factor": factor,
                        "activity": activity,
                        "scores": scores,
                        "baseline_deltas": deltas,
                        "feedback_decision": decision,
                        "feedback": {
                            "observations": getattr(feedback, "observations", ""),
                            "hypothesis_evaluation": getattr(feedback, "hypothesis_evaluation", ""),
                            "new_hypothesis": getattr(feedback, "new_hypothesis", ""),
                            "reason": getattr(feedback, "reason", ""),
                        },
                    }
                )
                known_ids.add(identity)
        if not new_docs:
            return 0
        _append_jsonl(ledger_records)
        if self.vector_base is None:
            return len(ledger_records)
        try:
            self.vector_base.add(self._prepare_documents(new_docs))
            self.vector_base.dump()
            logger.info(f"Medical embedding RAG added {len(new_docs)} factor memories to {self.path}")
            return len(new_docs)
        except Exception as exc:  # noqa: BLE001
            _disable_vector_embeddings(exc)
            self.vector_base = None
            return len(ledger_records)

    def ingest_summary_files(self) -> int:
        if not self.enabled or self.vector_base is None or self._seeded:
            return 0
        self._seeded = True
        patterns = [
            item.strip()
            for item in os.environ.get("MEDICAL_EMBEDDING_RAG_SEED_GLOB", "").split(",")
            if item.strip()
        ]
        if not patterns:
            return 0
        known_ids = self._known_ids()
        new_docs = []
        ledger_records = []
        for pattern in patterns:
            for path_str in glob(pattern):
                path = Path(path_str)
                try:
                    payload = json.loads(path.read_text())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Skipping invalid medical RAG seed file {path}: {exc}")
                    continue
                result = payload.get("result", {}) or {}
                factors = payload.get("factors", []) or []
                if not result or not factors:
                    continue
                task_name = result.get("task") or get_task_config().name
                profile = result.get("source_profile", "")
                current_profile = source_profile()
                allow_cross_profile = str_to_bool(
                    os.environ.get("MEDICAL_RAG_ALLOW_CROSS_PROFILE_SEEDS", "0")
                )
                if profile != current_profile and not allow_cross_profile:
                    continue
                scores = result.get("scores", {}) or {}
                deltas = result.get("baseline_deltas", {}) or {}
                factor_hash = result.get("factor_hash", path.stem)
                activity_by_name = {
                    item.get("name"): item
                    for item in result.get("factor_summary", []) or []
                    if item.get("name")
                }
                feedback = SimpleNamespace(
                    observations=str(payload.get("feedback", ""))[:1500],
                    hypothesis_evaluation="Seeded from prior workflow summary.",
                    new_hypothesis=str(payload.get("hypothesis", ""))[:1500],
                    decision=True,
                )
                perf_tags = _performance_error_tags(task_name, deltas)
                for factor in factors:
                    name = factor.get("name")
                    if not name:
                        continue
                    identity = f"medical-rag-seed:{profile}:{task_name}:{factor_hash}:{name}"
                    if identity in known_ids:
                        continue
                    activity = activity_by_name.get(name, {})
                    error_tags = sorted(set(_activity_error_tags(activity) + perf_tags))
                    memory_status = _memory_status(True, error_tags)
                    content = _document_content(
                        task_name=task_name,
                        factor=factor,
                        activity=activity,
                        scores=scores,
                        deltas=deltas,
                        feedback=feedback,
                        decision=True,
                        memory_status=memory_status,
                        error_tags=error_tags,
                        source_profile_name=profile,
                    )
                    new_docs.append(
                        Document(
                            content=content,
                            label=f"medical_factor_seed:{task_name}",
                            identity=identity,
                        )
                    )
                    ledger_records.append(
                        {
                            "id": identity,
                            "task": task_name,
                            "source_profile": profile,
                            "memory_status": memory_status,
                            "error_tags": error_tags,
                            "factor": factor,
                            "activity": activity,
                            "scores": scores,
                            "baseline_deltas": deltas,
                            "seed_summary_path": str(path),
                        }
                    )
                    known_ids.add(identity)
        if not new_docs:
            return 0
        _append_jsonl(ledger_records)
        if not self.enabled or self.vector_base is None:
            return len(ledger_records)
        try:
            self.vector_base.add(self._prepare_documents(new_docs))
            self.vector_base.dump()
            logger.info(f"Medical embedding RAG seeded {len(new_docs)} memories from summary files")
            return len(new_docs)
        except Exception as exc:  # noqa: BLE001
            _disable_vector_embeddings(exc)
            self.vector_base = None
            return len(ledger_records)

    def ingest_regulator_failure(
        self,
        *,
        hypothesis: Any,
        payload: dict[str, Any] | None,
        regulator_feedback: str,
        task_name: str | None = None,
    ) -> int:
        task_name = task_name or get_task_config().name
        profile = source_profile()
        raw_factors = []
        if isinstance(payload, dict) and isinstance(payload.get("factors"), list):
            raw_factors = [item for item in payload.get("factors", []) if isinstance(item, dict)]
        if not raw_factors:
            raw_factors = [
                {
                    "name": "empty_or_invalid_factor_batch",
                    "description": "LLM returned no valid medical DSL factors.",
                    "rationale": str(hypothesis)[:500],
                    "operation": "",
                    "sources": [],
                    "numeric_source": "",
                    "keywords": [],
                }
            ]
        tags = _regulator_error_tags(regulator_feedback)
        records = []
        docs = []
        known_ids = self._known_ids()
        safe_hyp = re.sub(r"[^a-zA-Z0-9_]+", "_", str(hypothesis)[:80]).strip("_")
        for idx, factor in enumerate(raw_factors, start=1):
            name = factor.get("name") or f"invalid_factor_{idx}"
            identity = f"medical-rag-regulator-failure:{profile}:{task_name}:{safe_hyp}:{idx}:{name}"
            if identity in known_ids:
                continue
            content = _document_content(
                task_name=task_name,
                factor=factor,
                activity={},
                scores={},
                deltas={},
                feedback=SimpleNamespace(
                    observations="Regulator rejected candidate medical DSL factors.",
                    hypothesis_evaluation=regulator_feedback,
                    new_hypothesis="Avoid repeating this invalid factor pattern.",
                ),
                decision=False,
                memory_status="failed",
                error_tags=tags,
                regulator_feedback=regulator_feedback,
                source_profile_name=profile,
            )
            records.append(
                {
                    "id": identity,
                    "task": task_name,
                    "source_profile": profile,
                    "memory_status": "failed",
                    "error_tags": tags,
                    "factor": factor,
                    "hypothesis": str(hypothesis),
                    "regulator_feedback": regulator_feedback,
                }
            )
            docs.append(Document(content=content, label=f"medical_factor_failure:{task_name}", identity=identity))
            known_ids.add(identity)
        _append_jsonl(records)
        if not self.enabled or self.vector_base is None or not docs:
            return len(records)
        try:
            self.vector_base.add(self._prepare_documents(docs))
            self.vector_base.dump()
            logger.info(f"Medical embedding RAG added {len(docs)} regulator-failure memories")
        except Exception as exc:  # noqa: BLE001
            _disable_vector_embeddings(exc)
            self.vector_base = None
        return len(records)

    def _ledger_by_id(self) -> dict[str, dict[str, Any]]:
        if not self.knowledge_log_path.exists():
            return {}
        records = {}
        try:
            with self.knowledge_log_path.open() as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record_id = record.get("id")
                    if record_id:
                        records[str(record_id)] = record
        except OSError as exc:
            logger.warning(f"Failed to read medical factor knowledge log: {exc}")
        return records

    def _trace_signatures(self, trace: Any) -> set[tuple[Any, ...]]:
        signatures = set()
        for _, exp, _ in getattr(trace, "hist", []) or []:
            for factor in _factor_by_name(exp).values():
                signatures.add(_factor_pattern_signature(factor))
        return signatures

    def _search(self, query_text: str, top_k: int) -> tuple[list[Document], list[float]]:
        if self._use_local_embedding():
            return self._search_local(query_text, top_k)
        return self.vector_base.search(  # type: ignore[union-attr]
            query_text,
            topk_k=top_k,
            similarity_threshold=self.threshold,
        )

    def _record_for_doc(
        self,
        doc: Document,
        ledger: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        record = ledger.get(str(getattr(doc, "id", "")), {})
        if record:
            return record
        content = str(getattr(doc, "content", ""))
        return {
            "id": getattr(doc, "id", ""),
            "task": _line_value(content, "Task"),
            "source_profile": _line_value(content, "Source profile"),
            "memory_status": _memory_status_from_doc(doc),
            "error_tags": _parsed_line_value(content, "Error tags", []),
            "factor": _factor_from_content(content),
            "activity": _parsed_line_value(content, "Activity", {}),
            "scores": _parsed_line_value(content, "Scores", {}),
            "baseline_deltas": {
                "combined_from_vector_content": {
                    "test": _parsed_line_value(
                        content,
                        "Combined-minus-baseline test delta",
                        {},
                    )
                }
            },
            "content": content,
        }

    def _bucket_for_memory(self, doc: Document, record: dict[str, Any]) -> str:
        label = str(getattr(doc, "label", ""))
        status = str(record.get("memory_status") or _memory_status_from_doc(doc, record)).lower()
        error_tags = record.get("error_tags") or []
        error_text = " ".join(error_tags) if isinstance(error_tags, list) else str(error_tags)
        if "failure" in label or status == "failed":
            if re.search(r"regulator|dsl|duplicate|horizon|definition|empty|invalid", error_text):
                return "error"
            return "failure"
        return "success"

    def _compact_memory(
        self,
        *,
        title: str,
        doc: Document,
        score: float,
        record: dict[str, Any],
        query_components: list[str],
    ) -> str:
        factor = record.get("factor") or _factor_from_content(str(getattr(doc, "content", "")))
        task_name = record.get("task") or _line_value(str(getattr(doc, "content", "")), "Task")
        tags = record.get("error_tags") or []
        activity = record.get("activity") or {}
        scores = record.get("scores") or {}
        deltas = _first_combined_test_delta(record.get("baseline_deltas") or {})
        factor_text = json.dumps(
            {
                "name": factor.get("name"),
                "operation": factor.get("operation"),
                "sources": factor.get("sources"),
                "numeric_source": factor.get("numeric_source"),
                "secondary_numeric_source": factor.get("secondary_numeric_source"),
                "aggregation": factor.get("aggregation"),
                "secondary_aggregation": factor.get("secondary_aggregation"),
                "operator": factor.get("operator"),
                "transform": factor.get("transform"),
                "keywords": factor.get("keywords"),
                "window_start_hours": factor.get("window_start_hours"),
                "window_end_hours": factor.get("window_end_hours"),
                "early_window_hours": factor.get("early_window_hours"),
                "late_window_hours": factor.get("late_window_hours"),
                "windows_hours": factor.get("windows_hours"),
                "abnormal_low": factor.get("abnormal_low"),
                "abnormal_high": factor.get("abnormal_high"),
                "threshold": factor.get("threshold"),
            },
            sort_keys=True,
        )
        report_metrics = list(get_task_config(task_name or None).report_metrics)
        components = sorted(set(query_components).intersection(_component_tags(str(factor) + str(record))))
        lines = [
            f"{title} (similarity={score:.3f}, status={record.get('memory_status')}, task={task_name})",
            f"Components: {components or _component_tags(str(factor) + str(record))}",
            f"Factor pattern: {_clip_text(factor_text, 450)}",
            f"Activity: {_clip_text(json.dumps(activity, sort_keys=True), 240)}",
            f"Scores: {_clip_text(json.dumps(_metric_subset(scores, report_metrics), sort_keys=True), 240)}",
            "Combined-minus-baseline test delta: "
            f"{_clip_text(json.dumps(_metric_subset(deltas, report_metrics), sort_keys=True), 240)}",
            f"Error tags: {tags}",
        ]
        observations = record.get("feedback", {}).get("observations", "")
        if observations:
            lines.append(f"Lesson: {_clip_text(observations, 280)}")
        regulator_feedback = record.get("regulator_feedback", "")
        if regulator_feedback:
            lines.append(f"Regulator lesson: {_clip_text(regulator_feedback, 280)}")
        lines.append("Usage rule: transfer the design lesson; do not copy this factor exactly.")
        return _clip_text("\n".join(lines), self.max_chars_per_item)

    def _legacy_query(self, query_text: str, top_k: int | None = None) -> str:
        docs, scores = self._search(query_text, top_k or self.top_k)
        if not docs:
            return ""
        blocks = []
        for idx, (doc, score) in enumerate(zip(docs, scores), start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Retrieved medical factor memory {idx} (similarity={float(score):.3f}):",
                        str(doc.content)[:2500],
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _costeer_query(self, trace: Any, query_text: str, top_k: int | None = None) -> str:
        candidate_k = max(
            top_k or self.top_k,
            self.candidate_limit,
            (self.success_limit + self.failure_limit + self.error_limit) * 10,
        )
        docs, scores = self._search(query_text, candidate_k)
        if not docs:
            return ""

        task_name = get_task_config().name
        current_profile = source_profile()
        ledger = self._ledger_by_id()
        query_components = _component_tags(query_text)
        previous_signatures = self._trace_signatures(trace)
        candidates: dict[str, list[tuple[Document, float, dict[str, Any]]]] = {
            "success": [],
            "failure": [],
            "error": [],
        }
        limits = {
            "success": self.success_limit,
            "failure": self.failure_limit,
            "error": self.error_limit,
        }
        seen_ids = set()
        seen_signatures = set()
        performance_weight = float(os.environ.get("MEDICAL_RAG_PERFORMANCE_WEIGHT", "0.25"))

        for doc, score in zip(docs, scores):
            doc_id = str(getattr(doc, "id", ""))
            if doc_id in seen_ids:
                continue
            record = self._record_for_doc(doc, ledger)
            record_task = record.get("task") or _line_value(str(getattr(doc, "content", "")), "Task")
            if record_task and str(record_task) != task_name:
                continue
            record_profile = record.get("source_profile") or _line_value(
                str(getattr(doc, "content", "")), "Source profile"
            )
            if record_profile and str(record_profile) != current_profile:
                continue
            factor = record.get("factor") or _factor_from_content(str(getattr(doc, "content", "")))
            signature = _factor_pattern_signature(factor)
            bucket = self._bucket_for_memory(doc, record)
            if self.block_dup_signatures and bucket == "success" and signature in previous_signatures:
                continue
            if signature in seen_signatures:
                continue
            candidates[bucket].append((doc, float(score), record))
            seen_ids.add(doc_id)
            seen_signatures.add(signature)
            if all(len(candidates[name]) >= max(limit * 4, limit) for name, limit in limits.items()):
                break

        def rank_item(item: tuple[Document, float, dict[str, Any]], bucket: str) -> float:
            _, score, record = item
            if bucket != "success":
                return score
            return score + performance_weight * _metric_gain_for_record(record, task_name)

        selected = {
            bucket: sorted(
                items,
                key=lambda item, bucket=bucket: rank_item(item, bucket),
                reverse=True,
            )[: limits[bucket]]
            for bucket, items in candidates.items()
        }

        blocks = []
        titles = {
            "failure": "CoSTEER former-failure memory",
            "success": "CoSTEER component-success memory",
            "error": "CoSTEER error-regulator memory",
        }
        order = ["failure", "success", "error"]
        for bucket in order:
            for doc, score, record in selected[bucket][: limits[bucket]]:
                blocks.append(
                    self._compact_memory(
                        title=titles[bucket],
                        doc=doc,
                        score=score,
                        record=record,
                        query_components=query_components,
                    )
                )
        if not blocks:
            return ""
        header = (
            "CoSTEER-style retrieved medical memories. These are sparse examples "
            "for error avoidance and design transfer, not factors to copy."
        )
        return header + "\n\n" + "\n\n".join(blocks)

    def query(self, trace: Any, query_text: str, top_k: int | None = None) -> str:
        if not self.enabled:
            return ""
        self.ingest_summary_files()
        self.ingest_trace(trace)
        if _VECTOR_EMBEDDING_DISABLED or self.vector_base is None:
            return ""
        if getattr(self.vector_base, "vector_df", None) is None or self.vector_base.vector_df.empty:
            return ""
        try:
            if self.style in {"legacy", "topk", "top-k"}:
                return self._legacy_query(query_text, top_k=top_k)
            return self._costeer_query(trace, query_text, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            _disable_vector_embeddings(exc)
            self.vector_base = None
            return ""
