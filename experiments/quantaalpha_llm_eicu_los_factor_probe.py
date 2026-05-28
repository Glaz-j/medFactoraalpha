"""
Generate QuantaAlpha-style medical factors with an LLM and evaluate in PyHealth.

This is the first small end-to-end bridge:

    gpt-5.5 -> symbolic factor JSON -> eICU LOS task -> QuantaAlphaPyHealthModel
    -> PyHealth Trainer

The LLM is constrained to a tiny, deterministic DSL. We do not execute
LLM-written Python code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.tasks import LengthOfStayPredictioneICU
from pyhealth.trainer import Trainer
from quantaalpha.pyhealth_model import QuantaAlphaPyHealthModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
QUANTAALPHA_ROOT = PROJECT_ROOT / "baselines" / "QuantaAlpha"

DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "quantaalpha_medical_factors"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "pyhealth_runs"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "pyhealth_cache"

VALID_FEATURES = {"conditions", "procedures", "drugs"}
VALID_OPS = {
    "log_count",
    "count_ratio",
    "keyword_any",
    "keyword_count",
    "keyword_density",
}


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

        sources = raw.get("sources", raw.get("source", []))
        sources = [str(item).strip() for item in as_list(sources)]
        sources = [item for item in sources if item in VALID_FEATURES]

        numerator_source = str(raw.get("numerator_source", "")).strip()
        denominator_source = str(raw.get("denominator_source", "")).strip()

        if op == "count_ratio":
            if numerator_source not in VALID_FEATURES or denominator_source not in VALID_FEATURES:
                continue
        elif not sources:
            continue

        keywords = [
            str(item).strip().lower()
            for item in as_list(raw.get("keywords", []))
            if str(item).strip()
        ][:16]
        if op.startswith("keyword") and not keywords:
            continue

        normalized.append(
            {
                "name": safe_name(str(raw.get("name", "")), f"factor_{idx}"),
                "description": str(raw.get("description", ""))[:500],
                "rationale": str(raw.get("rationale", ""))[:500],
                "operation": op,
                "sources": sources,
                "keywords": keywords,
                "numerator_source": numerator_source,
                "denominator_source": denominator_source,
            }
        )

    if not normalized:
        raise ValueError("No valid factors survived DSL validation.")
    return normalized[:12]


def generate_factors_with_llm(model: str) -> dict[str, Any]:
    env_path = Path(os.environ.get("QUANTAALPHA_ENV", QUANTAALPHA_ROOT / ".env"))
    if env_path.exists():
        load_dotenv(env_path, override=True)
    os.environ.setdefault("CHAT_MODEL", model)
    os.environ.setdefault("REASONING_MODEL", model)
    os.environ.setdefault("LOG_LLM_CHAT_CONTENT", "False")
    os.environ.setdefault("CHAT_STREAM", "False")

    # Keep optional dependency warnings out of the experiment log.
    logging.getLogger("quantaalpha").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

    from quantaalpha.llm.client import APIBackend, robust_json_parse

    system_prompt = (
        "You are a clinical informatics researcher designing symbolic risk "
        "factors for ICU length-of-stay prediction. Return only valid JSON."
    )
    user_prompt = """
We are running PyHealth LengthOfStayPredictioneICU on eICU.
Available current-stay features:
- conditions: diagnosis strings/codes
- procedures: physical exam/procedure strings/codes
- drugs: medication strings/codes

Goal: propose clinically plausible symbolic factors that may help predict the
current ICU stay length-of-stay class. Do not use future information.

Use only this DSL:
- log_count: needs sources = one of ["conditions", "procedures", "drugs"]
- count_ratio: needs numerator_source and denominator_source
- keyword_any: needs sources and keywords
- keyword_count: needs sources and keywords
- keyword_density: needs sources and keywords

Return exactly this JSON shape:
{
  "task": "eicu_length_of_stay",
  "model": "gpt-5.5",
  "factors": [
    {
      "name": "short_snake_case",
      "description": "human-readable factor description",
      "rationale": "why it might relate to ICU LOS",
      "operation": "keyword_any",
      "sources": ["conditions", "drugs"],
      "keywords": ["sepsis", "infection"]
    }
  ]
}

Generate 8 to 10 diverse factors. Prefer clinically meaningful concepts such as
infection burden, respiratory failure, shock/vasopressors, renal failure,
cardiac disease, neurologic disease, metabolic derangement, medication burden,
and procedure/exam complexity.
"""
    api = APIBackend(chat_model=model, reasoning_model=model)
    response = api.build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        reasoning_flag=False,
        max_retry=2,
        max_tokens=2400,
        temperature=0.2,
    )
    payload = robust_json_parse(response)
    payload["factors"] = normalize_factors(payload)
    payload["model"] = model
    return payload


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


def compute_factor(sample: dict[str, Any], factor: dict[str, Any]) -> float:
    op = factor["operation"]
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
    raise ValueError(f"Unsupported operation: {op}")


class LengthOfStayPredictioneICUWithLLMFactors(LengthOfStayPredictioneICU):
    task_name = "LengthOfStayPredictioneICUWithLLMFactors"
    input_schema = {
        **LengthOfStayPredictioneICU.input_schema,
        "symbolic_factors": "tensor",
    }
    output_schema = LengthOfStayPredictioneICU.output_schema

    def __init__(self, factors: list[dict[str, Any]]):
        self.factors = factors
        digest = hashlib.md5(
            json.dumps(factors, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()[:8]
        self.task_name = f"{self.task_name}_{digest}"

    def __call__(self, patient):
        samples = super().__call__(patient)
        for sample in samples:
            sample["symbolic_factors"] = [
                compute_factor(sample, factor) for factor in self.factors
            ]
        return samples


def label_counts(dataset, label_key: str) -> Counter:
    counts = Counter()
    for idx in range(len(dataset)):
        value = dataset[idx][label_key]
        if torch.is_tensor(value):
            value = int(value.item())
        counts[int(value)] += 1
    return counts


def summarize_factors(dataset, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [[] for _ in factors]
    for idx in range(len(dataset)):
        sample_values = dataset[idx]["symbolic_factors"]
        for factor_idx, value in enumerate(sample_values):
            values[factor_idx].append(float(value))

    summary = []
    for factor, vals in zip(factors, values, strict=True):
        mean = sum(vals) / max(len(vals), 1)
        var = sum((value - mean) ** 2 for value in vals) / max(len(vals), 1)
        nonzero = sum(1 for value in vals if abs(value) > 1e-12) / max(len(vals), 1)
        summary.append(
            {
                "name": factor["name"],
                "operation": factor["operation"],
                "mean": mean,
                "std": math.sqrt(var),
                "nonzero_rate": nonzero,
            }
        )
    return summary


def main() -> None:
    logging.getLogger("pyhealth").handlers.clear()
    logging.getLogger("pyhealth").propagate = True
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    model = os.environ.get("LLM_MODEL", "gpt-5.5")
    output_root = Path(os.environ.get("FACTOR_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    output_root.mkdir(parents=True, exist_ok=True)

    factor_path = Path(
        os.environ.get("FACTOR_JSON", output_root / "eicu_los_gpt55_factors.json")
    )
    reuse_factor_json = str_to_bool(os.environ.get("REUSE_FACTOR_JSON", "0"))

    if factor_path.exists() and reuse_factor_json:
        payload = json.loads(factor_path.read_text())
        payload["factors"] = normalize_factors(payload)
    else:
        logging.info("Generating medical symbolic factors with %s", model)
        payload = generate_factors_with_llm(model)
        factor_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    factors = payload["factors"]

    logging.info("Factor JSON: %s", factor_path)
    logging.info("Generated factors:")
    for factor in factors:
        logging.info(
            "  - %s [%s]: %s",
            factor["name"],
            factor["operation"],
            factor["description"],
        )

    run_train = str_to_bool(os.environ.get("RUN_TRAIN", "1"))
    if not run_train:
        return

    eicu_root = Path(os.environ.get("EICU_ROOT", DEFAULT_EICU_ROOT)).expanduser()
    require_eicu_files(eicu_root)

    dev = str_to_bool(os.environ.get("EICU_DEV", "1"))
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    seed = int(os.environ.get("SEED", "0"))
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = os.environ.get("DEVICE", default_device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    factor_hash = hashlib.md5(
        json.dumps(factors, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()[:8]
    cache_dir = Path(
        os.environ.get(
            "PYHEALTH_LOCAL_CACHE_DIR",
            DEFAULT_CACHE_ROOT / f"eicu_los_quantaalpha_llm_factors_{factor_hash}",
        )
    )
    run_root = Path(os.environ.get("PYHEALTH_LOCAL_OUTPUT_DIR", DEFAULT_RUN_ROOT))
    exp_name = (
        f"eicu_los_quantaalpha_llm_gpt55_dev_{factor_hash}"
        if dev
        else f"eicu_los_quantaalpha_llm_gpt55_full_{factor_hash}"
    )

    logging.info("Training PyHealth model with LLM-generated factors")
    logging.info("eICU root: %s", eicu_root)
    logging.info("Cache root: %s", cache_dir)
    logging.info("Output root: %s", run_root)
    logging.info("Epochs: %d", epochs)
    logging.info("Batch size: %d", batch_size)
    logging.info("Device: %s", device)

    base_dataset = eICUDataset(
        root=str(eicu_root),
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir=str(cache_dir),
        num_workers=num_workers,
        dev=dev,
    )
    base_dataset.stats()

    task = LengthOfStayPredictioneICUWithLLMFactors(factors)
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)
    if len(sample_dataset) == 0:
        raise RuntimeError("Task produced zero samples.")

    label_key = next(iter(sample_dataset.output_schema.keys()))
    summary = summarize_factors(sample_dataset, factors)
    summary_path = output_root / f"eicu_los_gpt55_factor_summary_{factor_hash}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logging.info("Factor summary: %s", summary_path)
    for item in summary:
        logging.info(
            "  - %s mean=%.4f std=%.4f nonzero=%.3f",
            item["name"],
            item["mean"],
            item["std"],
            item["nonzero_rate"],
        )

    logging.info("Input schema: %s", sample_dataset.input_schema)
    logging.info("Sample dataset size: %d", len(sample_dataset))
    logging.info(
        "Label counts: %s",
        dict(sorted(label_counts(sample_dataset, label_key).items())),
    )

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset,
        [0.8, 0.1, 0.1],
        seed=seed,
    )
    logging.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model_obj = QuantaAlphaPyHealthModel(dataset=sample_dataset)
    trainer = Trainer(
        model=model_obj,
        metrics=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
        device=device,
        output_path=str(run_root),
        exp_name=exp_name,
    )
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        monitor="f1_macro",
        monitor_criterion="max",
    )

    scores = trainer.evaluate(test_dataloader)
    scores_path = run_root / exp_name / "final_test_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    logging.info("Final test scores: %s", scores)
    logging.info("Final test scores written to: %s", scores_path)


if __name__ == "__main__":
    main()
