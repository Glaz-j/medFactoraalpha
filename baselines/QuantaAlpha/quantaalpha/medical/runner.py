"""PyHealth runner for medical QuantaAlpha factor experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.tasks import LengthOfStayPredictioneICU
from pyhealth.trainer import Trainer
from quantaalpha.core.developer import Developer
from quantaalpha.medical.dsl import compute_factor
from quantaalpha.medical.experiment import MedicalFactorExperiment
from quantaalpha.pyhealth_model import QuantaAlphaPyHealthModel


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "quantaalpha_medical_workflow"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "pyhealth_runs"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "pyhealth_cache"


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


class LengthOfStayPredictioneICUWithWorkflowFactors(LengthOfStayPredictioneICU):
    task_name = "LengthOfStayPredictioneICUWithWorkflowFactors"
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


class PyHealthFactorRunner(Developer[MedicalFactorExperiment]):
    """Evaluate medical symbolic factors using PyHealth Trainer."""

    def develop(
        self,
        exp: MedicalFactorExperiment,
        use_local: bool = True,
    ) -> MedicalFactorExperiment:
        logging.getLogger("pyhealth").handlers.clear()
        logging.getLogger("pyhealth").propagate = True

        factors = exp.factor_dicts()
        fhash = factor_hash(factors)
        output_root = Path(os.environ.get("MEDICAL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
        output_root.mkdir(parents=True, exist_ok=True)
        factor_path = output_root / f"workflow_factors_{fhash}.json"
        factor_path.write_text(json.dumps({"factors": factors}, indent=2, sort_keys=True) + "\n")

        eicu_root = Path(os.environ.get("EICU_ROOT", DEFAULT_EICU_ROOT)).expanduser()
        require_eicu_files(eicu_root)

        dev = str_to_bool(os.environ.get("EICU_DEV", "1"))
        epochs = int(os.environ.get("EPOCHS", os.environ.get("MEDICAL_EPOCHS", "1")))
        batch_size = int(os.environ.get("BATCH_SIZE", os.environ.get("MEDICAL_BATCH_SIZE", "64")))
        num_workers = int(os.environ.get("NUM_WORKERS", os.environ.get("MEDICAL_NUM_WORKERS", "4")))
        seed = int(os.environ.get("SEED", "0"))
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = os.environ.get("DEVICE", default_device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

        cache_dir = Path(
            os.environ.get(
                "PYHEALTH_LOCAL_CACHE_DIR",
                DEFAULT_CACHE_ROOT / f"eicu_los_quantaalpha_workflow_{fhash}",
            )
        )
        run_root = Path(os.environ.get("PYHEALTH_LOCAL_OUTPUT_DIR", DEFAULT_RUN_ROOT))
        exp_name = (
            f"eicu_los_quantaalpha_workflow_dev_{fhash}"
            if dev
            else f"eicu_los_quantaalpha_workflow_full_{fhash}"
        )

        base_dataset = eICUDataset(
            root=str(eicu_root),
            tables=["diagnosis", "medication", "physicalexam"],
            cache_dir=str(cache_dir),
            num_workers=num_workers,
            dev=dev,
        )
        base_dataset.stats()

        task = LengthOfStayPredictioneICUWithWorkflowFactors(factors)
        sample_dataset = base_dataset.set_task(task, num_workers=num_workers)
        if len(sample_dataset) == 0:
            raise RuntimeError("Task produced zero samples.")

        label_key = next(iter(sample_dataset.output_schema.keys()))
        summary = summarize_factors(sample_dataset, factors)
        summary_path = output_root / f"workflow_factor_summary_{fhash}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset,
            [0.8, 0.1, 0.1],
            seed=seed,
        )
        train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        model = QuantaAlphaPyHealthModel(dataset=sample_dataset)
        trainer = Trainer(
            model=model,
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

        exp.result = {
            "scores": scores,
            "factor_hash": fhash,
            "factor_path": str(factor_path),
            "factor_summary_path": str(summary_path),
            "scores_path": str(scores_path),
            "run_dir": str(run_root / exp_name),
            "cache_dir": str(cache_dir),
            "sample_size": len(sample_dataset),
            "label_counts": dict(sorted(label_counts(sample_dataset, label_key).items())),
            "split_sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "factor_summary": summary,
        }
        exp.sub_results = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        return exp

