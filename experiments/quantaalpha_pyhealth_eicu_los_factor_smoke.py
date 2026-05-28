"""
Smoke test for plugging QuantaAlpha-style symbolic factors into PyHealth.

This does not run QuantaAlpha's LLM loop yet. It mimics the expected output of
that loop with a small fixed set of symbolic factors, appends them to PyHealth's
eICU length-of-stay task as a tensor feature, and evaluates with PyHealth's
standard Trainer.

The point is to verify the evaluation side:

    eICU -> PyHealth task -> symbolic_factors tensor -> MultimodalRNN -> Trainer
"""

import json
import logging
import math
import os
from collections import Counter
from pathlib import Path

import torch

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.tasks import LengthOfStayPredictioneICU
from pyhealth.trainer import Trainer
from quantaalpha.pyhealth_model import QuantaAlphaPyHealthModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "results" / "pyhealth_cache" / "eicu_los_quantaalpha_factor_smoke"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "pyhealth_runs"


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


def label_counts(dataset, label_key: str) -> Counter:
    counts = Counter()
    for idx in range(len(dataset)):
        value = dataset[idx][label_key]
        if torch.is_tensor(value):
            value = int(value.item())
        counts[int(value)] += 1
    return counts


def has_any(texts: list[str], keywords: tuple[str, ...]) -> float:
    joined = " ".join(str(item).lower() for item in texts)
    return float(any(keyword in joined for keyword in keywords))


class LengthOfStayPredictioneICUWithSymbolicFactors(LengthOfStayPredictioneICU):
    """eICU LOS task plus fixed symbolic factor features.

    These factors are placeholders for the kind of expressions QuantaAlpha would
    eventually generate. They are intentionally simple and deterministic.
    """

    task_name = "LengthOfStayPredictioneICUWithSymbolicFactors"
    input_schema = {
        **LengthOfStayPredictioneICU.input_schema,
        "symbolic_factors": "tensor",
    }
    output_schema = LengthOfStayPredictioneICU.output_schema

    def __call__(self, patient):
        samples = super().__call__(patient)
        for sample in samples:
            conditions = sample.get("conditions", [])
            procedures = sample.get("procedures", [])
            drugs = sample.get("drugs", [])
            all_text = conditions + procedures + drugs

            n_conditions = len(conditions)
            n_procedures = len(procedures)
            n_drugs = len(drugs)

            sample["symbolic_factors"] = [
                math.log1p(n_conditions),
                math.log1p(n_procedures),
                math.log1p(n_drugs),
                n_conditions / max(n_drugs, 1),
                has_any(all_text, ("sepsis", "infection", "infectious")),
                has_any(all_text, ("respiratory", "pneumonia", "ventilator")),
                has_any(all_text, ("heart", "cardiac", "coronary")),
                has_any(all_text, ("renal", "kidney", "dialysis")),
            ]
        return samples


def main() -> None:
    logging.getLogger("pyhealth").handlers.clear()
    logging.getLogger("pyhealth").propagate = True
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    eicu_root = Path(os.environ.get("EICU_ROOT", DEFAULT_EICU_ROOT)).expanduser()
    cache_dir = Path(
        os.environ.get("PYHEALTH_LOCAL_CACHE_DIR", DEFAULT_CACHE_DIR)
    ).expanduser()
    output_dir = Path(
        os.environ.get("PYHEALTH_LOCAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    ).expanduser()

    dev = str_to_bool(os.environ.get("EICU_DEV", "1"))
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    seed = int(os.environ.get("SEED", "0"))

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = os.environ.get("DEVICE", default_device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device == "cpu" and not str_to_bool(os.environ.get("ALLOW_CPU", "0")):
        raise RuntimeError("GPU is required. Set ALLOW_CPU=1 only for debugging.")

    require_eicu_files(eicu_root)

    logging.info("QuantaAlpha-style symbolic factor smoke test")
    logging.info("eICU root: %s", eicu_root)
    logging.info("Cache root: %s", cache_dir)
    logging.info("Output root: %s", output_dir)
    logging.info("Dev mode: %s", dev)
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

    task = LengthOfStayPredictioneICUWithSymbolicFactors()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError("Task produced zero samples.")

    label_key = next(iter(sample_dataset.output_schema.keys()))
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

    model = QuantaAlphaPyHealthModel(dataset=sample_dataset)
    exp_name = (
        "eicu_los_quantaalpha_factor_smoke_dev"
        if dev
        else "eicu_los_quantaalpha_factor_smoke_full"
    )
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
        device=device,
        output_path=str(output_dir),
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
    logging.info("Final test scores: %s", scores)
    scores_path = output_dir / exp_name / "final_test_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    logging.info("Final test scores written to: %s", scores_path)


if __name__ == "__main__":
    main()
