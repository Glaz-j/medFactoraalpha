"""
Mortality prediction on local eICU with RNN.

This example mirrors mortality_prediction_eicu_rnn.py, but uses the local
eICU 2.0 files in this workspace and writes caches/results under
medFactoraalpha/results.

Environment variables:
    EICU_ROOT: local eICU root directory
    PYHEALTH_LOCAL_CACHE_DIR: cache root for parsed PyHealth data
    PYHEALTH_LOCAL_OUTPUT_DIR: output root for trainer checkpoints/logs
    EICU_DEV: 1/0, whether to limit to 1000 patients (default: 1)
    EPOCHS: number of training epochs (default: 1)
    BATCH_SIZE: dataloader batch size (default: 32)
    NUM_WORKERS: PyHealth preprocessing workers (default: 4)
    DEVICE: torch device (default: cuda:0 when available)
    ALLOW_CPU: set to 1 to allow CPU fallback
"""

import logging
import os
from collections import Counter
from pathlib import Path

import torch

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionEICU
from pyhealth.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_EICU_ROOT = WORKSPACE_ROOT / "eicu" / "EICU 2.0"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "pyhealth_cache" / "eicu_mortality_rnn_local"
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
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    seed = int(os.environ.get("SEED", "0"))

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = os.environ.get("DEVICE", default_device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device == "cpu" and not str_to_bool(os.environ.get("ALLOW_CPU", "0")):
        raise RuntimeError("GPU is required. Set ALLOW_CPU=1 only for debugging.")

    require_eicu_files(eicu_root)

    logging.info("Local eICU RNN mortality example")
    logging.info("eICU root: %s", eicu_root)
    logging.info("Cache root: %s", cache_dir)
    logging.info("Output root: %s", output_dir)
    logging.info("Dev mode: %s", dev)
    logging.info("Device: %s", device)

    base_dataset = eICUDataset(
        root=str(eicu_root),
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir=str(cache_dir),
        num_workers=num_workers,
        dev=dev,
    )
    base_dataset.stats()

    task = MortalityPredictionEICU()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "MortalityPredictionEICU produced zero samples. Try EICU_DEV=0 or "
            "switch to a task that does not require multiple ICU stays per patient."
        )

    label_key = next(iter(sample_dataset.output_schema.keys()))
    logging.info("Sample dataset size: %d", len(sample_dataset))
    logging.info("Label key: %s", label_key)
    logging.info("Label counts: %s", dict(label_counts(sample_dataset, label_key)))

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
    logging.info("Train labels: %s", dict(label_counts(train_dataset, label_key)))
    logging.info("Val labels: %s", dict(label_counts(val_dataset, label_key)))
    logging.info("Test labels: %s", dict(label_counts(test_dataset, label_key)))

    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RNN(dataset=sample_dataset)
    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1"],
        device=device,
        output_path=str(output_dir),
        exp_name="eicu_mortality_rnn_local_smoke",
    )
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        monitor="loss",
        monitor_criterion="min",
    )

    scores = trainer.evaluate(test_dataloader)
    logging.info("Final test scores: %s", scores)


if __name__ == "__main__":
    main()
