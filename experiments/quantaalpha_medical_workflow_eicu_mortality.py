"""
Run the QuantaAlpha medical workflow on PyHealth eICU mortality.

This is a thin task-specific entrypoint around the shared eICU medical workflow.
It defaults to the tabular alpha evaluator and PyHealth-standard source tables.
"""

from __future__ import annotations

import os
from pathlib import Path

from quantaalpha_medical_workflow_eicu_los import PROJECT_ROOT, main


os.environ.setdefault("MEDICAL_EICU_TASK", "mortality")
os.environ.setdefault("MEDICAL_EVALUATOR", "tabular")
os.environ.setdefault("MEDICAL_WORKFLOW_FLAVOR", "alpha")
os.environ.setdefault("MEDICAL_TABULAR_DIRECT_SOURCE", "1")
os.environ.setdefault("MEDICAL_SOURCE_PROFILE", "pyhealth_standard")
os.environ.setdefault("MEDICAL_TEMPORAL_SOURCE", "1")
os.environ.setdefault("MEDICAL_NUMERIC_TEMPORAL_SOURCE", "0")
os.environ.setdefault(
    "MEDICAL_OUTPUT_ROOT",
    str(PROJECT_ROOT / "results" / "quantaalpha_medical_workflow_mortality"),
)
os.environ.setdefault(
    "MEDICAL_DIRECTION",
    (
        "Mine clinically meaningful symbolic factors for PyHealth eICU mortality "
        "prediction, where current ICU stay information predicts whether the next "
        "hospital visit/stay discharge status is expired. Prioritize early "
        "severity, persistent shock, respiratory failure, renal failure, and "
        "physiologic instability without target leakage."
    ),
)


if __name__ == "__main__":
    main()
