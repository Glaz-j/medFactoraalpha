"""
Run the QuantaAlpha medical workflow on eICU readmission.

This is a thin task-specific entrypoint around the shared eICU medical workflow.
The readmission label is whether the next observed ICU stay belongs to the same
hospital system stay.
"""

from __future__ import annotations

import os

from quantaalpha_medical_workflow_eicu_los import PROJECT_ROOT, main


os.environ.setdefault("MEDICAL_EICU_TASK", "readmission")
os.environ.setdefault("MEDICAL_EVALUATOR", "tabular")
os.environ.setdefault("MEDICAL_WORKFLOW_FLAVOR", "alpha")
os.environ.setdefault("MEDICAL_TABULAR_DIRECT_SOURCE", "1")
os.environ.setdefault("MEDICAL_SOURCE_PROFILE", "pyhealth_standard")
os.environ.setdefault("MEDICAL_TEMPORAL_SOURCE", "1")
os.environ.setdefault("MEDICAL_NUMERIC_TEMPORAL_SOURCE", "0")
os.environ.setdefault(
    "MEDICAL_OUTPUT_ROOT",
    str(PROJECT_ROOT / "results" / "quantaalpha_medical_workflow_readmission"),
)
os.environ.setdefault(
    "MEDICAL_DIRECTION",
    (
        "Mine clinically meaningful symbolic factors for eICU readmission "
        "prediction, defined as whether the next observed ICU stay belongs to "
        "the same hospital system stay. Prioritize unresolved severity, "
        "persistent organ support, respiratory instability, renal support, and "
        "signals of incomplete recovery."
    ),
)


if __name__ == "__main__":
    main()
