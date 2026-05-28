"""Source-profile helpers for PyHealth-aligned and expanded eICU inputs."""

from __future__ import annotations

import os

from quantaalpha.medical.dsl import VALID_NUMERIC_SOURCES


PYHEALTH_STANDARD = "pyhealth_standard"
EXPANDED_V2 = "expanded_v2"
VALID_SOURCE_PROFILES = {PYHEALTH_STANDARD, EXPANDED_V2}


def source_profile(value: str | None = None) -> str:
    profile = (value or os.environ.get("MEDICAL_SOURCE_PROFILE", PYHEALTH_STANDARD)).strip().lower()
    aliases = {
        "pyhealth": PYHEALTH_STANDARD,
        "standard": PYHEALTH_STANDARD,
        "table15": PYHEALTH_STANDARD,
        "table_15": PYHEALTH_STANDARD,
        "expanded": EXPANDED_V2,
        "full": EXPANDED_V2,
        "expanded_numeric": EXPANDED_V2,
    }
    profile = aliases.get(profile, profile)
    if profile not in VALID_SOURCE_PROFILES:
        raise ValueError(
            "MEDICAL_SOURCE_PROFILE must be one of "
            f"{sorted(VALID_SOURCE_PROFILES)}; got {profile!r}"
        )
    return profile


def numeric_sources_for_profile(value: str | None = None) -> set[str]:
    if source_profile(value) == PYHEALTH_STANDARD:
        return set()
    return set(VALID_NUMERIC_SOURCES)


def numeric_sources_text(value: str | None = None) -> str:
    sources = sorted(numeric_sources_for_profile(value))
    if not sources:
        return "none for the PyHealth-standard source profile"
    return ", ".join(sources)
