"""Sepsis-3 cohort specification hooks for TRACE V4.

This module intentionally does not invent a new cohort definition. The MIMIC
path is designed around the MIT-LCP MIMIC concept implementation; eICU is a
mapping audit layer for the same conceptual components.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sepsis3Component:
    name: str
    mimic_status: str
    eicu_status: str
    notes: str


SEPSIS3_COMPONENTS = (
    Sepsis3Component(
        "suspected_infection",
        "MIT-LCP concept required",
        "requires medication/antibiotic and culture timing mapping",
        "Do not approximate without antibiotic/culture timestamp audit.",
    ),
    Sepsis3Component(
        "sofa_delta_ge_2",
        "MIT-LCP SOFA concept required",
        "requires component-level SOFA reproduction",
        "Baseline SOFA assumption must match the selected MIT-LCP concept.",
    ),
    Sepsis3Component(
        "icu_onset",
        "anchor to ICU admission time and redacted ICU stay key",
        "anchor to unit admission offset and redacted eICU stay key",
        "Onset must not depend on future target windows used for prediction.",
    ),
)


def mimic_sepsis3_plan() -> dict[str, object]:
    return {
        "dataset": "mimic",
        "definition_source": "MIT-LCP MIMIC Sepsis-3 concept implementation",
        "status": "requires_local_researcher_execution",
        "components": [component.__dict__ for component in SEPSIS3_COMPONENTS],
    }


def eicu_sepsis3_mapping_audit() -> list[dict[str, str]]:
    return [
        {
            "component": component.name,
            "mimic_reference": component.mimic_status,
            "eicu_reproducibility": component.eicu_status,
            "semantic_equivalence": "not_assumed",
            "notes": component.notes,
        }
        for component in SEPSIS3_COMPONENTS
    ]
