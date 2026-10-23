"""Deterministic patient-level splitting for MIMIC development cohorts."""

from __future__ import annotations

import hashlib


def assign_patient_split(patient_id: str, seed: int = 4317, train: float = 0.70, validation: float = 0.15) -> str:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "internal_test"


def assert_no_overlapping_patient_ids(split_rows: list[dict[str, str]]) -> None:
    seen: dict[str, str] = {}
    for row in split_rows:
        patient_id = row["patient_id"]
        split = row["split"]
        prior = seen.get(patient_id)
        if prior is not None and prior != split:
            raise ValueError(f"patient {patient_id} appears in both {prior} and {split}")
        seen[patient_id] = split

