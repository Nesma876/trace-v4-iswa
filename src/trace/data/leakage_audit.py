"""Automated checks for split and temporal leakage."""

from __future__ import annotations

from trace.data.preprocessing import read_patient_sequences


def audit_patient_splits(split_paths: dict[str, str]) -> list[str]:
    """Check that patients do not appear in multiple non-external splits."""
    seen: dict[str, str] = {}
    errors = []
    for split, path in split_paths.items():
        if split == "external":
            continue
        for patient_id in read_patient_sequences(path):
            if patient_id in seen:
                errors.append(f"patient {patient_id} appears in {seen[patient_id]} and {split}")
            seen[patient_id] = split
    return errors


def audit_temporal_order(path: str) -> list[str]:
    """Check that every patient sequence is strictly ordered by time index."""
    errors = []
    for patient_id, rows in read_patient_sequences(path).items():
        times = [row["time_index"] for row in rows]
        if times != sorted(times) or len(times) != len(set(times)):
            errors.append(f"patient {patient_id} has invalid temporal order")
    return errors

