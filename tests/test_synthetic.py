from pathlib import Path

from trace.data.leakage_audit import audit_patient_splits, audit_temporal_order
from trace.data.preprocessing import read_patient_sequences
from trace.data.synthetic import generate_synthetic_dataset


def test_synthetic_splits_are_patient_level(tmp_path: Path) -> None:
    files = generate_synthetic_dataset(tmp_path, n_patients=24, n_steps=8, seed=3)
    assert audit_patient_splits(files) == []
    assert audit_temporal_order(files["train"]) == []


def test_synthetic_rows_have_masks(tmp_path: Path) -> None:
    files = generate_synthetic_dataset(tmp_path, n_patients=8, n_steps=5, seed=4)
    patients = read_patient_sequences(files["train"])
    first_row = next(iter(patients.values()))[0]
    assert "heart_rate_mask" in first_row
    assert first_row["time_index"] == 0

