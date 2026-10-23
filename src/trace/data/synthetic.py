"""Synthetic longitudinal ICU data for TRACE dry-runs."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable

STATE_VARIABLES = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "temperature",
    "mean_arterial_pressure",
    "lactate",
    "creatinine",
]

ACTION_VARIABLES = ["iv_fluids", "vasopressors"]


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_patients: int = 96,
    n_steps: int = 16,
    seed: int = 17,
    missing_probability: float = 0.12,
    external_shift: float = 0.35,
    dynamics_shift: float = 0.0,
) -> dict[str, str]:
    """Create train/validation/test/external CSV files with masks and actions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for patient_idx in range(n_patients):
        rows.extend(_patient_rows(f"p{patient_idx:04d}", n_steps, rng, missing_probability, shift=0.0, dynamics_shift=dynamics_shift))
    ext_rows = []
    for patient_idx in range(max(12, n_patients // 4)):
        ext_rows.extend(_patient_rows(f"e{patient_idx:04d}", n_steps, rng, missing_probability + 0.05, shift=external_shift, dynamics_shift=dynamics_shift))

    patient_ids = sorted({row["patient_id"] for row in rows})
    rng.shuffle(patient_ids)
    n_train = int(0.6 * len(patient_ids))
    n_val = int(0.2 * len(patient_ids))
    split_map = {
        "train": set(patient_ids[:n_train]),
        "validation": set(patient_ids[n_train : n_train + n_val]),
        "test": set(patient_ids[n_train + n_val :]),
    }
    files = {}
    for split, ids in split_map.items():
        path = output / f"{split}.csv"
        _write_rows(path, [row for row in rows if row["patient_id"] in ids])
        files[split] = str(path)
    external_path = output / "external_shift.csv"
    _write_rows(external_path, ext_rows)
    files["external"] = str(external_path)
    manifest = {
        "seed": seed,
        "n_patients": n_patients,
        "n_steps": n_steps,
        "state_variables": STATE_VARIABLES,
        "action_variables": ACTION_VARIABLES,
        "splits": {split: len(ids) for split, ids in split_map.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files


def _patient_rows(
    patient_id: str,
    n_steps: int,
    rng: random.Random,
    missing_probability: float,
    shift: float,
    dynamics_shift: float = 0.0,
) -> Iterable[dict[str, str]]:
    severity = rng.uniform(0.2, 1.2) + shift
    state = {
        "heart_rate": 82 + 18 * severity + rng.gauss(0, 4),
        "respiratory_rate": 18 + 5 * severity + rng.gauss(0, 1.5),
        "spo2": 97 - 2.5 * severity + rng.gauss(0, 1),
        "temperature": 37 + 0.7 * severity + rng.gauss(0, 0.2),
        "mean_arterial_pressure": 82 - 12 * severity + rng.gauss(0, 4),
        "lactate": 1.2 + 1.5 * severity + rng.gauss(0, 0.2),
        "creatinine": 0.9 + 0.4 * severity + rng.gauss(0, 0.1),
    }
    for step in range(n_steps):
        iv_fluids = max(0.0, min(2.0, (75 - state["mean_arterial_pressure"]) / 20 + rng.random() * 0.25))
        vasopressors = max(0.0, min(1.5, (68 - state["mean_arterial_pressure"]) / 18 + severity * 0.15))
        row = {"patient_id": patient_id, "time_index": str(step)}
        for name in ACTION_VARIABLES:
            row[name] = f"{locals()[name]:.6f}"
        for name in STATE_VARIABLES:
            observed = rng.random() > missing_probability
            row[name] = f"{state[name]:.6f}" if observed else ""
            row[f"{name}_mask"] = "1" if observed else "0"
        yield row
        pressure_support = 4.0 * iv_fluids + 7.0 * vasopressors
        inflammatory = math.sin(step / 3.0 + severity) * 0.4 + rng.gauss(0, 0.25)
        state["mean_arterial_pressure"] += (0.25 + dynamics_shift) * (78 - state["mean_arterial_pressure"]) + pressure_support * (0.08 + dynamics_shift * 0.02) + rng.gauss(0, 1.8)
        state["heart_rate"] += 0.18 * (88 - state["heart_rate"]) + inflammatory + 1.5 * vasopressors + rng.gauss(0, 2)
        state["respiratory_rate"] += 0.2 * (20 - state["respiratory_rate"]) + inflammatory + rng.gauss(0, 0.8)
        state["spo2"] += 0.25 * (96 - state["spo2"]) - 0.15 * inflammatory + rng.gauss(0, 0.5)
        state["temperature"] += 0.15 * (37.2 - state["temperature"]) + 0.05 * inflammatory + rng.gauss(0, 0.08)
        state["lactate"] += (0.18 - dynamics_shift * 0.03) * (1.4 - state["lactate"]) - 0.08 * iv_fluids + max(0, 65 - state["mean_arterial_pressure"]) * (0.015 + dynamics_shift * 0.004)
        state["creatinine"] += 0.08 * (1.0 - state["creatinine"]) + max(0, 70 - state["mean_arterial_pressure"]) * (0.003 + dynamics_shift * 0.001)
        state["spo2"] = max(65.0, min(100.0, state["spo2"]))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["patient_id", "time_index", *ACTION_VARIABLES]
    for name in STATE_VARIABLES:
        fields.extend([name, f"{name}_mask"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
