"""Data loading and alignment helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES


def read_patient_sequences(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read a CSV split into ordered patient sequences."""
    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {"patient_id": row["patient_id"], "time_index": int(row["time_index"])}
            for name in ACTION_VARIABLES:
                parsed[name] = float(row[name])
            for name in STATE_VARIABLES:
                parsed[name] = float(row[name]) if row[name] else None
                parsed[f"{name}_mask"] = int(row[f"{name}_mask"])
            patients[parsed["patient_id"]].append(parsed)
    for rows in patients.values():
        rows.sort(key=lambda item: item["time_index"])
    return dict(patients)


def missingness_report(path: str | Path) -> dict[str, float]:
    """Return fraction missing per state variable."""
    sequences = read_patient_sequences(path)
    counts = {name: [0, 0] for name in STATE_VARIABLES}
    for rows in sequences.values():
        for row in rows:
            for name in STATE_VARIABLES:
                counts[name][1] += 1
                counts[name][0] += 1 - row[f"{name}_mask"]
    return {name: missing / total for name, (missing, total) in counts.items()}

