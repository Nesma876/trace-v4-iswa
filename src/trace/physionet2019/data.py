"""Strict PSV parsing, source discovery and immutable patient manifests."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np

from .schema import HEADER


def read_psv(path: Path) -> np.ndarray:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        if tuple(handle.readline().strip().split("|")) != HEADER:
            raise ValueError("PSV header differs from official 41-column schema")
        rows = list(csv.reader(handle, delimiter="|"))
    if not rows or any(len(row) != len(HEADER) for row in rows):
        raise ValueError("Empty or ragged PSV")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Non-numeric PSV measurement") from exc
    if np.isinf(values).any():
        raise ValueError("Infinite measurement")
    hours = values[:, HEADER.index("ICULOS")]
    if not np.isfinite(hours).all() or not np.equal(hours, np.floor(hours)).all() or hours[0] < 1 or not np.equal(np.diff(hours), 1).all():
        raise ValueError("ICULOS must be positive integer consecutive hours in original order")
    labels = values[:, -1]
    if not np.isin(labels, [0, 1]).all() or (np.diff(labels) < 0).any():
        raise ValueError("SepsisLabel must be binary and monotone")
    return values


def discover(root: Path, source: str, *, allow_duplicates=False) -> list[Path]:
    if source not in ("A", "B"):
        raise ValueError("Unknown source")
    aliases = {f"training_set{source}".lower(), f"training_set{source}".lower() + ".zip", f"set{source}".lower()}
    paths = sorted(p for p in Path(root).rglob("*.psv") if any(part.lower() in aliases for part in p.parts))
    names = [p.name for p in paths]
    if len(names) != len(set(names)) and not allow_duplicates:
        raise ValueError(f"Duplicate patient filenames in Set {source}; remove redundant extraction")
    return paths


def split_for(patient: str, seed: int = 4317) -> str:
    # Preserve the existing credentialed pipeline's hash-based 70/15/15 convention.
    value = int(hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest()[:12], 16) / 16**12
    return "train" if value < .70 else "validation" if value < .85 else "internal_test"


def patient_split(paths: list[Path], destination: Path, seed: int = 4317) -> dict[str, str]:
    ids = sorted(p.name for p in paths)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Split requires nonempty unique Set A patient IDs")
    assignments = {name: split_for(name, seed) for name in ids}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["file_id", "split"])
    writer.writerows(assignments.items())
    expected = stream.getvalue().encode()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != expected:
            raise ValueError("Frozen split differs from cohort/seed; will not regenerate silently")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(expected)
    return assignments
