"""Data-governance safeguards for credentialed real-world ICU datasets."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

RAW_DATA_ENV = {"mimic": "TRACE_MIMIC_ROOT", "eicu": "TRACE_EICU_ROOT"}
FORBIDDEN_OUTPUT_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "patient_id",
}


def raw_root_from_env(dataset: str) -> Path:
    env_name = RAW_DATA_ENV[dataset]
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is required for local {dataset} execution")
    return Path(value)


def env_status(dataset: str) -> dict[str, Any]:
    env_name = RAW_DATA_ENV[dataset]
    value = os.environ.get(env_name)
    return {"dataset": dataset, "env_var": env_name, "configured": bool(value), "path_recorded": False}


def scrub_inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    clean.pop("path", None)
    clean["path_recorded"] = False
    return clean


def assert_publication_safe_rows(rows: list[dict[str, Any]], context: str) -> None:
    if not rows:
        return
    keys = {key.lower() for row in rows for key in row}
    forbidden = sorted(keys & FORBIDDEN_OUTPUT_COLUMNS)
    if forbidden:
        raise ValueError(f"{context} contains forbidden identifier columns: {', '.join(forbidden)}")


def assert_publication_safe_file(path: Path) -> None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    forbidden = sorted({h.lower() for h in header} & FORBIDDEN_OUTPUT_COLUMNS)
    if forbidden:
        raise ValueError(f"{path} contains forbidden identifier columns: {', '.join(forbidden)}")


def write_safe_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2)
    for env_name in RAW_DATA_ENV.values():
        value = os.environ.get(env_name)
        if value and value in text:
            raise ValueError(f"refusing to write raw path from {env_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
