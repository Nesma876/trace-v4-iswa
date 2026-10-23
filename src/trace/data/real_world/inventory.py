"""Lightweight MIMIC-IV/eICU inventory without downloading or full-table loading."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

from trace.data.real_world.governance import FORBIDDEN_OUTPUT_COLUMNS, env_status, raw_root_from_env, scrub_inventory_row, write_safe_json


EXPECTED_TABLES = {
    "mimic": [
        "patients",
        "admissions",
        "icustays",
        "chartevents",
        "labevents",
        "inputevents",
        "outputevents",
        "d_items",
        "d_labitems",
    ],
    "eicu": [
        "patient",
        "vitalPeriodic",
        "vitalAperiodic",
        "lab",
        "infusionDrug",
        "intakeOutput",
        "medication",
        "respiratoryCharting",
        "nurseCharting",
    ],
}

RELEVANT_COLUMNS = {
    "mimic": {
        "patients": ["subject_id", "gender", "anchor_age"],
        "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
        "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        "chartevents": ["subject_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        "labevents": ["subject_id", "hadm_id", "charttime", "storetime", "itemid", "valuenum", "valueuom"],
        "inputevents": ["subject_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "rate", "rateuom"],
        "outputevents": ["subject_id", "stay_id", "charttime", "itemid", "value", "valueuom"],
    },
    "eicu": {
        "patient": ["patientunitstayid", "uniquepid", "hospitalid", "unitadmittime24", "unitdischargetime24"],
        "vitalPeriodic": ["patientunitstayid", "observationoffset", "heartrate", "respiration", "sao2", "temperature"],
        "vitalAperiodic": ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"],
        "lab": ["patientunitstayid", "labresultoffset", "labname", "labresult", "labmeasurenamesystem"],
        "infusionDrug": ["patientunitstayid", "infusionoffset", "drugname", "drugrate", "infusionrate"],
        "intakeOutput": ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"],
    },
}


def build_inventory(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return synthetic_inventory()
    datasets = {"mimic": None, "eicu": None}
    for dataset in datasets:
        try:
            datasets[dataset] = raw_root_from_env(dataset)
        except RuntimeError:
            datasets[dataset] = None
    return {
        "datasets": {name: inspect_dataset(name, root) for name, root in datasets.items()},
        "environment": {name: env_status(name) for name in datasets},
        "note": "No data were downloaded. Raw source paths are never written to outputs.",
    }


def inspect_dataset(dataset: str, root: Path | None) -> dict[str, Any]:
    if root is None or not root.exists():
        return {
            "dataset": dataset,
            "root": "not_recorded",
            "available": False,
            "version": "not_identified",
            "tables": [],
            "missing_expected_tables": EXPECTED_TABLES[dataset],
        }
    files = [p for p in root.rglob("*") if p.is_file() and _is_table_file(p)]
    table_rows = []
    seen = set()
    for path in files[:200]:
        table = _table_name(path)
        seen.add(table)
        columns = _read_header(path)
        expected = RELEVANT_COLUMNS.get(dataset, {}).get(table, [])
        safe_columns = _safe_columns(columns)
        table_rows.append(
            {
                "dataset": dataset,
                "version": _version_hint(root),
                "table": table,
                "rows": "not_counted",
                "relevant_columns": [c for c in expected if c in safe_columns],
                "timestamps": [c for c in safe_columns if "time" in c.lower() or "offset" in c.lower()],
                "identifier_columns": "redacted",
                "units": [c for c in safe_columns if "unit" in c.lower() or "uom" in c.lower()],
                "columns": safe_columns,
            }
        )
    return {
        "dataset": dataset,
        "root": "not_recorded",
        "available": True,
        "version": _version_hint(root),
        "tables": [scrub_inventory_row(row) for row in table_rows],
        "missing_expected_tables": [name for name in EXPECTED_TABLES[dataset] if name not in seen],
    }


def write_inventory(path: Path, inventory: dict[str, Any]) -> None:
    write_safe_json(path, inventory)


def synthetic_inventory() -> dict[str, Any]:
    return {
        "datasets": {
            dataset: {
                "dataset": dataset,
                "root": "dry_run_not_recorded",
                "available": True,
                "version": "dry_run_schema",
                "tables": [
                    {
                        "dataset": dataset,
                        "version": "dry_run_schema",
                        "table": table,
                        "rows": "dry_run_not_counted",
                        "relevant_columns": _safe_columns(RELEVANT_COLUMNS.get(dataset, {}).get(table, [])),
                        "timestamps": [c for c in _safe_columns(RELEVANT_COLUMNS.get(dataset, {}).get(table, [])) if "time" in c.lower() or "offset" in c.lower()],
                        "identifier_columns": "redacted",
                        "units": [c for c in _safe_columns(RELEVANT_COLUMNS.get(dataset, {}).get(table, [])) if "unit" in c.lower() or "uom" in c.lower()],
                        "path_recorded": False,
                    }
                    for table in EXPECTED_TABLES[dataset]
                ],
                "missing_expected_tables": [],
            }
            for dataset in ["mimic", "eicu"]
        },
        "environment": {dataset: {"dataset": dataset, "env_var": f"TRACE_{dataset.upper()}_ROOT", "configured": "dry_run", "path_recorded": False} for dataset in ["mimic", "eicu"]},
        "note": "Dry-run synthetic schema only; no credentialed data accessed.",
    }


def _is_table_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz")


def _safe_columns(columns: list[str]) -> list[str]:
    safe = []
    for column in columns:
        if column.lower() in FORBIDDEN_OUTPUT_COLUMNS:
            safe.append("IDENTIFIER_REDACTED")
        else:
            safe.append(column)
    return safe


def _table_name(path: Path) -> str:
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-7]
    if name.endswith(".csv"):
        return name[:-4]
    return path.stem


def _read_header(path: Path) -> list[str]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError, StopIteration):
        return []


def _version_hint(root: Path) -> str:
    for candidate in ["VERSION", "version.txt", "README.md", "SHA256SUMS.txt"]:
        path = root / candidate
        if path.exists():
            return candidate
    return "not_identified"
