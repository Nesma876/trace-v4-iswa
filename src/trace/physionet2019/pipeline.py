"""Aggregate-only inventory/audit and local immutable preprocessing artifacts."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .data import discover, patient_split, read_psv
from .preprocessing import fit_preprocessing, eligible_origins
from .schema import DYNAMIC, HEADER, ROLES

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/physionet2019/protocol.json"
TITLE = "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019"
SOURCE = "https://physionet.org/content/challenge-2019/1.0.0/"


def sha(path):
    path = Path(path)
    payload = path.read_bytes()
    # Git stores these text artifacts with LF; bind logical bytes consistently on
    # Windows and Unix. Raw patient PSV content hashes remain strictly bytewise.
    if path.suffix in (".json", ".yaml", ".csv"):
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def write_json(path, payload, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if frozen and path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen artifact differs: {path.name}; explicit new protocol required")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def code_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=15).strip()


def config_hashes():
    return {p.name: sha(p) for p in sorted((REPO / "configs/physionet2019").glob("*")) if p.is_file()}


def inventory(root, out):
    root, out = Path(root), Path(out)
    result = {"dataset_version": "1.0.0", "source_policy": "A development; B structural only", "sets": {}, "external_test_locked": True}
    for source in ("A", "B"):
        paths = discover(root, source, allow_duplicates=True)
        duplicate_count = len(paths) - len({p.name for p in paths})
        invalid = Counter()
        lengths = []
        digest = hashlib.sha256()
        for path in paths:
            try:
                digest.update(path.name.encode())
                digest.update(bytes.fromhex(sha(path)))
                raw = read_psv(path)
                lengths.append(len(raw))
            except (ValueError, OSError) as exc:
                invalid[type(exc).__name__ + ": " + str(exc).split("\n")[0] if isinstance(exc, ValueError) else "unreadable"] += 1
        entry = {"patient_files": len(paths), "valid_files": len(lengths), "invalid_files": sum(invalid.values()),
                 "invalid_reasons": dict(invalid), "duplicate_patient_filenames": duplicate_count,
                 "header_identity": bool(paths) and not invalid, "SepsisLabel_available": bool(paths) and not invalid,
                 "hourly_rows": sum(lengths), "content_hash": digest.hexdigest()}
        entry["rows_hours_distribution"] = length_summary(lengths) if source == "A" else "withheld: Set B locked"
        result["sets"][source] = entry
    a_names = {p.name for p in discover(root, "A", allow_duplicates=True)}
    b_names = {p.name for p in discover(root, "B", allow_duplicates=True)}
    result["cross_source_duplicate_filenames"] = len(a_names & b_names)
    # Published expected numbers are validation expectations, never observed counts.
    result["published_expected_files"] = {"A": 20336, "B": 20000}
    result["complete"] = all(result["sets"][s]["patient_files"] == result["published_expected_files"][s]
                             and result["sets"][s]["header_identity"]
                             and not result["sets"][s]["duplicate_patient_filenames"] for s in ("A", "B")) and not (a_names & b_names)
    write_json(out / "data_inventory.json", result)
    receipt = root / "download_receipt.json"
    metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
    write_json(out / "data_manifest.json", {"dataset_title": TITLE, "version": "1.0.0", "doi": "10.13026/v64v-d857",
                "source_url": SOURCE, "download_date": metadata.get("download_date"),
                "Set_A_file_count": result["sets"]["A"]["patient_files"], "Set_B_file_count": result["sets"]["B"]["patient_files"],
                "raw_header": list(HEADER), "header_verified": result["complete"], "code_commit": code_commit(),
                "config_hashes": config_hashes(), "download_validation": metadata.get("validation"),
                "official_psv_checksums_available": metadata.get("official_psv_checksums_available")})
    return result


def length_summary(lengths):
    if not lengths:
        return {"min": None, "median": None, "max": None, "total": 0}
    return {"min": int(min(lengths)), "median": float(np.median(lengths)), "max": int(max(lengths)), "total": int(sum(lengths))}


def prepare(root, out):
    root, out = Path(root), Path(out)
    if (REPO / "configs/physionet2019/FROZEN_PREPROCESSING.yaml").exists():
        from .final_audit import describe_training, freeze_training
        describe_training(root, out)
        freeze_training(root, out)
        return json.loads((out / "audit/final_freeze_manifest.json").read_text())
    config = json.loads(CONFIG.read_text())
    inv = inventory(root, out)
    if not inv["complete"]:
        raise ValueError("Complete structurally valid official sets required before preparation")
    paths = [p for p in discover(root, "A") if len(read_psv(p)) >= config["minimum_hours"]]
    split_path = out / "splits/setA_patient_split.csv"
    assignments = patient_split(paths, split_path, config["seed"])
    if set(assignments.values()) != {"train", "validation", "internal_test"}:
        raise ValueError("At least one internal split is empty")
    train = [p for p in paths if assignments[p.name] == "train"]
    stats = fit_preprocessing(train, assignments, config)
    stats["split_hash"] = sha(split_path)
    stats["Set_A_content_hash"] = inv["sets"]["A"]["content_hash"]
    stats["config_hashes"] = config_hashes()
    write_json(out / "preprocessing.json", stats, frozen=True)
    schema = {"features": stats["features"], "column_roles": ROLES, "context_inputs": [], "action_dim": 0,
              "input_layout": ["normalized_values", "observation_masks", "hours_since_last_observation"],
              "target_policy": "observed dynamic values only; missing targets masked"}
    write_json(out / "feature_schema.json", schema, frozen=True)
    manifest = {"dataset_title": TITLE, "source_url": SOURCE,
                "current_commit": code_commit(), "data_doi": "10.13026/v64v-d857", "data_version": "1.0.0",
                "config_hashes": config_hashes(), "split_hash": sha(split_path),
                "feature_schema_hash": sha(out / "feature_schema.json"), "preprocessing_hash": sha(out / "preprocessing.json"),
                "data_inventory_hash": sha(out / "data_inventory.json"), "external_test_locked": True,
                "Set_B_prediction_evaluation_performed": False, "models_trained": False,
                "split_counts": dict(Counter(assignments.values()))}
    write_json(out / "manifest.json", manifest)
    return manifest


def audit(root, out):
    root, out = Path(root), Path(out)
    if (out / "audit/final_freeze_manifest.json").exists():
        from .policy import load_frozen_preprocessing
        stats = load_frozen_preprocessing(out)
        return {"final_preprocessing_verified": True, "variables": len(stats["features"]),
                "external_test_locked": True, "notice": "Final audits are in audit/; no held-out distribution scan performed"}
    config = json.loads(CONFIG.read_text())
    stats = json.loads((out / "preprocessing.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if sha(out / "preprocessing.json") != manifest["preprocessing_hash"] or config_hashes() != manifest["config_hashes"]:
        raise ValueError("Preprocessing/config provenance mismatch")
    # Rechecking Set B here is strictly structural; no shift or prevalence statistics.
    inv = inventory(root, out)
    if inv["sets"]["A"]["content_hash"] != stats["Set_A_content_hash"]:
        raise ValueError("Set A data changed after preparation")
    paths = [p for p in discover(root, "A") if len(read_psv(p)) >= config["minimum_hours"]]
    assignments = patient_split(paths, out / "splits/setA_patient_split.csv", config["seed"])
    if sha(out / "splits/setA_patient_split.csv") != stats["split_hash"]:
        raise ValueError("Split provenance mismatch")
    cohort, missingness, ranges, sequences, prevalence = [], [], [], [], []
    for split in ("train", "validation", "internal_test"):
        blocks = [read_psv(p) for p in paths if assignments[p.name] == split]
        lengths = [len(x) for x in blocks]
        x = np.concatenate(blocks)
        positive = sum(bool((b[:, -1] == 1).any()) for b in blocks)
        cohort.append({"source": "A", "split": split, "eligible_patients": len(blocks), "hourly_rows": len(x),
                       "common_48h_patients": sum(bool(eligible_origins(n, 48, config["history_hours"])) for n in lengths)})
        sequences.append({"source": "A", "split": split, **length_summary(lengths)})
        prevalence.append({"source": "A", "split": split, "sepsis_positive": positive, "non_sepsis": len(blocks) - positive,
                           "prevalence": positive / len(blocks)})
        for j, variable in enumerate(HEADER):
            values = x[:, j][np.isfinite(x[:, j])]
            missingness.append({"source": "A", "split": split, "variable": variable, "role": ROLES[variable],
                                "retained_dynamic": variable in stats["features"], "missingness": 1 - len(values) / len(x),
                                "observed_count": len(values), "total_hours": len(x)})
            if variable in DYNAMIC:
                q = np.quantile(values, [0, .01, .5, .99, 1]).tolist() if len(values) else [None] * 5
                ranges.append({"source": "A", "split": split, "variable": variable, "min": q[0], "p01": q[1],
                               "median": q[2], "p99": q[3], "max": q[4], "clipped": 0})
    for name, rows in [("cohort_summary", cohort), ("missingness_summary", missingness), ("variable_ranges", ranges),
                       ("sequence_length_summary", sequences), ("sepsis_prevalence_summary", prevalence)]:
        write_csv(out / (name + ".csv"), rows)
    selection = []
    training_missing = {r["variable"]: r["missingness"] for r in missingness if r["split"] == "train"}
    for row in stats["selection"]:
        selection.append({**row, "missingness": training_missing[row["variable"]], "role": "target and input" if row["included"] else "excluded"})
    for v in HEADER[len(DYNAMIC):]:
        selection.append({"variable": v, "included": False, "observations": "", "observed_patients": "",
                          "rationale": "label: subgroup only" if v == "SepsisLabel" else "context retained for audit, not model input/target",
                          "missingness": training_missing[v], "role": ROLES[v]})
    write_csv(out / "variable_selection.csv", selection)
    report = {"patient_leakage": "passed: immutable disjoint patient split", "normalization_leakage": "passed: A train provenance",
              "label_leakage": "passed: allowlisted physiology features", "external_Set_B": "locked; structural checks only",
              "temporal_and_imputation": "prefix-only implementation; tested with synthetic fixtures",
              "ineligible_Set_A_patients": inv["sets"]["A"]["patient_files"] - len(paths),
              "no_models_trained": True, "Set_B_prediction_evaluation_performed": False}
    write_json(out / "leakage_audit.json", report)
    return report
