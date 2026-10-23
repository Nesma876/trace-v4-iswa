"""Frozen training-derived cleaning. No fitted cutoffs and no external decisions."""
from pathlib import Path
import json

import numpy as np

from .schema import HEADER
from .pipeline import sha

POLICY = Path(__file__).resolve().parents[3] / "configs/physionet2019/FROZEN_RANGE_POLICY.yaml"
FLAG_NAMES = {0: "missing", 1: "kept", 2: "kept_subambient_uncertain", 3: "rejected_to_missing"}


def load_policy():
    return json.loads(POLICY.read_text(encoding="utf-8"))


def canonicalize(values, features, policy=None):
    """Return canonical values and per-entry flags without altering raw input.

    The separate raw array is retained by the caller. No percent-to-fraction conversion
    is inferred. Flags distinguish missing, retained, uncertain subambient and rejected.
    """
    policy = load_policy() if policy is None else policy
    rules = {r["variable"]: r for r in policy["variables"]}
    x = np.asarray(values, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] != len(features) or set(features) - rules.keys():
        raise ValueError("Canonicalization requires the frozen dynamic feature schema")
    flags = np.isfinite(x).astype(np.uint8)
    for j, variable in enumerate(features):
        rule = rules[variable]
        if rule["action"] == "KEEP":
            if np.isinf(x[:, j]).any():
                raise ValueError("Infinite raw value should have failed parsing")
            continue
        if rule["action"] != "SET_TO_MISSING_IF_OUTSIDE_HARD_RANGE":
            raise ValueError("Unsupported/unresolved range policy")
        observed = np.isfinite(x[:, j])
        low, high = rule["lower_hard_bound"], rule["upper_hard_bound"]
        invalid = np.isinf(x[:, j])
        if low is not None:
            invalid |= observed & ((x[:, j] < low) if rule["lower_inclusive"] else (x[:, j] <= low))
        if high is not None:
            invalid |= observed & ((x[:, j] > high) if rule["upper_inclusive"] else (x[:, j] >= high))
        if variable == "FiO2":
            flags[observed & (x[:, j] > 0) & (x[:, j] < .21), j] = 2
        flags[invalid, j] = 3
        x[invalid, j] = np.nan
    return x, flags


def fio2_audit_view(raw_values):
    raw = np.asarray(raw_values, dtype=float).copy()
    canonical, flags = canonicalize(raw.reshape(-1, 1), ["FiO2"])
    return {"raw_FiO2": raw, "canonical_FiO2": canonical[:, 0], "conversion_flag": flags[:, 0], "flag_names": FLAG_NAMES}


def model_values(raw, stats):
    values = np.asarray(raw)[:, [HEADER.index(v) for v in stats["features"]]]
    if "FiO2" in stats["features"] and not stats.get("range_policy_hash"):
        raise ValueError("FiO2 requires final post-policy training statistics; historical raw scaler is not valid")
    if stats.get("range_policy_hash"):
        if stats["range_policy_hash"] != sha(POLICY):
            raise ValueError("Frozen range policy changed after scaling")
        values, _ = canonicalize(values, stats["features"])
    return values


def load_frozen_preprocessing(output):
    """Only the final verified post-policy scaler may be used in subsequent modelling."""
    from .pipeline import config_hashes
    out = Path(output)
    manifest = json.loads((out / "audit/final_freeze_manifest.json").read_text())
    path = out / "audit/frozen_training_stats.json"
    if sha(path) != manifest["preprocessing_statistics_hash"]:
        raise ValueError("Final preprocessing statistics were modified")
    stats = json.loads(path.read_text())
    current = config_hashes()
    # New modelling-only configurations do not invalidate the frozen data contract.
    # Every configuration recorded at freeze time must still exist byte-identically.
    if any(current.get(k) != v for k, v in stats["config_hashes"].items()) or stats["range_policy_hash"] != sha(POLICY):
        raise ValueError("Frozen configurations changed")
    if sha(out / "splits/setA_patient_split.csv") != stats["split_hash"]:
        raise ValueError("Frozen split changed")
    return stats
