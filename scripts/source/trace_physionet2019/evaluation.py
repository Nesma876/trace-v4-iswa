"""Pre-model metric interfaces and fail-closed external access protocol."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def external_evaluation(callback, *, unlock_external_test=False, freeze_file=None, event_file=None):
    """Invoke a future evaluator once, only after explicit unlock and artifact freeze.

    The callback is the only operation allowed to load Set B for prediction or shift
    characterization. A failed attempt is retained and cannot silently be repeated.
    """
    if not unlock_external_test:
        raise PermissionError("Set B locked; --unlock-external-test required")
    if freeze_file is None or event_file is None:
        raise PermissionError("Model/protocol freeze and first-evaluation log required")
    freeze_file = Path(freeze_file)
    freeze = json.loads(freeze_file.read_text())
    if freeze.get("development_source") != "A" or freeze.get("calibration_source") != "A_validation" or freeze.get("model_selection_complete") is not True:
        raise PermissionError("Set A-only model selection/calibration must be complete")
    for kind in ("model", "protocol", "preprocessing", "feature_schema", "split"):
        item = freeze["artifacts"][kind]
        file = (freeze_file.parent / item["file"]).resolve()
        if not file.is_relative_to(freeze_file.parent.resolve()):
            raise PermissionError("Freeze artifact must be inside freeze directory")
        if hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
            raise PermissionError("Frozen artifact hash mismatch")
    event = Path(event_file)
    event.parent.mkdir(parents=True, exist_ok=True)
    record = {"first_evaluation_started_at": datetime.now(timezone.utc).isoformat(),
              "freeze_sha256": hashlib.sha256(freeze_file.read_bytes()).hexdigest(), "status": "started"}
    with event.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    try:
        result = callback()
    except Exception:
        record["status"] = "failed; explicit review required before any repeat"
        event.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise
    record["status"] = "completed"
    event.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result


def correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def trajectory_metrics(truth, mean, std, mask, baseline, baseline_mask):
    """One patient's full hourly rollout, [time, variable], in a declared common scale.

    Targets remain naturally observed only. Dynamic differences require both adjacent
    observations. Never score imputed targets. Aggregate patient/variable metrics later,
    with eligible counts, paired patient bootstrap and common-cohort horizon comparisons.
    """
    truth, mean, std = map(np.asarray, (truth, mean, std))
    mask = np.asarray(mask, dtype=bool) & np.isfinite(truth)
    if truth.ndim != 2 or truth.shape != mean.shape or truth.shape != std.shape or truth.shape != mask.shape:
        raise ValueError("Metric array shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Finite predictive mean and strictly positive learned scale required")
    results = []
    for j in range(truth.shape[1]):
        observed = mask[:, j]
        y, mu, sigma = truth[observed, j], mean[observed, j], std[observed, j]
        if not len(y):
            results.append({"observed_targets": 0})
            continue
        error = np.abs(y - mu)
        z = (y - mu) / sigma
        cdf = np.asarray([NormalDist().cdf(float(v)) for v in z])
        pdf = np.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        valid_pair = mask[1:, j] & mask[:-1, j]
        dy, dm = np.diff(truth[:, j])[valid_pair], np.diff(mean[:, j])[valid_pair]
        b_valid = bool(np.asarray(baseline_mask)[j]) and np.isfinite(np.asarray(baseline)[j])
        row = {"observed_targets": len(y), "MAE": float(error.mean()), "RMSE": float(np.sqrt(np.mean(error**2))),
               "NLL": float(np.mean(.5 * math.log(2 * math.pi) + np.log(sigma) + .5 * z**2)),
               "CRPS": float(np.mean(sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)))),
               "DynamicMAE": float(np.mean(np.abs(dy - dm))) if len(dy) else None,
               "dynamic_observed_pairs": len(dy),
               "variance_ratio": float(np.var(mu) / np.var(y)) if len(y) > 1 and np.var(y) > 0 else None,
               "trajectory_correlation": correlation(y, mu),
               "change_from_baseline_error": float(np.mean(np.abs((mu - baseline[j]) - (y - baseline[j])))) if b_valid else None,
               "uncertainty_error_correlation": correlation(sigma, error)}
        for level in (.90, .95):
            q = NormalDist().inv_cdf((1 + level) / 2)
            row[f"coverage_{int(level*100)}"] = float(np.mean(error <= q * sigma))
            row[f"interval_width_{int(level*100)}"] = float(np.mean(2 * q * sigma))
        order = np.argsort(sigma, kind="stable")
        row["risk_coverage"] = [{"coverage": k / len(y), "MAE": float(error[order[:k]].mean())} for k in sorted({max(1, math.ceil(c * len(y))) for c in (.1, .25, .5, .75, 1)})]
        results.append(row)
    return results


def metric_change(internal, external):
    if internal is None or external is None:
        return {"absolute_change": None, "relative_change": None}
    return {"absolute_change": external - internal,
            "relative_change": (external - internal) / abs(internal) if internal != 0 else None}
