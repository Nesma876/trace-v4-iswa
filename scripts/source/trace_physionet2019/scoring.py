"""Frozen macro-variable primary and micro-observation secondary metrics."""
import json
from pathlib import Path

import numpy as np

from .schema import validate_features

SCORING = Path(__file__).resolve().parents[3] / "configs/physionet2019/FROZEN_SCORING_PROTOCOL.yaml"


def scoring_protocol():
    return json.loads(SCORING.read_text())


def common_eligible_variables(counts_by_horizon, features, minimum=100):
    if set(counts_by_horizon) != {4, 12, 24, 48}:
        raise ValueError("All four hourly horizons are required")
    return [v for j, v in enumerate(features) if all(np.asarray(counts_by_horizon[h])[j] >= minimum for h in (4, 12, 24, 48))]


def score_forecasts(truth, prediction, target_mask, features, stats, *, model, horizon, cohort, macro_variables=None):
    """Score native-scale endpoint arrays [target instance, variable]. No fitting.

    A target instance is a patient-origin endpoint. Within-variable errors pool its
    observed instances; between-variable primary aggregation gives equal weights.
    Native-unit per-variable diagnostics accompany standardized primary metrics.
    """
    protocol = scoring_protocol()
    validate_features(features)
    if horizon not in protocol["horizons"] or cohort not in protocol["cohorts"]:
        raise ValueError("Unknown horizon/cohort")
    if stats["fit_source"] != "A" or stats["fit_split"] != "train" or list(features) != stats["features"]:
        raise ValueError("Scoring requires the matching A-training scaler")
    y, pred, mask = np.asarray(truth), np.asarray(prediction), np.asarray(target_mask, dtype=bool)
    if y.ndim != 2 or y.shape != pred.shape or y.shape != mask.shape or y.shape[1] != len(features):
        raise ValueError("Forecast arrays must share [instance, variable] dimensions")
    observed = mask & np.isfinite(y)
    if not np.isfinite(pred[observed]).all():
        raise ValueError("Nonfinite predictions on observed targets; cannot drop model failures")
    scale = np.asarray(stats["stds"])
    if scale.shape != (len(features),) or not np.isfinite(scale).all() or (scale <= 1e-8).any():
        raise ValueError("Invalid frozen training scale")
    rows, eligible, sum_abs, sum_sq, total = [], [], 0., 0., 0
    minimum = protocol["min_observed_target_instances"]
    for j, variable in enumerate(features):
        residual = (pred[observed[:, j], j] - y[observed[:, j], j]) / scale[j]
        n = len(residual)
        enough = n >= minimum
        row = {"model": model, "horizon": horizon, "cohort": cohort, "variable": variable,
               "N_observed_targets": n, "status": "ELIGIBLE" if enough else "INSUFFICIENT_TARGET_OBSERVATIONS",
               "MAE": float(np.mean(np.abs(residual))) if enough else None,
               "RMSE": float(np.sqrt(np.mean(residual**2))) if enough else None,
               "native_MAE": float(np.mean(np.abs(residual)) * scale[j]) if enough else None,
               "native_RMSE": float(np.sqrt(np.mean(residual**2)) * scale[j]) if enough else None,
               "metric_scale": "A_training_std"}
        rows.append(row)
        if enough:
            eligible.append(variable)
        sum_abs += float(np.abs(residual).sum())
        sum_sq += float((residual**2).sum())
        total += n
    selected = eligible if macro_variables is None else list(macro_variables)
    if len(selected) != len(set(selected)) or set(selected) - set(eligible):
        raise ValueError("Common macro variables must meet target-only eligibility")
    used = [r for r in rows if r["variable"] in selected]
    aggregate = {"model": model, "horizon": horizon, "cohort": cohort,
                 "MacroMAE": float(np.mean([r["MAE"] for r in used])) if used else None,
                 "MacroRMSE": float(np.mean([r["RMSE"] for r in used])) if used else None,
                 "MicroMAE": sum_abs / total if total else None,
                 "MicroRMSE": float(np.sqrt(sum_sq / total)) if total else None,
                 "N_observed_targets": total, "eligible_variables": selected,
                 "insufficient_variables": [v for v in features if v not in eligible],
                 "status": "ELIGIBLE" if used else "INSUFFICIENT_TARGET_OBSERVATIONS",
                 "metric_scale": "A_training_std"}
    return {"per_variable": rows, "aggregate": aggregate}


def masked_training_nll(mu, logvar, target, target_mask):
    """Gaussian objective interface only: indexes true observations before arithmetic.

    Unlike NaN*0 masking, arbitrary missing target placeholders have no loss or gradient.
    This function does not create an optimizer or fit a model.
    """
    import math
    import torch
    if mu.shape != logvar.shape or mu.shape != target.shape or mu.shape != target_mask.shape:
        raise ValueError("Loss tensor shape mismatch")
    observed = target_mask.bool() & torch.isfinite(target)
    if not observed.any():
        return mu[observed].sum()
    m, lv, y = mu[observed], logvar[observed].clamp(-6, 3), target[observed]
    if not torch.isfinite(m).all() or not torch.isfinite(lv).all():
        raise ValueError("Nonfinite predictions at observed training targets")
    return (.5 * (math.log(2 * math.pi) + lv + (y - m).square() / lv.exp())).mean()
