"""Training-only statistics and prefix-invariant value/mask/elapsed representation."""
from __future__ import annotations

import numpy as np

from .data import read_psv
from .schema import DYNAMIC, HEADER, validate_features


def fit_preprocessing(paths, assignments, config):
    count = np.zeros(len(DYNAMIC), dtype=np.int64)
    patients = count.copy()
    mean = np.zeros(len(DYNAMIC))
    m2 = mean.copy()
    used = []
    for path in paths:
        if not any(part.lower() in ("training_seta", "seta") for part in path.parts):
            raise ValueError("Fit requires explicitly identified Set A source paths")
        if assignments.get(path.name) != "train":
            raise ValueError("Fit received a non-training patient")
        x = read_psv(path)[:, :len(DYNAMIC)]
        used.append(path.name)
        for j in range(len(DYNAMIC)):
            observed = x[:, j][np.isfinite(x[:, j])]
            n = len(observed)
            if not n:
                continue
            patients[j] += 1
            delta = observed.mean() - mean[j]
            total = count[j] + n
            m2[j] += ((observed - observed.mean()) ** 2).sum() + delta**2 * count[j] * n / total
            mean[j] += delta * n / total
            count[j] = total
    std = np.sqrt(m2 / np.maximum(count, 1))
    keep = (count >= config["minimum_observations"]) & (patients >= config["minimum_observed_patients"]) & (std > config["minimum_standard_deviation"]) & np.isfinite(std)
    features = [v for j, v in enumerate(DYNAMIC) if keep[j]]
    validate_features(features)
    if not used:
        raise ValueError("No training patients")
    return {"fit_source": "A", "fit_split": "train", "fit_patient_count": len(used),
            "features": features, "means": mean[keep].tolist(), "stds": std[keep].tolist(),
            "selection": [{"variable": v, "included": bool(keep[j]), "observations": int(count[j]),
                           "observed_patients": int(patients[j]),
                           "rationale": "physiology; availability and nondegenerate training scale" if keep[j] else "insufficient training observations/patients or degenerate scale"}
                          for j, v in enumerate(DYNAMIC)]}


def transform_prefix(raw, stats):
    validate_features(stats["features"])
    if stats["fit_source"] != "A" or stats["fit_split"] != "train":
        raise ValueError("Preprocessor must be fitted on Set A training")
    from .policy import model_values
    x = model_values(raw, stats)
    means, stds = np.asarray(stats["means"]), np.asarray(stats["stds"])
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Invalid frozen scaling")
    mask = np.isfinite(x)
    last = means.copy()
    last_time = np.full(len(means), -1)
    values, elapsed = np.empty_like(x), np.empty_like(x)
    for t in range(len(x)):
        last[mask[t]] = x[t, mask[t]]
        last_time[mask[t]] = t
        values[t] = (last - means) / stds
        # -1 explicitly distinguishes never-observed from stale observations.
        elapsed[t] = np.where(last_time >= 0, t - last_time, -1)
    return np.concatenate([values, mask.astype(float), elapsed], axis=-1).astype(np.float32)


def forecasting_window(raw, origin, horizon, stats, history_hours=4):
    if origin < history_hours - 1 or horizon < 1 or origin + horizon >= len(raw):
        raise ValueError("Insufficient causal context or future eligibility")
    # Causal imputation carries all information before t, then selects a finite context.
    inputs = transform_prefix(raw[:origin + 1], stats)[-history_hours:]
    from .policy import model_values
    target = model_values(raw[origin + 1:origin + horizon + 1], stats)
    mask = np.isfinite(target)
    targets = np.where(mask, (target - np.asarray(stats["means"])) / np.asarray(stats["stds"]), 0)
    return inputs, targets.astype(np.float32), mask


def eligible_origins(length, horizon, history_hours=4, common_48h=False):
    required = max(horizon, 48) if common_48h else horizon
    return range(history_hours - 1, max(history_hours - 1, length - required))


def mask_history(raw, fraction, seed):
    if fraction not in (0, .1, .2, .3, .5):
        raise ValueError("Mask fraction is not predefined")
    result = raw.copy()
    rng = np.random.default_rng(seed)
    dynamic = result[:, :len(DYNAMIC)]
    dynamic[rng.random(dynamic.shape) < fraction] = np.nan
    return result
