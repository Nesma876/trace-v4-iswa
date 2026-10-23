"""Leakage-safe preprocessing fit/apply helpers."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingOnlyStats:
    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]
    iqrs: dict[str, float]
    fit_split: str = "mimic_train"


def fit_training_only_stats(rows: list[dict[str, float]], variables: list[str], split_name: str) -> TrainingOnlyStats:
    if split_name != "mimic_train":
        raise ValueError("normalization/imputation statistics must be fitted on MIMIC train only")
    means, stds, medians, iqrs = {}, {}, {}, {}
    for variable in variables:
        values = [float(row[variable]) for row in rows if row.get(variable) is not None]
        if not values:
            means[variable] = 0.0
            stds[variable] = 1.0
            medians[variable] = 0.0
            iqrs[variable] = 1.0
            continue
        ordered = sorted(values)
        means[variable] = sum(values) / len(values)
        stds[variable] = statistics.pstdev(values) or 1.0
        medians[variable] = statistics.median(values)
        q1 = ordered[int(0.25 * (len(ordered) - 1))]
        q3 = ordered[int(0.75 * (len(ordered) - 1))]
        iqrs[variable] = (q3 - q1) or 1.0
    return TrainingOnlyStats(means, stds, medians, iqrs)

