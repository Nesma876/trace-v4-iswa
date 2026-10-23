"""Corrected bootstrap/statistical helpers for audit V2."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, seed: int = 123) -> tuple[float, float]:
    """Bootstrap CI for a mean by resampling the provided observations."""
    if not values:
        return 0.0, 0.0
    if len(set(values)) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    low_idx = int(0.025 * len(estimates))
    high_idx = int(0.975 * len(estimates)) - 1
    return estimates[low_idx], estimates[max(low_idx, high_idx)]


def summarize_metric(rows: list[dict[str, Any]], group_keys: list[str], metric: str) -> list[dict[str, Any]]:
    """Summarize raw rows without bootstrapping an already aggregated scalar."""
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value in ("", None, "not_applicable"):
            continue
        grouped[tuple(row[key] for key in group_keys)].append(float(value))
    output = []
    for key, values in grouped.items():
        low, high = bootstrap_ci(values)
        record = {name: value for name, value in zip(group_keys, key)}
        record.update(
            {
                f"{metric}_mean": sum(values) / len(values),
                f"{metric}_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
                "n_observations": len(values),
            }
        )
        output.append(record)
    return output


def paired_difference_ci(rows: list[dict[str, Any]], model_a: str, model_b: str, metric: str, key_fields: list[str]) -> dict[str, Any]:
    """Paired bootstrap CI for model A minus model B over shared keys."""
    by_model: dict[str, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        if value in ("", None, "not_applicable"):
            continue
        by_model[row["model"]][tuple(row[field] for field in key_fields)] = float(value)
    common = sorted(set(by_model[model_a]) & set(by_model[model_b]))
    diffs = [by_model[model_a][key] - by_model[model_b][key] for key in common]
    low, high = bootstrap_ci(diffs, seed=321)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "mean_difference_a_minus_b": sum(diffs) / len(diffs) if diffs else 0.0,
        "ci_low": low,
        "ci_high": high,
        "n_pairs": len(diffs),
    }

