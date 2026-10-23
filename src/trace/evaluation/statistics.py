"""Statistical analysis placeholders."""

from __future__ import annotations


def mean_confidence_interval(values: list[float]) -> dict[str, float]:
    """Return a simple normal-approximation interval for dry-run diagnostics."""
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * (variance ** 0.5) / (len(values) ** 0.5)
    return {"mean": mean, "lower": mean - half, "upper": mean + half}

