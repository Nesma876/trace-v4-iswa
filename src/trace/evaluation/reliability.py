"""Reliability and calibration metrics."""

from __future__ import annotations


def deterministic_reliability_summary(mae: float) -> dict[str, float | str]:
    """Return deterministic placeholder reliability values for non-probabilistic baselines."""
    return {
        "model_type": "deterministic",
        "mae_reference": mae,
        "coverage_90": "not_available",
        "coverage_95": "not_available",
        "interval_width": "not_available",
    }

