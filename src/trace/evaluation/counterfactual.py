"""Counterfactual simulator-grounded metrics."""

from __future__ import annotations

from trace.data.synthetic import STATE_VARIABLES


def trajectory_mae(predicted: list[dict[str, float]], target: list[dict[str, float]]) -> float:
    """Mean absolute error across shared state variables and time."""
    errors = []
    for pred_row, target_row in zip(predicted, target):
        for variable in STATE_VARIABLES:
            errors.append(abs(pred_row[variable] - target_row[variable]))
    return sum(errors) / len(errors) if errors else 0.0

