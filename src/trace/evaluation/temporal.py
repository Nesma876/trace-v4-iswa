"""Temporal TRACE evaluation."""

from __future__ import annotations

import math
from typing import Any

from trace.data.synthetic import STATE_VARIABLES


def evaluate_rollouts(model: Any, sequences: dict[str, list[dict[str, Any]]], horizons_steps: list[int]) -> list[dict[str, float]]:
    """Evaluate rollout MAE/RMSE by horizon in steps."""
    rows = []
    for horizon in horizons_steps:
        errors = []
        squared = []
        for sequence in sequences.values():
            for start in range(1, len(sequence) - horizon):
                preds = model.rollout(sequence[:start], horizon)
                target = sequence[start + horizon - 1]
                for variable in STATE_VARIABLES:
                    if target[variable] is None:
                        continue
                    err = preds[-1][variable] - target[variable]
                    errors.append(abs(err))
                    squared.append(err * err)
        mae = sum(errors) / len(errors) if errors else float("nan")
        rmse = math.sqrt(sum(squared) / len(squared)) if squared else float("nan")
        rows.append({"horizon_steps": horizon, "mae": mae, "rmse": rmse})
    if rows and rows[0]["mae"]:
        one = rows[0]["mae"]
        for row in rows:
            row["temporal_retention"] = one / row["mae"] if row["mae"] else float("nan")
    return rows

