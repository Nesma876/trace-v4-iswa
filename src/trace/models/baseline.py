"""Persistence and simple baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace.data.synthetic import STATE_VARIABLES


class PersistenceModel:
    """Last-observation-carried-forward model for state rollouts."""

    name = "persistence"

    def fit(self, sequences: dict[str, list[dict[str, Any]]]) -> "PersistenceModel":
        self.global_means = {}
        for variable in STATE_VARIABLES:
            values = [row[variable] for rows in sequences.values() for row in rows if row[variable] is not None]
            self.global_means[variable] = sum(values) / len(values)
        return self

    def predict_next(self, history: list[dict[str, Any]]) -> dict[str, float]:
        pred = {}
        for variable in STATE_VARIABLES:
            value = None
            for row in reversed(history):
                if row[variable] is not None:
                    value = row[variable]
                    break
            pred[variable] = float(value if value is not None else self.global_means[variable])
        return pred

    def rollout(self, history: list[dict[str, Any]], horizon_steps: int) -> list[dict[str, float]]:
        current = self.predict_next(history)
        return [dict(current) for _ in range(horizon_steps)]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.json").write_text(json.dumps({"name": self.name, "global_means": self.global_means}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PersistenceModel":
        payload = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
        model = cls()
        model.global_means = payload["global_means"]
        return model

