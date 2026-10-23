"""Lightweight comparative model families for synthetic TRACE V1.

These implementations intentionally avoid heavyweight dependencies so the
benchmark can run in a restricted dry-run environment. They are not substitutes
for the later PyTorch GRU/Transformer/RSSM implementations; they are compact,
action-conditioned experimental probes with the same rollout interface.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES


def latest_state(history: list[dict[str, Any]], means: dict[str, float]) -> dict[str, float]:
    """Return the latest observed state with train means as fallback."""
    state = {}
    for variable in STATE_VARIABLES:
        value = None
        for row in reversed(history):
            if row.get(variable) is not None:
                value = row[variable]
                break
        state[variable] = float(value if value is not None else means[variable])
    return state


def action_from_row(row: dict[str, Any] | None) -> dict[str, float]:
    """Extract action variables from a row."""
    if row is None:
        return {name: 0.0 for name in ACTION_VARIABLES}
    return {name: float(row.get(name, 0.0)) for name in ACTION_VARIABLES}


class FeatureAutoregressiveModel:
    """Per-variable ridge autoregression over causal history features."""

    def __init__(self, name: str, feature_fn: Callable[[list[dict[str, Any]], dict[str, float], dict[str, float]], list[float]]) -> None:
        self.name = name
        self.feature_fn = feature_fn
        self.global_means: dict[str, float] = {}
        self.coefficients: dict[str, list[float]] = {}
        self.residual_variance: dict[str, float] = {}
        self.parameter_count = 0
        self.training_time_seconds = 0.0
        self.best_validation_epoch = 1
        self.final_validation_loss = 0.0

    def fit(self, train: dict[str, list[dict[str, Any]]], validation: dict[str, list[dict[str, Any]]] | None = None) -> "FeatureAutoregressiveModel":
        start = time.perf_counter()
        self.global_means = _global_means(train)
        features: list[list[float]] = []
        targets: dict[str, list[float]] = {name: [] for name in STATE_VARIABLES}
        for rows in train.values():
            for idx in range(len(rows) - 1):
                x = self.feature_fn(rows[: idx + 1], action_from_row(rows[idx]), self.global_means)
                target = rows[idx + 1]
                if all(target[name] is not None for name in STATE_VARIABLES):
                    features.append(x)
                    for variable in STATE_VARIABLES:
                        targets[variable].append(float(target[variable]))
        if not features:
            raise ValueError("No complete training transitions available.")
        for variable in STATE_VARIABLES:
            self.coefficients[variable] = _ridge_fit(features, targets[variable], ridge=0.05)
            residuals = []
            for x, y in zip(features, targets[variable]):
                residuals.append(y - _dot(self.coefficients[variable], x))
            variance = sum(value * value for value in residuals) / max(1, len(residuals) - len(features[0]))
            self.residual_variance[variable] = max(variance, 1e-4)
        self.parameter_count = sum(len(weights) for weights in self.coefficients.values())
        self.training_time_seconds = time.perf_counter() - start
        if validation:
            self.final_validation_loss = self._validation_mae(validation)
        return self

    def predict_next(self, history: list[dict[str, Any]], action: dict[str, float] | None = None) -> dict[str, float]:
        used_action = action if action is not None else action_from_row(history[-1] if history else None)
        x = self.feature_fn(history, used_action, self.global_means)
        return {variable: _dot(weights, x) for variable, weights in self.coefficients.items()}

    def predict_next_with_std(self, history: list[dict[str, Any]], action: dict[str, float] | None = None, horizon_step: int = 1) -> tuple[dict[str, float], dict[str, float]]:
        pred = self.predict_next(history, action)
        scale = math.sqrt(max(1, horizon_step))
        std = {variable: math.sqrt(self.residual_variance[variable]) * scale for variable in STATE_VARIABLES}
        return pred, std

    def rollout(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[dict[str, float]]:
        rows = [dict(row) for row in history]
        outputs = []
        for step in range(horizon_steps):
            action = action_sequence[step] if action_sequence else action_from_row(rows[-1] if rows else None)
            pred = self.predict_next(rows, action)
            row = {"patient_id": rows[-1].get("patient_id", "synthetic"), "time_index": rows[-1].get("time_index", 0) + 1 if rows else step}
            row.update(action)
            for variable in STATE_VARIABLES:
                row[variable] = pred[variable]
                row[f"{variable}_mask"] = 1
            rows.append(row)
            outputs.append(pred)
        return outputs

    def rollout_with_uncertainty(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[tuple[dict[str, float], dict[str, float]]]:
        rows = [dict(row) for row in history]
        outputs = []
        for step in range(horizon_steps):
            action = action_sequence[step] if action_sequence else action_from_row(rows[-1] if rows else None)
            pred, std = self.predict_next_with_std(rows, action, step + 1)
            row = {"patient_id": rows[-1].get("patient_id", "synthetic"), "time_index": rows[-1].get("time_index", 0) + 1 if rows else step}
            row.update(action)
            for variable in STATE_VARIABLES:
                row[variable] = pred[variable]
                row[f"{variable}_mask"] = 1
            rows.append(row)
            outputs.append((pred, std))
        return outputs

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "global_means": self.global_means,
            "coefficients": self.coefficients,
            "residual_variance": self.residual_variance,
            "parameter_count": self.parameter_count,
            "training_time_seconds": self.training_time_seconds,
            "best_validation_epoch": self.best_validation_epoch,
            "final_validation_loss": self.final_validation_loss,
        }
        (output / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _validation_mae(self, validation: dict[str, list[dict[str, Any]]]) -> float:
        errors = []
        for rows in validation.values():
            for idx in range(1, len(rows) - 1):
                pred = self.predict_next(rows[: idx + 1], action_from_row(rows[idx]))
                target = rows[idx + 1]
                for variable in STATE_VARIABLES:
                    if target[variable] is not None:
                        errors.append(abs(pred[variable] - float(target[variable])))
        return sum(errors) / len(errors) if errors else 0.0


class PersistenceProbe:
    """Deterministic persistence baseline with the comparative interface."""

    name = "persistence"

    def fit(self, train: dict[str, list[dict[str, Any]]], validation: dict[str, list[dict[str, Any]]] | None = None) -> "PersistenceProbe":
        start = time.perf_counter()
        self.global_means = _global_means(train)
        self.residual_variance = {name: 1.0 for name in STATE_VARIABLES}
        residuals = {name: [] for name in STATE_VARIABLES}
        for rows in train.values():
            for idx in range(1, len(rows) - 1):
                pred = latest_state(rows[: idx + 1], self.global_means)
                target = rows[idx + 1]
                for variable in STATE_VARIABLES:
                    if target[variable] is not None:
                        residuals[variable].append(float(target[variable]) - pred[variable])
        for variable, values in residuals.items():
            self.residual_variance[variable] = max(sum(v * v for v in values) / max(1, len(values)), 1e-4)
        self.parameter_count = len(self.global_means)
        self.training_time_seconds = time.perf_counter() - start
        self.best_validation_epoch = 0
        self.final_validation_loss = 0.0
        return self

    def predict_next(self, history: list[dict[str, Any]], action: dict[str, float] | None = None) -> dict[str, float]:
        return latest_state(history, self.global_means)

    def predict_next_with_std(self, history: list[dict[str, Any]], action: dict[str, float] | None = None, horizon_step: int = 1) -> tuple[dict[str, float], dict[str, float]]:
        pred = self.predict_next(history, action)
        return pred, {variable: math.sqrt(self.residual_variance[variable]) * math.sqrt(max(1, horizon_step)) for variable in STATE_VARIABLES}

    def rollout(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[dict[str, float]]:
        pred = self.predict_next(history)
        return [dict(pred) for _ in range(horizon_steps)]

    def rollout_with_uncertainty(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[tuple[dict[str, float], dict[str, float]]]:
        return [self.predict_next_with_std(history, horizon_step=step + 1) for step in range(horizon_steps)]


def build_model(name: str) -> PersistenceProbe | FeatureAutoregressiveModel:
    """Construct one of the four V1 model-family probes."""
    if name == "persistence":
        return PersistenceProbe()
    if name == "linear_ar":
        return FeatureAutoregressiveModel(name, linear_features)
    if name == "gru":
        return FeatureAutoregressiveModel(name, gru_like_features)
    if name == "transformer":
        return FeatureAutoregressiveModel(name, transformer_like_features)
    if name == "rssm":
        return FeatureAutoregressiveModel(name, rssm_like_features)
    raise ValueError(f"unknown model: {name}")


def linear_features(history: list[dict[str, Any]], action: dict[str, float], means: dict[str, float]) -> list[float]:
    state = latest_state(history, means)
    return [1.0, *[state[name] / _scale(name) for name in STATE_VARIABLES], *[action[name] for name in ACTION_VARIABLES]]


def gru_like_features(history: list[dict[str, Any]], action: dict[str, float], means: dict[str, float]) -> list[float]:
    state = latest_state(history, means)
    ema = _ema_state(history, means, alpha=0.55)
    values = [1.0]
    values.extend(state[name] / _scale(name) for name in STATE_VARIABLES)
    values.extend(ema[name] / _scale(name) for name in STATE_VARIABLES)
    values.extend(action[name] for name in ACTION_VARIABLES)
    values.extend(math.tanh((state[name] - ema[name]) / _scale(name)) for name in STATE_VARIABLES)
    values.extend((state["mean_arterial_pressure"] / 100.0) * action[name] for name in ACTION_VARIABLES)
    return values


def transformer_like_features(history: list[dict[str, Any]], action: dict[str, float], means: dict[str, float]) -> list[float]:
    state = latest_state(history, means)
    prev = latest_state(history[:-1], means) if len(history) > 1 else state
    window = _window_mean(history, means, width=4)
    values = [1.0]
    values.extend(state[name] / _scale(name) for name in STATE_VARIABLES)
    values.extend(window[name] / _scale(name) for name in STATE_VARIABLES)
    values.extend((state[name] - prev[name]) / _scale(name) for name in STATE_VARIABLES)
    values.extend(action[name] for name in ACTION_VARIABLES)
    values.extend((idx + 1) / 16.0 for idx in range(3))
    return values


def rssm_like_features(history: list[dict[str, Any]], action: dict[str, float], means: dict[str, float]) -> list[float]:
    state = latest_state(history, means)
    ema = _ema_state(history, means, alpha=0.35)
    latent = _latent_severity(state)
    latent_ema = _latent_severity(ema)
    values = [1.0, latent, latent_ema, latent - latent_ema]
    values.extend(state[name] / _scale(name) for name in STATE_VARIABLES)
    values.extend(action[name] for name in ACTION_VARIABLES)
    values.extend(latent * action[name] for name in ACTION_VARIABLES)
    values.extend([math.sin(latent), math.cos(latent)])
    return values


def _global_means(sequences: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    means = {}
    for variable in STATE_VARIABLES:
        values = [float(row[variable]) for rows in sequences.values() for row in rows if row.get(variable) is not None]
        means[variable] = sum(values) / len(values)
    return means


def _ema_state(history: list[dict[str, Any]], means: dict[str, float], alpha: float) -> dict[str, float]:
    ema = dict(means)
    for row in history:
        for variable in STATE_VARIABLES:
            if row.get(variable) is not None:
                ema[variable] = alpha * float(row[variable]) + (1 - alpha) * ema[variable]
    return ema


def _window_mean(history: list[dict[str, Any]], means: dict[str, float], width: int) -> dict[str, float]:
    rows = history[-width:]
    output = {}
    for variable in STATE_VARIABLES:
        values = [float(row[variable]) for row in rows if row.get(variable) is not None]
        output[variable] = sum(values) / len(values) if values else means[variable]
    return output


def _latent_severity(state: dict[str, float]) -> float:
    return (
        0.45 * state["lactate"]
        + 0.35 * state["creatinine"]
        + 0.015 * state["heart_rate"]
        + 0.025 * state["respiratory_rate"]
        - 0.012 * state["mean_arterial_pressure"]
        - 0.01 * state["spo2"]
    )


def _scale(variable: str) -> float:
    return {
        "heart_rate": 100.0,
        "respiratory_rate": 25.0,
        "spo2": 100.0,
        "temperature": 40.0,
        "mean_arterial_pressure": 90.0,
        "lactate": 5.0,
        "creatinine": 2.0,
    }[variable]


def _ridge_fit(x_rows: list[list[float]], y: list[float], ridge: float) -> list[float]:
    n_features = len(x_rows[0])
    xtx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    xty = [0.0 for _ in range(n_features)]
    for x, target in zip(x_rows, y):
        for i in range(n_features):
            xty[i] += x[i] * target
            for j in range(n_features):
                xtx[i][j] += x[i] * x[j]
    for i in range(n_features):
        xtx[i][i] += ridge
    return _solve(xtx, xty)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [value] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        if abs(denom) < 1e-12:
            denom = 1e-12
        for idx in range(col, n + 1):
            aug[col][idx] /= denom
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for idx in range(col, n + 1):
                aug[row][idx] -= factor * aug[col][idx]
    return [aug[row][n] for row in range(n)]


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))
