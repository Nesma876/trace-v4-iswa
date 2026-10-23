"""Comparative TRACE V1 evaluation helpers."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES
from trace.models.comparative import action_from_row
from trace.simulators.synthetic_physiology import SyntheticPhysiologySimulator


def temporal_metrics(model: Any, sequences: dict[str, list[dict[str, Any]]], horizons: list[int], bin_hours: int, seed: int) -> list[dict[str, Any]]:
    """Evaluate one-step and recursive rollout metrics."""
    output = []
    one_step_mae = None
    for horizon in horizons:
        abs_errors: list[float] = []
        sq_errors: list[float] = []
        by_variable = {name: [] for name in STATE_VARIABLES}
        for rows in sequences.values():
            for start in range(1, len(rows) - horizon):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred = model.rollout(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                for variable in STATE_VARIABLES:
                    if target[variable] is None:
                        continue
                    err = pred[variable] - float(target[variable])
                    abs_errors.append(abs(err))
                    sq_errors.append(err * err)
                    by_variable[variable].append(abs(err))
        mae = _mean(abs_errors)
        rmse = math.sqrt(_mean(sq_errors))
        if horizon == 1:
            one_step_mae = mae
        row = {
            "model": model.name,
            "seed": seed,
            "horizon_steps": horizon,
            "horizon_hours": horizon * bin_hours,
            "mae": mae,
            "rmse": rmse,
            "temporal_retention": (one_step_mae / mae) if one_step_mae and mae else 0.0,
            "rollout_degradation_slope": 0.0,
            "cumulative_trajectory_error": mae * horizon,
        }
        for variable, values in by_variable.items():
            row[f"{variable}_mae"] = _mean(values)
        output.append(row)
    if len(output) > 1:
        first = output[0]
        last = output[-1]
        slope = (last["mae"] - first["mae"]) / max(1, last["horizon_steps"] - first["horizon_steps"])
        for row in output:
            row["rollout_degradation_slope"] = slope
    return output


def reliability_metrics(model: Any, sequences: dict[str, list[dict[str, Any]]], horizons: list[int], bin_hours: int, seed: int) -> list[dict[str, Any]]:
    """Evaluate Gaussian interval coverage and uncertainty-error association."""
    rows_out = []
    for horizon in horizons:
        records = []
        for rows in sequences.values():
            for start in range(1, len(rows) - horizon):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred, std = model.rollout_with_uncertainty(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                for variable in STATE_VARIABLES:
                    if target[variable] is None:
                        continue
                    y = float(target[variable])
                    sigma = max(std[variable], 1e-3)
                    error = abs(pred[variable] - y)
                    records.append((error, sigma, _gaussian_nll(y, pred[variable], sigma)))
        coverage_90 = _coverage(records, 1.6448536269514722)
        coverage_95 = _coverage(records, 1.959963984540054)
        rows_out.append(
            {
                "model": model.name,
                "seed": seed,
                "horizon_steps": horizon,
                "horizon_hours": horizon * bin_hours,
                "coverage_90": coverage_90,
                "coverage_95": coverage_95,
                "avg_interval_width_90": _mean([2 * 1.6448536269514722 * sigma for _, sigma, _ in records]),
                "avg_interval_width_95": _mean([2 * 1.959963984540054 * sigma for _, sigma, _ in records]),
                "gaussian_nll": _mean([nll for _, _, nll in records]),
                "uncertainty_error_corr": _corr([sigma for _, sigma, _ in records], [error for error, _, _ in records]),
            }
        )
    return rows_out


def selective_prediction(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int, horizon: int = 1) -> list[dict[str, Any]]:
    """Risk-coverage curve based on predictive uncertainty."""
    records = []
    for rows in sequences.values():
        for start in range(1, len(rows) - horizon):
            actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
            pred, std = model.rollout_with_uncertainty(rows[:start], horizon, actions)[-1]
            target = rows[start + horizon - 1]
            err = []
            unc = []
            for variable in STATE_VARIABLES:
                if target[variable] is not None:
                    err.append(abs(pred[variable] - float(target[variable])))
                    unc.append(std[variable])
            if err:
                records.append((_mean(err), _mean(unc)))
    records.sort(key=lambda pair: pair[1])
    output = []
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.5]:
        keep = max(1, int(len(records) * coverage))
        output.append({"model": model.name, "seed": seed, "retained_fraction": coverage, "mae": _mean([error for error, _ in records[:keep]])})
    return output


def counterfactual_metrics(model: Any, seed: int) -> list[dict[str, Any]]:
    """Evaluate simulator-grounded intervention branches."""
    initial = {
        "heart_rate": 96.0,
        "respiratory_rate": 22.0,
        "spo2": 94.0,
        "temperature": 37.8,
        "mean_arterial_pressure": 64.0,
        "lactate": 3.2,
        "creatinine": 1.35,
    }
    intensities = [
        ("none", 0.0, 0.0),
        ("low", 0.5, 0.15),
        ("moderate", 1.0, 0.4),
        ("high", 1.5, 0.8),
    ]
    simulator = SyntheticPhysiologySimulator()
    simulator.reset(initial)
    action_sequences = [[{"iv_fluids": fluid, "vasopressors": pressor} for _ in range(12)] for _, fluid, pressor in intensities]
    true_branches = simulator.branch(action_sequences)
    pred_branches = [model.rollout([_initial_row(initial)], 12, actions) for actions in action_sequences]
    baseline_error = _trajectory_mae(pred_branches[0], true_branches[0])
    output = []
    baseline_true = true_branches[0][-1]
    baseline_pred = pred_branches[0][-1]
    for (label, _, _), true, pred in zip(intensities, true_branches, pred_branches):
        error = _trajectory_mae(pred, true)
        te_errors = []
        sign_hits = []
        effect_corr_true = []
        effect_corr_pred = []
        for variable in ["mean_arterial_pressure", "lactate", "heart_rate"]:
            true_delta = true[-1][variable] - baseline_true[variable]
            pred_delta = pred[-1][variable] - baseline_pred[variable]
            te_errors.append(abs(true_delta - pred_delta))
            if abs(true_delta) > 1e-6:
                sign_hits.append(1.0 if (true_delta > 0) == (pred_delta > 0) else 0.0)
            effect_corr_true.append(true_delta)
            effect_corr_pred.append(pred_delta)
        output.append(
            {
                "model": model.name,
                "seed": seed,
                "branch": label,
                "trajectory_mae": error,
                "counterfactual_penalty": error / (baseline_error + 1e-6),
                "treatment_effect_error": _mean(te_errors),
                "intervention_sign_accuracy": _mean(sign_hits),
                "trajectory_effect_correlation": _corr(effect_corr_true, effect_corr_pred),
            }
        )
    map_true = [branch[-1]["mean_arterial_pressure"] for branch in true_branches]
    map_pred = [branch[-1]["mean_arterial_pressure"] for branch in pred_branches]
    ranking_accuracy = _pairwise_order_accuracy(map_true, map_pred)
    for row in output:
        row["intervention_ranking_accuracy"] = ranking_accuracy
    return output


def clinical_stress_metrics(model: Any, seed: int, corrupted_control: bool = False) -> list[dict[str, Any]]:
    """Evaluate discriminative stress scenarios and optional positive controls."""
    base = {
        "heart_rate": 88.0,
        "respiratory_rate": 20.0,
        "spo2": 96.0,
        "temperature": 37.2,
        "mean_arterial_pressure": 78.0,
        "lactate": 1.6,
        "creatinine": 1.0,
    }
    scenarios = [
        ("hypotension_mild", "mild", {"mean_arterial_pressure": 68.0}),
        ("hypotension_moderate", "moderate", {"mean_arterial_pressure": 58.0}),
        ("hypotension_severe", "severe", {"mean_arterial_pressure": 48.0}),
        ("elevated_lactate", "moderate", {"lactate": 4.5}),
        ("renal_deterioration", "moderate", {"creatinine": 2.2}),
        ("hypoxemia", "moderate", {"spo2": 84.0}),
        ("nuisance_negative_control", "control", {"temperature": 37.21}),
    ]
    output = []
    for scenario, severity, changes in scenarios:
        state = dict(base)
        state.update(changes)
        actions = [[{"iv_fluids": 0.0, "vasopressors": dose} for _ in range(6)] for dose in [0.0, 0.15, 0.4, 0.8]]
        branches = [model.rollout([_initial_row(state)], 6, seq) for seq in actions]
        if corrupted_control:
            for idx, branch in enumerate(branches):
                for row in branch:
                    row["mean_arterial_pressure"] -= idx * 8.0
                    row["spo2"] = 104.0 if scenario == "hypoxemia" else row["spo2"]
        checks, violations, contradictions = _clinical_checks(scenario, branches)
        output.append(
            {
                "model": f"{model.name}_corrupted_control" if corrupted_control else model.name,
                "seed": seed,
                "scenario": scenario,
                "severity": severity,
                "horizon_steps": 6,
                "total_checks": checks,
                "total_violations": violations,
                "physiological_violation_rate": violations / checks if checks else 0.0,
                "clinical_contradiction_rate": contradictions / checks if checks else 0.0,
            }
        )
    return output


def action_reliance_metrics(model: Any, seed: int) -> list[dict[str, Any]]:
    """Compare normal, zeroed, and shuffled actions for intervention branches."""
    normal = counterfactual_metrics(model, seed)
    initial = {
        "heart_rate": 96.0,
        "respiratory_rate": 22.0,
        "spo2": 94.0,
        "temperature": 37.8,
        "mean_arterial_pressure": 64.0,
        "lactate": 3.2,
        "creatinine": 1.35,
    }
    altered = []
    for mode, action in [("zeroed", {"iv_fluids": 0.0, "vasopressors": 0.0}), ("shuffled", {"iv_fluids": 0.2, "vasopressors": 1.0})]:
        pred = model.rollout([_initial_row(initial)], 12, [action for _ in range(12)])
        sim = SyntheticPhysiologySimulator()
        sim.reset(initial)
        true = sim.rollout([{"iv_fluids": 1.0, "vasopressors": 0.4} for _ in range(12)], 12)
        altered.append({"model": model.name, "seed": seed, "action_mode": mode, "intervention_mae": _trajectory_mae(pred, true)})
    baseline = next(row for row in normal if row["branch"] == "moderate")
    altered.append({"model": model.name, "seed": seed, "action_mode": "normal", "intervention_mae": baseline["trajectory_mae"]})
    normal_mae = baseline["trajectory_mae"]
    for row in altered:
        row["delta_mae_action"] = row["intervention_mae"] - normal_mae
    return altered


def _clinical_checks(scenario: str, branches: list[list[dict[str, float]]]) -> tuple[int, int, int]:
    checks = 0
    violations = 0
    contradictions = 0
    for branch in branches:
        for row in branch:
            for variable, lower, upper in [
                ("mean_arterial_pressure", 25.0, 140.0),
                ("spo2", 60.0, 100.0),
                ("temperature", 32.0, 42.5),
                ("lactate", 0.0, 15.0),
                ("creatinine", 0.1, 8.0),
            ]:
                checks += 1
                if row[variable] < lower or row[variable] > upper:
                    violations += 1
    final_maps = [branch[-1]["mean_arterial_pressure"] for branch in branches]
    checks += 3
    for a, b in zip(final_maps, final_maps[1:]):
        if b + 1e-6 < a:
            violations += 1
            contradictions += 1
    if scenario == "hypoxemia":
        checks += 1
        if branches[0][-1]["spo2"] > 100.0:
            violations += 1
    return checks, violations, contradictions


def _trajectory_mae(predicted: list[dict[str, float]], target: list[dict[str, float]]) -> float:
    errors = []
    for pred_row, target_row in zip(predicted, target):
        for variable in STATE_VARIABLES:
            errors.append(abs(pred_row[variable] - target_row[variable]))
    return _mean(errors)


def _initial_row(state: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"patient_id": "stress", "time_index": 0, "iv_fluids": 0.0, "vasopressors": 0.0}
    for variable, value in state.items():
        row[variable] = value
        row[f"{variable}_mask"] = 1
    return row


def _coverage(records: list[tuple[float, float, float]], z_value: float) -> float:
    if not records:
        return 0.0
    return sum(1 for error, sigma, _ in records if error <= z_value * sigma) / len(records)


def _gaussian_nll(y: float, mean: float, sigma: float) -> float:
    variance = sigma * sigma
    return 0.5 * (math.log(2 * math.pi * variance) + ((y - mean) ** 2) / variance)


def _pairwise_order_accuracy(true_values: list[float], pred_values: list[float]) -> float:
    hits = []
    for i, j in combinations(range(len(true_values)), 2):
        true_order = true_values[i] < true_values[j]
        pred_order = pred_values[i] < pred_values[j]
        hits.append(1.0 if true_order == pred_order else 0.0)
    return _mean(hits)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0

