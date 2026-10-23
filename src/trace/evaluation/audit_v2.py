"""Audit V2 evaluation functions."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any

from trace.data.actions import action_levels, raw_no_intervention_action, transform_action
from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES
from trace.models.comparative import action_from_row
from trace.simulators.synthetic_physiology import SyntheticPhysiologySimulator

SIGN_EPSILON = 0.05


def trajectory_rows(model: Any, sequences: dict[str, list[dict[str, Any]]], horizons: list[int], bin_hours: int, seed: int) -> list[dict[str, Any]]:
    """Return trajectory-level temporal metrics for correct bootstrap units."""
    rows_out = []
    for patient_id, rows in sequences.items():
        for start in range(1, len(rows) - max(horizons)):
            for horizon in horizons:
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred = model.rollout(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                errors = []
                sq_errors = []
                for variable in STATE_VARIABLES:
                    if target[variable] is None:
                        continue
                    err = pred[variable] - float(target[variable])
                    errors.append(abs(err))
                    sq_errors.append(err * err)
                if errors:
                    rows_out.append(
                        {
                            "model": model.name,
                            "seed": seed,
                            "patient_id": patient_id,
                            "start_index": start,
                            "horizon_steps": horizon,
                            "horizon_hours": horizon * bin_hours,
                            "mae": _mean(errors),
                            "rmse": math.sqrt(_mean(sq_errors)),
                        }
                    )
    return rows_out


def uncertainty_rows(model: Any, sequences: dict[str, list[dict[str, Any]]], horizons: list[int], bin_hours: int, seed: int) -> list[dict[str, Any]]:
    """Return trajectory-level reliability metrics with nominal intervals."""
    out = []
    for patient_id, rows in sequences.items():
        for start in range(1, len(rows) - max(horizons)):
            for horizon in horizons:
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred, std = model.rollout_with_uncertainty(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                records = []
                for variable in STATE_VARIABLES:
                    if target[variable] is None:
                        continue
                    sigma = max(float(std[variable]), 1e-3)
                    error = abs(pred[variable] - float(target[variable]))
                    records.append((error, sigma, _nll(float(target[variable]), pred[variable], sigma)))
                if records:
                    out.append(
                        {
                            "model": model.name,
                            "seed": seed,
                            "patient_id": patient_id,
                            "start_index": start,
                            "horizon_steps": horizon,
                            "horizon_hours": horizon * bin_hours,
                            "coverage_90": _coverage(records, 1.6448536269514722),
                            "coverage_95": _coverage(records, 1.959963984540054),
                            "coverage_error_90": abs(_coverage(records, 1.6448536269514722) - 0.9),
                            "coverage_error_95": abs(_coverage(records, 1.959963984540054) - 0.95),
                            "avg_interval_width_95": _mean([2 * 1.959963984540054 * sigma for _, sigma, _ in records]),
                            "gaussian_nll": _mean([nll for _, _, nll in records]),
                            "uncertainty_error_corr": _corr([sigma for _, sigma, _ in records], [error for error, _, _ in records]),
                        }
                    )
    return out


def simulator_dose_response_rows() -> list[dict[str, Any]]:
    """Directly query the simulator for frozen intervention levels."""
    initial_states = [
        ("stable", _state(88, 20, 96, 37.2, 78, 1.6, 1.0)),
        ("hypotensive", _state(96, 22, 94, 37.8, 64, 3.2, 1.35)),
        ("severe", _state(110, 27, 90, 38.5, 52, 5.0, 2.0)),
    ]
    rows = []
    for state_name, initial in initial_states:
        for level, raw_action in action_levels():
            sim = SyntheticPhysiologySimulator()
            sim.reset(initial)
            trajectory = sim.rollout([raw_action for _ in range(12)], 12)
            for step, row in enumerate(trajectory, start=1):
                record = {"initial_state": state_name, "action_level": level, "step": step}
                record.update({variable: row[variable] for variable in STATE_VARIABLES})
                rows.append(record)
    return rows


def action_distribution_rows(sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    """Summarize action distributions and severity association."""
    flat = [row for rows in sequences.values() for row in rows]
    rows = []
    severity = [_severity(row) for row in flat]
    for action in ACTION_VARIABLES:
        values = [float(row[action]) for row in flat]
        rows.append(
            {
                "seed": seed,
                "action_variable": action,
                "mean": _mean(values),
                "median": _quantile(values, 0.5),
                "std": _std(values),
                "q10": _quantile(values, 0.1),
                "q25": _quantile(values, 0.25),
                "q75": _quantile(values, 0.75),
                "q90": _quantile(values, 0.9),
                "proportion_zero": sum(1 for value in values if abs(value) < 1e-12) / len(values),
                "low_or_above_frequency": sum(1 for value in values if value >= 0.15) / len(values),
                "temporal_persistence_corr": _lag_corr(sequences, action),
                "severity_correlation": _corr(values, severity),
            }
        )
    return rows


def action_reliance_rows_v2(model: Any, seed: int) -> list[dict[str, Any]]:
    """Compare controlled branches using semantically defined raw actions."""
    initial = _state(96, 22, 94, 37.8, 64, 3.2, 1.35)
    target_action = transform_action({"iv_fluids": 1.0, "vasopressors": 0.4})
    modes = [
        ("normal_moderate", target_action),
        ("true_no_intervention", transform_action(raw_no_intervention_action())),
        ("shuffled_actions", transform_action({"iv_fluids": 0.2, "vasopressors": 1.0})),
        ("low", transform_action({"iv_fluids": 0.5, "vasopressors": 0.15})),
        ("high", transform_action({"iv_fluids": 1.5, "vasopressors": 0.8})),
    ]
    sim = SyntheticPhysiologySimulator()
    sim.reset(initial)
    true_moderate = sim.rollout([target_action for _ in range(12)], 12)
    base_pred = model.rollout([_initial_row(initial)], 12, [transform_action(raw_no_intervention_action()) for _ in range(12)])
    rows = []
    for mode, action in modes:
        pred = model.rollout([_initial_row(initial)], 12, [action for _ in range(12)])
        delta = _trajectory_mae(pred, base_pred)
        rows.append(
            {
                "model": model.name,
                "seed": seed,
                "action_mode": mode,
                "target_branch": "moderate_simulator",
                "mae_to_moderate_true": _trajectory_mae(pred, true_moderate),
                "mean_abs_delta_vs_no_intervention_prediction": delta,
            }
        )
    return rows


def counterfactual_rows_v2(model: Any, seed: int) -> list[dict[str, Any]]:
    """Counterfactual metrics with not-applicable reference-branch fields."""
    initial = _state(96, 22, 94, 37.8, 64, 3.2, 1.35)
    action_items = action_levels()
    sim = SyntheticPhysiologySimulator()
    sim.reset(initial)
    true_branches = sim.branch([[transform_action(action) for _ in range(12)] for _, action in action_items])
    pred_branches = [model.rollout([_initial_row(initial)], 12, [transform_action(action) for _ in range(12)]) for _, action in action_items]
    baseline_error = _trajectory_mae(pred_branches[0], true_branches[0])
    baseline_true = true_branches[0][-1]
    baseline_pred = pred_branches[0][-1]
    rows = []
    variable_ranking_accuracy = _dose_response_order_accuracy(true_branches, pred_branches)
    for (level, _), true, pred in zip(action_items, true_branches, pred_branches):
        te_errors = []
        sign_hits = []
        true_effects = []
        pred_effects = []
        for variable in STATE_VARIABLES:
            true_delta = true[-1][variable] - baseline_true[variable]
            pred_delta = pred[-1][variable] - baseline_pred[variable]
            if abs(true_delta) > SIGN_EPSILON:
                te_errors.append(abs(true_delta - pred_delta))
                sign_hits.append(1.0 if (true_delta > 0) == (pred_delta > 0) else 0.0)
                true_effects.append(true_delta)
                pred_effects.append(pred_delta)
        is_reference = level == "none"
        rows.append(
            {
                "model": model.name,
                "seed": seed,
                "branch": level,
                "trajectory_mae": _trajectory_mae(pred, true),
                "counterfactual_penalty": _trajectory_mae(pred, true) / (baseline_error + 1e-6),
                "treatment_effect_error": "not_applicable" if is_reference else _mean(te_errors),
                "intervention_sign_accuracy": "not_applicable" if is_reference else _mean(sign_hits),
                "trajectory_effect_correlation": "not_applicable" if is_reference else _corr(true_effects, pred_effects),
                "dose_response_ordering_accuracy": "not_applicable" if is_reference else variable_ranking_accuracy,
                "sign_epsilon": SIGN_EPSILON,
            }
        )
    return rows


def clinical_stress_rows_v2(model: Any, seed: int, corruption: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scenario-specific TRACE-A audit split into A1/A2/A3."""
    scenarios = stress_scenarios()
    model_rows = []
    scenario_rows = []
    rule_rows = []
    for scenario in scenarios:
        branches = _scenario_branches(model, scenario)
        if corruption:
            _apply_corruption(branches, corruption)
        applicable = [rule for rule in stress_rules() if scenario["family"] in rule["relevant_scenarios"] or "all" in rule["relevant_scenarios"]]
        rule_counts = _evaluate_rules(branches, applicable)
        checks = sum(value["checks"] for value in rule_counts.values())
        violations = sum(value["violations"] for value in rule_counts.values())
        clinical_sensitivity = _mean_abs_change(branches[-1], branches[0])
        nuisance_reference = 1e-6 if scenario["family"] == "nuisance" else None
        name = f"{model.name}_{corruption}_control" if corruption else model.name
        scenario_rows.append(
            {
                "model": name,
                "seed": seed,
                "scenario": scenario["name"],
                "severity": scenario["severity"],
                "family": scenario["family"],
                "A1_physiological_plausibility_rate": violations / checks if checks else 0.0,
                "A2_perturbation_response_sensitivity": clinical_sensitivity,
                "A3_intervention_response_consistency": _vasopressor_monotonicity(branches),
                "specificity_like_sensitivity": clinical_sensitivity / (nuisance_reference or 1.0),
                "total_checks": checks,
                "total_violations": violations,
            }
        )
        for rule_id, counts in rule_counts.items():
            rule_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "scenario": scenario["name"],
                    "rule_id": rule_id,
                    "checks": counts["checks"],
                    "violations": counts["violations"],
                    "violation_rate": counts["violations"] / counts["checks"] if counts["checks"] else 0.0,
                }
            )
    grouped = defaultdict(lambda: {"checks": 0, "violations": 0})
    for row in scenario_rows:
        grouped[row["model"]]["checks"] += row["total_checks"]
        grouped[row["model"]]["violations"] += row["total_violations"]
    for model_name, counts in grouped.items():
        model_rows.append({"model": model_name, "seed": seed, "total_checks": counts["checks"], "total_violations": counts["violations"], "physiological_violation_rate": counts["violations"] / counts["checks"] if counts["checks"] else 0.0})
    return model_rows, scenario_rows, rule_rows


def stress_scenarios() -> list[dict[str, Any]]:
    """Frozen stress scenario definitions."""
    base = _state(88, 20, 96, 37.2, 78, 1.6, 1.0)
    return [
        {"name": "hypotension_mild", "family": "hypotension", "severity": "mild", "state": {**base, "mean_arterial_pressure": 68.0}},
        {"name": "hypotension_moderate", "family": "hypotension", "severity": "moderate", "state": {**base, "mean_arterial_pressure": 58.0}},
        {"name": "hypotension_severe", "family": "hypotension", "severity": "severe", "state": {**base, "mean_arterial_pressure": 48.0}},
        {"name": "elevated_lactate", "family": "lactate", "severity": "moderate", "state": {**base, "lactate": 4.5}},
        {"name": "renal_deterioration", "family": "renal", "severity": "moderate", "state": {**base, "creatinine": 2.2}},
        {"name": "hypoxemia", "family": "oxygenation", "severity": "moderate", "state": {**base, "spo2": 84.0}},
        {"name": "nuisance_negative_control", "family": "nuisance", "severity": "control", "state": {**base, "temperature": 37.21}},
    ]


def stress_rules() -> list[dict[str, Any]]:
    """Frozen scenario-specific audit rules."""
    return [
        {"id": "phys_bounds_map", "variable": "mean_arterial_pressure", "lower": 25.0, "upper": 140.0, "relevant_scenarios": ["all"]},
        {"id": "phys_bounds_spo2", "variable": "spo2", "lower": 60.0, "upper": 100.0, "relevant_scenarios": ["all", "oxygenation"]},
        {"id": "phys_bounds_lactate", "variable": "lactate", "lower": 0.0, "upper": 15.0, "relevant_scenarios": ["all", "lactate"]},
        {"id": "phys_bounds_creatinine", "variable": "creatinine", "lower": 0.1, "upper": 8.0, "relevant_scenarios": ["all", "renal"]},
        {"id": "vasopressor_map_monotonic", "variable": "mean_arterial_pressure", "kind": "monotonic_increasing_by_branch", "relevant_scenarios": ["hypotension"]},
        {"id": "nuisance_low_response", "variable": "mean_abs_state_delta", "upper": 4.0, "kind": "nuisance_sensitivity_bound", "relevant_scenarios": ["nuisance"]},
    ]


def handcrafted_rule_test_rows() -> list[dict[str, Any]]:
    """Audit rule engine with handcrafted states."""
    valid = _state(85, 18, 97, 37, 78, 1.2, 1.0)
    map_bad = {**valid, "mean_arterial_pressure": 20.0}
    multi_bad = {**valid, "mean_arterial_pressure": 20.0, "spo2": 105.0}
    boundary = {**valid, "spo2": 100.0}
    rows = []
    for name, state in [("valid", valid), ("single_rule_violation", map_bad), ("multi_rule_violation", multi_bad), ("boundary", boundary)]:
        branches = [[state]]
        counts = _evaluate_rules(branches, [rule for rule in stress_rules() if rule.get("kind") is None])
        rows.append({"case": name, "total_violations": sum(item["violations"] for item in counts.values())})
    return rows


def temporal_alignment_toy_result() -> dict[str, Any]:
    """Deterministic proof that A_t is used for S_t -> S_t+1 in current windowing."""
    state = _state(80, 20, 96, 37, 60, 2, 1)
    sim_none = SyntheticPhysiologySimulator()
    sim_high = SyntheticPhysiologySimulator()
    sim_none.reset(state)
    sim_high.reset(state)
    next_none = sim_none.step({"iv_fluids": 0.0, "vasopressors": 0.0})
    next_high = sim_high.step({"iv_fluids": 0.0, "vasopressors": 1.0})
    return {
        "intended_alignment": "S_t_plus_1 = f(S_t, A_t)",
        "map_next_none": next_none["mean_arterial_pressure"],
        "map_next_high_pressor": next_high["mean_arterial_pressure"],
        "action_t_changes_t_plus_1": next_high["mean_arterial_pressure"] > next_none["mean_arterial_pressure"],
    }


def shift_characterization_rows(source: dict[str, list[dict[str, Any]]], shifted: dict[str, list[dict[str, Any]]], shift_type: str, seed: int) -> list[dict[str, Any]]:
    """Compare source and shifted cohorts without assuming degradation."""
    return [_cohort_characterization(source, "source", shift_type, seed), _cohort_characterization(shifted, "shifted", shift_type, seed)]


def _cohort_characterization(sequences: dict[str, list[dict[str, Any]]], cohort: str, shift_type: str, seed: int) -> dict[str, Any]:
    flat = [row for rows in sequences.values() for row in rows]
    severities = [_severity(row) for row in flat]
    volatility = []
    for rows in sequences.values():
        for a, b in zip(rows, rows[1:]):
            vals = []
            for variable in STATE_VARIABLES:
                if a[variable] is not None and b[variable] is not None:
                    vals.append(abs(float(b[variable]) - float(a[variable])))
            if vals:
                volatility.append(_mean(vals))
    return {
        "seed": seed,
        "shift_type": shift_type,
        "cohort": cohort,
        "severity_mean": _mean(severities),
        "severity_std": _std(severities),
        "trajectory_volatility": _mean(volatility),
        "missingness": _mean([1.0 - float(row[f"{variable}_mask"]) for row in flat for variable in STATE_VARIABLES]),
        "iv_fluids_mean": _mean([float(row["iv_fluids"]) for row in flat]),
        "vasopressors_mean": _mean([float(row["vasopressors"]) for row in flat]),
    }


def _scenario_branches(model: Any, scenario: dict[str, Any]) -> list[list[dict[str, float]]]:
    doses = [0.0, 0.15, 0.4, 0.8]
    return [model.rollout([_initial_row(scenario["state"])], 6, [{"iv_fluids": 0.0, "vasopressors": dose} for _ in range(6)]) for dose in doses]


def _evaluate_rules(branches: list[list[dict[str, float]]], rules: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {rule["id"]: {"checks": 0, "violations": 0} for rule in rules}
    for rule in rules:
        if rule.get("kind") == "monotonic_increasing_by_branch":
            values = [branch[-1][rule["variable"]] for branch in branches]
            for a, b in zip(values, values[1:]):
                counts[rule["id"]]["checks"] += 1
                counts[rule["id"]]["violations"] += int(b < a)
        elif rule.get("kind") == "nuisance_sensitivity_bound":
            delta = _mean_abs_change(branches[-1], branches[0])
            counts[rule["id"]]["checks"] += 1
            counts[rule["id"]]["violations"] += int(delta > rule["upper"])
        else:
            for branch in branches:
                for row in branch:
                    counts[rule["id"]]["checks"] += 1
                    value = row[rule["variable"]]
                    counts[rule["id"]]["violations"] += int(value < rule["lower"] or value > rule["upper"])
    return counts


def _apply_corruption(branches: list[list[dict[str, float]]], corruption: str) -> None:
    for idx, branch in enumerate(branches):
        for row in branch:
            if corruption == "implausible_oxygenation":
                row["spo2"] = 104.0
            elif corruption == "reversed_intervention_response":
                row["mean_arterial_pressure"] -= idx * 10.0
            elif corruption == "random_physiological_variable":
                row["lactate"] = 20.0
            elif corruption == "impossible_cross_variable":
                row["spo2"] = 55.0
                row["respiratory_rate"] = 6.0


def _dose_response_order_accuracy(true_branches: list[list[dict[str, float]]], pred_branches: list[list[dict[str, float]]]) -> float:
    hits = []
    for variable in STATE_VARIABLES:
        true_values = [branch[-1][variable] for branch in true_branches]
        if max(true_values) - min(true_values) <= SIGN_EPSILON:
            continue
        pred_values = [branch[-1][variable] for branch in pred_branches]
        for i, j in combinations(range(len(true_values)), 2):
            hits.append(1.0 if (true_values[i] < true_values[j]) == (pred_values[i] < pred_values[j]) else 0.0)
    return _mean(hits)


def _vasopressor_monotonicity(branches: list[list[dict[str, float]]]) -> float:
    maps = [branch[-1]["mean_arterial_pressure"] for branch in branches]
    checks = [1.0 if b >= a else 0.0 for a, b in zip(maps, maps[1:])]
    return _mean(checks)


def _mean_abs_change(a: list[dict[str, float]], b: list[dict[str, float]]) -> float:
    values = []
    for row_a, row_b in zip(a, b):
        for variable in STATE_VARIABLES:
            values.append(abs(row_a[variable] - row_b[variable]))
    return _mean(values)


def _trajectory_mae(predicted: list[dict[str, float]], target: list[dict[str, float]]) -> float:
    return _mean([abs(pred_row[var] - target_row[var]) for pred_row, target_row in zip(predicted, target) for var in STATE_VARIABLES])


def _initial_row(state: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"patient_id": "audit", "time_index": 0, "iv_fluids": 0.0, "vasopressors": 0.0}
    for variable, value in state.items():
        row[variable] = value
        row[f"{variable}_mask"] = 1
    return row


def _state(hr: float, rr: float, spo2: float, temp: float, map_value: float, lactate: float, creatinine: float) -> dict[str, float]:
    return {"heart_rate": hr, "respiratory_rate": rr, "spo2": spo2, "temperature": temp, "mean_arterial_pressure": map_value, "lactate": lactate, "creatinine": creatinine}


def _severity(row: dict[str, Any]) -> float:
    return 0.02 * float(row.get("heart_rate") or 0) + 0.1 * float(row.get("lactate") or 0) + 0.1 * float(row.get("creatinine") or 0) - 0.01 * float(row.get("mean_arterial_pressure") or 0)


def _lag_corr(sequences: dict[str, list[dict[str, Any]]], variable: str) -> float:
    xs = []
    ys = []
    for rows in sequences.values():
        for a, b in zip(rows, rows[1:]):
            xs.append(float(a[variable]))
            ys.append(float(b[variable]))
    return _corr(xs, ys)


def _coverage(records: list[tuple[float, float, float]], z: float) -> float:
    return sum(1 for error, sigma, _ in records if error <= z * sigma) / len(records) if records else 0.0


def _nll(y: float, mean: float, sigma: float) -> float:
    variance = sigma * sigma
    return 0.5 * (math.log(2 * math.pi * variance) + (y - mean) ** 2 / variance)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values])) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not den_x or not den_y:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (den_x * den_y)

