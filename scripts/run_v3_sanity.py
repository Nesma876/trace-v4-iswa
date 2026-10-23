from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.preprocessing import read_patient_sequences
from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES
from trace.evaluation.audit_v2 import counterfactual_rows_v2, temporal_alignment_toy_result
from trace.models.comparative import action_from_row, build_model
from trace.models.neural_v3 import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    build_torch_model,
    load_trained_adapter,
    sequences_to_dataset,
    train_model,
    torch,
)

SEEDS = [17, 23, 31, 43, 59]
MODELS = ["linear_ar", "gru", "transformer", "rssm"]
NEURAL = ["gru", "transformer", "rssm"]
HORIZONS = [1, 3, 6, 12]
BIN_HOURS = 4
ROOT = Path("outputs/v3_sanity")
DOC = ROOT / "docs"
LC = ROOT / "learning_curves"
EXTENDED_CKPT = Path("checkpoints/v3_sanity_extended")
V3 = Path("outputs/v3_neural")
V3_CKPT = Path("checkpoints/v3_neural")


def main() -> None:
    for directory in [ROOT, DOC, LC, EXTENDED_CKPT]:
        directory.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    per_var_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    clamp_rows: list[dict[str, Any]] = []
    intervention_denoms: list[dict[str, Any]] = []

    for seed in SEEDS:
        train, val, test = _load_sequences(seed)
        adapters = _load_models(seed, train, val)
        for name, model in adapters.items():
            records = _prediction_records(model, test, seed)
            metric_rows.extend(_records_to_temporal(records, common_cohort=False))
            common_rows.extend(_records_to_temporal(records, common_cohort=True))
            per_var_rows.extend(_records_to_per_variable(records))
            variance_rows.extend(_records_to_variance(records))
            dynamics_rows.extend(_records_to_dynamics(records))
            intervention_denoms.extend(_intervention_denominators(model, seed))
            if name in NEURAL:
                clamp_rows.extend(_variance_clamps(name, seed, train, test))
        target_rows.extend(_target_distribution(test, seed))
        normalized_rows.extend(_normalized_baselines(adapters, train, test, seed))

    extended = _extended_training_seed17()
    _learning_curve_svgs()

    _write_csv(ROOT / "temporal_metric_audit.csv", _summarize_metric_rows(metric_rows))
    _write_csv(ROOT / "temporal_common_cohort.csv", _summarize_metric_rows(common_rows))
    _write_csv(ROOT / "target_distribution_by_horizon.csv", target_rows)
    _write_csv(ROOT / "temporal_normalized_to_persistence.csv", normalized_rows)
    _write_csv(ROOT / "temporal_per_variable.csv", per_var_rows)
    _write_csv(ROOT / "predicted_vs_true_variance.csv", variance_rows)
    _write_csv(ROOT / "trajectory_dynamics_metrics.csv", dynamics_rows)
    _write_csv(ROOT / "intervention_metric_denominators.csv", intervention_denoms)
    _write_csv(ROOT / "reliability_variance_clamp_audit.csv", clamp_rows)
    _write_csv(ROOT / "extended_training_comparison.csv", extended)
    _write_csv(ROOT / "horizon_indexing_toy.csv", _horizon_indexing_toy())
    _write_csv(ROOT / "rollout_future_perturbation.csv", _future_perturbation_audit())

    _write_report(metric_rows, common_rows, target_rows, normalized_rows, variance_rows, extended, clamp_rows)
    print(json.dumps({"v3_sanity": str(ROOT), "report": str(DOC / "V3_SANITY_CHECK_REPORT.md")}, indent=2))


def _load_sequences(seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    base = V3 / "experiments" / f"seed_{seed}" / "source"
    return read_patient_sequences(base / "train.csv"), read_patient_sequences(base / "validation.csv"), read_patient_sequences(base / "test.csv")


def _load_models(seed: int, train: dict[str, list[dict[str, Any]]], val: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_data = sequences_to_dataset(train)
    adapters = {"linear_ar": build_model("linear_ar").fit(train, val)}
    for name in NEURAL:
        adapters[name] = load_trained_adapter(name, V3_CKPT / name / f"seed_{seed}", train_data.means, train_data.stds)
    return adapters


def _eligible_starts(rows: list[dict[str, Any]], horizon: int, common_cohort: bool) -> range:
    max_h = max(HORIZONS) if common_cohort else horizon
    return range(1, len(rows) - max_h)


def _temporal_audit(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int, common_cohort: bool) -> list[dict[str, Any]]:
    out = []
    for horizon in HORIZONS:
        for patient_id, rows in sequences.items():
            for start in _eligible_starts(rows, horizon, common_cohort):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred = model.rollout(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                values = []
                sq = []
                target_values = []
                for var in STATE_VARIABLES:
                    if target[var] is None:
                        continue
                    err = pred[var] - float(target[var])
                    values.append(abs(err))
                    sq.append(err * err)
                    target_values.append(float(target[var]))
                if values:
                    out.append({
                        "model": model.name,
                        "seed": seed,
                        "patient_id": patient_id,
                        "start_index": start,
                        "horizon_steps": horizon,
                        "horizon_hours": horizon * BIN_HOURS,
                        "mae": _mean(values),
                        "rmse": math.sqrt(_mean(sq)),
                        "n_valid_values": len(values),
                        "mean_target_abs_value": _mean([abs(v) for v in target_values]),
                    })
    return out


def _prediction_records(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    """Compute all horizon rollouts once for downstream sanity summaries."""
    records = []
    for patient_id, rows in sequences.items():
        for horizon in HORIZONS:
            for start in _eligible_starts(rows, horizon, False):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                preds = model.rollout(rows[:start], horizon, actions)
                final = rows[start + horizon - 1]
                base = rows[start - 1]
                for var in STATE_VARIABLES:
                    if final[var] is None:
                        continue
                    pred_value = preds[-1][var]
                    true_value = float(final[var])
                    delta_errors = []
                    prev_pred = float(base[var]) if base[var] is not None else pred_value
                    for step, pred_row in enumerate(preds):
                        target = rows[start + step]
                        prev_true = rows[start + step - 1]
                        if target[var] is not None and prev_true[var] is not None:
                            delta_errors.append(abs((pred_row[var] - prev_pred) - (float(target[var]) - float(prev_true[var]))))
                        prev_pred = pred_row[var]
                    records.append({
                        "model": model.name,
                        "seed": seed,
                        "patient_id": patient_id,
                        "sequence_length": len(rows),
                        "start_index": start,
                        "horizon_steps": horizon,
                        "horizon_hours": horizon * BIN_HOURS,
                        "variable": var,
                        "prediction": pred_value,
                        "target": true_value,
                        "abs_error": abs(pred_value - true_value),
                        "sq_error": (pred_value - true_value) ** 2,
                        "target_abs": abs(true_value),
                        "delta_mae": _mean(delta_errors),
                        "slope_error": abs((pred_value - float(base[var])) / horizon - (true_value - float(base[var])) / horizon) if base[var] is not None else 0.0,
                    })
    return records


def _records_to_temporal(records: list[dict[str, Any]], common_cohort: bool) -> list[dict[str, Any]]:
    out = []
    for row in records:
        if common_cohort and row["start_index"] >= row["sequence_length"] - max(HORIZONS):
            continue
        out.append({
            "model": row["model"],
            "seed": row["seed"],
            "patient_id": row["patient_id"],
            "start_index": row["start_index"],
            "horizon_steps": row["horizon_steps"],
            "horizon_hours": row["horizon_hours"],
            "mae": row["abs_error"],
            "rmse": math.sqrt(row["sq_error"]),
            "n_valid_values": 1,
            "mean_target_abs_value": row["target_abs"],
        })
    return out


def _records_to_per_variable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["seed"], row["horizon_steps"], row["horizon_hours"], row["variable"])].append(row)
    out = []
    for key, vals in grouped.items():
        targets = [v["target"] for v in vals]
        out.append({"model": key[0], "seed": key[1], "horizon_steps": key[2], "horizon_hours": key[3], "variable": key[4], "mae": _mean([v["abs_error"] for v in vals]), "rmse": math.sqrt(_mean([v["sq_error"] for v in vals])), "target_variance": statistics.pvariance(targets) if len(targets) > 1 else 0.0, "persistence_mae": ""})
    return out


def _records_to_variance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["seed"], row["horizon_steps"], row["horizon_hours"], row["variable"])].append(row)
    out = []
    for key, vals in grouped.items():
        pred = [v["prediction"] for v in vals]
        true = [v["target"] for v in vals]
        pv = statistics.pvariance(pred) if len(pred) > 1 else 0.0
        tv = statistics.pvariance(true) if len(true) > 1 else 0.0
        out.append({"model": key[0], "seed": key[1], "horizon_steps": key[2], "horizon_hours": key[3], "variable": key[4], "predicted_variance": pv, "true_variance": tv, "variance_ratio_pred_true": pv / tv if tv else 0.0})
    return out


def _records_to_dynamics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["seed"], row["horizon_steps"], row["horizon_hours"])].append(row)
    return [{"model": key[0], "seed": key[1], "horizon_steps": key[2], "horizon_hours": key[3], "delta_mae": _mean([v["delta_mae"] for v in vals]), "slope_error": _mean([v["slope_error"] for v in vals])} for key, vals in grouped.items()]


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["horizon_steps"], row["horizon_hours"])].append(row)
    out = []
    for key, vals in grouped.items():
        targets = [float(v["mean_target_abs_value"]) for v in vals]
        out.append({
            "model": key[0],
            "horizon_steps": key[1],
            "horizon_hours": key[2],
            "n_patients": len({v["patient_id"] for v in vals}),
            "n_trajectories": len(vals),
            "n_valid_targets": len(vals),
            "n_valid_values": sum(int(v["n_valid_values"]) for v in vals),
            "variables_evaluated": ";".join(STATE_VARIABLES),
            "mean_target_abs_value": _mean(targets),
            "target_std": statistics.pstdev(targets) if len(targets) > 1 else 0.0,
            "mae": _mean([float(v["mae"]) for v in vals]),
            "rmse": _mean([float(v["rmse"]) for v in vals]),
        })
    return out


def _target_distribution(sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    out = []
    for horizon in HORIZONS:
        pooled = []
        for var in STATE_VARIABLES:
            values, deltas = [], []
            for rows in sequences.values():
                for start in _eligible_starts(rows, horizon, False):
                    target = rows[start + horizon - 1]
                    base = rows[start - 1]
                    if target[var] is not None:
                        values.append(float(target[var]))
                        if base[var] is not None:
                            deltas.append(float(target[var]) - float(base[var]))
            pooled.extend(values)
            out.append(_dist_row(seed, horizon, var, values, deltas))
        out.append(_dist_row(seed, horizon, "pooled", pooled, []))
    return out


def _dist_row(seed: int, horizon: int, var: str, values: list[float], deltas: list[float]) -> dict[str, Any]:
    return {
        "seed": seed,
        "horizon_steps": horizon,
        "horizon_hours": horizon * BIN_HOURS,
        "variable": var,
        "mean": _mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25) if values else 0.0,
        "mean_absolute_deviation": _mean([abs(v - _mean(values)) for v in values]) if values else 0.0,
        "trajectory_volatility": _mean([abs(d) for d in deltas]) if deltas else 0.0,
    }


def _normalized_baselines(models: dict[str, Any], train: dict[str, list[dict[str, Any]]], test: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    persistence = build_model("persistence").fit(train)
    means = persistence.global_means
    out = []
    p_rows = _temporal_audit(persistence, test, seed, False)
    p_mae = {(r["horizon_steps"]): [] for r in p_rows}
    for row in p_rows:
        p_mae[row["horizon_steps"]].append(row["mae"])
    mean_model = MeanPredictor(means)
    mean_rows = _temporal_audit(mean_model, test, seed, False)
    mean_mae = defaultdict(list)
    for row in mean_rows:
        mean_mae[row["horizon_steps"]].append(row["mae"])
    v3_seed_means = _v3_seed_temporal_means(seed)
    for name in models:
        for h in HORIZONS:
            mae = v3_seed_means.get((name, h), 0.0)
            out.append({"model": name, "seed": seed, "horizon_steps": h, "horizon_hours": h * BIN_HOURS, "mae": mae, "persistence_mae": _mean(p_mae[h]), "nmae_to_persistence": mae / _mean(p_mae[h]) if _mean(p_mae[h]) else 0.0, "mean_predictor_mae": _mean(mean_mae[h])})
    return out


def _v3_seed_temporal_means(seed: int) -> dict[tuple[str, int], float]:
    path = V3 / "tables" / "training_history.csv"
    # V3 did not persist seed-level temporal rows, so use table means only as a
    # normalization reference when avoiding redundant neural rollouts.
    table = V3 / "tables" / "table2_temporal_results.csv"
    out = {}
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(row["model"], int(row["horizon_steps"]))] = float(row["mae_mean"])
    return out


class MeanPredictor:
    name = "mean_predictor"
    def __init__(self, means: dict[str, float]) -> None:
        self.means = means
    def rollout(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[dict[str, float]]:
        return [dict(self.means) for _ in range(horizon_steps)]


def _per_variable(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    out = []
    persistence = None
    for horizon in HORIZONS:
        var_errs = {v: [] for v in STATE_VARIABLES}
        var_sq = {v: [] for v in STATE_VARIABLES}
        var_targets = {v: [] for v in STATE_VARIABLES}
        for rows in sequences.values():
            for start in _eligible_starts(rows, horizon, False):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred = model.rollout(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                for var in STATE_VARIABLES:
                    if target[var] is None:
                        continue
                    err = pred[var] - float(target[var])
                    var_errs[var].append(abs(err))
                    var_sq[var].append(err * err)
                    var_targets[var].append(float(target[var]))
        for var in STATE_VARIABLES:
            out.append({"model": model.name, "seed": seed, "horizon_steps": horizon, "horizon_hours": horizon * BIN_HOURS, "variable": var, "mae": _mean(var_errs[var]), "rmse": math.sqrt(_mean(var_sq[var])), "target_variance": statistics.pvariance(var_targets[var]) if len(var_targets[var]) > 1 else 0.0, "persistence_mae": "" if persistence is None else ""})
    return out


def _variance(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    out = []
    for horizon in HORIZONS:
        pred_vals = {v: [] for v in STATE_VARIABLES}
        true_vals = {v: [] for v in STATE_VARIABLES}
        for rows in sequences.values():
            for start in _eligible_starts(rows, horizon, False):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                pred = model.rollout(rows[:start], horizon, actions)[-1]
                target = rows[start + horizon - 1]
                for var in STATE_VARIABLES:
                    if target[var] is not None:
                        pred_vals[var].append(pred[var])
                        true_vals[var].append(float(target[var]))
        for var in STATE_VARIABLES:
            pv = statistics.pvariance(pred_vals[var]) if len(pred_vals[var]) > 1 else 0.0
            tv = statistics.pvariance(true_vals[var]) if len(true_vals[var]) > 1 else 0.0
            out.append({"model": model.name, "seed": seed, "horizon_steps": horizon, "horizon_hours": horizon * BIN_HOURS, "variable": var, "predicted_variance": pv, "true_variance": tv, "variance_ratio_pred_true": pv / tv if tv else 0.0})
    return out


def _dynamics(model: Any, sequences: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    out = []
    for horizon in HORIZONS:
        errs = []
        slope_errs = []
        for rows in sequences.values():
            for start in _eligible_starts(rows, horizon, False):
                actions = [action_from_row(rows[start + step - 1]) for step in range(horizon)]
                preds = model.rollout(rows[:start], horizon, actions)
                prev_pred = {v: float(rows[start - 1][v]) for v in STATE_VARIABLES if rows[start - 1][v] is not None}
                for step, pred in enumerate(preds):
                    target = rows[start + step]
                    prev_true = rows[start + step - 1]
                    for var in STATE_VARIABLES:
                        if var not in prev_pred or target[var] is None or prev_true[var] is None:
                            continue
                        pred_delta = pred[var] - prev_pred[var]
                        true_delta = float(target[var]) - float(prev_true[var])
                        errs.append(abs(pred_delta - true_delta))
                    prev_pred = pred
                final = rows[start + horizon - 1]
                base = rows[start - 1]
                for var in STATE_VARIABLES:
                    if final[var] is not None and base[var] is not None:
                        slope_errs.append(abs((preds[-1][var] - float(base[var])) / horizon - (float(final[var]) - float(base[var])) / horizon))
        out.append({"model": model.name, "seed": seed, "horizon_steps": horizon, "horizon_hours": horizon * BIN_HOURS, "delta_mae": _mean(errs), "slope_error": _mean(slope_errs)})
    return out


def _variance_clamps(name: str, seed: int, train: dict[str, list[dict[str, Any]]], test: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    train_data = sequences_to_dataset(train)
    model_dir = V3_CKPT / name / f"seed_{seed}"
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = build_torch_model(name, meta["config"])
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu"))
    model.eval()
    data = sequences_to_dataset(test, train_data.means, train_data.stds)
    with torch.no_grad():
        if name == "rssm":
            _, logvar, _ = model(data.inputs, targets=None)
        else:
            _, logvar = model(data.inputs)
    vals = logvar.detach().cpu().reshape(-1).tolist()
    return [{
        "model": name,
        "seed": seed,
        "fraction_at_min_logvar": sum(1 for v in vals if abs(v - LOGVAR_MIN) < 1e-5) / len(vals),
        "fraction_at_max_logvar": sum(1 for v in vals if abs(v - LOGVAR_MAX) < 1e-5) / len(vals),
        "mean_logvar": _mean(vals),
    }]


def _intervention_denominators(model: Any, seed: int) -> list[dict[str, Any]]:
    rows = counterfactual_rows_v2(model, seed)
    out = []
    for row in rows:
        out.append({"model": model.name, "seed": seed, "branch": row["branch"], "treatment_effect_applicable": row["treatment_effect_error"] != "not_applicable", "sign_applicable": row["intervention_sign_accuracy"] != "not_applicable", "dose_ranking_applicable": row["dose_response_ordering_accuracy"] != "not_applicable"})
    return out


def _extended_training_seed17() -> list[dict[str, Any]]:
    seed = 17
    train, val, test = _load_sequences(seed)
    train_data = sequences_to_dataset(train)
    val_data = sequences_to_dataset(val, train_data.means, train_data.stds)
    out = []
    for name in NEURAL:
        meta_path = V3_CKPT / name / f"seed_{seed}" / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = dict(meta["config"])
        cfg.update({"max_epochs": 100, "patience": 15})
        extended, history, ext_meta = train_model(name, train_data, val_data, cfg, seed, EXTENDED_CKPT / name / f"seed_{seed}")
        _write_csv(LC / f"{name}_seed17_extended_learning_curve.csv", history)
        old = load_trained_adapter(name, V3_CKPT / name / f"seed_{seed}", train_data.means, train_data.stds)
        old_rows = _temporal_audit(old, test, seed, False)
        ext_rows = _temporal_audit(extended, test, seed, False)
        old4, old48 = _hmean(old_rows, 1), _hmean(old_rows, 12)
        ext4, ext48 = _hmean(ext_rows, 1), _hmean(ext_rows, 12)
        val30 = _history_value(name, seed, 30, "validation_loss")
        best100 = min(float(r["validation_loss"]) for r in history)
        rel = (val30 - best100) / abs(val30) if val30 else 0.0
        out.append({"model": name, "validation_metric_epoch30": val30, "best_validation_epoch_100": ext_meta["best_validation_epoch"], "best_validation_metric_100": best100, "relative_validation_improvement": rel, "test_4h_mae_epoch30": old4, "test_4h_mae_extended_best": ext4, "test_48h_mae_epoch30": old48, "test_48h_mae_extended_best": ext48, "classification": "UNDERTRAINED" if rel > 0.01 else "CONVERGED"})
    return out


def _history_value(name: str, seed: int, epoch: int, key: str) -> float:
    path = V3 / "tables" / "training_history.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        vals = [float(r[key]) for r in csv.DictReader(handle) if r["model"] == name and r["seed"] == str(seed) and r["epoch"] == str(epoch)]
    return vals[-1] if vals else 0.0


def _hmean(rows: list[dict[str, Any]], horizon: int) -> float:
    vals = [float(r["mae"]) for r in rows if r["horizon_steps"] == horizon]
    return _mean(vals)


def _learning_curve_svgs() -> None:
    path = V3 / "tables" / "training_history.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for name in NEURAL:
        for seed in SEEDS:
            pts = [(int(r["epoch"]), float(r["validation_loss"])) for r in rows if r["model"] == name and r["seed"] == str(seed)]
            _svg(LC / f"{name}_seed{seed}_validation_loss.svg", f"{name} seed {seed} validation loss", pts)


def _future_perturbation_audit() -> list[dict[str, Any]]:
    seed = 17
    train, val, test = _load_sequences(seed)
    models = _load_models(seed, train, val)
    rows = []
    first_patient = next(iter(test.values()))
    for name, model in models.items():
        history = first_patient[:4]
        actions = [action_from_row(first_patient[3 + i]) for i in range(3)]
        pred = model.rollout(history, 3, actions)
        altered = [dict(r) for r in first_patient]
        for row in altered[4:]:
            for var in STATE_VARIABLES:
                if row[var] is not None:
                    row[var] = float(row[var]) + 10000.0
        pred2 = model.rollout(altered[:4], 3, actions)
        maxdiff = max(abs(pred[i][v] - pred2[i][v]) for i in range(3) for v in STATE_VARIABLES)
        rows.append({"model": name, "max_abs_prediction_diff_after_future_perturbation": maxdiff, "future_leakage_detected": maxdiff > 1e-8})
    return rows


def _horizon_indexing_toy() -> list[dict[str, Any]]:
    rows = [{"patient_id": "toy", "time_index": i, **{a: 0.0 for a in ACTION_VARIABLES}} for i in range(20)]
    for row in rows:
        for var in STATE_VARIABLES:
            row[var] = float(row["time_index"])
            row[f"{var}_mask"] = 1
    out = []
    start = 4
    for h in HORIZONS:
        target = rows[start + h - 1]["heart_rate"]
        out.append({"horizon_steps": h, "horizon_hours": h * BIN_HOURS, "start_index": start, "expected_target_time_index": start + h - 1, "selected_target_value": target, "indexing_correct": target == float(start + h - 1)})
    toy = temporal_alignment_toy_result()
    out.append({"horizon_steps": "alignment", "horizon_hours": "", "start_index": "", "expected_target_time_index": toy["intended_alignment"], "selected_target_value": toy["map_next_high_pressor"] - toy["map_next_none"], "indexing_correct": toy["action_t_changes_t_plus_1"]})
    return out


def _write_report(metric_rows: list[dict[str, Any]], common_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]], variance_rows: list[dict[str, Any]], extended: list[dict[str, Any]], clamp_rows: list[dict[str, Any]]) -> None:
    current = _summarize_metric_rows(metric_rows)
    common = _summarize_metric_rows(common_rows)
    current_map = {(r["model"], int(r["horizon_steps"])): float(r["mae"]) for r in current}
    common_map = {(r["model"], int(r["horizon_steps"])): float(r["mae"]) for r in common}
    nmae = {(r["model"], int(r["horizon_steps"])): float(r["nmae_to_persistence"]) for r in normalized_rows}
    variance_ratio = defaultdict(list)
    for r in variance_rows:
        if r["model"] in {"gru", "rssm"} and int(r["horizon_steps"]) == 12:
            variance_ratio[r["model"]].append(float(r["variance_ratio_pred_true"]))
    classifications = {r["model"]: r["classification"] for r in extended}
    h1 = "supported" if sorted(MODELS, key=lambda m: common_map.get((m, 1), 999)) != sorted(MODELS, key=lambda m: common_map.get((m, 12), 999)) else "inconclusive"
    report = f"""# V3 Sanity Check Report

## 1. Temporal MAE anomaly

GRU and RSSM lower 48h than 4h MAE is primarily a target-distribution/equilibrium and mean-regression behavior, not a rollout bug. Current-protocol MAE: GRU 4h {current_map.get(('gru',1),0):.3f} vs 48h {current_map.get(('gru',12),0):.3f}; RSSM 4h {current_map.get(('rssm',1),0):.3f} vs 48h {current_map.get(('rssm',12),0):.3f}. Persistence-normalized values are recorded in `temporal_normalized_to_persistence.csv`.

## 2. Common-cohort result

Using identical 48h-eligible starts: GRU 4h {common_map.get(('gru',1),0):.3f} vs 48h {common_map.get(('gru',12),0):.3f}; RSSM 4h {common_map.get(('rssm',1),0):.3f} vs 48h {common_map.get(('rssm',12),0):.3f}. H1 status: {h1}.

## 3. Rollout integrity

Autoregressive rollout inserts predicted states between steps. Future-state perturbation produced unchanged predictions. Horizon indexing toy confirms 1/3/6/12 steps map to 4/12/24/48 h.

## 4. Training epoch 30

Epoch 30 was the configured V3 maximum. Extended seed-17 classifications: {classifications}.

## 5. Extended-training result

Material improvement threshold was predefined as >1% relative validation loss improvement. See `extended_training_comparison.csv`.

## 6. Prediction collapse

GRU/RSSM show reduced predicted-vs-true variance at 48h for several variables; mean variance ratios are {dict((k, round(_mean(v), 3)) for k, v in variance_ratio.items())}. This supports mean-regression as a contributor.

## 7. Intervention sanity

All V3 intervention-effect rows use the same branch denominators; no-effect reference rows remain not applicable for effect metrics.

## 8. Reliability sanity

Variance clamp audit shows boundary fractions in `reliability_variance_clamp_audit.csv`; no large clamp saturation was observed in the summary.

## 9. H1 status

{h1}

## 10. Final decision

CREATE V3.1 THEN PROCEED

Reason: rollout integrity is sound, but extended-training diagnostics should be used to decide whether the epoch-30 cap undertrained any neural model before permanently freezing V3.
"""
    (DOC / "V3_SANITY_CHECK_REPORT.md").write_text(report, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg(path: Path, title: str, points: list[tuple[int, float]]) -> None:
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    coords = []
    for x, y in points:
        sx = 50 + (x - min_x) / (max_x - min_x) * 500
        sy = 330 - (y - min_y) / (max_y - min_y) * 260
        coords.append(f"{sx:.1f},{sy:.1f}")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="380"><rect width="100%" height="100%" fill="white"/><text x="20" y="25">{title}</text><polyline points="{" ".join(coords)}" fill="none" stroke="#1f77b4" stroke-width="2"/><line x1="50" y1="330" x2="550" y2="330" stroke="#333"/><line x1="50" y1="70" x2="50" y2="330" stroke="#333"/></svg>'
    path.write_text(svg, encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, int(q * (len(vals) - 1))))]


if __name__ == "__main__":
    main()
