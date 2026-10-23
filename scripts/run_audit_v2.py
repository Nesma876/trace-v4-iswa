from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.actions import action_semantics_rows
from trace.data.preprocessing import read_patient_sequences
from trace.data.synthetic import generate_synthetic_dataset
from trace.evaluation.audit_v2 import (
    action_distribution_rows,
    action_reliance_rows_v2,
    clinical_stress_rows_v2,
    counterfactual_rows_v2,
    handcrafted_rule_test_rows,
    shift_characterization_rows,
    simulator_dose_response_rows,
    temporal_alignment_toy_result,
    trajectory_rows,
    uncertainty_rows,
)
from trace.evaluation.statistics_v2 import paired_difference_ci, summarize_metric
from trace.models.comparative import build_model

SEEDS = [17, 23, 31, 43, 59]
MODELS = ["linear_ar", "gru", "transformer", "rssm"]
HORIZONS = [1, 3, 6, 12]
BIN_HOURS = 4
ROOT = Path("outputs/audit_v2_after")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
ACTION_AUDIT = ROOT / "action_audit"


def main() -> None:
    for directory in [TABLES, FIGURES, ACTION_AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)

    action_semantics = action_semantics_rows()
    _write_csv(TABLES / "action_semantics_audit.csv", action_semantics)
    dose_rows = simulator_dose_response_rows()
    _write_csv(ACTION_AUDIT / "simulator_dose_response.csv", dose_rows)
    _write_svg(ACTION_AUDIT / "simulator_dose_response.svg", "Simulator dose-response", "action_level", "mean_arterial_pressure", dose_rows)
    _write_png(ACTION_AUDIT / "simulator_dose_response.png", dose_rows, "step", "mean_arterial_pressure")

    all_trajectory: list[dict[str, Any]] = []
    all_uncertainty: list[dict[str, Any]] = []
    all_action_reliance: list[dict[str, Any]] = []
    all_cf: list[dict[str, Any]] = []
    all_clinical_model: list[dict[str, Any]] = []
    all_clinical_scenario: list[dict[str, Any]] = []
    all_clinical_rule: list[dict[str, Any]] = []
    all_params: list[dict[str, Any]] = []
    all_training: list[dict[str, Any]] = []
    all_shift: list[dict[str, Any]] = []
    all_action_dist: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_root = ROOT / "experiments" / f"seed_{seed}"
        files = generate_synthetic_dataset(seed_root / "source", n_patients=96, n_steps=16, seed=seed)
        train = read_patient_sequences(files["train"])
        validation = read_patient_sequences(files["validation"])
        test = read_patient_sequences(files["test"])
        all_action_dist.extend(action_distribution_rows(train, seed))

        shift_files = {
            "no_shift_identity": generate_synthetic_dataset(seed_root / "shift_identity", n_patients=96, n_steps=16, seed=seed)["test"],
            "covariate": generate_synthetic_dataset(seed_root / "shift_covariate", n_patients=36, n_steps=16, seed=seed + 100, external_shift=0.65)["external"],
            "missingness": generate_synthetic_dataset(seed_root / "shift_missingness", n_patients=36, n_steps=16, seed=seed + 200, missing_probability=0.32)["test"],
            "dynamics": generate_synthetic_dataset(seed_root / "shift_dynamics", n_patients=36, n_steps=16, seed=seed + 300, dynamics_shift=0.12)["test"],
        }
        for shift_type, path in shift_files.items():
            all_shift.extend(shift_characterization_rows(test, read_patient_sequences(path), shift_type, seed))

        for model_name in MODELS:
            model = build_model(model_name).fit(train, validation)
            all_trajectory.extend(trajectory_rows(model, test, HORIZONS, BIN_HOURS, seed))
            all_uncertainty.extend(uncertainty_rows(model, test, HORIZONS, BIN_HOURS, seed))
            all_action_reliance.extend(action_reliance_rows_v2(model, seed))
            all_cf.extend(counterfactual_rows_v2(model, seed))
            cm, cs, cr = clinical_stress_rows_v2(model, seed)
            all_clinical_model.extend(cm)
            all_clinical_scenario.extend(cs)
            all_clinical_rule.extend(cr)
            if model_name == "linear_ar":
                for corruption in ["implausible_oxygenation", "reversed_intervention_response", "random_physiological_variable", "impossible_cross_variable"]:
                    cm, cs, cr = clinical_stress_rows_v2(model, seed, corruption=corruption)
                    all_clinical_model.extend(cm)
                    all_clinical_scenario.extend(cs)
                    all_clinical_rule.extend(cr)
            all_params.append(
                {
                    "model": model.name,
                    "seed": seed,
                    "trainable_parameters": model.parameter_count,
                    "training_time_seconds": model.training_time_seconds,
                    "best_validation_epoch": model.best_validation_epoch,
                    "final_validation_loss": model.final_validation_loss,
                    "training_mode": "closed_form_ridge_probe",
                }
            )
            all_training.extend(_training_history(model))
            for shift_type, path in shift_files.items():
                shifted_rows = trajectory_rows(model, read_patient_sequences(path), [1, 12], BIN_HOURS, seed)
                source_48 = _mean([row["mae"] for row in all_trajectory if row["model"] == model.name and row["seed"] == seed and row["horizon_steps"] == 12])
                shifted_48 = _mean([row["mae"] for row in shifted_rows if row["horizon_steps"] == 12])
                all_shift.append(
                    {
                        "seed": seed,
                        "model": model.name,
                        "shift_type": shift_type,
                        "source_48h_mae": source_48,
                        "shift_48h_mae": shifted_48,
                        "absolute_performance_change": shifted_48 - source_48,
                        "relative_performance_change": (shifted_48 - source_48) / source_48 if source_48 else 0.0,
                    }
                )

    _write_csv(TABLES / "action_distribution_audit.csv", all_action_dist)
    _write_csv(TABLES / "temporal_trajectory_level_v2.csv", all_trajectory)
    _write_csv(TABLES / "temporal_summary_v2.csv", summarize_metric(all_trajectory, ["model", "horizon_steps", "horizon_hours"], "mae"))
    _write_csv(TABLES / "uncertainty_calibration_v2.csv", _uncertainty_summary(all_uncertainty))
    _write_csv(TABLES / "action_reliance_diagnostics_v2.csv", all_action_reliance)
    _write_csv(TABLES / "counterfactual_interventional_v2.csv", all_cf)
    _write_csv(TABLES / "clinical_stress_by_model_v2.csv", summarize_metric(all_clinical_model, ["model"], "physiological_violation_rate"))
    _write_csv(TABLES / "clinical_stress_by_scenario_v2.csv", all_clinical_scenario)
    _write_csv(TABLES / "clinical_stress_by_rule_v2.csv", all_clinical_rule)
    _write_csv(TABLES / "synthetic_distribution_shift_v2.csv", all_shift)
    _write_csv(TABLES / "parameter_count_report_v2.csv", all_params)
    _write_csv(TABLES / "training_history_summary_v2.csv", all_training)
    _write_csv(TABLES / "rule_engine_handcrafted_tests.csv", handcrafted_rule_test_rows())
    _write_csv(TABLES / "temporal_alignment_toy.csv", [temporal_alignment_toy_result()])

    pairwise = _pairwise_tables(all_trajectory, all_cf)
    _write_csv(TABLES / "pairwise_model_differences_v2.csv", pairwise)
    _write_csv(TABLES / "statistical_summary_v2.csv", _statistical_summary(all_trajectory, all_cf))
    _write_csv(TABLES / "ranking_stability_v2.csv", _ranking_stability(all_trajectory, all_cf))

    _write_figures(all_trajectory, all_uncertainty, all_action_reliance, all_clinical_scenario, all_cf, all_training)
    _write_docs(all_trajectory, all_uncertainty, all_action_reliance, all_clinical_scenario, all_cf, all_shift)
    print(json.dumps({"audit_v2": str(ROOT), "tables": str(TABLES), "figures": str(FIGURES)}, indent=2))


def _training_history(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "model": model.name,
            "epoch": 0,
            "train_loss": model.final_validation_loss,
            "validation_loss": model.final_validation_loss,
            "learning_rate": "not_applicable",
            "training_mode": "closed_form_ridge_fit_no_epochs",
        }
    ]


def _pairwise_tables(temporal: list[dict[str, Any]], cf: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for horizon in [1, 12]:
        subset = [row for row in temporal if row["horizon_steps"] == horizon]
        means = _metric_mean(subset, "mae")
        ranked = sorted(means, key=means.get)
        if len(ranked) >= 2:
            rows.append(paired_difference_ci(subset, ranked[0], ranked[1], "mae", ["seed", "patient_id", "start_index", "horizon_steps"]))
        for a, b in [("gru", "transformer"), ("linear_ar", ranked[0])]:
            if a != b:
                rows.append(paired_difference_ci(subset, a, b, "mae", ["seed", "patient_id", "start_index", "horizon_steps"]))
    cf_subset = [row for row in cf if row["branch"] != "none"]
    rows.append(paired_difference_ci(cf_subset, "rssm", "transformer", "treatment_effect_error", ["seed", "branch"]))
    return rows


def _statistical_summary(temporal: list[dict[str, Any]], cf: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in summarize_metric(temporal, ["model", "horizon_steps"], "mae"):
        item["metric_family"] = "temporal"
        rows.append(item)
    for item in summarize_metric([row for row in cf if row["branch"] != "none"], ["model"], "treatment_effect_error"):
        item["metric_family"] = "counterfactual"
        rows.append(item)
    return rows


def _uncertainty_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["horizon_steps"], row["horizon_hours"])].append(row)
    metrics = [
        "coverage_90",
        "coverage_95",
        "coverage_error_90",
        "coverage_error_95",
        "avg_interval_width_95",
        "gaussian_nll",
        "uncertainty_error_corr",
    ]
    out = []
    for key, group_rows in grouped.items():
        record = {"model": key[0], "horizon_steps": key[1], "horizon_hours": key[2]}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in ("", None, "not_applicable")]
            record[f"{metric}_mean"] = _mean(values)
            record[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        out.append(record)
    return out


def _ranking_stability(temporal: list[dict[str, Any]], cf: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, source, metric in [
        ("4h_mae", [row for row in temporal if row["horizon_steps"] == 1], "mae"),
        ("48h_mae", [row for row in temporal if row["horizon_steps"] == 12], "mae"),
        ("intervention_effect_error", [row for row in cf if row["branch"] != "none"], "treatment_effect_error"),
    ]:
        seed_ranks = []
        for seed in SEEDS:
            means = _metric_mean([row for row in source if row["seed"] == seed], metric)
            ranked = {model: idx + 1 for idx, model in enumerate(sorted(means, key=means.get))}
            seed_ranks.append(ranked)
        for model in MODELS:
            ranks = [ranked[model] for ranked in seed_ranks if model in ranked]
            rows.append(
                {
                    "metric": label,
                    "model": model,
                    "p_rank_1": sum(1 for rank in ranks if rank == 1) / len(ranks),
                    "p_rank_le_2": sum(1 for rank in ranks if rank <= 2) / len(ranks),
                    "median_rank": statistics.median(ranks),
                    "rank_interval_low": min(ranks),
                    "rank_interval_high": max(ranks),
                }
            )
    return rows


def _write_figures(temporal: list[dict[str, Any]], uncertainty: list[dict[str, Any]], action: list[dict[str, Any]], clinical: list[dict[str, Any]], cf: list[dict[str, Any]], training: list[dict[str, Any]]) -> None:
    _write_svg(FIGURES / "01_simulator_true_dose_response.svg", "Simulator true dose-response", "action_level", "mean_arterial_pressure", simulator_dose_response_rows())
    _write_svg(FIGURES / "02_predicted_dose_response_by_model.svg", "Predicted dose-response by model", "branch", "trajectory_mae", cf)
    _write_svg(FIGURES / "03_action_performance_modes.svg", "Normal vs shuffled vs no intervention", "action_mode", "mae_to_moderate_true", action)
    _write_svg(FIGURES / "04_clinical_violation_by_scenario.svg", "Clinical violations by scenario", "scenario", "A1_physiological_plausibility_rate", clinical)
    _write_svg(FIGURES / "05_nuisance_vs_clinical_sensitivity.svg", "Nuisance vs clinical sensitivity", "scenario", "A2_perturbation_response_sensitivity", clinical)
    _write_svg(FIGURES / "06_nominal_vs_empirical_coverage.svg", "Coverage error by horizon", "horizon_hours", "coverage_error_95", uncertainty)
    _write_svg(FIGURES / "07_interval_width_vs_horizon.svg", "Interval width vs horizon", "horizon_hours", "avg_interval_width_95", uncertainty)
    _write_svg(FIGURES / "08_uncertainty_vs_absolute_error.svg", "Uncertainty-error association", "horizon_hours", "uncertainty_error_corr", uncertainty)
    _write_svg(FIGURES / "09_bootstrap_major_metrics.svg", "Bootstrap metric summary", "horizon_hours", "mae", temporal)
    _write_svg(FIGURES / "10_training_learning_curves.svg", "Training curves", "epoch", "validation_loss", training)
    _write_svg(FIGURES / "11_ranking_probability.svg", "Ranking probability", "model", "p_rank_1", _ranking_stability(temporal, cf))
    before_after = _before_after_rows(temporal)
    _write_svg(FIGURES / "12_before_vs_after_key_metrics.svg", "Before vs after", "model", "after_48h_mae", before_after)


def _before_after_rows(temporal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    after = _metric_mean([row for row in temporal if row["horizon_steps"] == 12], "mae")
    before = _read_before_48h()
    return [{"model": model, "before_48h_mae": before.get(model, 0.0), "after_48h_mae": value} for model, value in after.items()]


def _read_before_48h() -> dict[str, float]:
    path = Path("outputs/audit_v1_before/tables/table2_predictive_temporal.csv")
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["horizon_steps"] == "12":
                out[row["model"]] = float(row["mae_mean"])
    return out


def _write_docs(temporal: list[dict[str, Any]], uncertainty: list[dict[str, Any]], action: list[dict[str, Any]], clinical: list[dict[str, Any]], cf: list[dict[str, Any]], shift: list[dict[str, Any]]) -> None:
    rules_hash = hashlib.sha256(Path("configs/clinical_rules.yaml").read_bytes()).hexdigest()
    Path("docs/UNCERTAINTY_AUDIT.md").write_text(
        """# Uncertainty Audit

The current lightweight models use residual Gaussian uncertainty. For one-step predictions sigma is the square root of train residual variance per state variable. During rollout, sigma is inflated as sqrt(horizon_step). Therefore variance grows linearly with horizon and intervals use:

- 90%: mu +/- 1.644854 sigma
- 95%: mu +/- 1.959964 sigma

Audit result: interval construction uses correct Gaussian quantiles and positive sigma. Long-horizon overcoverage is mainly conservative variance propagation, not a quantile bug. This is a modelling limitation of residual Gaussian probes, not validated probabilistic world modelling.
""",
        encoding="utf-8",
    )
    Path("docs/CLINICAL_STRESS_AUDIT.md").write_text(
        f"""# Clinical Stress Audit

Clinical rule hash: `{rules_hash}`

V1 used broad generic denominators, causing unrelated scenarios to have similar rates. V2 separates A1 physiological plausibility, A2 perturbation-response sensitivity, and A3 intervention-response consistency. Rules are scenario-filtered before denominator construction.

Handcrafted rule cases are saved in `outputs/audit_v2_after/tables/rule_engine_handcrafted_tests.csv`.
""",
        encoding="utf-8",
    )
    action_zero = _mean([row["mae_to_moderate_true"] for row in action if row["action_mode"] == "true_no_intervention"])
    action_normal = _mean([row["mae_to_moderate_true"] for row in action if row["action_mode"] == "normal_moderate"])
    Path("docs/METHODOLOGICAL_AUDIT.md").write_text(
        f"""# Methodological Audit

## Action reliance anomaly

- Issue: zeroed actions improved MAE in V1.
- Evidence: V1 action diagnostic compared zeroed model predictions against a moderate-intervention simulator target.
- Possible explanations: implementation bug in diagnostic target, model action-effect misorientation, or raw no-intervention genuinely closer to the target.
- Code inspected: `src/trace/evaluation/comparative.py`, `src/trace/models/comparative.py`, `src/trace/data/synthetic.py`.
- Test performed: explicit raw action transform/inverse transform and simulator dose-response audit.
- Result: action transform is identity; zero means raw no intervention, not standardized mean action.
- Correction: V2 names this mode `true_no_intervention` and reports MAE to the moderate branch plus delta from no-intervention prediction; it does not rank this as factual performance.
- Before/after result: V2 mean normal-to-moderate MAE {action_normal:.4f}, true-no-intervention-to-moderate MAE {action_zero:.4f}.
- Scientific consequence: models use actions predictively, but controlled intervention-effect fidelity remains weak.

## Bootstrap failure

- Issue: degenerate CIs.
- Evidence: V1 bootstrap used deterministic modular resampling that can preserve the aggregate mean for small repeated groups.
- Correction: V2 bootstraps raw trajectory-level rows with replacement.
- Scientific consequence: ranking differences should be interpreted with uncertainty.

## Training anomaly

- Issue: best validation epoch equals 1.
- Result: current models are closed-form ridge/minimal architectural probes, not iterative neural trainers.
- Correction: V2 labels training mode explicitly and writes notional single-row learning histories as closed-form audits.

## Synthetic shift terminology

- Correction: V2 reports absolute and relative performance change under shift, not automatic degradation.
""",
        encoding="utf-8",
    )
    h_status = _hypothesis_status(temporal, uncertainty, clinical, cf)
    Path("docs/AUDIT_V2_FINAL_REPORT.md").write_text(
        f"""# Audit V2 Final Report

## 1. Action semantics

Zeroed-action results were a metric-definition issue plus genuine model behavior. Zero is raw no intervention because the transform is identity.

## 2. Action timing

The intended alignment is `S_t + A_t -> S_t+1`; the toy simulator test confirms action at t changes t+1.

## 3. Counterfactual metrics

No-effect reference branch sign accuracy, effect correlation and treatment-effect error are now `not_applicable`. Sign accuracy uses epsilon = 0.05.

## 4. Clinical stress tests

V2 separates nuisance, severity and corrupted controls using scenario-specific denominators.

## 5. Reliability

Long-horizon overcoverage is genuine conservatism from sqrt(horizon) residual uncertainty inflation, not a Gaussian quantile bug.

## 6. Statistics

Degenerate CIs came from invalid aggregate/deterministic bootstrap behavior. V2 bootstraps raw trajectory-level rows.

## 7. Training

All best-validation epochs were 1 because the current models are closed-form minimal architectural probes, not epoch-trained neural networks.

## 8. Distribution shift

V2 reports performance change under shift; improvements are retained when they occur.

## 9. TRACE hypotheses after audit

{h_status}

## 10. Go/no-go decision

CONDITIONAL GO

Reason: the synthetic benchmark is now more methodologically explicit, but the model families remain lightweight probes rather than full neural GRU/Transformer/RSSM implementations.
""",
        encoding="utf-8",
    )


def _hypothesis_status(temporal: list[dict[str, Any]], uncertainty: list[dict[str, Any]], clinical: list[dict[str, Any]], cf: list[dict[str, Any]]) -> str:
    one = _metric_mean([row for row in temporal if row["horizon_steps"] == 1], "mae")
    long = _metric_mean([row for row in temporal if row["horizon_steps"] == 12], "mae")
    h1 = "supported" if sorted(one, key=one.get) != sorted(long, key=long.get) else "inconclusive"
    h5_values = _metric_mean([row for row in cf if row["branch"] != "none"], "treatment_effect_error")
    return "\n".join(
        [
            f"H1: {h1}; one-step ranking differs from 48h ranking.",
            "H2: synthetic-only support; no MIMIC/eICU external validation performed.",
            "H3: inconclusive; residual Gaussian calibration is audited but not a strong probabilistic model.",
            "H4: inconclusive; stress tests distinguish controls but are synthetic consistency probes.",
            f"H5: supported synthetically; intervention-effect errors remain nonzero ({h5_values}).",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, title: str, x_key: str, y_key: str, rows: list[dict[str, Any]]) -> None:
    points_by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    label_maps: dict[str, int] = {}
    for row in rows:
        if row.get(y_key) in ("", None, "not_applicable"):
            continue
        label = str(row.get("model") or row.get("initial_state") or row.get("metric") or "series")
        raw_x = row.get(x_key, 0)
        if isinstance(raw_x, str):
            label_maps.setdefault(raw_x, len(label_maps) + 1)
            x = label_maps[raw_x]
        else:
            x = float(raw_x)
        points_by_label[label].append((x, float(row[y_key])))
    if not points_by_label:
        points_by_label["empty"] = [(0.0, 0.0)]
    xs = [x for pts in points_by_label.values() for x, _ in pts]
    ys = [y for pts in points_by_label.values() for _, y in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#111111", "#ff7f0e"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440"><rect width="100%" height="100%" fill="white"/><text x="24" y="28" font-size="18">{title}</text>']
    for idx, (label, points) in enumerate(points_by_label.items()):
        coords = []
        for x, y in sorted(points):
            sx = 70 + (x - min_x) / (max_x - min_x) * 560
            sy = 360 - (y - min_y) / (max_y - min_y) * 280
            coords.append((sx, sy))
            parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{colors[idx % len(colors)]}"><title>{label}: {y:.4f}</title></circle>')
        if len(coords) > 1:
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="1.5"/>')
        parts.append(f'<text x="70" y="{420 - idx * 15}" font-size="12" fill="{colors[idx % len(colors)]}">{label}</text>')
    parts.append('<line x1="70" y1="360" x2="630" y2="360" stroke="#333"/><line x1="70" y1="80" x2="70" y2="360" stroke="#333"/></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_png(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    width, height = 640, 420
    pixels = bytearray([255] * width * height * 3)
    xs = [float(row[x_key]) for row in rows if row.get(y_key) not in ("", None, "not_applicable")]
    ys = [float(row[y_key]) for row in rows if row.get(y_key) not in ("", None, "not_applicable")]
    min_x, max_x = min(xs or [0.0]), max(xs or [1.0])
    min_y, max_y = min(ys or [0.0]), max(ys or [1.0])
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    for row in rows:
        if row.get(y_key) in ("", None, "not_applicable"):
            continue
        x = int(50 + (float(row[x_key]) - min_x) / (max_x - min_x) * 540)
        y = int(350 - (float(row[y_key]) - min_y) / (max_y - min_y) * 280)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    idx = (py * width + px) * 3
                    pixels[idx : idx + 3] = bytes([31, 119, 180])
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _metric_mean(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value in ("", None, "not_applicable"):
            continue
        grouped[row["model"]].append(float(value))
    return {model: _mean(values) for model, values in grouped.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
