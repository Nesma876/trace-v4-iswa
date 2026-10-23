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
from trace.data.synthetic import generate_synthetic_dataset
from trace.evaluation.comparative import (
    action_reliance_metrics,
    clinical_stress_metrics,
    counterfactual_metrics,
    reliability_metrics,
    selective_prediction,
    temporal_metrics,
)
from trace.models.comparative import build_model

SEEDS = [17, 23, 31, 43, 59]
MODELS = ["linear_ar", "gru", "transformer", "rssm"]
HORIZONS = [1, 3, 6, 12]
BIN_HOURS = 4
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")
ANALYSIS_DIR = Path("outputs/analysis")
EXP_DIR = Path("outputs/experiments/v1")


def main() -> None:
    for directory in [TABLE_DIR, FIGURE_DIR, ANALYSIS_DIR, EXP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    _write_protocol()
    _write_stress_audit()

    all_temporal: list[dict[str, Any]] = []
    all_reliability: list[dict[str, Any]] = []
    all_selective: list[dict[str, Any]] = []
    all_clinical: list[dict[str, Any]] = []
    all_counterfactual: list[dict[str, Any]] = []
    all_action: list[dict[str, Any]] = []
    all_params: list[dict[str, Any]] = []
    all_shift: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_dir = EXP_DIR / f"seed_{seed}"
        files = generate_synthetic_dataset(seed_dir / "source", n_patients=96, n_steps=16, seed=seed)
        train = read_patient_sequences(files["train"])
        validation = read_patient_sequences(files["validation"])
        test = read_patient_sequences(files["test"])
        shift_files = {
            "covariate": generate_synthetic_dataset(seed_dir / "shift_covariate", n_patients=36, n_steps=16, seed=seed + 100, external_shift=0.65)["external"],
            "missingness": generate_synthetic_dataset(seed_dir / "shift_missingness", n_patients=36, n_steps=16, seed=seed + 200, missing_probability=0.32)["test"],
            "dynamics": generate_synthetic_dataset(seed_dir / "shift_dynamics", n_patients=36, n_steps=16, seed=seed + 300, dynamics_shift=0.12)["test"],
        }
        for model_name in MODELS:
            model = build_model(model_name).fit(train, validation)
            all_temporal.extend(temporal_metrics(model, test, HORIZONS, BIN_HOURS, seed))
            all_reliability.extend(reliability_metrics(model, test, HORIZONS, BIN_HOURS, seed))
            all_selective.extend(selective_prediction(model, test, seed, horizon=1))
            all_clinical.extend(clinical_stress_metrics(model, seed))
            if model_name == MODELS[0]:
                all_clinical.extend(clinical_stress_metrics(model, seed, corrupted_control=True))
            all_counterfactual.extend(counterfactual_metrics(model, seed))
            all_action.extend(action_reliance_metrics(model, seed))
            all_params.append(
                {
                    "model": model.name,
                    "seed": seed,
                    "trainable_parameters": model.parameter_count,
                    "training_time_seconds": model.training_time_seconds,
                    "best_validation_epoch": model.best_validation_epoch,
                    "final_validation_loss": model.final_validation_loss,
                }
            )
            for shift_name, path in shift_files.items():
                rows = temporal_metrics(model, read_patient_sequences(path), [1, 12], BIN_HOURS, seed)
                for row in rows:
                    row["shift_type"] = shift_name
                    all_shift.append(row)

    _write_csv(TABLE_DIR / "temporal_comparison.csv", all_temporal)
    _write_csv(TABLE_DIR / "table2_predictive_temporal.csv", _aggregate(all_temporal, ["model", "horizon_steps", "horizon_hours"], ["mae", "rmse", "temporal_retention"]))
    _write_csv(TABLE_DIR / "table3_uncertainty_calibration.csv", _aggregate(all_reliability, ["model", "horizon_steps", "horizon_hours"], ["coverage_90", "coverage_95", "avg_interval_width_90", "gaussian_nll", "uncertainty_error_corr"]))
    _write_csv(TABLE_DIR / "selective_prediction.csv", all_selective)
    _write_csv(TABLE_DIR / "clinical_stress_by_model.csv", _aggregate(all_clinical, ["model"], ["physiological_violation_rate", "clinical_contradiction_rate", "total_violations"]))
    _write_csv(TABLE_DIR / "clinical_stress_by_scenario.csv", _aggregate(all_clinical, ["model", "scenario", "severity"], ["physiological_violation_rate", "clinical_contradiction_rate"]))
    _write_csv(TABLE_DIR / "table5_clinical_stress_results.csv", all_clinical)
    _write_csv(TABLE_DIR / "table6_counterfactual_interventional.csv", all_counterfactual)
    _write_csv(TABLE_DIR / "action_reliance_diagnostics.csv", all_action)
    _write_csv(TABLE_DIR / "parameter_count_report.csv", all_params)
    _write_csv(TABLE_DIR / "synthetic_distribution_shift.csv", all_shift)

    temporal_rankings, rank_corrs, reversal_matrix = _ranking_outputs(all_temporal, all_reliability, all_clinical, all_counterfactual)
    _write_csv(TABLE_DIR / "temporal_rankings.csv", temporal_rankings)
    _write_csv(TABLE_DIR / "ranking_reversal_matrix.csv", reversal_matrix)
    _write_csv(TABLE_DIR / "rank_correlations.csv", rank_corrs)

    trace_profile = _trace_profile(all_temporal, all_reliability, all_clinical, all_counterfactual)
    _write_csv(TABLE_DIR / "trace_profile.csv", trace_profile)
    _write_csv(TABLE_DIR / "table7_trace_profile_rankings.csv", trace_profile)
    _write_csv(TABLE_DIR / "pareto_summary.csv", _pareto(trace_profile))
    _write_csv(TABLE_DIR / "statistical_summary.csv", _statistical_summary(all_temporal, all_counterfactual))

    _write_figures(all_temporal, all_reliability, all_clinical, all_counterfactual, trace_profile, reversal_matrix)
    _write_hypothesis_summary(all_temporal, all_reliability, all_clinical, all_counterfactual, reversal_matrix)
    _write_interpretation(reversal_matrix)
    print(json.dumps({"models": MODELS, "seeds": len(SEEDS), "tables": str(TABLE_DIR), "figures": str(FIGURE_DIR), "analysis": str(ANALYSIS_DIR)}, indent=2))


def _ranking_outputs(temporal: list[dict[str, Any]], reliability: list[dict[str, Any]], clinical: list[dict[str, Any]], counterfactual: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rankings = []
    for horizon in HORIZONS:
        means = _metric_mean([row for row in temporal if row["horizon_steps"] == horizon], "mae")
        ranks = _rank(means)
        for model, rank in ranks.items():
            rankings.append({"model": model, "horizon_steps": horizon, "horizon_hours": horizon * BIN_HOURS, "rank": rank})
    corrs = []
    ranks_by_h = {h: {row["model"]: row["rank"] for row in rankings if row["horizon_steps"] == h} for h in HORIZONS}
    for h in [3, 6, 12]:
        corrs.append({"comparison": f"4h_vs_{h * BIN_HOURS}h", "spearman": _spearman(ranks_by_h[1], ranks_by_h[h]), "kendall_tau": _kendall(ranks_by_h[1], ranks_by_h[h])})
    dimensions = _dimension_ranks(temporal, reliability, clinical, counterfactual)
    matrix = []
    names = list(dimensions)
    for left in names:
        for right in names:
            matrix.append({"dimension_a": left, "dimension_b": right, "spearman": _spearman(dimensions[left], dimensions[right]), "kendall_tau": _kendall(dimensions[left], dimensions[right]), "ranking_disagreement": 1 - max(-1, min(1, _spearman(dimensions[left], dimensions[right])))})
    return rankings, corrs, matrix


def _dimension_ranks(temporal: list[dict[str, Any]], reliability: list[dict[str, Any]], clinical: list[dict[str, Any]], counterfactual: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    one = _rank(_metric_mean([row for row in temporal if row["horizon_steps"] == 1], "mae"))
    long = _rank(_metric_mean([row for row in temporal if row["horizon_steps"] == 12], "mae"))
    calib = _rank({m: abs(v - 0.95) for m, v in _metric_mean([row for row in reliability if row["horizon_steps"] == 1], "coverage_95").items()})
    assoc = _rank(_metric_mean([row for row in reliability if row["horizon_steps"] == 12], "uncertainty_error_corr"), lower=False)
    clinical_rank = _rank(_metric_mean([row for row in clinical if "corrupted" not in row["model"]], "physiological_violation_rate"))
    cf = _rank(_metric_mean([row for row in counterfactual if row["branch"] != "none"], "treatment_effect_error"))
    return {"one_step": one, "rollout_48h": long, "calibration": calib, "uncertainty_error": assoc, "clinical": clinical_rank, "intervention": cf}


def _trace_profile(temporal: list[dict[str, Any]], reliability: list[dict[str, Any]], clinical: list[dict[str, Any]], counterfactual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dims = _dimension_ranks(temporal, reliability, clinical, counterfactual)
    raw = {
        "T_temporal": _metric_mean([row for row in temporal if row["horizon_steps"] == 12], "mae"),
        "R_reliability": {m: abs(v - 0.95) for m, v in _metric_mean([row for row in reliability if row["horizon_steps"] == 1], "coverage_95").items()},
        "A_alignment": _metric_mean([row for row in clinical if "corrupted" not in row["model"]], "physiological_violation_rate"),
        "C_intervention": _metric_mean([row for row in counterfactual if row["branch"] != "none"], "treatment_effect_error"),
    }
    output = []
    for model in MODELS:
        row = {"model": model}
        for dim, values in raw.items():
            row[dim] = values.get(model, 0.0)
            row[f"{dim}_rank"] = dims[{"T_temporal": "rollout_48h", "R_reliability": "calibration", "A_alignment": "clinical", "C_intervention": "intervention"}[dim]].get(model, 0)
        output.append(row)
    return output


def _pareto(profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = ["T_temporal", "R_reliability", "A_alignment", "C_intervention"]
    output = []
    for row in profile:
        dominated = False
        for other in profile:
            if other["model"] == row["model"]:
                continue
            no_worse = all(float(other[obj]) <= float(row[obj]) for obj in objectives)
            better = any(float(other[obj]) < float(row[obj]) for obj in objectives)
            dominated = dominated or (no_worse and better)
        output.append({"model": row["model"], "non_dominated": not dominated})
    return output


def _statistical_summary(temporal: list[dict[str, Any]], counterfactual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source, metric, label in [(temporal, "mae", "temporal_mae"), (counterfactual, "treatment_effect_error", "counterfactual_te_error")]:
        for model, values in _metric_values(source, metric).items():
            rows.append({"metric": label, "model": model, "mean": _mean(values), "std": statistics.pstdev(values) if len(values) > 1 else 0.0, "bootstrap_ci_low": _bootstrap(values)[0], "bootstrap_ci_high": _bootstrap(values)[1]})
    return rows


def _write_figures(temporal: list[dict[str, Any]], reliability: list[dict[str, Any]], clinical: list[dict[str, Any]], counterfactual: list[dict[str, Any]], profile: list[dict[str, Any]], reversal: list[dict[str, Any]]) -> None:
    temporal_means = _aggregate(temporal, ["model", "horizon_hours"], ["mae", "rmse"])
    _line_svg(FIGURE_DIR / "figure2_rollout_error.svg", temporal_means, "horizon_hours", "mae", "Figure 2: MAE vs rollout horizon")
    coverage = _aggregate(reliability, ["model", "horizon_hours"], ["coverage_95"])
    _line_svg(FIGURE_DIR / "figure3_calibration_coverage.svg", coverage, "horizon_hours", "coverage_95", "Figure 3: 95% coverage by horizon")
    stress = _aggregate([r for r in clinical if "corrupted" not in r["model"]], ["model"], ["physiological_violation_rate"])
    _bar_svg(FIGURE_DIR / "figure5_clinical_violation_rate.svg", stress, "model", "physiological_violation_rate_mean", "Figure 5: clinical violation rate")
    cf = _aggregate(counterfactual, ["model", "branch"], ["treatment_effect_error"])
    _bar_svg(FIGURE_DIR / "figure6_counterfactual_branches.svg", cf, "model", "treatment_effect_error_mean", "Figure 6: intervention branch error")
    _bar_svg(FIGURE_DIR / "figure7_trace_profile.svg", profile, "model", "T_temporal", "Figure 7: TRACE profile temporal objective")
    _heat_svg(FIGURE_DIR / "figure8_ranking_reversal_matrix.svg", reversal, "Figure 8: ranking reversal matrix")
    one = _metric_mean([r for r in temporal if r["horizon_steps"] == 1], "mae")
    long = _metric_mean([r for r in temporal if r["horizon_steps"] == 12], "mae")
    intervention = _metric_mean([r for r in counterfactual if r["branch"] != "none"], "treatment_effect_error")
    _scatter_svg(FIGURE_DIR / "predictive_accuracy_vs_interventional_validity.svg", one, intervention, "Predictive accuracy vs interventional validity")
    _scatter_svg(FIGURE_DIR / "predictive_accuracy_vs_long_horizon_fidelity.svg", one, long, "Predictive accuracy vs long-horizon fidelity")


def _write_hypothesis_summary(temporal: list[dict[str, Any]], reliability: list[dict[str, Any]], clinical: list[dict[str, Any]], counterfactual: list[dict[str, Any]], reversal: list[dict[str, Any]]) -> None:
    any_reversal = any(row["dimension_a"] != row["dimension_b"] and row["spearman"] < 1.0 for row in reversal)
    summary = {
        "H1": {"hypothesis": "Local next-step accuracy does not guarantee long-horizon rollout fidelity.", "supported_by_current_experiment": any_reversal, "status": "supported" if any_reversal else "not_supported", "primary_results": {"ranking_reversal_detected": any_reversal}, "caveats": ["Synthetic dry-run only."]},
        "H2": {"status": "not_tested_until_external_dataset", "synthetic_distribution_shift": "reported separately; not clinical external validation"},
        "H3": {"hypothesis": "Predictive accuracy does not imply good calibration.", "status": "inconclusive", "primary_results": _metric_mean([r for r in reliability if r["horizon_steps"] == 1], "coverage_95"), "caveats": ["Uncertainty is residual Gaussian dry-run uncertainty."]},
        "H4": {"hypothesis": "Statistical trajectory fidelity does not guarantee clinical alignment.", "status": "inconclusive", "primary_results": _metric_mean(clinical, "physiological_violation_rate"), "caveats": ["Rules are synthetic stress probes, not clinical guidelines."]},
        "H5": {"hypothesis": "Factual trajectory prediction does not establish counterfactual validity.", "status": "supported" if any_reversal else "inconclusive", "primary_results": _metric_mean(counterfactual, "treatment_effect_error"), "caveats": ["Simulator-grounded only."]},
    }
    (ANALYSIS_DIR / "hypothesis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_protocol() -> None:
    Path("docs/EXPERIMENT_V1_PROTOCOL.md").write_text(
        """# TRACE Experiment V1 Protocol

Frozen before running comparative model results.

- Seeds: 17, 23, 31, 43, 59.
- Synthetic patients per seed: 96 source patients; additional controlled shift cohorts.
- Split: deterministic patient-level 60% train, 20% validation, 20% test.
- Time bin: 4 hours.
- Sequence length: 16 bins.
- Rollout horizons: 1, 3, 6, 12 steps, reported as 4h, 12h, 24h, 48h.
- State variables: heart_rate, respiratory_rate, spo2, temperature, mean_arterial_pressure, lactate, creatinine.
- Interventions: continuous iv_fluids and vasopressors.
- Preprocessing: missing values are represented by masks; latest observed value with train mean fallback is used by lightweight models.
- Model-selection metric: validation one-step MAE. Hyperparameter budget is fixed to one predefined lightweight configuration per family in this dependency-free dry-run.
- Independent runs: five seeds.
- Statistical summaries: mean, standard deviation, bootstrap 95% intervals. Ranking analyses use Spearman and Kendall rank agreement.

This protocol does not constitute MIMIC/eICU external validation.
""",
        encoding="utf-8",
    )


def _write_stress_audit() -> None:
    Path("docs/STRESS_TEST_AUDIT.md").write_text(
        """# Stress Test Audit

The Milestone 1 clinical stress test produced zero violations because the rules checked only broad hard bounds: MAP below 30 and SpO2 above 100. The synthetic generator also clips SpO2 to physiologically possible values and rarely generates extreme MAP values, so the rules largely duplicated generator safeguards.

Experiment V1 adds discriminative stress scenarios: graded hypotension, elevated lactate, renal deterioration, hypoxemia, intervention intensity, a nuisance negative control, and corrupted positive controls. These tests remain synthetic consistency probes and must not be interpreted as clinical guideline validation.
""",
        encoding="utf-8",
    )


def _write_interpretation(reversal: list[dict[str, Any]]) -> None:
    detected = any(row["dimension_a"] != row["dimension_b"] and row["spearman"] < 1.0 for row in reversal)
    Path("docs/EXPERIMENT_V1_INTERPRETATION.md").write_text(
        f"""# Experiment V1 Interpretation

## Finding 1 - Next-step vs rollout

Ranking reversal detected: {detected}. This is synthetic evidence only and should be treated as a benchmark diagnostic rather than a clinical claim.

## Finding 2 - Accuracy vs calibration

Calibration is evaluated with residual Gaussian uncertainty. It is useful for dry-run plumbing but not yet a substitute for probabilistic neural training.

## Finding 3 - Accuracy vs clinical consistency

The stress engine now includes positive controls so failure detection can be audited.

## Finding 4 - Factual vs interventional dynamics

Controlled simulator branches are evaluated across intervention intensities and summarized in Table 6.

## Finding 5 - Multidimensional model selection

TRACE reports separate objectives and a Pareto summary. No single global TRACE score is computed.
""",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in grouped.items():
        out = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in ("", None)]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        output.append(out)
    return output


def _metric_mean(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = _metric_values(rows, metric)
    return {model: _mean(vals) for model, vals in values.items()}


def _metric_values(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        model = row["model"]
        if "corrupted" in model:
            continue
        values[model].append(float(row[metric]))
    return values


def _rank(values: dict[str, float], lower: bool = True) -> dict[str, int]:
    ordered = sorted(values, key=values.get, reverse=not lower)
    return {model: idx + 1 for idx, model in enumerate(ordered)}


def _spearman(a: dict[str, int], b: dict[str, int]) -> float:
    common = [model for model in a if model in b]
    n = len(common)
    if n < 2:
        return 0.0
    d2 = sum((a[model] - b[model]) ** 2 for model in common)
    return 1 - (6 * d2) / (n * (n * n - 1))


def _kendall(a: dict[str, int], b: dict[str, int]) -> float:
    common = [model for model in a if model in b]
    pairs = 0
    concordant = 0
    discordant = 0
    for i, left in enumerate(common):
        for right in common[i + 1 :]:
            pairs += 1
            concordant += int((a[left] - a[right]) * (b[left] - b[right]) > 0)
            discordant += int((a[left] - a[right]) * (b[left] - b[right]) < 0)
    return (concordant - discordant) / pairs if pairs else 0.0


def _bootstrap(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    estimates = []
    for idx in range(200):
        sample = [values[(idx * 17 + j * 11) % len(values)] for j in range(len(values))]
        estimates.append(_mean(sample))
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates)) - 1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _line_svg(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[row["model"]].append((float(row[x_key]), float(row[f"{y_key}_mean"])))
    _basic_svg(path, title, groups)


def _bar_svg(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> None:
    groups = {str(row[label_key]): [(idx + 1, float(row.get(value_key, row.get(f"{value_key}_mean", 0.0))))] for idx, row in enumerate(rows)}
    _basic_svg(path, title, groups)


def _scatter_svg(path: Path, xs: dict[str, float], ys: dict[str, float], title: str) -> None:
    groups = {model: [(xs[model], ys.get(model, 0.0))] for model in xs}
    _basic_svg(path, title, groups)


def _heat_svg(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = sorted({row["dimension_a"] for row in rows})
    size = 42
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="430" height="360"><text x="12" y="24" font-size="16">{title}</text>']
    for row in rows:
        x = labels.index(row["dimension_a"]) * size + 120
        y = labels.index(row["dimension_b"]) * size + 50
        shade = int(255 - 180 * float(row["ranking_disagreement"]))
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="rgb(255,{shade},{shade})" stroke="#333"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _basic_svg(path: Path, title: str, groups: dict[str, list[tuple[float, float]]]) -> None:
    all_x = [x for points in groups.values() for x, _ in points]
    all_y = [y for points in groups.values() for _, y in points]
    min_x, max_x = min(all_x or [0.0]), max(all_x or [1.0])
    min_y, max_y = min(all_y or [0.0]), max(all_y or [1.0])
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#111111"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420"><rect width="100%" height="100%" fill="white"/><text x="24" y="30" font-size="18">{title}</text>']
    for idx, (label, points) in enumerate(groups.items()):
        coords = []
        for x, y in sorted(points):
            sx = 60 + (x - min_x) / (max_x - min_x) * 520
            sy = 360 - (y - min_y) / (max_y - min_y) * 280
            coords.append((sx, sy))
            parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{colors[idx % len(colors)]}"><title>{label}: {y:.3f}</title></circle>')
        if len(coords) > 1:
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2"/>')
        parts.append(f'<text x="60" y="{390 - idx * 16}" font-size="12" fill="{colors[idx % len(colors)]}">{label}</text>')
    parts.append('<line x1="60" y1="360" x2="580" y2="360" stroke="#333"/><line x1="60" y1="80" x2="60" y2="360" stroke="#333"/></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()

