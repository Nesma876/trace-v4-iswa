"""Complete prespecified post-lock CIs from immutable sufficient statistics only.

This script performs no inference and does not alter frozen metric definitions.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from trace.physionet2019.internal import OUT, config, verify
from trace.physionet2019.internal_evaluation import F, divide
from trace.physionet2019.pipeline import write_csv, write_json

LABEL = "POST-LOCK REPORTING COMPLETION — PRESPECIFIED ANALYSIS, MISSING CI"


def interval(delta, point=None):
    valid = np.isfinite(delta)
    lo, hi = np.quantile(delta[valid], [0.025, 0.975])
    return (float(delta[valid].mean() if point is None else point), float(lo), float(hi),
            bool(lo > 0 or hi < 0), int(valid.sum()))


def main():
    stats = verify()
    cfg = config()
    names = list(cfg["models"])
    seeds = cfg["seeds"]
    B = 2000

    required = [("transformer", "rssm"), ("transformer", "gru"),
                ("rssm", "gru"), ("linear_ar", "gru"),
                ("linear_ar", "transformer"), ("linear_ar", "rssm")]

    density_definition = json.loads((OUT / "analysis/jbi_density_definition.json").read_text())
    feature_index = {name: i for i, name in enumerate(stats["features"])}
    groups = {g: np.array([feature_index[v] for v in variables], int)
              for g, variables in density_definition["strata"].items()}
    values = []
    for name in names:
        model = []
        for seed in seeds:
            z = np.load(OUT / "local" / f"{name}_{seed}_patient_stats.npz")
            model.append(z["statistics"][1, 4, :, :, [F["n"], F["abs"], F["dn"], F["da"]]].transpose(1, 2, 0))
        values.append(np.stack(model))
    values = np.stack(values)  # model, seed, patient, variable, field
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    seed_draws = np.empty((B, len(seeds)), dtype=np.int16)
    patient_counts = np.empty((B, values.shape[2]), dtype=np.int16)
    for b in range(B):
        seed_ids = rng.integers(0, len(seeds), len(seeds))
        patient_ids = rng.integers(0, values.shape[2], values.shape[2])
        seed_draws[b] = seed_ids
        patient_counts[b] = np.bincount(patient_ids, minlength=values.shape[2])

    all_seed_sums = np.einsum("bp,mspvf->bmsvf", patient_counts, values, optimize=True)
    draw_index = np.broadcast_to(seed_draws[:, None, :, None, None], all_seed_sums.shape)
    sampled = np.take_along_axis(all_seed_sums, draw_index, axis=2)

    primary = np.array(json.loads((OUT / "analysis/common_variable_eligibility.json").read_text())["eligible"])
    dynamic_support = values[0, 0, :, :, 2].sum(0) >= 100
    support = primary & dynamic_support
    dynamic = divide(sampled[..., 3], sampled[..., 2])[:, :, :, support].mean(-1).mean(-1)
    summary = list(csv.DictReader((OUT / "tables/table_3_dynamic_fidelity.csv").open()))
    dynamic_point = {r["model"]: float(r["dynamic_mae_mean"]) for r in summary
                     if r["cohort"] == "common_48h" and int(r["horizon"]) == 48}
    dynamic_rows = []
    for a, b in required:
        delta = dynamic[:, names.index(a)] - dynamic[:, names.index(b)]
        point, lo, hi, excludes, valid = interval(delta, dynamic_point[a] - dynamic_point[b])
        dynamic_rows.append({"comparison": f"{a}_minus_{b}", "difference": point,
                             "ci_low": lo, "ci_high": hi, "excludes_zero": excludes,
                             "resamples": B, "valid_resamples": valid, "label": LABEL})
    write_csv(OUT / "tables/jbi_48h_dynamic_paired_ci.csv", dynamic_rows)

    density_samples = {g: np.empty((B, len(names))) for g in groups}
    for g, indices in groups.items():
        # Eligibility was frozen on the original common cohort; never redefine it
        # from a bootstrap draw with duplicated/omitted patients.
        group_values = np.take(values, indices, axis=3)
        eligible = group_values[0, 0, :, :, 0].sum(0) >= 100
        s = sampled[:, :, :, indices, :]
        density_samples[g] = divide(s[..., 1], s[..., 0])[:, :, :, eligible].mean(-1).mean(-1)

    ranking_rows = list(csv.DictReader((OUT / "tables/jbi_density_rankings.csv").open()))
    selected = {g: next(r["top_model"] for r in ranking_rows if r["density_group"] == g)
                for g in groups}
    density_rows = []
    for g, sample in density_samples.items():
        pairs = [(deep, "linear_ar") for deep in names if deep != "linear_ar"]
        winner = selected[g]
        pairs += [(winner, other) for other in names if other not in (winner, "linear_ar")]
        for a, b in pairs:
            point_values = {r["model"]: float(r["normalized_macro_mae"]) for r in ranking_rows
                            if r["density_group"] == g}
            point, lo, hi, excludes, valid = interval(sample[:, names.index(a)] - sample[:, names.index(b)],
                                                       point_values[a] - point_values[b])
            density_rows.append({"density_group": g, "comparison": f"{a}_minus_{b}",
                                 "difference": point, "ci_low": lo, "ci_high": hi,
                                 "excludes_zero": excludes, "numerical_selected_model": winner,
                                 "resamples": B, "valid_resamples": valid, "label": LABEL})
    write_csv(OUT / "tables/jbi_density_paired_ci.csv", density_rows)

    consistency = list(csv.DictReader((OUT / "tables/jbi_model_selection_consistency.csv").open()))
    reference = next(r for r in consistency if r["dimension"] == "validation_nll")
    reference_order, reference_top = reference["complete_ranking"], reference["top_model"]
    existing_dimensions = {r["dimension"] for r in consistency}
    reference_names = reference_order.split(" > ")
    reference_ranks = {name: i for i, name in enumerate(reference_names)}
    for group in groups:
        dimension = f"density_{group}_48h"
        if dimension in existing_dimensions:
            continue
        source = [r for r in ranking_rows if r["density_group"] == group]
        order = source[0]["complete_ranking"].split(" > ")
        ranks = {name: i for i, name in enumerate(order)}
        signs = [np.sign(reference_ranks[a] - reference_ranks[b]) * np.sign(ranks[a] - ranks[b])
                 for a, b in itertools.combinations(names, 2)]
        tau = float(np.mean(signs))
        for item in source:
            consistency.append({"dimension": dimension, "model": item["model"],
                                "value": item["normalized_macro_mae"], "rank": item["rank"],
                                "complete_ranking": item["complete_ranking"], "top_model": item["top_model"],
                                "kendall_tau_vs_validation_nll": tau, "rank_reversal": "YES"})
    for row in consistency:
        if row["model"] == "NOT_APPLICABLE":
            row["ranking_changed"] = row["top_model_reversal"] = "NOT APPLICABLE"
        else:
            row["ranking_changed"] = "YES" if row["complete_ranking"] != reference_order else "NO"
            row["top_model_reversal"] = "YES" if row["top_model"] != reference_top else "NO"
    write_csv(OUT / "tables/jbi_model_selection_consistency.csv", consistency)
    write_json(OUT / "analysis/post_lock_reporting_completion.json",
               {"label": LABEL, "resamples": B, "bootstrap_seed": cfg["bootstrap"]["seed"],
                "inference_rerun": False, "Set_B_locked": True,
                "outputs": ["jbi_48h_dynamic_paired_ci.csv", "jbi_density_paired_ci.csv",
                            "jbi_model_selection_consistency.csv"]})


if __name__ == "__main__":
    main()
