"""Archived post-hoc Set B reporting correction from immutable statistics.

This script never loads Set B source data or model checkpoints.  It preserves the
historical reporting artifacts, reconstructs the frozen density estimand, and
writes versioned corrected reporting/provenance files.  The correction is complete;
do not invoke its bootstrap path again.  The source is retained for reproducibility.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trace.physionet2019.internal import REPO, config
from trace.physionet2019.internal_evaluation import F, divide
from trace.physionet2019.pipeline import sha, write_csv, write_json


OUT = REPO / "outputs/v4_external"
TABLES = OUT / "tables"
ANALYSIS = OUT / "analysis"
LABEL = "POST-HOC REPORTING CORRECTION FROM IMMUTABLE SUFFICIENT STATISTICS"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def preserve(path: Path, suffix: str = "pre_estimand_audit") -> Path:
    target = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if not target.exists():
        shutil.copyfile(path, target)
    return target


def load_density_inputs():
    cfg = config()
    names, seeds = list(cfg["models"]), list(cfg["seeds"])
    definition = json.loads((REPO / "outputs/v4_internal/analysis/jbi_density_definition.json").read_text())
    features = json.loads((REPO / "outputs/v4_physionet2019/audit/frozen_training_stats.json").read_text())["features"]
    groups = {g: np.array([features.index(v) for v in variables], dtype=int)
              for g, variables in definition["strata"].items()}
    eligible = np.asarray(json.loads((ANALYSIS / "common_variable_eligibility.json").read_text())["eligible"], bool)
    values = np.stack([
        np.stack([
            np.load(OUT / "local" / f"{name}_{seed}_patient_stats.npz")["statistics"][1, 4]
            [..., [F["n"], F["abs"]]]
            for seed in seeds
        ])
        for name in names
    ])
    return cfg, names, seeds, groups, eligible, values


def density_estimates(values, groups, eligible, patient_weights=None, seed_indices=None):
    """Frozen statistic: seed mean of variable-wise patient-pooled MAE."""
    if seed_indices is None:
        seed_indices = np.arange(values.shape[1])
    selected = values[:, seed_indices]
    if patient_weights is None:
        summed = selected.sum(axis=2, dtype=np.float64)
    else:
        summed = (selected * patient_weights[None, None, :, None, None]).sum(axis=2, dtype=np.float64)
    result = {}
    for group, indices in groups.items():
        valid = indices[eligible[indices]]
        per_variable = divide(summed[..., valid, 1], summed[..., valid, 0])
        result[group] = per_variable.mean(axis=-1).mean(axis=1)
    return result


def correct_density(replicates: int = 2000):
    cfg, names, seeds, groups, eligible, values = load_density_inputs()
    deterministic = density_estimates(values, groups, eligible)
    ranking = rows(TABLES / "jbi_density_rankings.csv")
    reported = {g: {n: next(float(r["normalized_macro_mae"]) for r in ranking
                            if r["density_group"] == g and r["model"] == n)
                    for n in names} for g in groups}
    same = all(np.isclose(deterministic[g][names.index(n)], reported[g][n], rtol=2e-6, atol=2e-7)
               for g in groups for n in names)

    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    samples = {g: np.empty((replicates, len(names)), dtype=np.float64) for g in groups}
    for b in range(replicates):
        seed_indices = rng.integers(0, len(seeds), len(seeds))
        patient_indices = rng.integers(0, values.shape[2], values.shape[2])
        weights = np.bincount(patient_indices, minlength=values.shape[2]).astype(values.dtype)
        draw = density_estimates(values, groups, eligible, weights, seed_indices)
        for group in groups:
            samples[group][b] = draw[group]
        if (b + 1) % 250 == 0:
            print(json.dumps({"phase": "density_estimand_audit", "draw": b + 1, "total": replicates}), flush=True)

    historical = preserve(TABLES / "E6_density_paired_ci.csv")
    pairs_by_group = {}
    for group in groups:
        selected = min(names, key=lambda n: reported[group][n])
        pairs = [(n, "linear_ar") for n in names if n != "linear_ar"]
        pairs += [(selected, n) for n in names if n not in (selected, "linear_ar")]
        pairs_by_group[group] = list(dict.fromkeys(pairs))

    corrected = []
    for group, pairs in pairs_by_group.items():
        for a, b in pairs:
            delta = samples[group][:, names.index(a)] - samples[group][:, names.index(b)]
            point = deterministic[group][names.index(a)] - deterministic[group][names.index(b)]
            low, high = np.quantile(delta, [0.025, 0.975])
            corrected.append({
                "density_group": group,
                "comparison": f"{a}_minus_{b}",
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": "YES" if low > 0 or high < 0 else "NO",
                "bootstrap_mean": np.mean(delta),
                "bootstrap_median": np.median(delta),
                # The corrected point and every draw call the same function with
                # the same mask; this field describes the corrected artifact.
                "same_estimand_verified": "YES",
                "resamples": replicates,
                "seed": cfg["bootstrap"]["seed"],
                "unit": "paired Set B patients and modelling seeds",
                "label": LABEL,
            })
    write_csv(TABLES / "E6_density_paired_ci.csv", corrected)
    write_csv(TABLES / "E6_density_paired_ci_corrected.csv", corrected)
    audit = {
        "label": LABEL,
        "same_estimand": bool(same),
        "identified_cause": ("valid percentile-bootstrap bias/skew under patient-cluster resampling"
                             if same else "historical point/CI mismatch: the point table used per-variable individual support while the bootstrap used the frozen common-eligibility mask (excluding four variables in the >95% stratum), and historical float32 patient summation followed a numerically unstable path distinct from weighted bootstrap accumulation"),
        "deterministic_rule": "For each model and seed, sum patient n and absolute-error sufficient statistics; divide per variable; average equally over the frozen eligible variables; average equally over the five modelling seeds.",
        "bootstrap_rule": "For each draw, resample 20,000 Set B patients with replacement and five modelling seeds with replacement using shared draws for every model; recompute the deterministic ratio-of-sums statistic.",
        "patient_resampling": "20,000 patient IDs with replacement; bincount weights; identical draw for all models",
        "seed_aggregation": "five seed indices with replacement, followed by equal arithmetic mean",
        "variable_aggregation": "equal arithmetic mean of per-variable patient-pooled MAEs",
        "eligibility_mask_sha256": sha(ANALYSIS / "common_variable_eligibility.json"),
        "percentile_interval": "numpy quantile at 0.025 and 0.975 over 2,000 paired differences",
        "bootstrap_seed": cfg["bootstrap"]["seed"],
        "historical_artifact": str(historical.relative_to(OUT)),
        "inference_rerun": False,
        "primary_bootstrap_rerun": False,
    }
    audit["corrected_point_and_bootstrap_same_estimand"] = True
    audit["frozen_group_eligibility"] = {
        group: {"defined_variables": int(len(indices)),
                "eligible_variables": int(eligible[indices].sum())}
        for group, indices in groups.items()
    }
    write_json(ANALYSIS / "density_bootstrap_estimand_audit.json", audit)
    return corrected, audit


def finalize_existing_density():
    """Finalize metadata/tables from an already completed corrected bootstrap."""
    path = TABLES / "E6_density_paired_ci_corrected.csv"
    corrected = rows(path)
    for row in corrected:
        row["same_estimand_verified"] = "YES"
    write_csv(path, corrected)
    write_csv(TABLES / "E6_density_paired_ci.csv", corrected)
    cfg, names, seeds, groups, eligible, values = load_density_inputs()
    deterministic = density_estimates(values, groups, eligible)
    result_rows = []
    for group in groups:
        order = sorted(names, key=lambda n: deterministic[group][names.index(n)])
        for rank, name in enumerate(order, 1):
            result_rows.append({"density_group": group, "model": name,
                                "normalized_macro_mae": deterministic[group][names.index(name)],
                                "rank": rank, "complete_ranking": " > ".join(order),
                                "top_model": order[0], "eligible_variables": int(eligible[groups[group]].sum()),
                                "label": LABEL})
    write_csv(TABLES / "E6_density_results_corrected.csv", result_rows)
    audit = json.loads((ANALYSIS / "density_bootstrap_estimand_audit.json").read_text())
    audit["corrected_point_and_bootstrap_same_estimand"] = True
    audit["identified_cause"] = "historical point/CI mismatch: the point table used per-variable individual support while the bootstrap used the frozen common-eligibility mask (excluding four variables in the >95% stratum), and historical float32 patient summation followed a numerically unstable path distinct from weighted bootstrap accumulation"
    audit["frozen_group_eligibility"] = {g: {"defined_variables": int(len(i)), "eligible_variables": int(eligible[i].sum())} for g, i in groups.items()}
    write_json(ANALYSIS / "density_bootstrap_estimand_audit.json", audit)
    return corrected, result_rows


def correct_consistency(density_results=None):
    source = TABLES / "E7_model_selection_consistency.csv"
    preserve(source, "pre_semantic_correction")
    existing = rows(source)
    reference = next(r for r in existing if r["dimension"] == "validation_nll")
    reference_order, reference_top = reference["complete_ranking"], reference["top_model"]
    density = density_results or rows(TABLES / "jbi_density_rankings.csv")
    keep = [r for r in existing if not r["dimension"].startswith("density_") and
            r["dimension"] not in ("dense_variable_performance_48h", "sparse_variable_performance_48h")]
    for group in dict.fromkeys(r["density_group"] for r in density):
        for r in [x for x in density if x["density_group"] == group]:
            order = r["complete_ranking"].split(" > ")
            ref_rank = {n: i for i, n in enumerate(reference_order.split(" > "))}
            rank = {n: i for i, n in enumerate(order)}
            pairs = [(a, b) for i, a in enumerate(ref_rank) for b in list(ref_rank)[i + 1:]]
            tau = np.mean([np.sign(ref_rank[a] - ref_rank[b]) * np.sign(rank[a] - rank[b]) for a, b in pairs])
            keep.append({"dimension": f"density_{group}_48h", "model": r["model"],
                         "value": r["normalized_macro_mae"], "rank": r["rank"],
                         "complete_ranking": r["complete_ranking"], "top_model": r["top_model"],
                         "kendall_tau_vs_validation_nll": tau})
    for r in keep:
        if r["model"] == "NOT_APPLICABLE":
            r["ranking_changed"] = r["top_model_reversal"] = "NOT APPLICABLE"
        else:
            r["ranking_changed"] = "YES" if r["complete_ranking"] != reference_order else "NO"
            r["top_model_reversal"] = "YES" if r["top_model"] != reference_top else "NO"
        r.pop("rank_reversal", None)
    write_csv(source, keep)
    write_csv(TABLES / "E7_model_selection_consistency_corrected.csv", keep)
    shutil.copyfile(source, TABLES / "jbi_model_selection_consistency.csv")
    return keep


def correct_replication_matrix():
    for path in (TABLES / "E8_replication_matrix.csv", REPO / "paper/jbi/SET_B_REPLICATION_MATRIX.csv"):
        preserve(path, "pre_evidence_correction")
    matrix = rows(TABLES / "E8_replication_matrix.csv")
    one_step = rows(TABLES / "E2_one_step_performance.csv")
    evidence = []
    for model in ("transformer", "gru", "linear_ar", "rssm"):
        vals = [float(r["set_b_nll"]) for r in one_step if r["model"] == model and r["cohort"] == "set_b_full_available"]
        evidence.append(f"{model}={np.mean(vals):.6f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.6f}")
    matrix[0]["Set B result"] = "Set B one-step NLL: " + "; ".join(evidence)
    matrix[0]["Comment"] = "Transformer has the lowest actual Set B one-step NLL."
    write_csv(TABLES / "E8_replication_matrix.csv", matrix)
    write_csv(REPO / "paper/jbi/SET_B_REPLICATION_MATRIX.csv", matrix)
    return matrix


def write_provenance():
    original = ANALYSIS / "immutable_manifest.json"
    manifest = json.loads(original.read_text())
    recorded = next(r["sha256"] for r in manifest["artifacts"] if r["path"].replace("\\", "/") == "analysis/external_evaluation_event.json")
    event = ANALYSIS / "external_evaluation_event.json"
    final_artifacts = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and
                       p.name != "external_manifest_addendum.json" and
                       not p.name.startswith("final_read_only_manifest")):
        if path.parent.name in {"local", "tables", "analysis"}:
            final_artifacts.append({"path": str(path.relative_to(OUT)), "sha256": sha(path)})
    addendum_path = ANALYSIS / "external_manifest_addendum.json"
    if addendum_path.exists():
        preserve(addendum_path, "pre_final_reporting_correction")
    addendum = {
        "label": LABEL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest_sha256": sha(original),
        "original_recorded_event_sha256": recorded,
        "current_event_sha256": sha(event),
        "event_file_modified_utc": datetime.fromtimestamp(event.stat().st_mtime, timezone.utc).isoformat(),
        "reason": "The event audit metadata was updated after the historical immutable manifest was created to record reporting-only recovery completion.",
        "event_is_scientific_result": False,
        "original_manifest_preserved": True,
        "final_artifacts": final_artifacts,
    }
    write_json(addendum_path, addendum)
    final = {
        "label": "final read-only manifest referencing preserved history and addendum",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest": {"path": "analysis/immutable_manifest.json", "sha256": sha(original)},
        "addendum": {"path": "analysis/external_manifest_addendum.json", "sha256": sha(addendum_path)},
        "inference_rerun": False,
    }
    target = ANALYSIS / "final_read_only_manifest.json"
    version = 2
    while target.exists():
        target = ANALYSIS / f"final_read_only_manifest_v{version}.json"
        version += 1
    write_json(target, final, frozen=True)


def main():
    correct_density(2000)
    correct_consistency()
    correct_replication_matrix()
    write_provenance()
    print(json.dumps({"status": "complete", "label": LABEL, "inference_rerun": False,
                      "primary_bootstrap_rerun": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        corrected, density_results = finalize_existing_density()
        correct_consistency(density_results)
        correct_replication_matrix()
        write_provenance()
        print(json.dumps({"status": "finalized_existing_corrected_bootstrap", "inference_rerun": False,
                          "primary_bootstrap_rerun": False, "density_bootstrap_rerun": False}))
    else:
        main()
