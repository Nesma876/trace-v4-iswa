"""Train-only descriptive audit. No predictive or external-distribution analysis."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .data import discover, read_psv, split_for, patient_split
from .schema import HEADER
from .pipeline import sha, write_csv, write_json

SPLIT_SHA = "2ef72de30dff98ac4c67da1eb4f68ba0e40156953fa76ffaade7371c0eaa9484"
QUANTILES = {"min": 0, "p0_1": .001, "p1": .01, "p5": .05, "p25": .25, "median": .5,
             "p75": .75, "p95": .95, "p99": .99, "p99_9": .999, "max": 1}


def training_paths(root, out):
    split = Path(out) / "splits/setA_patient_split.csv"
    if not split.exists():
        # Clone reproducibility: restore only the exact original ID manifest, never
        # invent a new split or use held-out values for eligibility/decisions.
        candidate = Path(out) / "local/reconstructed_split.csv"
        patient_split(discover(Path(root), "A"), candidate)
        if sha(candidate) != SPLIT_SHA:
            raise ValueError("Available file IDs cannot reconstruct the original frozen split")
        split.parent.mkdir(parents=True, exist_ok=True)
        with split.open("xb") as handle:
            handle.write(candidate.read_bytes())
    if sha(split) != SPLIT_SHA:
        raise ValueError("Original Set A patient split changed")
    with split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["file_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate split patients")
    if any(r["split"] != split_for(r["file_id"]) for r in rows):
        raise ValueError("Non-deterministic split assignment")
    ids = {r["file_id"] for r in rows if r["split"] == "train"}
    paths = [p for p in discover(Path(root), "A") if p.name in ids]
    if {p.name for p in paths} != ids:
        raise ValueError("Missing training files")
    return paths


def summary(values, total):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"N_observed": len(finite), "missingness": 1 - len(finite) / total,
            **{name: float(np.quantile(finite, q)) if len(finite) else None for name, q in QUANTILES.items()}}


def describe_training(root, out):
    out = Path(out)
    features = json.loads((out / "feature_schema.json").read_text())["features"]
    paths = training_paths(root, out)
    index = [HEADER.index(v) for v in features]

    def read(path):
        raw = read_psv(path)
        return raw[:, index], hashlib.sha256(path.read_bytes()).digest()

    arrays, lengths = [], []
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (array, content_hash) in enumerate(pool.map(read, paths), 1):
            arrays.append(array)
            lengths.append(len(array))
            digest.update(content_hash)
            if i % 2000 == 0:
                print(f"Read {i}/{len(paths)} training files only", flush=True)
    data = np.concatenate(arrays)
    audit = out / "audit"
    write_csv(audit / "trainA_variable_range_audit.csv", [{"variable": v, **summary(data[:, j], len(data))} for j, v in enumerate(features)])
    fio2 = data[:, features.index("FiO2")]
    observed = fio2[np.isfinite(fio2)]
    rows = [{"statistic": k, "value": v, "percentage_of_observed": ""} for k, v in summary(fio2, len(data)).items()]
    groups = {"<=0": observed <= 0, ">0_to_1": (observed > 0) & (observed <= 1),
              ">1_to_10": (observed > 1) & (observed <= 10), ">10_to_100": (observed > 10) & (observed <= 100),
              ">100": observed > 100, "fraction_0.21_to_1": (observed >= .21) & (observed <= 1),
              "percent_21_to_100": (observed >= 21) & (observed <= 100)}
    rows += [{"statistic": k, "value": int(mask.sum()), "percentage_of_observed": float(mask.mean() * 100)} for k, mask in groups.items()]
    rounded, counts = np.unique(np.round(observed, 3), return_counts=True)
    rows += [{"statistic": f"rounded_mode_{rounded[i]:.3f}", "value": int(counts[i]), "percentage_of_observed": float(counts[i] / len(observed) * 100)} for i in np.argsort(-counts)[:25]]
    write_csv(audit / "fio2_distribution_trainA.csv", rows)
    write_csv(audit / "fio2_distinct_values_trainA.csv", [{"value": float(v), "count": int(n)} for v, n in zip(*np.unique(observed, return_counts=True))])
    os.environ.setdefault("MPLCONFIGDIR", str((out / "local/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(observed, bins=np.linspace(0, 1, 51), color="#277da1")
    axes[0].set(title="Set A training FiO2: values 0–1", xlabel="Released FiO2 value", ylabel="Observed count")
    axes[1].hist(observed, bins=np.linspace(min(0, observed.min()), observed.max(), 101), color="#b56576")
    axes[1].set(title="Full observed range (log count)", xlabel="Released FiO2 value", ylabel="Observed count", yscale="log")
    fig.savefig(audit / "fio2_distribution_trainA.png", dpi=160)
    plt.close(fig)
    local = out / "local"
    local.mkdir(exist_ok=True)
    np.savez_compressed(local / "trainA_audit_cache.npz", data=data, lengths=lengths, features=features)
    write_json(audit / "training_audit_provenance.json", {"source": "A", "split": "train", "patients": len(paths),
               "hours": len(data), "split_hash": SPLIT_SHA, "training_content_digest": digest.hexdigest(),
               "cache_hash": sha(local / "trainA_audit_cache.npz"), "validation_test_or_B_values_read": False})
    print(json.dumps({"training_patients": len(paths), "training_hours": len(data), "fio2": rows}, indent=2))


def freeze_training(root, out):
    """Fit only scaler statistics after frozen policy; never fit predictive models."""
    from .pipeline import CONFIG, REPO, config_hashes, code_commit
    from .policy import canonicalize, load_policy, POLICY
    from .preprocessing import eligible_origins
    from .scoring import common_eligible_variables
    out = Path(out)
    audit = out / "audit"
    provenance = json.loads((audit / "training_audit_provenance.json").read_text())
    cache_path = out / "local/trainA_audit_cache.npz"
    if sha(cache_path) != provenance["cache_hash"] or provenance["split_hash"] != SPLIT_SHA:
        raise ValueError("Audit cache/split provenance changed")
    # Verify source bytes anew, without reading any held-out patient's contents.
    digest = hashlib.sha256()
    paths = training_paths(root, out)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for value in pool.map(lambda p: hashlib.sha256(p.read_bytes()).digest(), paths):
            digest.update(value)
    if digest.hexdigest() != provenance["training_content_digest"]:
        raise ValueError("Training source changed since descriptive audit")
    with np.load(cache_path, allow_pickle=False) as cache:
        raw, lengths, features = cache["data"], cache["lengths"], cache["features"].tolist()
    expected = json.loads((REPO / "configs/physionet2019/FROZEN_STATE_SCHEMA.yaml").read_text())["dynamic_variables"]
    if features != expected:
        raise ValueError("State schema differs from original retained features")
    canonical, flags = canonicalize(raw, features)
    policy = load_policy()
    effects, missingness, scaling, taxonomy = [], [], [], []
    means, stds, medians, iqrs = [], [], [], []
    for j, variable in enumerate(features):
        prior = raw[:, j][np.isfinite(raw[:, j])]
        clean = canonical[:, j][np.isfinite(canonical[:, j])]
        rejected = int((flags[:, j] == 3).sum())
        effects.append({"variable": variable, "observations_total": len(prior), "observations_flagged": rejected,
                        "percentage_flagged": 100 * rejected / len(prior), "converted": 0,
                        "uncertain_subambient_retained": int((flags[:, j] == 2).sum())})
        missing = 1 - len(clean) / len(raw)
        label = "high availability" if missing <= .2 else "moderate availability" if missing <= .8 else "sparse" if missing <= .95 else "very sparse"
        missingness.append({"variable": variable, "N_observed": len(clean), "hours": len(raw), "missingness_percentage": missing * 100,
                            "raw_missingness_percentage": (1 - len(prior) / len(raw)) * 100, "availability": label})
        mean, std, median = float(clean.mean()), float(clean.std()), float(np.median(clean))
        iqr = float(np.quantile(clean, .75) - np.quantile(clean, .25))
        if not np.isfinite(std) or std <= 1e-8:
            raise ValueError("Near-zero/nonfinite scale; explicit review required")
        ratio = std / (iqr / 1.349) if iqr > 0 else None
        shift = abs(mean - median) / iqr if iqr > 0 else None
        scaling.append({"variable": variable, "mean": mean, "std": std, "median": median, "IQR": iqr,
                        "std_to_IQR_normal_equivalent": ratio, "mean_median_difference_over_IQR": shift,
                        "heavy_tail_review_flag": iqr == 0 or ratio > 5 or shift > 2,
                        "primary_scaler_changed": False})
        means.append(mean); stds.append(std); medians.append(median); iqrs.append(iqr)
    write_csv(audit / "range_policy_effect_trainA.csv", effects)
    write_csv(audit / "trainA_missingness.csv", sorted(missingness, key=lambda r: -r["missingness_percentage"]))
    write_csv(audit / "trainA_scaling_sensitivity.csv", scaling)
    if any(row["percentage_flagged"] > policy["unexpected_flagged_percentage_review_threshold"] for row in effects):
        raise ValueError("Range rejection exceeds 1% of a variable; manual review required before freeze")
    stats = {"version": "final_pre_model_v1", "fit_source": "A", "fit_split": "train", "fit_patient_count": len(paths),
             "features": features, "means": means, "stds": stds, "medians": medians, "iqrs": iqrs,
             "range_policy_hash": sha(POLICY), "split_hash": SPLIT_SHA,
             "training_content_digest": provenance["training_content_digest"], "config_hashes": config_hashes()}
    write_json(audit / "frozen_training_stats.json", stats, frozen=True)
    # Ground-truth SUPPORT ONLY: never produce or score predictions.
    counts = {(cohort, h): np.zeros(len(features), dtype=np.int64) for cohort in ("full_available", "common_48h") for h in (4, 12, 24, 48)}
    start = 0
    for length in lengths:
        block = canonical[start:start + length]
        for (cohort, h), count in counts.items():
            origins = np.asarray(list(eligible_origins(int(length), h, 4, cohort == "common_48h")), dtype=int)
            count += np.isfinite(block[origins + h]).sum(axis=0)
        start += length
    support = [{"source": "A", "split": "train", "cohort": cohort, "horizon": h, "variable": v,
                "N_observed_targets": int(count[j]), "status": "ELIGIBLE" if count[j] >= 100 else "INSUFFICIENT_TARGET_OBSERVATIONS"}
               for (cohort, h), count in counts.items() for j, v in enumerate(features)]
    write_csv(audit / "trainA_horizon_target_support.csv", support)
    common = common_eligible_variables({h: counts[("common_48h", h)] for h in (4, 12, 24, 48)}, features)
    frozen_hashes = {name: sha(REPO / "configs/physionet2019" / name) for name in
                     ("FROZEN_STATE_SCHEMA.yaml", "FROZEN_RANGE_POLICY.yaml", "FROZEN_SCORING_PROTOCOL.yaml", "FROZEN_PREPROCESSING.yaml")}
    write_json(audit / "final_freeze_manifest.json", {"repository_commit": code_commit(), "split_hash": SPLIT_SHA,
               "frozen_artifacts": frozen_hashes, "preprocessing_statistics_hash": sha(audit / "frozen_training_stats.json"),
               "training_common_horizon_eligible_variables": common, "external_test_locked": True,
               "validation_test_or_B_values_used": False, "real_models_trained": False,
               "heavy_tail_flags": [r["variable"] for r in scaling if r["heavy_tail_review_flag"]],
               "range_review_threshold_exceeded": False})
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update({"current_commit": code_commit(), "config_hashes": config_hashes(),
                     "historical_preprocessing_artifact": "preprocessing.json",
                     "historical_preprocessing_hash": sha(out / "preprocessing.json"),
                     "preprocessing_artifact": "audit/frozen_training_stats.json",
                     "preprocessing_hash": sha(audit / "frozen_training_stats.json"),
                     "historical_feature_schema_artifact": "feature_schema.json",
                     "historical_feature_schema_hash": sha(out / "feature_schema.json"),
                     "feature_schema_artifact": "configs/physionet2019/FROZEN_STATE_SCHEMA.yaml",
                     "feature_schema_hash": frozen_hashes["FROZEN_STATE_SCHEMA.yaml"],
                     "frozen_artifact_hashes": frozen_hashes,
                     "final_audit_source": "Set A training only", "preprocessing_status": "frozen_final_pre_model_v1"})
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"effects": [r for r in effects if r["observations_flagged"]], "frozen_hashes": frozen_hashes,
                      "heavy_tail_flags": [r["variable"] for r in scaling if r["heavy_tail_review_flag"]],
                      "common_horizon_eligible": len(common)}, indent=2))
