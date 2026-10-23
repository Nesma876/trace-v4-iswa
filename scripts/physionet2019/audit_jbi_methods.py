"""Additive read-only audit of frozen TRACE artifacts; no clinical inference API.

Writes only artifacts/jbi_* and artifacts/TRACE_JBI_REPRODUCIBILITY_MANIFEST.json.
Run `evidence` before `manifest`; `verify` checks a completed manifest without edits.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts"


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT/path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_out(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_out(name, values):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def statistic_slices(path):
    """Stream one cohort/horizon slab, avoiding a >1 GB external allocation."""
    with zipfile.ZipFile(path) as archive, archive.open("statistics.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported npy version {version}")
        if fortran or len(shape) != 5 or shape[:2] != (2, 5):
            raise ValueError("Unexpected frozen statistic layout")
        count = int(np.prod(shape[2:]))
        size = count*dtype.itemsize
        for ci in range(2):
            for hi in range(5):
                data = f.read(size)
                if len(data) != size:
                    raise ValueError("Truncated statistic slab")
                yield ci, hi, np.frombuffer(data, dtype=dtype).reshape(shape[2:])


def evidence():
    cfg = read("configs/physionet2019/INTERNAL_MODELLING.json")
    features = read("outputs/v4_physionet2019/audit/frozen_training_stats.json")["features"]
    common = np.array(read("outputs/v4_internal/analysis/common_variable_eligibility.json")["eligible"])
    support = []
    field_evidence = {}
    finite_bootstrap = {}
    # Reference counts are model-independent. Never load raw patient files.
    for source, folder in (("internal", "v4_internal"), ("Set_B", "v4_external")):
        path = ROOT/f"outputs/{folder}/local/linear_ar_17_patient_stats.npz"
        with np.load(path, allow_pickle=False) as z:
            fields = list(z["fields"])
            field_evidence[source] = {"npz_keys": list(z.files), "fields":fields}
        index = {k:i for i,k in enumerate(fields)}
        origin_lookup = {}
        if source == "internal":
            for r in rows("outputs/v4_internal/tables/selective_gru_17.csv"):
                if float(r["retained_fraction"]) == 1:
                    origin_lookup[(r["cohort"],int(r["horizon"]))] = int(r["origins_retained"])
        for ci, hi, a in statistic_slices(path):
            h = (1,4,12,24,48)[hi]
            if h == 1:
                continue
            cohort = ("full_available", "common_48h")[ci]
            counts = a[...,index["n"]].astype(np.float64)
            pair_counts = a[...,index["dn"]].astype(np.float64)
            totals = counts.sum(0)
            eligible = (totals >= 100) & (common if ci else True)
            patients_scored = int((counts.sum(1)>0).sum())
            # Temporal counts must never be guessed from masks or scored counts.
            temporal_patients = ""
            temporal_source = "not recoverable from retained mask-aggregated statistics"
            if source == "internal" and (ci or h == 48):
                temporal_patients = 473
                temporal_source = "outputs/v4_physionet2019/cohort_summary.csv"
            elif source == "internal" and h == 4 and not ci:
                temporal_patients = 3127
                temporal_source = "cohort/inventory: all stays have at least 8 hourly records"
            origins = origin_lookup.get((cohort,h), "")
            if source == "Set_B" and not ci and h == 4:
                origins = read("outputs/v4_external/analysis/evaluation_integrity.json")["origins"]
            support.append(dict(dataset=source,cohort=cohort,horizon_h=h,
                                eligible_patients=temporal_patients,
                                patients_with_scored_endpoint=patients_scored,
                                eligible_origins=origins,
                                scored_endpoint_targets_all_variables=int(totals.sum()),
                                scored_endpoint_targets_macro_variables=int(totals[eligible].sum()),
                                eligible_macro_variables=int(eligible.sum()),
                                frozen_primary_variables=int(common.sum()) if ci else "",
                                targets_on_frozen_primary_support=int(totals[common].sum()) if ci else "",
                                observed_adjacent_pairs_all_variables=int(pair_counts.sum()),
                                median_origins_per_patient="",temporal_patient_source=temporal_source))
        with np.load(ROOT/f"outputs/{folder}/local/bootstrap_replicates.npz") as z:
            b = z["values"]
            finite_bootstrap[source] = {"shape": list(b.shape),
                "finite_48h_dynamic_draws_per_model": np.isfinite(b[:,:,2]).sum(0).tolist(),
                "finite_48h_dynamic_draws_all_models":int(np.isfinite(b[:,:,2]).all(1).sum()),
                "finite_48h_state_draws_all_models":int(np.isfinite(b[:,:,1]).all(1).sum())}
        print(f"Read frozen {source} support and bootstrap arrays", flush=True)
    csv_out("jbi_rollout_support.csv", support)
    tex = [r"\begin{tabular}{llrrrrr}",r"\hline",
           r"Dataset / cohort & Horizon & Eligible patients & Scored patients & Origins & Targets & Variables \\",r"\hline"]
    for r in support:
        vals = [r["horizon_h"],r["eligible_patients"],r["patients_with_scored_endpoint"],r["eligible_origins"],r["scored_endpoint_targets_all_variables"],r["eligible_macro_variables"]]
        tex.append(r["dataset"].replace("_",r"\_")+" / "+r["cohort"].replace("_",r"\_")+" & "+" & ".join(str(v) if v!="" else "--" for v in vals)+r" \\")
    tex += [r"\hline",r"\end{tabular}",r"% Targets are observed patient-origin endpoints; -- means unavailable, not zero.",r"% Scored patients need not equal temporally eligible patients. See JBI_COHORT_SUPPORT.md.",r"% Variables here use the descriptive >=100 endpoint threshold. Frozen primary common support has 29 variables, including on Set B."]
    (OUT/"jbi_rollout_support.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    # Reproduce RSSM evidence from published per-variable tables; no predictions.
    rssm = []
    for r in rows("outputs/v4_internal/tables/per_variable_temporal_metrics.csv"):
        if r["model"]=="rssm" and r["cohort"]=="common_48h" and int(r["horizon"]) in (1,48) and r["variable"] in ("Resp","MAP","O2Sat","HR"):
            rssm.append({k:r[k] for k in ("model","seed","cohort","horizon","variable","n_targets","nll","coverage95","width95","mae")})
    csv_out("jbi_rssm_per_variable_sanity.csv",rssm)
    histories = []
    import torch
    for r in rows("outputs/v4_internal/tables/model_capacity.csv"):
        name, seed = r["model"], int(r["seed"])
        item = {k:r[k] for k in ("model","seed","best_epoch","epochs_run","convergence_status","validation_nll")}
        if name != "linear_ar":
            weights = torch.load(ROOT/f"checkpoints/v4_internal/{name}/{seed}/model.pt", map_location="cpu", weights_only=True)
            item["checkpoint_tensors_finite"] = all(bool(torch.isfinite(v).all()) for v in weights.values())
            item["checkpoint_tensor_shapes"] = {k:list(v.shape) for k,v in weights.items() if "decoder" in k or "head" in k}
            history = read(f"logs/v4_internal/{name}_{seed}.json")
            best = min(history,key=lambda x:x["validation_nll"])
            item.update(all_logged_numeric_losses_finite=all(np.isfinite(x[k]) for x in history for k in ("reconstruction","kl","total","validation_nll","validation_mae")),
                        selected_epoch_matches_log=best["epoch"]==int(r["best_epoch"]),
                        selected_nll_matches_log=best["validation_nll"]==float(r["validation_nll"]),
                        total_loss_identity_max_error=float(max(abs(x["total"]-x["reconstruction"]-cfg["rssm_kl_weight"]*x["kl"]) for x in history)),
                        first=history[0],last=history[-1])
        histories.append(item)
    json_out("jbi_audit_evidence.json",dict(classification="reporting/reproducibility completion",inference_performed=False,
             fields=field_evidence,bootstrap=finite_bootstrap,training=histories,
             common_variables=[v for v,e in zip(features,common) if e],
             excluded_common_variables=[v for v,e in zip(features,common) if not e]))
    # Frozen training-only density labels and range policy, mechanically joined.
    density=read("outputs/v4_internal/analysis/jbi_density_definition.json")["strata"]
    group={v:g for g,vs in density.items() for v in vs}
    missing={r["variable"]:r for r in rows("outputs/v4_physionet2019/audit/trainA_missingness.csv")}
    density_rows=[{"variable":v,"training_observed":missing[v]["N_observed"],
                   "training_hours":missing[v]["hours"],
                   "training_missingness_percentage":missing[v]["missingness_percentage"],
                   "stratum":group[v]} for v in features]
    csv_out("jbi_density_strata.csv",density_rows)
    tex=[r"\begin{tabular}{lrl}",r"\hline",r"Variable & Training missingness (\%) & Stratum \\",r"\hline"]
    for r in density_rows:
        tex.append(f"{r['variable'].replace('_',r'\_')} & {float(r['training_missingness_percentage']):.3f} & {r['stratum'].replace('_',r'\_')} \\\\")
    tex += [r"\hline",r"\end{tabular}"]
    (OUT/"jbi_density_strata.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    policy=read("configs/physionet2019/FROZEN_RANGE_POLICY.yaml")["variables"]
    effects={r["variable"]:r for r in rows("outputs/v4_physionet2019/audit/range_policy_effect_trainA.csv")}
    range_rows=[]
    for r in policy:
        e=effects.get(r["variable"],{})
        range_rows.append({"variable":r["variable"],"unit":r["canonical_unit"],
            "lower_bound":"" if r["lower_hard_bound"] is None else r["lower_hard_bound"],
            "lower_inclusive":r["lower_inclusive"],
            "upper_bound":"" if r["upper_hard_bound"] is None else r["upper_hard_bound"],
            "upper_inclusive":r["upper_inclusive"],"action":r["action"],
            "train_observations_total":e.get("observations_total",""),
            "train_observations_excluded":e.get("observations_flagged",""),
            "train_percentage_excluded":e.get("percentage_flagged",""),
            "validation_excluded":"","internal_test_excluded":"","Set_B_excluded":""})
    csv_out("jbi_range_policy.csv",range_rows)
    # Supplementary checkpoint provenance table.
    capacity=rows("outputs/v4_internal/tables/model_capacity.csv")
    seed_manifest={(r["model"],str(r["seed"])):r for r in read("outputs/v4_internal/frozen_seed_checkpoint_manifest.json")}
    model_rows=[]
    for r in capacity:
        m=seed_manifest[(r["model"],r["seed"])]
        model_rows.append({"model":r["model"],"seed":r["seed"],"best_epoch":r["best_epoch"],
            "stop_epoch":r["epochs_run"],"best_validation_nll":r["validation_nll"],
            "checkpoint_sha256":m["checkpoint_hash"]})
    csv_out("jbi_model_checkpoint_provenance.csv",model_rows)


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,encoding="utf-8").strip()


def manifest():
    files = set()
    for folder in ("configs/physionet2019","src/trace/physionet2019","scripts/physionet2019",
                   "logs/v4_internal","checkpoints/v4_internal","outputs/v4_physionet2019",
                   "outputs/v4_internal","outputs/v4_external","paper/jbi"):
        files.update(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update(ROOT/p for p in ("src/trace/models/neural_v3.py","pyproject.toml","requirements.txt"))
    files.update((ROOT/"tests").glob("*.py"))
    records = []
    for i,p in enumerate(sorted(files)):
        records.append({"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p),"bytes":p.stat().st_size})
        if i%100==0:
            print(f"Hashed {i+1}/{len(files)} artifacts",flush=True)
    hashes = {r["path"]:r["sha256"] for r in records}
    checkpoints=[]
    for r in read("outputs/v4_internal/frozen_seed_checkpoint_manifest.json"):
        p="checkpoints/v4_internal/"+r["checkpoint"].replace("\\","/")
        checkpoints.append({"model":r["model"],"seed":r["seed"],"path":p,"sha256":hashes.get(p),
                            "frozen_sha256":r["checkpoint_hash"],"matches_frozen":hashes.get(p)==r["checkpoint_hash"],
                            "model_config_sha256":r["config_hash"]})
    references={}
    for label,p in {
        "model_config":"configs/physionet2019/INTERNAL_MODELLING.json",
        "preprocessing":"configs/physionet2019/FROZEN_PREPROCESSING.yaml",
        "range_policy":"configs/physionet2019/FROZEN_RANGE_POLICY.yaml",
        "state_schema":"configs/physionet2019/FROZEN_STATE_SCHEMA.yaml",
        "normalization":"outputs/v4_physionet2019/audit/frozen_training_stats.json",
        "internal_cohort":"outputs/v4_physionet2019/splits/setA_patient_split.csv",
        "internal_cohort_summary":"outputs/v4_physionet2019/cohort_summary.csv",
        "set_b_manifest":"outputs/v4_external/analysis/immutable_manifest.json",
        "set_b_final_addendum":"outputs/v4_external/analysis/external_manifest_addendum.json",
        "set_b_final_manifest":"outputs/v4_external/analysis/final_read_only_manifest_v4.json",
        "metric_config":"configs/physionet2019/FROZEN_SCORING_PROTOCOL.yaml",
        "internal_bootstrap_execution":"outputs/v4_internal/analysis/bootstrap_provenance.json",
        "external_bootstrap_execution":"outputs/v4_external/analysis/bootstrap_provenance.json"}.items():
        references[label]={"path":p,"sha256":hashes.get(p)}
    references["internal_figure_generation"]={"path":"src/trace/physionet2019/internal_reporting.py",
        "sha256":hashes["src/trace/physionet2019/internal_reporting.py"],"commit":git("log","-1","--format=%H","--","src/trace/physionet2019/internal_reporting.py")}
    references["external_figure_generation"]={"path":"src/trace/physionet2019/external_reporting.py",
        "sha256":hashes["src/trace/physionet2019/external_reporting.py"],"commit":git("log","-1","--format=%H","--","src/trace/physionet2019/external_reporting.py")}
    checks=[]
    intermediate_checks=[]
    for folder in ("v4_internal","v4_external"):
        signature=read(f"outputs/{folder}/analysis/"+("internal_ground_truth_signature.json" if folder=="v4_internal" else "external_ground_truth_signature.json"))
        for r in checkpoints:
            p=f"outputs/{folder}/local/{r['model']}_{r['seed']}_patient_stats.npz"
            with np.load(ROOT/p,allow_pickle=False) as z:
                intermediate_checks.append({"path":p,"checkpoint_matches":str(z["checkpoint_hash"])==r["sha256"],
                    "config_matches":str(z["config_hash"])==hashes["configs/physionet2019/INTERNAL_MODELLING.json"],
                    "evaluation_mask_matches_signature":str(z["evaluation_mask_hash"])==signature["mask_sha256"],
                    "patient_count":int(len(z["subgroup"]))})
    manifests=[("outputs/v4_internal/analysis/jbi_immutable_intermediate_manifest.json","artifacts","outputs/v4_internal/"),
               ("outputs/v4_external/analysis/immutable_manifest.json","artifacts","outputs/v4_external/"),
               ("outputs/v4_external/analysis/external_manifest_addendum.json","final_artifacts","outputs/v4_external/")]
    for name,key,prefix in manifests:
        for r in read(name)[key]:
            p=prefix+r["path"].replace("\\","/")
            checks.append({"manifest":name,"path":p,"matches":hashes.get(p)==r["sha256"],
                           "expected":r["sha256"],"actual":hashes.get(p)})
    import torch, pytest
    result=dict(classification="reporting/reproducibility completion",scientific_results_modified=False,
                git_commit=git("rev-parse","HEAD"),git_describe=git("log","-1","--format=%s"),
                working_tree_status_at_audit=git("status","--short"),
                environment={"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"pytest":pytest.__version__},
                models=read("configs/physionet2019/INTERNAL_MODELLING.json")["models"],
                seeds=[17,23,31,43,59],checkpoints=checkpoints,references=references,
                bootstrap={"actual_replicates":2000,"historical_config_replicates":1000,"seed":4317,
                           "unit":"paired patient clusters and paired modeling-seed draws"},
                existing_manifest_checks=checks,intermediate_checks=intermediate_checks,files=records,
                limitations=["Current snapshot hashes do not prove historical prespecification.",
                             "Historical external manifest contains preserved superseded reporting hashes; use final addendum.",
                             "Raw clinical source files are not packaged or accessed by this audit.",
                             "No saved per-origin predictions or increment cross-products identified."])
    json_out("TRACE_JBI_FINAL_REPRODUCIBILITY_MANIFEST.json",result)
    print(json.dumps({"files":len(records),"checkpoint_matches":sum(r["matches_frozen"] for r in checkpoints),
                      "historical_manifest_mismatches":sum(not r["matches"] for r in checks)}),flush=True)


def verify():
    m=read("artifacts/TRACE_JBI_FINAL_REPRODUCIBILITY_MANIFEST.json")
    mismatches=[r["path"] for r in m["files"] if not (ROOT/r["path"]).exists() or digest(ROOT/r["path"])!=r["sha256"]]
    result={"checked_files":len(m["files"]),"changed_since_manifest":mismatches}
    json_out("jbi_final_integrity_check.json",result)
    print(json.dumps(result))
    if mismatches:
        raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("evidence","manifest","verify"))
    args=parser.parse_args()
    {"evidence":evidence,"manifest":manifest,"verify":verify}[args.phase]()
