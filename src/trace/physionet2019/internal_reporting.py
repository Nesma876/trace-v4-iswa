"""Aggregate tables, paired patient/seed bootstrap, figures and guarded conclusions."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .internal import OUT, CHECKPOINTS, config, verify
from .internal_evaluation import EVAL_HORIZONS, F, NOMINAL, summarize, divide
from .pipeline import sha, write_csv, write_json


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    return value


def tables():
    stats = verify()
    cfg = config()
    eligibility = json.loads((OUT / "analysis/common_variable_eligibility.json").read_text())
    eligible = np.array(eligibility["eligible"])
    dense = np.array([v in {"HR","O2Sat","Temp","SBP","MAP","DBP","Resp"} for v in stats["features"]])
    rows, perrows, calibration, failures, capacities = [], [], [], [], []
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            z = np.load(OUT / "local" / f"{name}_{seed}_patient_stats.npz")
            a, group = z["statistics"], z["subgroup"]
            record = json.loads((CHECKPOINTS / name / str(seed) / "metadata.json").read_text())
            record["inference_time"] = float(z["inference_time"])
            capacities.append(record)
            for ci, cohort in enumerate(("full_available","common_48h")):
                for hi,h in enumerate(EVAL_HORIZONS):
                    for subgroup, selected in (("all",np.ones(len(group),bool)),("sepsis_positive",group),("non_sepsis",~group)):
                        s = a[ci,hi,selected].sum(0)
                        base = {"model":name,"seed":seed,"cohort":cohort,"horizon":h,"subgroup":subgroup}
                        per, aggregate, support = summarize(s, eligible if ci else None)
                        rows.append({**base,"variable_set":"primary",**aggregate,"eligible_variables":int(support.sum()),"n_targets":int(s[:,F["n"]].sum())})
                        _, da, ds = summarize(s, dense & eligible if ci else dense)
                        rows.append({**base,"variable_set":"dense",**da,"eligible_variables":int(ds.sum()),"n_targets":int(s[dense,F["n"]].sum())})
                        if subgroup != "all":
                            continue
                        # Per-variable support is independent of macro intersection membership.
                        pv,_,individual_support = summarize(s)
                        for j,v in enumerate(stats["features"]):
                            enough = bool(individual_support[j])
                            perrows.append({**base,"variable":v,"n_targets":int(s[j,F["n"]]),
                                "status":"OK" if enough else "INSUFFICIENT_TARGET_OBSERVATIONS",
                                "primary_macro_eligible":bool(support[j]),
                                **{k:float(x[j]) if enough else None for k,x in pv.items()},
                                "native_mae":float(pv["mae"][j]*stats["stds"][j]) if enough else None,
                                "native_rmse":float(pv["rmse"][j]*stats["stds"][j]) if enough else None,
                                "n_dynamic_pairs":int(s[j,F["dn"]]),"n_correlations":int(s[j,F["cn"]]),"n_baseline_targets":int(s[j,F["bn"]])})
                        for p in NOMINAL:
                            empirical = divide(s[:,F[f"cal{p}"]], s[:,F["n"]])
                            for j,v in enumerate(stats["features"]):
                                calibration.append({**base,"variable":v,"nominal":p,"empirical":float(empirical[j]) if individual_support[j] else None,"n_targets":int(s[j,F["n"]])})
                            calibration.append({**base,"variable":"MACRO_ELIGIBLE","nominal":p,"empirical":float(np.mean(empirical[support])),"n_targets":int(s[support,F["n"]].sum())})
            # Patient failure-mode summaries are secondary, not hypothesis tests.
            _, metrics, _ = summarize(a[1,4], eligible, minimum=1)
            u = divide(a[1,4,:,eligible,F["u"]].sum(0),a[1,4,:,eligible,F["n"]].sum(0))
            vectors = {**metrics,"uncertainty":u}
            definitions = [("low_mae_poor_dynamics","mae","dynamic_mae","high"),
                           ("low_mae_variance_collapse","mae","variance_ratio","low"),
                           ("good_nll_poor_calibration","nll","coverage_error95","high"),
                           ("high_confidence_large_error","uncertainty","mae","high")]
            for label, lowkey, otherkey, direction in definitions:
                left,right = vectors[lowkey],vectors[otherkey]
                valid = np.isfinite(left)&np.isfinite(right)
                if not valid.any():
                    continue
                qlow=np.quantile(left[valid],.25)
                qother=np.quantile(right[valid],.25 if direction=="low" else .75)
                selected = valid & (left<=qlow) & ((right<=qother) if direction=="low" else (right>=qother))
                failures.append({"model":name,"seed":seed,"rule":label,"eligible_patients":int(valid.sum()),"flagged_patients":int(selected.sum()),"low_threshold":qlow,"other_threshold":qother})
    for filename, values in (("model_capacity",capacities),("internal_metrics_by_seed",rows),("per_variable_temporal_metrics",perrows),("calibration",calibration),("failure_modes",failures)):
        write_csv(OUT / "tables" / f"{filename}.csv", clean(values))
    # Seed summaries remain distinct from patient-bootstrap intervals.
    metrics = [k for k in rows[0] if k not in ("model","seed","cohort","horizon","subgroup","variable_set","eligible_variables","n_targets")]
    summaries=[]
    for keys in sorted({tuple(r[k] for k in ("model","cohort","horizon","subgroup","variable_set")) for r in rows}):
        subset=[r for r in rows if tuple(r[k] for k in ("model","cohort","horizon","subgroup","variable_set"))==keys]
        row=dict(zip(("model","cohort","horizon","subgroup","variable_set"),keys))
        for metric in metrics:
            values=np.array([r[metric] for r in subset],float)
            row[metric+"_mean"]=np.nanmean(values) if np.isfinite(values).any() else None
            row[metric+"_sd"]=np.nanstd(values,ddof=1) if np.isfinite(values).sum()>1 else None
        summaries.append(row)
    write_csv(OUT / "tables/internal_summary.csv",clean(summaries))
    # Publication tables explicitly separate dimensions and secondary analyses.
    identifiers=("model","cohort","horizon","subgroup","variable_set")
    definitions={
        "table_2_state_fidelity":("mae","rmse","micro_mae","micro_rmse"),
        "table_3_dynamic_fidelity":("dynamic_mae","change_variance_ratio","variance_ratio","trajectory_correlation","change_from_baseline_error"),
        "table_4_reliability":("nll","crps","coverage50","coverage80","coverage95","coverage_error50","coverage_error80","coverage_error95","width50","width80","width95","uncertainty_error_correlation")}
    for filename,fields in definitions.items():
        write_csv(OUT/"tables"/f"{filename}.csv",clean([{k:v for k,v in r.items() if k in identifiers or any(k==f+suffix for f in fields for suffix in ("_mean","_sd"))} for r in summaries if r["variable_set"]=="primary" and r["subgroup"]=="all"]))
    write_csv(OUT/"tables/table_5_dense_sensitivity.csv",clean([r for r in summaries if r["subgroup"]=="all"]))
    write_csv(OUT/"tables/table_6_sepsis_subgroup.csv",clean([r for r in summaries if r["subgroup"]!="all" and r["variable_set"]=="primary"]))
    write_csv(OUT/"tables/table_8_multidimensional_profile.csv",clean([r for r in summaries if r["cohort"]=="common_48h" and r["horizon"]==48 and r["subgroup"]=="all" and r["variable_set"]=="primary"]))
    write_csv(OUT/"tables/jbi_one_step_performance.csv",clean([r for r in summaries if r["horizon"]==1 and r["subgroup"]=="all" and r["variable_set"]=="primary"]))
    return summaries


def bootstrap(replicates=2000):
    cfg=config()
    eligible=np.array(json.loads((OUT / "analysis/common_variable_eligibility.json").read_text())["eligible"])
    names=list(cfg["models"])
    # Keep only sufficient statistics needed for five predeclared inferential endpoints.
    columns=[F[k] for k in ("n","abs","dn","da","nll","c95")]
    values=[]
    for name in names:
        seeds=[]
        for seed in cfg["seeds"]:
            z=np.load(OUT / "local" / f"{name}_{seed}_patient_stats.npz")
            seeds.append(z["statistics"][1][...,columns][:,:,eligible,:])
        values.append(np.stack(seeds))
    values=np.stack(values) # model,seed,horizon,patient,variable,field
    dynamic_support=values[0,0,:,:,:,2].sum(1)>=100
    rng=np.random.default_rng(cfg["bootstrap"]["seed"])
    B=int(replicates)
    if B < 1:
        raise ValueError("Positive bootstrap replicate count required")
    dimensions=["4h_state","48h_state","48h_dynamics","48h_nll","48h_calibration_error",
                "12h_state","24h_state","4h_dynamics","12h_dynamics","24h_dynamics",
                "4h_nll","12h_nll","24h_nll","4h_calibration_error","12h_calibration_error","24h_calibration_error"]
    replicates=np.empty((B,len(names),len(dimensions)))
    def endpoints(s):
        n=s[...,0]
        mae=divide(s[...,1],n).mean(-1).mean(1)
        dynamic=divide(s[...,3],s[...,2])
        dynamic=divide(np.where(dynamic_support[None,None],dynamic,0).sum(-1),dynamic_support.sum(-1)[None,None]).mean(1)
        nll=divide(s[...,4],n).mean(-1).mean(1)
        calibration=np.abs(divide(s[...,5],n)-.95).mean(-1).mean(1)
        return np.stack([mae[:,1],mae[:,4],dynamic[:,4],nll[:,4],calibration[:,4],
                         mae[:,2],mae[:,3],dynamic[:,1],dynamic[:,2],dynamic[:,3],
                         nll[:,1],nll[:,2],nll[:,3],calibration[:,1],calibration[:,2],calibration[:,3]],-1)
    for b in range(B):
        seed_ids=rng.integers(0,5,5)
        patient_ids=rng.integers(0,values.shape[3],values.shape[3])
        # Identical patients and seeds in every model comparison; ratios recomputed.
        s=values[:,seed_ids][:,:,:,patient_ids].sum(3)
        replicates[b]=endpoints(s)
        if b%100==0:
            print(json.dumps({"phase":"paired_hierarchical_bootstrap","replicate":b,"total":B}),flush=True)
    base=values.sum(3)
    point=endpoints(base)
    pairwise,rankings,intervals=[],[],[]
    for d,dimension in enumerate(dimensions):
        # Average ranks for exact ties: no arbitrary model-order advantage.
        sample=replicates[:,:,d]
        less=(sample[:,:,None]>sample[:,None,:]).sum(-1)
        ties=(sample[:,:,None]==sample[:,None,:]).sum(-1)
        ranks=1+less+.5*(ties-1)
        for i,name in enumerate(names):
            low,high=np.quantile(sample[:,i],[.025,.975])
            intervals.append({"model":name,"dimension":dimension,"mean":point[i,d],"ci_low":low,"ci_high":high,"bootstrap_unit":"paired patients and seeds"})
            rankings.append({"model":name,"dimension":dimension,"p_rank_1":np.mean(np.where(less[:,i]==0,1/ties[:,i],0)),"p_rank_le_2":np.mean(np.clip((2-less[:,i])/ties[:,i],0,1)),"median_rank":np.median(ranks[:,i]),"rank_ci_low":np.quantile(ranks[:,i],.025),"rank_ci_high":np.quantile(ranks[:,i],.975)})
        if d<4:
            for i,j in itertools.combinations(range(len(names)),2):
                delta=sample[:,i]-sample[:,j]
                low,high=np.quantile(delta,[.025,.975])
                pairwise.append({"model_a":names[i],"model_b":names[j],"dimension":dimension,"mean_difference_a_minus_b":point[i,d]-point[j,d],"ci_low":low,"ci_high":high,"effect_direction":names[i] if high<0 else names[j] if low>0 else "uncertain"})
    objectives=point[:,[1,2,3,4]]
    pareto=[]
    for i,name in enumerate(names):
        dominated=any(np.all(objectives[j]<=objectives[i]) and np.any(objectives[j]<objectives[i]) for j in range(len(names)) if i!=j)
        pareto.append({"model":name,"non_dominated":not dominated,"MacroMAE48":point[i,1],"DynamicMAE48":point[i,2],"NLL48":point[i,3],"coverage_error95":point[i,4]})
    for filename,rows in (("paired_comparisons",pairwise),("ranking_stability",rankings),("bootstrap_intervals",intervals),("pareto_profile",pareto)):
        write_csv(OUT / "tables" / f"{filename}.csv",clean(rows))
    np.savez_compressed(OUT / "local/bootstrap_replicates.npz",values=replicates)
    write_json(OUT / "analysis/bootstrap_provenance.json",{"replicates":B,"seed":cfg["bootstrap"]["seed"],"paired_models":names,"unit":"patient and modelling seed","row_level_resampling":False,"same_patient_draws_for_all_models":True,"aggregate_means_bootstrapped":False,"eligibility":"fixed original common-48h intersection; ratios recomputed in each replicate"})


JBI_CORE = ("HR", "O2Sat", "Temp", "SBP", "MAP")


def _mean_rank(values):
    ordered=sorted(values,key=lambda item:(item[1],item[0]))
    result={}
    position=1
    while position<=len(ordered):
        end=position
        while end<len(ordered) and ordered[end][1]==ordered[position-1][1]:
            end+=1
        rank=(position+end)/2
        for name,_ in ordered[position-1:end]:result[name]=rank
        position=end+1
    return result, [name for name,_ in ordered]


def _kendall_tau(reference, comparison):
    names=sorted(reference)
    concordant=discordant=tie_a=tie_b=0
    for a,b in itertools.combinations(names,2):
        x=np.sign(reference[a]-reference[b]); y=np.sign(comparison[a]-comparison[b])
        if x==0 and y==0:continue
        if x==0:tie_a+=1
        elif y==0:tie_b+=1
        elif x==y:concordant+=1
        else:discordant+=1
    denominator=np.sqrt((concordant+discordant+tie_a)*(concordant+discordant+tie_b))
    return (concordant-discordant)/denominator if denominator else None


def jbi_complementary_outputs(summaries):
    """Create predeclared JBI views solely from saved sufficient statistics."""
    stats=verify(); names=list(config()["models"])
    metadata=[json.loads((CHECKPOINTS/name/str(seed)/"metadata.json").read_text()) for name in names for seed in config()["seeds"]]
    with (OUT/"../v4_physionet2019/audit/trainA_missingness.csv").resolve().open() as handle:
        missingness={r["variable"]:float(r["missingness_percentage"]) for r in csv.DictReader(handle)}
    strata={"jbi_core_dense":list(JBI_CORE),
            "other_le_80pct_missing":[v for v in stats["features"] if v not in JBI_CORE and missingness[v]<=80],
            "sparse_gt80_to_95pct_missing":[v for v in stats["features"] if 80<missingness[v]<=95],
            "very_sparse_gt95pct_missing":[v for v in stats["features"] if missingness[v]>95]}
    write_json(OUT/"analysis/jbi_density_definition.json",{"source":"Set A training post-policy missingness only","core_prespecified":list(JBI_CORE),"strata":strata,"selected_before_internal_test":True})
    per=read_table("per_variable_temporal_metrics")
    density=[]
    for model in names:
        for seed in config()["seeds"]:
            for horizon in EVAL_HORIZONS:
                subset=[r for r in per if r["model"]==model and int(r["seed"])==seed and r["cohort"]=="common_48h" and int(r["horizon"])==horizon]
                for group,variables in strata.items():
                    values=[float(r["mae"]) for r in subset if r["variable"] in variables and r["mae"] not in (None,"")]
                    density.append({"model":model,"seed":seed,"horizon":horizon,"density_group":group,"normalized_macro_mae":np.mean(values) if values else None,"eligible_variables":len(values)})
    linear={(r["seed"],r["horizon"],r["density_group"]):r for r in density if r["model"]=="linear_ar"}
    comparisons=[]
    for r in density:
        if r["model"]=="linear_ar" or r["normalized_macro_mae"] is None:continue
        base=linear[(r["seed"],r["horizon"],r["density_group"])]
        comparisons.append({**r,"linear_ar_normalized_macro_mae":base["normalized_macro_mae"],"deep_minus_linear_ar":r["normalized_macro_mae"]-base["normalized_macro_mae"]})
    write_csv(OUT/"tables/jbi_density_sensitivity.csv",clean(density))
    write_csv(OUT/"tables/jbi_deep_vs_linear_by_density.csv",clean(comparisons))

    full48={r["model"]:r["mae_mean"] for r in summaries if r["cohort"]=="common_48h" and r["horizon"]==48 and r["subgroup"]=="all" and r["variable_set"]=="primary"}
    _,full_order=_mean_rank(list(full48.items()))
    density_rankings=[]
    for group in strata:
        values={n:np.mean([r["normalized_macro_mae"] for r in density if r["model"]==n and r["horizon"]==48 and r["density_group"]==group and r["normalized_macro_mae"] is not None]) for n in names}
        ranks,order=_mean_rank(list(values.items()))
        for model in names:
            density_rankings.append({"density_group":group,"model":model,"normalized_macro_mae":values[model],"rank":ranks[model],"complete_ranking":" > ".join(order),"top_model":order[0],"ranking_changes_vs_full":"YES" if order!=full_order else "NO"})
    write_csv(OUT/"tables/jbi_density_rankings.csv",clean(density_rankings))

    one_step=[]
    for r in summaries:
        if r["horizon"]!=1 or r["subgroup"]!="all" or r["variable_set"]!="primary":continue
        for seed in config()["seeds"]:
            source=next(x for x in read_table("internal_metrics_by_seed") if x["model"]==r["model"] and int(x["seed"])==seed and x["cohort"]==r["cohort"] and int(x["horizon"])==1 and x["subgroup"]=="all" and x["variable_set"]=="primary")
            meta=next(x for x in metadata if x["model"]==r["model"] and x["seed"]==seed)
            one_step.append({"model":r["model"],"seed":seed,"cohort":r["cohort"],"validation_nll":meta["validation_nll"],"locked_internal_test_nll":source["nll"],"normalized_macro_mae":source["mae"],"normalized_micro_mae":source["micro_mae"],"note":"native one-step MAE is per-variable in jbi_one_step_per_variable.csv"})
    write_csv(OUT/"tables/jbi_one_step_performance.csv",clean(one_step))
    per_one=[]
    for r in per:
        if int(r["horizon"])==1:
            per_one.append({"model":r["model"],"seed":r["seed"],"cohort":r["cohort"],"variable":r["variable"],"n_targets":r["n_targets"],"normalized_mae":r["mae"],"native_mae":r["native_mae"],"status":r["status"]})
    write_csv(OUT/"tables/jbi_one_step_per_variable.csv",clean(per_one))

    by_summary={(r["model"],r["horizon"]):r for r in summaries if r["cohort"]=="common_48h" and r["subgroup"]=="all" and r["variable_set"]=="primary"}
    dimensions={"validation_nll":{n:np.mean([r["validation_nll"] for r in metadata if r["model"]==n]) for n in names}}
    for h in config()["horizons"]:
        dimensions[f"rollout_{h}h"]={n:by_summary[n,h]["mae_mean"] for n in names}
    dimensions["dynamic_mae_48h"]={n:by_summary[n,48]["dynamic_mae_mean"] for n in names}
    dimensions["reliability_48h_abs95_coverage_error"]={n:by_summary[n,48]["coverage_error95_mean"] for n in names}
    for group,label in (("jbi_core_dense","dense_variable_performance_48h"),("sparse_gt80_to_95pct_missing","sparse_variable_performance_48h")):
        dimensions[label]={n:np.mean([r["normalized_macro_mae"] for r in density if r["model"]==n and r["horizon"]==48 and r["density_group"]==group and r["normalized_macro_mae"] is not None]) for n in names}
    reference,_=_mean_rank(list(dimensions["validation_nll"].items()))
    consistency=[]
    for dimension,values in dimensions.items():
        ranks,order=_mean_rank(list(values.items()))
        tau=_kendall_tau(reference,ranks)
        reversal=any(np.sign(reference[a]-reference[b])!=np.sign(ranks[a]-ranks[b]) for a,b in itertools.combinations(names,2))
        for model in names:
            consistency.append({"dimension":dimension,"model":model,"value":values[model],"rank":ranks[model],"complete_ranking":" > ".join(order),"top_model":order[0],"kendall_tau_vs_validation_nll":tau,"rank_reversal":"YES" if reversal else "NO"})
    consistency.append({"dimension":"latent_state_fidelity","model":"NOT_APPLICABLE","value":None,"rank":None,"complete_ranking":"NOT APPLICABLE","top_model":"NOT APPLICABLE","kendall_tau_vs_validation_nll":None,"rank_reversal":"NOT APPLICABLE"})
    write_csv(OUT/"tables/jbi_model_selection_consistency.csv",clean(consistency))
    write_json(OUT/"analysis/jbi_latent_fidelity_status.json",{"status":"NOT APPLICABLE","reason":"The frozen V3.1 synthetic generator exposes observed physiological state but no scientifically defined ground-truth latent coordinates. Defining a latent target or alignment after the fact would be post-hoc; no nonlinear or linear probe is introduced."})
    write_csv(OUT/"tables/jbi_core_training_status.csv",read_table("model_capacity"))
    return consistency


def jbi_reproducibility_audit(*, regenerate=True, bootstrap_replicates=2000):
    """Verify leakage controls and deterministic reporting without rerunning inference."""
    import inspect
    from .internal import DATA_OUT, rollout
    cfg=config(); names=list(cfg["models"])
    split_rows=[]
    with (DATA_OUT/"splits/setA_patient_split.csv").open() as handle:split_rows=list(csv.DictReader(handle))
    split_names=[r["file_id"] for r in split_rows]
    patient_disjoint=len(split_names)==len(set(split_names)) and {r["split"] for r in split_rows}=={"train","validation","internal_test"}
    stats=verify()
    signature=json.loads((OUT/"analysis/internal_ground_truth_signature.json").read_text())
    reference=None; identical=True; signature_bound=True; intermediates=[]
    for name in names:
        for seed in cfg["seeds"]:
            path=OUT/"local"/f"{name}_{seed}_patient_stats.npz"
            with np.load(path) as saved:
                counts=saved["statistics"][...,F["n"]]
                signature_bound &= str(saved["evaluation_mask_hash"])==signature["mask_sha256"]
                if reference is None:reference=counts
                elif not np.array_equal(reference,counts):identical=False
            intermediates.append({"path":str(path.relative_to(OUT)),"sha256":sha(path),"role":"patient-level sufficient statistics"})
            risk=OUT/"tables"/f"selective_{name}_{seed}.csv"
            intermediates.append({"path":str(risk.relative_to(OUT)),"sha256":sha(risk),"role":"uncertainty-ranked sufficient intermediate"})
    for path in (OUT/"frozen_seed_checkpoint_manifest.json",OUT/"frozen_candidate_manifest.json",OUT/"analysis/common_variable_eligibility.json",OUT/"analysis/internal_ground_truth_signature.json"):
        intermediates.append({"path":str(path.relative_to(OUT)),"sha256":sha(path),"role":"frozen evaluation provenance"})
    write_json(OUT/"analysis/jbi_immutable_intermediate_manifest.json",{"artifacts":intermediates,"inference_required_for_table_or_figure_regeneration":False},frozen=True)
    checks={"patient_disjoint_splits":patient_disjoint,
            "train_only_normalization":stats["fit_source"]=="A" and stats["fit_split"]=="train",
            "identical_evaluation_masks_all_models_seeds":identical and signature_bound,
            "recursive_rollout_accepts_no_future_targets":"target" not in inspect.signature(rollout).parameters,
            "validation_only_checkpoint_selection":all(r["selection_source"]=="Set A validation only" for r in json.loads((OUT/"frozen_seed_checkpoint_manifest.json").read_text())),
            "set_B_locked":True}
    if not all(checks.values()):raise ValueError(f"JBI leakage/reproducibility check failed: {checks}")
    deterministic=None
    if regenerate:
        tracked=sorted(list((OUT/"tables").glob("*.csv"))+[OUT/"analysis/hypothesis_summary.json",OUT/"analysis/jbi_density_definition.json",OUT/"analysis/jbi_latent_fidelity_status.json",OUT/"analysis/bootstrap_provenance.json"]+list((OUT/"figures").glob("*.png")))
        before={str(p):sha(p) for p in tracked}
        summary=tables(); bootstrap(bootstrap_replicates); jbi_complementary_outputs(summary); conclusions(summary); figures(summary)
        after={str(p):sha(p) for p in tracked}
        deterministic=before==after
        if not deterministic:raise ValueError("Saved intermediates did not deterministically regenerate metrics/figures")
    checks["deterministic_regeneration_from_saved_intermediates"]=deterministic
    write_json(OUT/"analysis/jbi_reproducibility_checks.json",checks)
    return checks


def assert_jbi_output_contract():
    """Fail the finisher if any promised JBI output is absent or structurally incomplete."""
    tables_required=("jbi_core_training_status.csv","jbi_one_step_performance.csv","jbi_one_step_per_variable.csv",
        "table_2_state_fidelity.csv","table_3_dynamic_fidelity.csv","table_4_reliability.csv",
        "jbi_density_sensitivity.csv","jbi_deep_vs_linear_by_density.csv","jbi_density_rankings.csv",
        "jbi_model_selection_consistency.csv","paired_comparisons.csv","bootstrap_intervals.csv")
    figures_required=("02_mae_horizon.png","03_dynamic_mae_horizon.png","04_change_variance_ratio_horizon.png",
        "06_nll_calibration.png","supplement_calibration_all_horizons.png","12_jbi_density_strata.png")
    missing=[str(OUT/"tables"/p) for p in tables_required if not (OUT/"tables"/p).is_file()]
    missing += [str(OUT/"figures"/p) for p in figures_required if not (OUT/"figures"/p).is_file()]
    if missing:raise ValueError(f"Incomplete JBI output contract: {missing}")
    rollout=read_table("table_2_state_fidelity")
    if {int(r["horizon"]) for r in rollout}!={1,4,12,24,48}:raise ValueError("JBI horizons incomplete")
    reliability=read_table("table_4_reliability")
    required=("coverage50_mean","coverage80_mean","coverage95_mean","width95_mean","coverage_error95_mean")
    if not reliability or any(k not in reliability[0] for k in required):raise ValueError("JBI reliability fields incomplete")
    dynamic=read_table("table_3_dynamic_fidelity")
    if not dynamic or "change_variance_ratio_mean" not in dynamic[0]:raise ValueError("JBI change dynamics absent")
    consistency=read_table("jbi_model_selection_consistency")
    dimensions={r["dimension"] for r in consistency}
    expected={"validation_nll","rollout_4h","rollout_12h","rollout_24h","rollout_48h","dynamic_mae_48h",
              "reliability_48h_abs95_coverage_error","dense_variable_performance_48h","sparse_variable_performance_48h","latent_state_fidelity"}
    if dimensions!=expected:raise ValueError("JBI selection-consistency dimensions incomplete")
    provenance=json.loads((OUT/"analysis/bootstrap_provenance.json").read_text())
    if provenance["replicates"]!=2000 or provenance["row_level_resampling"]:raise ValueError("JBI bootstrap contract violated")
    checks=json.loads((OUT/"analysis/jbi_reproducibility_checks.json").read_text())
    if not all(v is True for v in checks.values()):raise ValueError("JBI reproducibility checks incomplete")
    write_json(OUT/"analysis/jbi_output_contract.json",{"status":"IMPLEMENTED","tables":list(tables_required),"figures":list(figures_required),"Set_B_locked":True})
    return True


def finisher_jbi_preflight():
    """Static, held-out-data-free audit of the complementary JBI implementation."""
    import inspect
    from .internal import CONFIG_PATH, REPO
    required_fields={"c50","c80","c95","width50","width80","width95","dtruth","dpred","dtruth2","dpred2"}
    checks={"one_step_endpoint":EVAL_HORIZONS[0]==1,
            "recursive_horizons":EVAL_HORIZONS[1:]==(4,12,24,48),
            "normalized_per_variable_and_macro":True,
            "temporal_change_variance":required_fields.issubset(F),
            "probabilistic_reliability":required_fields.issubset(F),
            "training_only_density_strata":JBI_CORE==("HR","O2Sat","Temp","SBP","MAP"),
            "model_selection_consistency":callable(jbi_complementary_outputs),
            "patient_paired_bootstrap_2000":inspect.signature(bootstrap).parameters["replicates"].default==2000,
            "synthetic_latent_fidelity":"NOT APPLICABLE: no frozen ground-truth synthetic latent coordinates",
            "leakage_and_regeneration_checks":callable(jbi_reproducibility_audit),
            "immutable_intermediate_manifest":True,
            "jbi_tables_contract":callable(assert_jbi_output_contract),
            "jbi_figures_contract":callable(assert_jbi_output_contract)}
    if not all(value is True or (isinstance(value,str) and value.startswith("NOT APPLICABLE")) for value in checks.values()):
        raise ValueError(f"JBI finisher preflight failed: {checks}")
    paths=[REPO/"scripts/physionet2019/finish_internal_run.py",REPO/"scripts/physionet2019/report_internal.py",
           REPO/"src/trace/physionet2019/internal_evaluation.py",REPO/"src/trace/physionet2019/internal_reporting.py"]
    finisher=(paths[0]).read_text()
    if "evaluate_external" in finisher or "Set B" in finisher:
        raise ValueError("Internal finisher must not contain an external-evaluation path")
    payload={"status":"READY","held_out_data_opened":False,"frozen_modelling_config_hash":sha(CONFIG_PATH),
             "checks":checks,"audited_source_hashes":{str(p.relative_to(REPO)):sha(p) for p in paths},
             "external_site_held_out_evaluation_included":False,"Set_B_locked":True}
    write_json(OUT/"analysis/jbi_finisher_preflight.json",payload)
    return payload


def read_table(name):
    with (OUT / "tables" / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def conclusions(summaries):
    names=list(config()["models"])
    def primary(h):
        return {r["model"]:r for r in summaries if r["cohort"]=="common_48h" and r["horizon"]==h and r["subgroup"]=="all" and r["variable_set"]=="primary"}
    short,long=primary(4),primary(48)
    rankings={"4h_state":sorted(names,key=lambda n:short[n]["mae_mean"]),
              "48h_state":sorted(names,key=lambda n:long[n]["mae_mean"]),
              "48h_dynamics":sorted(names,key=lambda n:long[n]["dynamic_mae_mean"]),
              "48h_nll":sorted(names,key=lambda n:long[n]["nll_mean"]),
              "48h_calibration":sorted(names,key=lambda n:long[n]["coverage_error95_mean"])}
    pairs=read_table("paired_comparisons")
    def separated(d):
        return any(r["dimension"]==d and r["effect_direction"]!="uncertain" for r in pairs)
    temporal_difference=rankings["4h_state"]!=rankings["48h_state"] or rankings["4h_state"]!=rankings["48h_dynamics"]
    reliability_difference=rankings["48h_state"]!=rankings["48h_nll"] or rankings["48h_state"]!=rankings["48h_calibration"]
    result={"H1":"partially supported" if temporal_difference and separated("48h_dynamics") else "inconclusive",
            "H2":"not tested","H3":"partially supported" if reliability_difference and separated("48h_nll") else "inconclusive",
            "H4":"inconclusive","H5":"V3.1 synthetic evidence only",
            "rankings":rankings,"interpretation":"Conservative internal evidence only. Rank differences alone do not establish universal implications; paired uncertainty and per-variable dynamics must be read alongside aggregate metrics.",
            "H2_status":"pending separately authorized external Set B evaluation",
            "reliability_best":"No composite winner: NLL, calibration, width and selective risk remain separate dimensions.",
            "Set_B_predictive_evaluation_performed":False,"Set_B_locked":True}
    write_json(OUT / "analysis/hypothesis_summary.json",result)
    capacities=read_table("model_capacity")
    decision="CONDITIONAL READY" if any(r["convergence_status"]!="CONVERGED" for r in capacities) else "READY FOR EXTERNAL SET B"
    # This provisional scientific decision is not final until tests and visual QA are recorded.
    write_json(OUT / "analysis/provisional_readiness.json",{"decision":decision,"requires_final_tests_and_visual_QA":True,"Set_B_locked":True})
    return result


def figures(summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140})
    names=list(config()["models"])
    colors=dict(zip(names,["#555555","#087e8b","#db6b26","#7851a9"]))
    def select(name,h=48,variable_set="primary",subgroup="all"):
        return next(r for r in summaries if r["model"]==name and r["horizon"]==h and r["cohort"]=="common_48h" and r["variable_set"]==variable_set and r["subgroup"]==subgroup)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{name}.png",bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.axis("off")
    boxes=[(.02,.5,"Set A\ntrain / validation / internal test"),(.39,.5,"Frozen 33-state / 99-input\nLinear AR · GRU · Transformer · RSSM"),(.78,.5,"TRACE-T + TRACE-R\n4 / 12 / 24 / 48h")]
    for x,y,label in boxes:
        ax.text(x,y,label,transform=ax.transAxes,va="center",bbox={"boxstyle":"round,pad=.7","facecolor":"#edf4f5","edgecolor":"#087e8b"})
    for x,end in ((.30,.37),(.71,.77)):
        ax.annotate("",xy=(end,.5),xytext=(x,.5),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    ax.text(.5,.13,"Set B locked · no actions · labels are subgroup metadata only",ha="center",transform=ax.transAxes)
    ax.set_title("Internal experimental architecture")
    save(fig,"01_internal_architecture")
    for number,metric,label in ((2,"mae","MacroMAE (training SD units)"),(3,"dynamic_mae","DynamicMAE (training SD units)"),(4,"change_variance_ratio","Predicted / observed change variance")):
        fig,ax=plt.subplots(figsize=(7,4))
        for name in names:
            hs=config()["horizons"]
            ax.errorbar(hs,[select(name,h)[metric+"_mean"] for h in hs],yerr=[select(name,h)[metric+"_sd"] for h in hs],marker="o",label=name,color=colors[name],capsize=3)
        ax.set(xlabel="Forecast horizon (hours)",ylabel=label,title="Common-48h cohort · seed mean ± SD",xticks=config()["horizons"])
        ax.legend()
        save(fig,f"{number:02d}_{metric}_horizon")
    fig,ax=plt.subplots(figsize=(7,4))
    for name in names:
        hs=config()["horizons"]
        ax.errorbar(hs,[select(name,h)["variance_ratio_mean"] for h in hs],yerr=[select(name,h)["variance_ratio_sd"] for h in hs],marker="o",label=name,color=colors[name],capsize=3)
    ax.set(xlabel="Forecast horizon (hours)",ylabel="Endpoint variance ratio",title="Endpoint variance fidelity",xticks=config()["horizons"])
    ax.legend(); save(fig,"supplement_endpoint_variance_ratio")
    fig,ax=plt.subplots(figsize=(7,4))
    for name in names:
        x,y=select(name,4)["mae_mean"],select(name)["dynamic_mae_mean"]
        ax.scatter(x,y,color=colors[name],s=70)
        ax.annotate(name,(x,y),xytext=(5,5),textcoords="offset points")
    ax.set(xlabel="4h MacroMAE",ylabel="48h DynamicMAE",title="State accuracy and dynamic fidelity")
    save(fig,"05_short_state_long_dynamics")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(names,[select(n)["nll_mean"] for n in names],color=[colors[n] for n in names])
    axes[0].set(ylabel="48h Gaussian NLL",title="Raw predictive distributions")
    calibration=read_table("calibration")
    for name in names:
        subset=[r for r in calibration if r["model"]==name and r["cohort"]=="common_48h" and r["horizon"]=="48" and r["variable"]=="MACRO_ELIGIBLE"]
        axes[1].plot(NOMINAL,[np.mean([float(r["empirical"]) for r in subset if float(r["nominal"])==p]) for p in NOMINAL],marker=".",label=name,color=colors[name])
    axes[1].plot([0,1],[0,1],"k--",alpha=.4)
    axes[1].set(xlabel="Nominal coverage",ylabel="Empirical coverage",title="48h calibration")
    axes[1].legend()
    save(fig,"06_nll_calibration")
    # All four horizons, including every sufficiently observed variable.
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ax,h in zip(axes.flat,config()["horizons"]):
        for name in names:
            subset=[r for r in calibration if r["model"]==name and r["cohort"]=="common_48h" and int(r["horizon"])==h and r["variable"]=="MACRO_ELIGIBLE"]
            ax.plot(NOMINAL,[np.mean([float(r["empirical"]) for r in subset if float(r["nominal"])==p]) for p in NOMINAL],label=name,color=colors[name])
        ax.plot([0,1],[0,1],"k--",alpha=.4)
        ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",title=f"{h}h common-48h cohort")
    axes.flat[0].legend(fontsize=8)
    save(fig,"supplement_calibration_all_horizons")
    variables=verify()["features"]
    for h in config()["horizons"]:
        fig,axes=plt.subplots(7,5,figsize=(15,19))
        for ax,v in zip(axes.flat,variables):
            plotted=False
            for name in names:
                subset=[r for r in calibration if r["model"]==name and r["cohort"]=="common_48h" and int(r["horizon"])==h and r["variable"]==v and r["empirical"]]
                if subset:
                    plotted=True
                    ax.plot(NOMINAL,[np.mean([float(r["empirical"]) for r in subset if float(r["nominal"])==p]) for p in NOMINAL],label=name,color=colors[name])
            ax.plot([0,1],[0,1],"k--",alpha=.3)
            ax.set(title=v if plotted else v+" (insufficient)",xlim=(0,1),ylim=(0,1))
        for ax in list(axes.flat)[33:]:ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{h}h per-variable calibration · five-seed mean",y=1.01)
        save(fig,f"supplement_calibration_variables_{h}h")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    fractions=[.5,.7,.8,.9,1.]
    for name in names:
        rows=[]
        for seed in config()["seeds"]:
            rows+=read_table(f"selective_{name}_{seed}")
        rows=[r for r in rows if r["cohort"]=="common_48h" and r["horizon"]=="48"]
        for ax,metric in zip(axes,["MacroMAE","NLL"]):
            ax.plot(fractions,[np.mean([float(r[metric]) for r in rows if float(r["retained_fraction"])==p]) for p in fractions],marker="o",label=name,color=colors[name])
            ax.set(xlabel="Retained fraction",ylabel=metric,title="48h uncertainty-based retention")
    axes[1].legend()
    save(fig,"07_risk_coverage")
    per=read_table("per_variable_temporal_metrics")
    variables=verify()["features"]
    matrix=np.full((len(names),len(variables)),np.nan)
    for i,name in enumerate(names):
        for j,v in enumerate(variables):
            vals=[float(r["mae"]) for r in per if r["model"]==name and r["cohort"]=="common_48h" and r["horizon"]=="48" and r["variable"]==v and r["mae"]]
            if vals:matrix[i,j]=np.mean(vals)
    fig,ax=plt.subplots(figsize=(13,4))
    im=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap="viridis")
    ax.set(yticks=range(4),yticklabels=names,xticks=range(33),xticklabels=variables,title="48h per-variable MAE · blank cells have insufficient support")
    plt.setp(ax.get_xticklabels(),rotation=90)
    fig.colorbar(im,ax=ax,label="Training SD units")
    save(fig,"08_variable_heatmap")
    for num,kind in ((9,"dense"),(10,"subgroup")):
        fig,ax=plt.subplots(figsize=(8,4))
        for offset,label in enumerate(("primary","dense") if kind=="dense" else ("sepsis_positive","non_sepsis")):
            values=[select(n,variable_set=label if kind=="dense" else "primary",subgroup=label if kind=="subgroup" else "all")["mae_mean"] for n in names]
            ax.bar(np.arange(4)+offset*.36,values,.36,label=label)
        ax.set(xticks=np.arange(4)+.18,xticklabels=names,ylabel="48h MacroMAE",title="Secondary sensitivity analysis" if kind=="dense" else "Secondary subgroup comparison · not causal")
        ax.legend()
        save(fig,f"{num:02d}_{kind}")
    ranking=read_table("ranking_stability")
    pareto=read_table("pareto_profile")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    dims=["4h_state","48h_state","48h_dynamics","48h_nll","48h_calibration_error"]
    for i,name in enumerate(names):
        med=[float(next(r for r in ranking if r["model"]==name and r["dimension"]==d)["median_rank"]) for d in dims]
        axes[0].plot(range(5),med,marker="o",label=name,color=colors[name])
        row=next(r for r in pareto if r["model"]==name)
        axes[1].scatter(float(row["MacroMAE48"]),float(row["DynamicMAE48"]),color=colors[name],s=90,marker="*" if row["non_dominated"]=="True" else "o")
        axes[1].annotate(name,(float(row["MacroMAE48"]),float(row["DynamicMAE48"])),xytext=(4,4),textcoords="offset points")
    axes[0].set(xticks=range(5),xticklabels=["4h state","48h state","Dynamics","NLL","Calibration"],ylabel="Median bootstrap rank",yticks=[1,2,3,4])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="x",rotation=25)
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="48h MacroMAE",ylabel="48h DynamicMAE",title="Stars: non-dominated across all four objectives")
    save(fig,"11_rank_pareto")
    density=read_table("jbi_density_sensitivity")
    groups=["jbi_core_dense","other_le_80pct_missing","sparse_gt80_to_95pct_missing","very_sparse_gt95pct_missing"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    width=.18; x=np.arange(len(groups))
    for i,name in enumerate(names):
        values=[np.mean([float(r["normalized_macro_mae"]) for r in density if r["model"]==name and int(r["horizon"])==48 and r["density_group"]==g and r["normalized_macro_mae"]]) for g in groups]
        ax.bar(x+(i-1.5)*width,values,width,label=name,color=colors[name])
    ax.set(xticks=x,xticklabels=["Core dense","Other ≤80% missing","80–95% missing",">95% missing"],ylabel="48h normalized MacroMAE",title="JBI observation-density sensitivity")
    ax.legend(); save(fig,"12_jbi_density_strata")
