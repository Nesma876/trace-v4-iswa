"""Frozen Set B reporting and transportability classification."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from . import internal_reporting as ir
from .external_evaluation import OUT
from .internal import REPO, config
from .internal_evaluation import F, divide
from .pipeline import sha, write_csv, write_json


def _rows(name, root=OUT):
    with (root/"tables"/name).open(newline="",encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _f(value):
    return float(value) if value not in (None, "") else np.nan


def _density_bootstrap(replicates=2000):
    cfg=config(); names=list(cfg["models"]); seeds=cfg["seeds"]
    definition=json.loads((REPO/"outputs/v4_internal/analysis/jbi_density_definition.json").read_text())
    features=json.loads((REPO/"outputs/v4_physionet2019/audit/frozen_training_stats.json").read_text())["features"]
    groups={g:np.array([features.index(v) for v in vs]) for g,vs in definition["strata"].items()}
    eligible=np.asarray(json.loads((OUT/"analysis/common_variable_eligibility.json").read_text())["eligible"],bool)
    values=[]
    for name in names:
        per_seed=[]
        for seed in seeds:
            raw=np.load(OUT/"local"/f"{name}_{seed}_patient_stats.npz")["statistics"][1,4]
            per_seed.append(raw[..., [F["n"],F["abs"]]])
        values.append(np.stack(per_seed))
    values=np.stack(values) # model, seed, patient, variable, field
    rng=np.random.default_rng(cfg["bootstrap"]["seed"]); samples={g:np.empty((replicates,len(names))) for g in groups}
    for b in range(replicates):
        si=rng.integers(0,len(seeds),len(seeds)); pi=rng.integers(0,values.shape[2],values.shape[2])
        weights=np.bincount(pi,minlength=values.shape[2]).astype(values.dtype)
        s=(values[:,si]*weights[None,None,:,None,None]).sum(2)
        for g,idx in groups.items():
            valid=idx[eligible[idx]]
            samples[g][b]=divide(s[...,valid,1],s[...,valid,0]).mean(-1).mean(1)
    ranking=_rows("jbi_density_rankings.csv")
    out=[]
    for group,sample in samples.items():
        selected=next(r["top_model"] for r in ranking if r["density_group"]==group)
        comparisons=[(n,"linear_ar") for n in names if n!="linear_ar"]
        comparisons += [(a,b) for a in names for b in names if a<b and (a==selected or b==selected) and "linear_ar" not in (a,b)]
        seen=set()
        for a,b in comparisons:
            if (a,b) in seen: continue
            seen.add((a,b)); d=sample[:,names.index(a)]-sample[:,names.index(b)]
            point=next(float(r["normalized_macro_mae"]) for r in ranking if r["density_group"]==group and r["model"]==a)-next(float(r["normalized_macro_mae"]) for r in ranking if r["density_group"]==group and r["model"]==b)
            lo,hi=np.quantile(d,[.025,.975])
            out.append({"density_group":group,"comparison":f"{a}_minus_{b}","mean_difference":point,
                        "ci_low":lo,"ci_high":hi,"favored":a if hi<0 else b if lo>0 else "uncertain",
                        "resamples":replicates,"unit":"paired Set B patients and modelling seeds"})
    write_csv(OUT/"tables/E6_density_paired_ci.csv",ir.clean(out))


def _copy_tables():
    mapping={
        "E2_one_step_performance.csv":"jbi_one_step_performance.csv",
        "E3_rollout_fidelity.csv":"table_2_state_fidelity.csv",
        "E4_dynamic_fidelity.csv":"table_3_dynamic_fidelity.csv",
        "E5_reliability.csv":"table_4_reliability.csv",
        "E6_density_sensitivity.csv":"jbi_density_sensitivity.csv",
        "E7_model_selection_consistency.csv":"jbi_model_selection_consistency.csv"}
    for dst,src in mapping.items():
        if dst=="E2_one_step_performance.csv":
            rows=_rows(src)
            for row in rows:
                row["set_b_nll"]=row.pop("locked_internal_test_nll")
                row["cohort"]=row["cohort"].replace("full_available","set_b_full_available").replace("common_48h","set_b_common_48h")
            write_csv(OUT/"tables"/dst,rows)
        else:
            shutil.copyfile(OUT/"tables"/src,OUT/"tables"/dst)


def _replication_matrix():
    ext_summary=_rows("internal_summary.csv")
    ext_consistency=_rows("jbi_model_selection_consistency.csv")
    ext_pair=_rows("paired_comparisons.csv")
    ext_density=_rows("jbi_density_rankings.csv")
    def summary(model,h):
        return next(r for r in ext_summary if r["model"]==model and r["cohort"]=="common_48h" and int(r["horizon"])==h and r["subgroup"]=="all" and r["variable_set"]=="primary")
    def order(dim): return next(r["complete_ranking"] for r in ext_consistency if r["dimension"]==dim)
    def pair(a,b,dim):
        r=next(r for r in ext_pair if r["model_a"]==a and r["model_b"]==b and r["dimension"]==dim)
        return _f(r["ci_low"]),_f(r["ci_high"])
    val_order=order("validation_nll"); rollout_order=order("rollout_48h")
    dyn_ci=pair("transformer","rssm","48h_dynamics")
    reliability_top=order("reliability_48h_abs95_coverage_error").split(" > ")[0]
    density_tops={r["density_group"]:r["top_model"] for r in ext_density}
    linear=[_f(summary("linear_ar",h)["mae_mean"]) for h in (4,12,24,48)]
    neural_vr=[_f(summary(n,48)["change_variance_ratio_mean"]) for n in ("gru","transformer","rssm")]
    linear_vr=_f(summary("linear_ar",48)["change_variance_ratio_mean"])
    checks=[
      ("Transformer best predictive NLL","Transformer ranks first by frozen Set A validation NLL",val_order.split(" > ")[0]=="transformer",f"frozen validation ordering: {val_order}"),
      ("48 h rollout ordering differs from validation-NLL ordering","Internal complete rankings differ",rollout_order!=val_order,f"Set B rollout: {rollout_order}; frozen validation: {val_order}"),
      ("RSSM > Transformer on 48 h DynamicMAE","Internal paired CI excludes zero in favor of RSSM",dyn_ci[0]>0,f"Transformer-RSSM paired CI {dyn_ci}"),
      ("Transformer strongest reliability","Internal frozen absolute-95%-coverage-error ranking",reliability_top=="transformer",f"Set B reliability top: {reliability_top}"),
      ("Density-dependent architecture selection","Internal top architecture changes by frozen density stratum",len(set(density_tops.values()))>1,f"Set B tops: {density_tops}"),
      ("Linear AR long-horizon degradation","Internal rollout error increases strongly with horizon",linear[-1]>linear[0],f"Set B Linear AR 4h={linear[0]:.6g}, 48h={linear[-1]:.6g}"),
      ("Neural variance contraction","Internal neural change-variance ratios below one",all(x<1 for x in neural_vr),f"Set B 48h ratios: {neural_vr}"),
      ("Linear AR variance inflation","Internal Linear AR change-variance ratio above one",linear_vr>1,f"Set B 48h ratio: {linear_vr:.6g}")]
    rows=[{"Internal finding":a,"Internal support":b,"Set B result":c,"Replicated":"YES" if ok else "NO","Comment":c} for a,b,ok,c in checks]
    target=REPO/"paper/jbi/SET_B_REPLICATION_MATRIX.csv"; write_csv(target,rows); write_csv(OUT/"tables/E8_replication_matrix.csv",rows)
    central=sum(r["Replicated"]=="YES" for r in rows)
    primary=[r["Replicated"]=="YES" for r in rows[:5]]
    strong=sum(primary)>=4 and primary[1] and primary[2] and (primary[3] or primary[4])
    classification="G1" if strong else "G2" if central>=3 else "G3"
    write_json(OUT/"analysis/external_generalization.json",{"classification":classification,"replicated":central,"total":len(rows),
               "rule":"G1 if >=4/5 central findings including rollout disagreement, dynamics, and density or reliability; G2 if >=3/8 total; otherwise G3",
               "internal_scenario":"B (unchanged)"})
    return classification


def report():
    ir.OUT=OUT
    summaries=ir.tables(); ir.bootstrap(2000); ir.jbi_complementary_outputs(summaries); ir.figures(summaries)
    _density_bootstrap(2000); _copy_tables(); classification=_replication_matrix()
    artifacts=[]
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="immutable_manifest.json"):
        artifacts.append({"path":str(path.relative_to(OUT)),"sha256":sha(path)})
    write_json(OUT/"analysis/immutable_manifest.json",{"artifacts":artifacts,"inference_required_for_regeneration":False,
               "patient_level_sufficient_statistics_saved":True,"classification":classification},frozen=True)
    return classification
