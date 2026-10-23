"""One-shot Set B evaluation using the frozen Set A protocol and checkpoints."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import read_psv
from .internal import CHECKPOINTS, CONFIG_PATH, DATA_OUT, REPO, config, make_model, rollout, verify
from .internal_evaluation import EVAL_HORIZONS, F, FIELDS, add_statistics, divide
from .pipeline import sha, write_csv, write_json
from .policy import model_values
from .preprocessing import transform_prefix

OUT = REPO / "outputs/v4_external"
SET_B = REPO / "data_external/physionet2019/training_setB"


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def preflight():
    """No Set B file is opened here."""
    stats = verify()
    event = OUT / "analysis/external_evaluation_event.json"
    if event.exists() or OUT.exists():
        raise FileExistsError("Set B output/event already exists; refusing a second evaluation")
    candidates = json.loads((REPO / "outputs/v4_internal/frozen_candidate_manifest.json").read_text())
    seeds = json.loads((REPO / "outputs/v4_internal/frozen_seed_checkpoint_manifest.json").read_text())
    if len(candidates) != 4 or len(seeds) != 20:
        raise ValueError("Frozen checkpoint manifests are incomplete")
    for row in seeds:
        path = CHECKPOINTS / row["checkpoint"]
        if row["config_hash"] != sha(CONFIG_PATH) or sha(path) != row["checkpoint_hash"]:
            raise ValueError("Frozen checkpoint/config hash mismatch")
        if row["selection_source"] != "Set A validation only":
            raise ValueError("Checkpoint selection was not validation-only")
    if stats["fit_source"] != "A" or stats["fit_split"] != "train":
        raise ValueError("Preprocessing is not Set A training-derived")
    internal = json.loads((REPO / "outputs/v4_internal/analysis/evaluation_integrity.json").read_text())
    if internal["Set_B_predictive_evaluation_performed"] or not internal["Set_B_locked"]:
        raise ValueError("Prior Set B access is indicated")
    return {"config_hash": sha(CONFIG_PATH), "preprocessing_hash": sha(DATA_OUT / "audit/frozen_training_stats.json"),
            "checkpoint_manifest_hash": sha(REPO / "outputs/v4_internal/frozen_seed_checkpoint_manifest.json")}


def _load_set_b(stats):
    names = sorted(p.name for p in SET_B.glob("*.psv"))
    if len(names) != 20000:
        raise ValueError(f"Expected 20000 Set B patients, found {len(names)}")
    patients, observed = [], np.zeros(33, dtype=np.int64)
    rows = 0
    for index, name in enumerate(names):
        raw = read_psv(SET_B / name)
        values = model_values(raw, stats)
        mask = np.isfinite(values)
        x = transform_prefix(raw, stats)
        y = np.where(mask, (values-stats["means"])/stats["stds"], 0).astype(np.float32)
        patients.append((x, y, mask, bool(np.any(raw[:, -1] == 1))))
        observed += mask.sum(0); rows += len(raw)
        if index % 2000 == 0:
            print(json.dumps({"phase":"load_Set_B","patients_done":index+1,"patients_total":len(names)}), flush=True)
    write_csv(OUT / "tables/E1_set_b_cohort_missingness.csv",
              [{"variable":v,"patients":len(names),"rows":rows,"observed":int(observed[j]),
                "missingness_percentage":100*(1-observed[j]/rows)} for j,v in enumerate(stats["features"])])
    return patients


def run_once():
    event = OUT / "analysis/external_evaluation_event.json"
    recovery = event.exists() and not any((OUT/"local").glob("*.npz")) and not (OUT/"tables/E1_set_b_cohort_missingness.csv").exists()
    if recovery:
        receipt=json.loads(event.read_text())
        if receipt.get("status") != "started":
            raise FileExistsError("Set B event is not an eligible pre-inference interruption")
        receipt.update(status="resumed_after_aborted_data_load",recovery_unix=time.time(),
                       recovery_reason="automated preflight process interrupted before cohort table or inference artifacts")
        event.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    else:
        receipt = preflight()
        for folder in (OUT, OUT/"analysis", OUT/"local", OUT/"tables", OUT/"figures"):
            folder.mkdir(parents=True, exist_ok=True)
        with event.open("x", encoding="utf-8") as handle:
            json.dump({**receipt,"status":"started","started_unix":time.time(),"prior_evaluation":False},handle,indent=2)
    try:
        stats=verify(); patients=_load_set_b(stats)
        subgroup=np.array([p[3] for p in patients]); origin_count=sum(max(0,len(p[0])-7) for p in patients)
        mask_hash=hashlib.sha256(); truth_hash=hashlib.sha256(); set_b_counts={str(h):np.zeros(33,dtype=np.int64) for h in config()["horizons"]}
        for _,y,m,_ in patients:
            truth_hash.update(np.ascontiguousarray(y).view(np.uint8)); mask_hash.update(np.ascontiguousarray(m).view(np.uint8))
            for origin in range(3,len(y)-4):
                if len(y)-origin-1>=48:
                    for h in config()["horizons"]: set_b_counts[str(h)]+=m[origin+h]
        signature={"processed_truth_stream_sha256":truth_hash.hexdigest(),"processed_mask_stream_sha256":mask_hash.hexdigest(),
                   "mask_sha256":mask_hash.hexdigest(),"subgroup_sha256":_digest(subgroup),
                   "patient_count":int(len(subgroup)),"origin_count":int(origin_count),"streamed_without_cross_patient_windows":True}
        write_json(OUT/"analysis/external_ground_truth_signature.json",signature,frozen=True)
        internal_elig=json.loads((REPO/"outputs/v4_internal/analysis/common_variable_eligibility.json").read_text())
        eligible=np.asarray(internal_elig["eligible"],bool)
        write_json(OUT/"analysis/common_variable_eligibility.json",{**internal_elig,"source":"frozen internal/common-48h definition; not refit on Set B",
                   "set_b_counts":{h:v.tolist() for h,v in set_b_counts.items()}},frozen=True)
        for name in config()["models"]:
            for seed in config()["seeds"]:
                folder=CHECKPOINTS/name/str(seed); checkpoint=folder/("model.npz" if name=="linear_ar" else "model.pt")
                model=make_model(name)
                if name=="linear_ar":
                    weights=np.load(checkpoint); model.weights,model.variance=weights["weights"],weights["variance"]
                else:
                    model.load_state_dict(torch.load(checkpoint,weights_only=True)); model.eval()
                sufficient=np.zeros((2,len(EVAL_HORIZONS),len(subgroup),33,len(FIELDS)),dtype=np.float32)
                started=time.perf_counter()
                done=0
                for pstart in range(0,len(patients),64):
                    contexts=[]; truths=[]; masks=[]; baselines=[]; baseline_masks=[]; owners=[]; available=[]
                    for pid,(px,py,pmask,_) in enumerate(patients[pstart:pstart+64],pstart):
                        for origin in range(3,len(px)-4):
                            h=min(48,len(px)-origin-1); contexts.append(px[origin-3:origin+1])
                            truths.append(np.pad(py[origin+1:origin+1+h],((0,48-h),(0,0))))
                            masks.append(np.pad(pmask[origin+1:origin+1+h],((0,48-h),(0,0))))
                            baselines.append(py[origin]); baseline_masks.append(pmask[origin]); owners.append(pid); available.append(h)
                    if not contexts: continue
                    bx=torch.from_numpy(np.asarray(contexts)); truth=np.asarray(truths); mask=np.asarray(masks)
                    baseline=np.asarray(baselines); bm=np.asarray(baseline_masks); owner=np.asarray(owners); available=np.asarray(available)
                    for start in range(0,len(bx),512):
                        end=min(start+512,len(bx)); context=bx[start:end].clone()
                        if name=="linear_ar":
                            xp=context.numpy(); predictions=[]; scales=[]
                            for _ in range(48):
                                mu,lv=model.predict(xp[:,-1]); predictions.append(mu); scales.append(np.exp(.5*lv))
                                elapsed=xp[:,-1,66:]; row=np.concatenate([mu,np.zeros_like(mu),np.where(elapsed>=0,elapsed+1,elapsed)],-1)
                                xp=np.concatenate([xp[:,1:],row[:,None]],1)
                            pred,sd=np.stack(predictions,1),np.stack(scales,1)
                        else:
                            pm,pv=rollout(model,context,48); pred,sd=pm.numpy().astype(float),pv.sqrt().numpy().astype(float)
                        for ci in range(2):
                            for hi,h in enumerate(EVAL_HORIZONS):
                                keep=available[start:end]>=(48 if ci else h)
                                if keep.any():
                                    ids=np.arange(start,end)[keep]
                                    add_statistics(sufficient[ci,hi],owner[ids],truth[ids,:h],pred[keep,:h],sd[keep,:h],mask[ids,:h],baseline[ids],bm[ids])
                    done+=len(bx)
                    if pstart%512==0:
                        print(json.dumps({"phase":"Set_B_inference","model":name,"seed":seed,"origins_done":done,"origins_total":origin_count}),flush=True)
                np.savez_compressed(OUT/"local"/f"{name}_{seed}_patient_stats.npz",statistics=sufficient,subgroup=subgroup,
                    fields=np.array(FIELDS),inference_time=time.perf_counter()-started,config_hash=sha(CONFIG_PATH),
                    checkpoint_hash=sha(checkpoint),evaluation_mask_hash=signature["mask_sha256"])
        write_json(OUT/"analysis/evaluation_integrity.json",{**receipt,"Set_B_predictive_evaluation_performed":True,
                   "frozen_preprocessing_applied":True,"frozen_metrics_applied":True,"patients":len(subgroup),"origins":origin_count})
        record=json.loads(event.read_text()); record.update(status="completed",completed_unix=time.time()); write_json(event,record)
    except Exception as exc:
        record=json.loads(event.read_text()); record.update(status="failed; explicit review required before any repeat",error=str(exc)); write_json(event,record)
        raise
