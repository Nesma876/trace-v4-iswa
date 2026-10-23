"""Frozen-candidate, observed-only internal evaluation with patient sufficient statistics."""
import json
import hashlib
import math
import shutil
import time
from statistics import NormalDist

import numpy as np
import torch

from .internal import (OUT, CHECKPOINTS, DATA_OUT, CONFIG_PATH, config, verify,
                       load_patients, make_model, rollout)
from .pipeline import sha, write_json, write_csv

EVAL_HORIZONS = (1, 4, 12, 24, 48)
FIELDS = ["n", "abs", "sq", "nll", "crps", "c50", "c80", "c90", "c95",
          "width50", "width80", "width95", "u", "u2", "e2", "ue",
          "truth", "pred", "truth2", "pred2", "dn", "da", "dtruth", "dpred",
          "dtruth2", "dpred2", "bn", "ba", "cn", "cs"]
NOMINAL = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
FIELDS += [f"cal{p}" for p in NOMINAL]
F = {v: j for j, v in enumerate(FIELDS)}


def freeze_candidates():
    stats = verify()
    manifest = []
    seed_manifest = []
    for name in config()["models"]:
        records = [json.loads((CHECKPOINTS / name / str(seed) / "metadata.json").read_text()) for seed in config()["seeds"]]
        if any(r["convergence_status"] not in ("CONVERGED", "MAX_EPOCH_REACHED") or r["config_hash"] != sha(CONFIG_PATH) for r in records):
            raise ValueError("Missing/invalid training record")
        for record in records:
            seed = record["seed"]
            if record["model"] != name or seed not in config()["seeds"] or not np.isfinite(record["validation_nll"]):
                raise ValueError("Invalid model/seed validation metadata")
            checkpoint = CHECKPOINTS / name / str(seed) / ("model.npz" if name == "linear_ar" else "model.pt")
            if name != "linear_ar":
                history = json.loads((CHECKPOINTS.parents[1] / "logs/v4_internal" / f"{name}_{seed}.json").read_text())
                selected = min(history, key=lambda r: r["validation_nll"])
                if selected["epoch"] != record["best_epoch"] or selected["validation_nll"] != record["validation_nll"] or len(history) != record["epochs_run"]:
                    raise ValueError("Checkpoint selection metadata disagrees with validation history")
            seed_manifest.append({**record, "checkpoint": str(checkpoint.relative_to(CHECKPOINTS)),
                                  "checkpoint_hash": sha(checkpoint), "selection_source": "Set A validation only"})
        best = min(records, key=lambda r: (r["validation_nll"], r["seed"]))
        suffix = "npz" if name == "linear_ar" else "pt"
        source = CHECKPOINTS / name / str(best["seed"]) / f"model.{suffix}"
        destination = CHECKPOINTS / "frozen_candidates" / f"{name}.{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError("Frozen candidate differs")
        if not destination.exists():
            shutil.copyfile(source, destination)
        manifest.append({"model": name, "config_hash": sha(CONFIG_PATH), "checkpoint_hash": sha(source),
                         "checkpoint": str(destination.relative_to(CHECKPOINTS)), "training_seed": best["seed"],
                         "validation_criterion": config()["checkpoint_criterion"], "validation_nll": best["validation_nll"],
                         "preprocessing_hash": sha(DATA_OUT / "audit/frozen_training_stats.json"),
                         "state_schema_hash": stats["config_hashes"]["FROZEN_STATE_SCHEMA.yaml"],
                         "calibration": "none", "selection_used_internal_test": False})
    if len(seed_manifest) != 20 or len({(r["model"],r["seed"]) for r in seed_manifest}) != 20:
        raise ValueError("All twenty distinct validation-selected fits are required")
    write_json(OUT / "frozen_seed_checkpoint_manifest.json", seed_manifest, frozen=True)
    write_json(OUT / "frozen_candidate_manifest.json", manifest, frozen=True)
    return manifest


def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=np.asarray(b) > 0)


def summarize(s, eligible=None, minimum=100):
    """Last axes are variables and fields. No missing variable is represented by zero."""
    n = s[..., F["n"]]
    support = n >= minimum
    if eligible is not None:
        support = support & np.asarray(eligible)
    get = lambda k: s[..., F[k]]
    mae = divide(get("abs"), n)
    dyn = divide(get("da"), get("dn"))
    tv = divide(get("truth2"), n) - divide(get("truth"), n)**2
    pv = divide(get("pred2"), n) - divide(get("pred"), n)**2
    cov = divide(get("ue"), n) - divide(get("u"), n)*mae
    uv = divide(get("u2"), n) - divide(get("u"), n)**2
    ev = divide(get("e2"), n) - mae**2
    dtv = divide(get("dtruth2"), get("dn")) - divide(get("dtruth"), get("dn"))**2
    dpv = divide(get("dpred2"), get("dn")) - divide(get("dpred"), get("dn"))**2
    per = {"mae": mae, "rmse": np.sqrt(divide(get("sq"), n)),
           "dynamic_mae": np.where(get("dn") >= minimum, dyn, np.nan),
           "change_variance_ratio": np.where(get("dn") >= minimum, divide(np.maximum(dpv, 0), dtv), np.nan),
           "variance_ratio": divide(np.maximum(pv, 0), tv),
           "trajectory_correlation": np.where(get("cn") >= minimum, divide(get("cs"), get("cn")), np.nan),
           "change_from_baseline_error": np.where(get("bn") >= minimum, divide(get("ba"), get("bn")), np.nan),
           "nll": divide(get("nll"), n), "crps": divide(get("crps"), n),
           "coverage50": divide(get("c50"), n), "coverage80": divide(get("c80"), n),
           "coverage90": divide(get("c90"), n), "coverage95": divide(get("c95"), n),
           "width50": divide(get("width50"), n), "width80": divide(get("width80"), n),
           "width95": divide(get("width95"), n), "uncertainty_error_correlation": divide(cov, np.sqrt(np.maximum(uv*ev, 0)))}
    per["coverage_error50"] = np.abs(per["coverage50"]-.5)
    per["coverage_error80"] = np.abs(per["coverage80"]-.8)
    per["coverage_error90"] = np.abs(per["coverage90"]-.9)
    per["coverage_error95"] = np.abs(per["coverage95"]-.95)
    aggregates = {}
    for k, v in per.items():
        valid = support & np.isfinite(v)
        aggregates[k] = divide(np.where(valid, v, 0).sum(-1), valid.sum(-1))
    aggregates["micro_mae"] = divide(get("abs").sum(-1), n.sum(-1))
    aggregates["micro_rmse"] = np.sqrt(divide(get("sq").sum(-1), n.sum(-1)))
    return per, aggregates, support


def add_statistics(destination, patient, truth, pred, sd, mask, baseline, baseline_mask):
    """Accumulate per-patient sufficient statistics; all arguments are standardized."""
    endpoint, pm, uncertainty, observed = truth[:, -1], pred[:, -1], sd[:, -1], mask[:, -1]
    if not np.isfinite(pred[mask]).all() or not np.isfinite(sd[mask]).all() or (sd[mask] <= 0).any():
        raise FloatingPointError("Invalid forecast at observed targets")
    err = np.where(observed, endpoint-pm, 0)
    ae = np.abs(err)
    z = err/uncertainty
    # torch erf avoids an optional SciPy dependency.
    erf = torch.erf(torch.from_numpy(z/math.sqrt(2))).numpy()
    crps = uncertainty*(z*erf + math.sqrt(2/math.pi)*np.exp(-z*z/2)-1/math.sqrt(math.pi))
    arrays = {"n": observed, "abs": ae, "sq": err**2,
              "nll": .5*(math.log(2*math.pi)+2*np.log(uncertainty)+z*z), "crps": crps,
              "c50": np.abs(z) <= NormalDist().inv_cdf(.75), "c80": np.abs(z) <= NormalDist().inv_cdf(.9),
              "c90": np.abs(z) <= NormalDist().inv_cdf(.95), "c95": np.abs(z) <= NormalDist().inv_cdf(.975),
              "width50": 2*NormalDist().inv_cdf(.75)*uncertainty,
              "width80": 2*NormalDist().inv_cdf(.9)*uncertainty,
              "width95": 2*NormalDist().inv_cdf(.975)*uncertainty,
              "u": uncertainty, "u2": uncertainty**2, "e2": ae**2, "ue": uncertainty*ae,
              "truth": endpoint, "pred": pm, "truth2": endpoint**2, "pred2": pm**2}
    arrays.update({f"cal{p}": np.abs(z) <= NormalDist().inv_cdf((1+p)/2) for p in NOMINAL})
    for key, value in arrays.items():
        np.add.at(destination[..., F[key]], patient, np.where(observed, value, 0))
    pairs = mask[:, 1:] & mask[:, :-1]
    dynamic = np.abs(np.diff(pred, axis=1)-np.diff(truth, axis=1))
    dtruth, dpred = np.diff(truth, axis=1), np.diff(pred, axis=1)
    np.add.at(destination[..., F["dn"]], patient, pairs.sum(1))
    np.add.at(destination[..., F["da"]], patient, np.where(pairs, dynamic, 0).sum(1))
    for key, value in (("dtruth", dtruth), ("dpred", dpred), ("dtruth2", dtruth**2), ("dpred2", dpred**2)):
        np.add.at(destination[..., F[key]], patient, np.where(pairs, value, 0).sum(1))
    bm = observed & baseline_mask
    np.add.at(destination[..., F["bn"]], patient, bm)
    np.add.at(destination[..., F["ba"]], patient, np.where(bm, ae, 0))
    count = mask.sum(1)
    tx, px = np.where(mask, truth, 0), np.where(mask, pred, 0)
    tv = (tx*tx).sum(1) - divide(tx.sum(1)**2, count)
    pv = (px*px).sum(1) - divide(px.sum(1)**2, count)
    covariance = (tx*px).sum(1)-divide(tx.sum(1)*px.sum(1), count)
    meaningful = (count >= 3) & (tv > 1e-10) & (pv > 1e-10)
    corr = divide(covariance, np.sqrt(np.maximum(tv*pv, 0)))
    np.add.at(destination[..., F["cn"]], patient, meaningful)
    np.add.at(destination[..., F["cs"]], patient, np.where(meaningful, corr, 0))
    return ae, arrays["nll"]


def evaluate_all():
    stats = verify()
    freeze_candidates()
    event = OUT / "analysis/internal_evaluation_event.json"
    # Exclusive creation: a failed or completed attempt cannot silently run again.
    # This happens before the first internal-test patient is opened.
    with event.open("x", encoding="utf-8") as handle:
        json.dump({"status": "started", "started_unix": time.time(),
                   "config_hash": sha(CONFIG_PATH),
                   "twenty_checkpoint_manifest_hash": sha(OUT / "frozen_seed_checkpoint_manifest.json"),
                   "Set_B_locked": True}, handle, indent=2)
    patients = load_patients("internal_test")
    cfg = config()
    contexts, truths, masks, baselines, baseline_masks, owners, available = [], [], [], [], [], [], []
    for pid, (x, y, m, _) in enumerate(patients):
        for origin in range(3, len(x)-4):
            h = min(48, len(x)-origin-1)
            contexts.append(x[origin-3:origin+1])
            truths.append(np.pad(y[origin+1:origin+1+h], ((0,48-h),(0,0))))
            masks.append(np.pad(m[origin+1:origin+1+h], ((0,48-h),(0,0))))
            baselines.append(y[origin])
            baseline_masks.append(m[origin])
            owners.append(pid)
            available.append(h)
    x = torch.from_numpy(np.asarray(contexts))
    truth, mask = np.asarray(truths), np.asarray(masks)
    baseline, bm = np.asarray(baselines), np.asarray(baseline_masks)
    owner, available = np.asarray(owners), np.asarray(available)
    subgroup = np.array([p[3] for p in patients])
    del contexts, truths, masks, baselines, baseline_masks, patients
    digest=lambda value:hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    ground_truth_signature={"truth_sha256":digest(truth),"mask_sha256":digest(mask),"owner_sha256":digest(owner),
                            "available_sha256":digest(available),"baseline_sha256":digest(baseline),
                            "baseline_mask_sha256":digest(bm),"subgroup_sha256":digest(subgroup),
                            "patient_count":int(len(subgroup)),"origin_count":int(len(x))}
    write_json(OUT/"analysis/internal_ground_truth_signature.json",ground_truth_signature,frozen=True)
    evaluation_mask_hash=ground_truth_signature["mask_sha256"]
    support = np.stack([mask[available >= 48, h-1].sum(0) >= 100 for h in cfg["horizons"]]).all(0)
    write_json(OUT / "analysis/common_variable_eligibility.json", {
        "variables": stats["features"], "eligible": support.tolist(), "minimum": 100,
        "counts": {str(h): mask[available >= 48,h-1].sum(0).tolist() for h in cfg["horizons"]}})
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            dest = OUT / "local" / f"{name}_{seed}_patient_stats.npz"
            folder = CHECKPOINTS / name / str(seed)
            checkpoint = folder / ("model.npz" if name == "linear_ar" else "model.pt")
            checkpoint_hash = sha(checkpoint)
            risk_path = OUT / "tables" / f"selective_{name}_{seed}.csv"
            if dest.exists() and risk_path.exists():
                with np.load(dest) as saved:
                    if str(saved["config_hash"]) != sha(CONFIG_PATH) or str(saved["checkpoint_hash"]) != checkpoint_hash:
                        raise ValueError("Saved evaluation provenance mismatch; refusing reuse")
                    if str(saved["evaluation_mask_hash"]) != evaluation_mask_hash:
                        raise ValueError("Saved evaluation mask provenance mismatch; refusing reuse")
                continue
            model = make_model(name)
            if name == "linear_ar":
                weights = np.load(folder / "model.npz")
                model.weights, model.variance = weights["weights"], weights["variance"]
            else:
                model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
                model.eval()
            sufficient = np.zeros((2,len(EVAL_HORIZONS),len(subgroup),33,len(FIELDS)))
            risks = {(c,h): [] for c in range(2) for h in range(len(EVAL_HORIZONS))}
            start_time = time.perf_counter()
            for start in range(0, len(x), 512):
                end = min(start+512, len(x))
                context = x[start:end].clone()
                if name == "linear_ar":
                    xp = context.numpy()
                    predictions, scales = [], []
                    for step in range(48):
                        mu, lv = model.predict(xp[:, -1])
                        predictions.append(mu)
                        scales.append(np.exp(.5*lv))
                        elapsed = xp[:, -1, 66:]
                        row = np.concatenate([mu, np.zeros_like(mu), np.where(elapsed >= 0, elapsed+1, elapsed)], -1)
                        xp = np.concatenate([xp[:, 1:], row[:,None]], 1)
                    pred, sd = np.stack(predictions,1), np.stack(scales,1)
                else:
                    pm, pv = rollout(model, context, 48)
                    pred, sd = pm.numpy().astype(float), pv.sqrt().numpy().astype(float)
                for ci in range(2):
                    for hi, h in enumerate(EVAL_HORIZONS):
                        keep = available[start:end] >= (48 if ci else h)
                        if not keep.any():
                            continue
                        ids = np.arange(start,end)[keep]
                        ae, nl = add_statistics(sufficient[ci,hi], owner[ids], truth[ids,:h], pred[keep,:h],
                                                sd[keep,:h], mask[ids,:h], baseline[ids], bm[ids])
                        uncertainty = sd[keep,h-1][:,support].mean(-1)
                        risks[ci,hi].append((uncertainty, ae, nl, mask[ids,h-1]))
                if start % 10240 == 0:
                    print(json.dumps({"phase":"internal_evaluation","model":name,"seed":seed,"origins_done":end,"origins_total":len(x)}), flush=True)
            np.savez_compressed(dest, statistics=sufficient, subgroup=subgroup, fields=np.array(FIELDS),
                                inference_time=time.perf_counter()-start_time, config_hash=sha(CONFIG_PATH), checkpoint_hash=checkpoint_hash,
                                evaluation_mask_hash=evaluation_mask_hash)
            risk_rows = []
            for (ci,hi), chunks in risks.items():
                uncertainty, ae, nl, observed = [np.concatenate([a[j] for a in chunks]) for j in range(4)]
                order = np.argsort(uncertainty, kind="stable")
                for fraction in (1.,.9,.8,.7,.5):
                    chosen = order[:max(1,math.ceil(len(order)*fraction))]
                    counts = observed[chosen].sum(0)
                    eligible = support & (counts >= 100) if ci else counts >= 100
                    mae = divide(np.where(observed[chosen],ae[chosen],0).sum(0), counts)
                    nll = divide(np.where(observed[chosen],nl[chosen],0).sum(0), counts)
                    risk_rows.append({"model":name,"seed":seed,"cohort":("full_available","common_48h")[ci],
                                      "horizon":EVAL_HORIZONS[hi],"retained_fraction":fraction,
                                      "MacroMAE":float(np.mean(mae[eligible])),"NLL":float(np.mean(nll[eligible])),
                                      "eligible_variables":int(eligible.sum()),"origins_retained":len(chosen)})
            write_csv(risk_path, risk_rows)
    write_json(OUT / "analysis/evaluation_integrity.json", {"Set_B_predictive_evaluation_performed":False,
               "Set_B_locked":True,"internal_test_patients":len(subgroup),"origins":len(x),"config_hash":sha(CONFIG_PATH)})
    receipt = json.loads(event.read_text())
    receipt.update(status="completed", completed_unix=time.time())
    write_json(event, receipt)
