"""Auditable tiny-set gates and full-origin Set A training."""
import json
import math
import time

import numpy as np
import torch

from trace.models.neural_v3 import set_seed
from .internal import (OUT, LOGS, CHECKPOINTS, CONFIG_PATH, config, verify,
                       load_patients, windows, make_model, posterior_forward, prior)
from .pipeline import sha, write_json, write_csv
from .scoring import masked_training_nll


@torch.no_grad()
def validate(model, data):
    model.eval()
    x, y, mask = data
    nll = mae = count = 0.
    for start in range(0, len(x), 1024):
        mu, lv = prior(model, x[start:start+1024])
        m = mask[start:start+1024, -1]
        target = y[start:start+1024, -1]
        n = int(m.sum())
        nll += float(masked_training_nll(mu, lv, target, m)) * n
        mae += float((mu[m] - target[m]).abs().sum())
        count += n
    return {"nll": nll/count, "mae": mae/count, "observed_targets": int(count)}


def step(model, name, x, y, mask, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if name == "rssm":
        mu, lv, kl = posterior_forward(model, x, y, mask)
        reconstruction = masked_training_nll(mu[:, -1], lv[:, -1], y[:, -1], mask[:, -1])
        kl_loss = kl.mean()
        loss = reconstruction + config()["rssm_kl_weight"] * kl_loss
    else:
        mu, lv = prior(model, x)
        reconstruction = masked_training_nll(mu, lv, y[:, -1], mask[:, -1])
        kl_loss = reconstruction.new_zeros(())
        loss = reconstruction
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradients")
    optimizer.step()
    return np.array([float(reconstruction.detach()), float(kl_loss.detach()), float(loss.detach())])


def sanity():
    cfg = config()
    verify()
    data = windows(load_patients("train", limit=cfg["sanity"]["patients"]))
    results = []
    for name in ("gru", "transformer", "rssm"):
        set_seed(cfg["sanity"]["seed"])
        model = make_model(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        initial = validate(model, data)
        best = initial.copy()
        passed = False
        for update in range(1, cfg["sanity"]["max_updates"]+1):
            indices = torch.randperm(len(data[0]))[:cfg["batch_size"]]
            losses = step(model, name, *(a[indices] for a in data), optimizer)
            if update % 50 == 0:
                metrics = validate(model, data)
                if metrics["mae"] < best["mae"]:
                    best = metrics
                print(json.dumps({"phase": "sanity", "model": name, "update": update, **metrics,
                                  "initial_mae": initial["mae"], "kl": losses[1]}), flush=True)
                if metrics["mae"] <= initial["mae"] * cfg["sanity"]["required_mae_ratio"]:
                    passed = True
                    break
        results.append({"model": name, "patients": 16, "windows": len(data[0]), "updates": update,
                        "initial_prior_mae": initial["mae"], "best_prior_mae": best["mae"],
                        "passed": passed, "config_hash": sha(CONFIG_PATH)})
        write_json(OUT / "analysis/overfit_sanity.json", results)
        if not passed:
            raise RuntimeError(f"STOP: {name} failed the predefined tiny-set overfit gate")
    return results


def train_all():
    cfg = config()
    verify()
    gates = json.loads((OUT / "analysis/overfit_sanity.json").read_text())
    if {r["model"] for r in gates if r["passed"] and r["config_hash"] == sha(CONFIG_PATH)} != {"gru", "transformer", "rssm"}:
        raise ValueError("All three current-config sanity gates must pass before full training")
    print("Loading all Set A training and validation origins; internal test unopened", flush=True)
    train = windows(load_patients("train"))
    val = windows(load_patients("validation"))
    records = []
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            folder = CHECKPOINTS / name / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            metadata = folder / "metadata.json"
            if metadata.exists():
                record = json.loads(metadata.read_text())
                if record["config_hash"] != sha(CONFIG_PATH):
                    raise ValueError("Cannot resume different configuration")
                records.append(record)
                continue
            set_seed(seed)
            model = make_model(name)
            start = time.perf_counter()
            history = []
            if name == "linear_ar":
                model.fit(train[0][:, -1].numpy(), train[1][:, -1].numpy(), train[2][:, -1].numpy(), source="A", split="train")
                mu, lv = model.predict(val[0][:, -1].numpy())
                mask = val[2][:, -1].numpy()
                err = val[1][:, -1].numpy()[mask] - mu[mask]
                best = float(np.mean(.5*(math.log(2*math.pi)+lv[mask]+err**2/np.exp(lv[mask]))))
                if not np.isfinite(best):
                    raise FloatingPointError("Nonfinite linear validation NLL")
                np.savez(folder / "model.npz", weights=model.weights, variance=model.variance)
                parameters = model.weights.size + model.variance.size
                best_epoch = epoch = 1
                status = "CONVERGED"
            else:
                parameters = sum(p.numel() for p in model.parameters())
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
                best = float("inf")
                best_epoch = 0
                status = "MAX_EPOCH_REACHED"
                for epoch in range(1, cfg["max_epochs"]+1):
                    sums = np.zeros(3)
                    batches = 0
                    order = torch.randperm(len(train[0]))
                    for ids in order.split(cfg["batch_size"]):
                        sums += step(model, name, *(a[ids] for a in train), optimizer)
                        batches += 1
                    metrics = validate(model, val)
                    if not np.isfinite(metrics["nll"]):
                        raise FloatingPointError("Nonfinite validation NLL")
                    row = {"epoch": epoch, "reconstruction": sums[0]/batches, "kl": sums[1]/batches,
                           "total": sums[2]/batches, "validation_nll": metrics["nll"], "validation_mae": metrics["mae"]}
                    history.append(row)
                    write_json(LOGS / f"{name}_{seed}.json", history)
                    print(json.dumps({"model": name, "seed": seed, **row}), flush=True)
                    if metrics["nll"] < best:
                        best, best_epoch = metrics["nll"], epoch
                        torch.save(model.state_dict(), folder / "model.pt")
                    if epoch-best_epoch >= cfg["patience"]:
                        status = "CONVERGED"
                        break
            record = {"model": name, "seed": seed, "parameters": parameters, "trainable_parameters": parameters,
                      "best_epoch": best_epoch, "epochs_run": epoch, "convergence_status": status,
                      "validation_nll": best, "training_time": time.perf_counter()-start,
                      "inference_time": None, "config_hash": sha(CONFIG_PATH)}
            write_json(metadata, record)
            records.append(record)
            write_csv(OUT / "tables/model_capacity.csv", records)
    write_csv(OUT / "tables/hyperparameter_selection.csv", [
        {"model": name, "config_hash": sha(CONFIG_PATH), "search_trials": 1, "selection_data": "Set A validation only", "configuration": json.dumps(params, sort_keys=True)}
        for name, params in cfg["models"].items()])
    return records
