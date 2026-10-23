"""Set A-only modelling utilities. No external data entry point is provided."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import read_psv
from .models import build_model
from .pipeline import REPO, sha, write_json
from .policy import load_frozen_preprocessing, model_values
from .preprocessing import transform_prefix
from .schema import DYNAMIC
from trace.models.neural_v3 import _split_latent, _sample, _kl_normal

OUT = REPO / "outputs/v4_internal"
CHECKPOINTS = REPO / "checkpoints/v4_internal"
LOGS = REPO / "logs/v4_internal"
DATA_OUT = REPO / "outputs/v4_physionet2019"
CONFIG_PATH = REPO / "configs/physionet2019/INTERNAL_MODELLING.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def verify():
    manifest = json.loads((DATA_OUT / "audit/final_freeze_manifest.json").read_text())
    for name, expected in manifest["frozen_artifacts"].items():
        if sha(REPO / "configs/physionet2019" / name) != expected:
            raise ValueError(f"Frozen hash mismatch: {name}")
    stats = load_frozen_preprocessing(DATA_OUT)
    assert stats["features"] == [v for v in DYNAMIC if v != "EtCO2"]
    assert len(stats["features"]) == 33
    assert not set(stats["features"]) & {"SepsisLabel", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}
    if not manifest["external_test_locked"]:
        raise ValueError("External lock is not asserted")
    for p in (OUT, CHECKPOINTS, LOGS, OUT / "local", OUT / "tables", OUT / "analysis", OUT / "figures"):
        p.mkdir(parents=True, exist_ok=True)
    receipt = {"frozen_artifacts": manifest["frozen_artifacts"], "split_hash": stats["split_hash"],
               "statistics_hash": manifest["preprocessing_statistics_hash"],
               "modelling_config_hash": sha(CONFIG_PATH),
               "dense_config_hash": sha(REPO / "configs/physionet2019/DENSE_VARIABLE_SENSITIVITY.yaml"),
               "Set_B_predictive_evaluation_performed": False, "Set_B_locked": True}
    path = OUT / "initial_integrity.json"
    write_json(path, receipt, frozen=True)
    return stats


def load_patients(split, *, limit=None):
    """Only explicitly assigned Set A files can be opened. Metadata returned separately."""
    if split not in ("train", "validation", "internal_test"):
        raise ValueError("Only Set A development splits are supported")
    if split == "internal_test" and not (OUT / "frozen_candidate_manifest.json").exists():
        raise ValueError("Internal test remains locked until candidates are frozen")
    stats = verify()
    with (DATA_OUT / "splits/setA_patient_split.csv").open() as f:
        names = sorted(r["file_id"] for r in csv.DictReader(f) if r["split"] == split)
    if limit is not None:
        if split != "train":
            raise ValueError("Tiny-set selection is training only")
        names = names[:limit]
    root = REPO / "data_external/physionet2019/training_setA"
    patients = []
    for name in names:
        if Path(name).name != name or not name.endswith(".psv"):
            raise ValueError("Invalid split filename")
        raw = read_psv(root / name)
        x = transform_prefix(raw, stats)
        values = model_values(raw, stats)
        mask = np.isfinite(values)
        y = np.where(mask, (values - stats["means"]) / stats["stds"], 0).astype(np.float32)
        assert x.shape[1] == 99 and y.shape[1] == 33
        patients.append((x, y, mask, bool(np.any(raw[:, -1] == 1))))
    return patients


def windows(patients):
    """Context and aligned next-row targets; no windows cross patient boundaries."""
    xs, ys, ms = [], [], []
    for x, y, m, _ in patients:
        if len(x) <= 4:
            continue
        xs.append(np.stack([x[t-3:t+1] for t in range(3, len(x)-1)]))
        ys.append(np.stack([y[t-2:t+2] for t in range(3, len(x)-1)]))
        ms.append(np.stack([m[t-2:t+2] for t in range(3, len(x)-1)]))
    return tuple(torch.from_numpy(np.concatenate(a)) for a in (xs, ys, ms))


def make_model(name):
    return build_model(name, 33, **config()["models"][name])


def posterior_forward(model, x, targets, masks):
    """Mask-safe V4 posterior; leaves frozen V3 forward/architecture unchanged."""
    batch, length, width = x.shape
    assert width == 99 and targets.shape == masks.shape == (batch, length, 33)
    h = x.new_zeros(batch, model.deterministic_dim)
    z = x.new_zeros(batch, model.latent_dim)
    mus, lvs, kls = [], [], []
    for t in range(length):
        h = model.rnn(torch.cat([x[:, t], z], -1), h)
        pm, pl = _split_latent(model.prior(h))
        prior_mean = model.decoder(torch.cat([h, pm], -1))[:, :33]
        safe = torch.where(masks[:, t], targets[:, t], prior_mean.detach())
        qm, ql = _split_latent(model.posterior(torch.cat([h, safe], -1)))
        has_target = masks[:, t].any(-1, keepdim=True)
        qm, ql = torch.where(has_target, qm, pm), torch.where(has_target, ql, pl)
        z = _sample(qm, ql)
        raw = model.decoder(torch.cat([h, z], -1))
        mus.append(raw[:, :33])
        lvs.append(raw[:, 33:].clamp(-6, 3))
        kls.append(_kl_normal(qm, ql, pm, pl))
    return torch.stack(mus, 1), torch.stack(lvs, 1), torch.stack(kls, 1)


def prior(model, x):
    result = model(x)
    return result[0][:, -1], result[1][:, -1]


def generated_row(mean, previous):
    assert mean.shape[-1] == 33 and previous.shape[-1] == 99
    elapsed = previous[..., 66:]
    elapsed = torch.where(elapsed >= 0, elapsed + 1, elapsed)
    return torch.cat([mean, torch.zeros_like(mean), elapsed], -1)


@torch.no_grad()
def rollout(model, x, horizon):
    """Only context is accepted; future ground truth cannot enter this API."""
    model.eval()
    means, variances = [], []
    for _ in range(horizon):
        mu, lv = prior(model, x)
        means.append(mu)
        variances.append(lv.exp())
        x = torch.cat([x[:, 1:], generated_row(mu, x[:, -1])[:, None]], 1)
    return torch.stack(means, 1), torch.stack(variances, 1)
