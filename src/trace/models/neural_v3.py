"""PyTorch neural sequence/world models for TRACE V3."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from torch import nn
except Exception:  # pragma: no cover - tests skip when torch is unavailable
    np = None
    torch = None
    nn = object

from trace.data.synthetic import ACTION_VARIABLES, STATE_VARIABLES

STATE_DIM = len(STATE_VARIABLES)
ACTION_DIM = len(ACTION_VARIABLES)
INPUT_DIM = STATE_DIM * 2 + ACTION_DIM
LOGVAR_MIN = -6.0
LOGVAR_MAX = 3.0


@dataclass
class TensorDatasetV3:
    inputs: Any
    targets: Any
    target_masks: Any
    means: dict[str, float]
    stds: dict[str, float]


def require_torch() -> None:
    if torch is None or np is None:
        raise RuntimeError("TRACE V3 requires numpy and torch. Use Python 3.12 in this workspace.")


def set_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sequences_to_dataset(sequences: dict[str, list[dict[str, Any]]], means: dict[str, float] | None = None, stds: dict[str, float] | None = None) -> TensorDatasetV3:
    """Convert patient sequences to tensors without changing V2 semantics."""
    require_torch()
    if means is None:
        means = {}
        stds = {}
        for variable in STATE_VARIABLES:
            values = [float(row[variable]) for rows in sequences.values() for row in rows if row[variable] is not None]
            means[variable] = float(np.mean(values))
            stds[variable] = float(np.std(values) + 1e-6)
    inputs = []
    targets = []
    masks = []
    for rows in sequences.values():
        x_rows = []
        y_rows = []
        m_rows = []
        for idx in range(len(rows) - 1):
            current = rows[idx]
            target = rows[idx + 1]
            state_values = []
            mask_values = []
            target_values = []
            target_mask = []
            for variable in STATE_VARIABLES:
                observed = current[variable] is not None
                value = float(current[variable]) if observed else means[variable]
                state_values.append((value - means[variable]) / stds[variable])
                mask_values.append(1.0 if observed else 0.0)
                target_observed = target[variable] is not None
                target_value = float(target[variable]) if target_observed else means[variable]
                target_values.append((target_value - means[variable]) / stds[variable])
                target_mask.append(1.0 if target_observed else 0.0)
            actions = [float(current[name]) for name in ACTION_VARIABLES]
            x_rows.append(state_values + mask_values + actions)
            y_rows.append(target_values)
            m_rows.append(target_mask)
        inputs.append(x_rows)
        targets.append(y_rows)
        masks.append(m_rows)
    return TensorDatasetV3(
        inputs=torch.tensor(inputs, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        target_masks=torch.tensor(masks, dtype=torch.float32),
        means=means,
        stds=stds or {},
    )


def denormalize_state(values: Any, means: dict[str, float], stds: dict[str, float]) -> dict[str, float]:
    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    return {variable: float(arr[idx] * stds[variable] + means[variable]) for idx, variable in enumerate(STATE_VARIABLES)}


def normalize_state(state: dict[str, float], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [(float(state[variable]) - means[variable]) / stds[variable] for variable in STATE_VARIABLES]


class GaussianHeadMixin:
    def split_output(self, raw: Any) -> tuple[Any, Any]:
        mu, logvar = raw[..., :self.state_dim], raw[..., self.state_dim:]
        return mu, torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)


class GRUWorldModel(nn.Module, GaussianHeadMixin):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1, *, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, auxiliary_dim: int | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + (state_dim if auxiliary_dim is None else auxiliary_dim) + action_dim
        self.net = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, state_dim * 2))

    def forward(self, x: Any) -> tuple[Any, Any]:
        h, _ = self.net(x)
        return self.split_output(self.head(h))


class CausalTransformerWorldModel(nn.Module, GaussianHeadMixin):
    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1, max_len: int = 32, *, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, auxiliary_dim: int | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + (state_dim if auxiliary_dim is None else auxiliary_dim) + action_dim
        self.input = nn.Linear(input_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, state_dim * 2))

    def forward(self, x: Any) -> tuple[Any, Any]:
        length = x.shape[1]
        z = self.input(x) + self.positional[:length].unsqueeze(0)
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(z, mask=mask)
        return self.split_output(self.head(encoded))


class RSSMWorldModel(nn.Module):
    """Small variational recurrent state-space model."""

    def __init__(self, deterministic_dim: int = 32, latent_dim: int = 16, hidden_dim: int = 64, *, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, auxiliary_dim: int | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.auxiliary_dim = state_dim if auxiliary_dim is None else auxiliary_dim
        self.consume_auxiliary = auxiliary_dim is not None
        self.deterministic_dim = deterministic_dim
        self.latent_dim = latent_dim
        self.rnn = nn.GRUCell(state_dim + action_dim + latent_dim + (self.auxiliary_dim if self.consume_auxiliary else 0), deterministic_dim)
        self.prior = nn.Sequential(nn.Linear(deterministic_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.posterior = nn.Sequential(nn.Linear(deterministic_dim + state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(deterministic_dim + latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim * 2))

    def forward(self, x: Any, targets: Any | None = None) -> tuple[Any, Any, Any]:
        batch, length, _ = x.shape
        h = torch.zeros(batch, self.deterministic_dim, device=x.device)
        z = torch.zeros(batch, self.latent_dim, device=x.device)
        mus, logvars, kls = [], [], []
        for t in range(length):
            state = x[:, t, :self.state_dim + (self.auxiliary_dim if self.consume_auxiliary else 0)]
            action = x[:, t, self.state_dim + self.auxiliary_dim :]
            h = self.rnn(torch.cat([state, action, z], dim=-1), h)
            prior_mu, prior_logvar = _split_latent(self.prior(h))
            if targets is not None:
                post_mu, post_logvar = _split_latent(self.posterior(torch.cat([h, targets[:, t]], dim=-1)))
                z = _sample(post_mu, post_logvar)
                kls.append(_kl_normal(post_mu, post_logvar, prior_mu, prior_logvar))
            else:
                z = prior_mu
                kls.append(torch.zeros(batch, device=x.device))
            raw = self.decoder(torch.cat([h, z], dim=-1))
            mu, logvar = raw[..., :self.state_dim], torch.clamp(raw[..., self.state_dim:], LOGVAR_MIN, LOGVAR_MAX)
            mus.append(mu)
            logvars.append(logvar)
        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1), torch.stack(kls, dim=1)


def _split_latent(raw: Any) -> tuple[Any, Any]:
    mu, logvar = raw.chunk(2, dim=-1)
    return mu, torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)


def _sample(mu: Any, logvar: Any) -> Any:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def _kl_normal(q_mu: Any, q_logvar: Any, p_mu: Any, p_logvar: Any) -> Any:
    return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) / torch.exp(p_logvar) - 1).sum(dim=-1)


def gaussian_nll(mu: Any, logvar: Any, target: Any, mask: Any) -> Any:
    logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
    nll = 0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 / torch.exp(logvar))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(mu: Any, target: Any, mask: Any) -> float:
    return float(((mu - target).abs() * mask).sum().detach().cpu() / mask.sum().clamp_min(1.0).detach().cpu())


class NeuralModelAdapter:
    """Adapter exposing the V2 rollout interface for trained PyTorch models."""

    def __init__(self, name: str, model: Any, means: dict[str, float], stds: dict[str, float], device: str = "cpu") -> None:
        self.name = name
        self.model = model.to(device)
        self.model.eval()
        self.means = means
        self.stds = stds
        self.device = device
        self.parameter_count = sum(p.numel() for p in model.parameters())

    def rollout(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[dict[str, float]]:
        return [pred for pred, _ in self.rollout_with_uncertainty(history, horizon_steps, action_sequence)]

    def rollout_with_uncertainty(self, history: list[dict[str, Any]], horizon_steps: int, action_sequence: list[dict[str, float]] | None = None) -> list[tuple[dict[str, float], dict[str, float]]]:
        rows = [dict(row) for row in history]
        output = []
        for step in range(horizon_steps):
            action = action_sequence[step] if action_sequence else {name: float(rows[-1].get(name, 0.0)) for name in ACTION_VARIABLES}
            x = _history_to_tensor(rows, action, self.means, self.stds).to(self.device)
            with torch.no_grad():
                if isinstance(self.model, RSSMWorldModel):
                    mu, logvar, _ = self.model(x, targets=None)
                else:
                    mu, logvar = self.model(x)
            pred_norm = mu[0, -1]
            std_norm = torch.exp(0.5 * logvar[0, -1])
            pred = denormalize_state(pred_norm, self.means, self.stds)
            std = {variable: float(std_norm[idx].cpu() * self.stds[variable]) for idx, variable in enumerate(STATE_VARIABLES)}
            new_row = {"patient_id": rows[-1].get("patient_id", "v3"), "time_index": rows[-1].get("time_index", 0) + 1}
            new_row.update(action)
            for variable in STATE_VARIABLES:
                new_row[variable] = pred[variable]
                new_row[f"{variable}_mask"] = 1
            rows.append(new_row)
            output.append((pred, std))
        return output


def _history_to_tensor(rows: list[dict[str, Any]], action: dict[str, float], means: dict[str, float], stds: dict[str, float]) -> Any:
    vectors = []
    for idx, row in enumerate(rows):
        state = []
        mask = []
        for variable in STATE_VARIABLES:
            observed = row.get(variable) is not None
            value = float(row[variable]) if observed else means[variable]
            state.append((value - means[variable]) / stds[variable])
            mask.append(1.0 if observed else 0.0)
        row_action = action if idx == len(rows) - 1 else {name: float(row.get(name, 0.0)) for name in ACTION_VARIABLES}
        vectors.append(state + mask + [row_action[name] for name in ACTION_VARIABLES])
    return torch.tensor([vectors], dtype=torch.float32)


def train_model(model_name: str, train_data: TensorDatasetV3, val_data: TensorDatasetV3, config: dict[str, Any], seed: int, output_dir: Path) -> tuple[NeuralModelAdapter, list[dict[str, Any]], dict[str, Any]]:
    require_torch()
    set_seed(seed)
    device = "cpu"
    if model_name == "gru":
        model = GRUWorldModel(hidden_size=config.get("hidden_size", 32), num_layers=config.get("num_layers", 1), dropout=config.get("dropout", 0.1))
    elif model_name == "transformer":
        model = CausalTransformerWorldModel(d_model=config.get("d_model", 32), n_layers=config.get("n_layers", 2), n_heads=config.get("n_heads", 4), dim_feedforward=config.get("dim_feedforward", 128), dropout=config.get("dropout", 0.1))
    elif model_name == "rssm":
        model = RSSMWorldModel(deterministic_dim=config.get("deterministic_dim", 32), latent_dim=config.get("latent_dim", 16), hidden_dim=config.get("hidden_dim", 64))
    else:
        raise ValueError(model_name)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 1e-4))
    max_epochs = int(config.get("max_epochs", 25))
    patience = int(config.get("patience", 6))
    kl_weight = float(config.get("kl_weight", 0.01))
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        if model_name == "rssm":
            mu, logvar, kl = model(train_data.inputs, train_data.targets)
            recon = gaussian_nll(mu, logvar, train_data.targets, train_data.target_masks)
            kl_loss = kl.mean()
            loss = recon + kl_weight * kl_loss
        else:
            mu, logvar = model(train_data.inputs)
            recon = gaussian_nll(mu, logvar, train_data.targets, train_data.target_masks)
            kl_loss = torch.tensor(0.0)
            loss = recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
        opt.step()
        model.eval()
        with torch.no_grad():
            if model_name == "rssm":
                v_mu, v_logvar, v_kl = model(val_data.inputs, val_data.targets)
                val_loss = gaussian_nll(v_mu, v_logvar, val_data.targets, val_data.target_masks) + kl_weight * v_kl.mean()
                val_nll = gaussian_nll(v_mu, v_logvar, val_data.targets, val_data.target_masks)
                val_kl = v_kl.mean()
            else:
                v_mu, v_logvar = model(val_data.inputs)
                val_loss = gaussian_nll(v_mu, v_logvar, val_data.targets, val_data.target_masks)
                val_nll = val_loss
                val_kl = torch.tensor(0.0)
            val_mae = masked_mae(v_mu, val_data.targets, val_data.target_masks)
        record = {
            "model": model_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "validation_loss": float(val_loss.detach().cpu()),
            "validation_mae": val_mae,
            "validation_nll": float(val_nll.detach().cpu()),
            "learning_rate": config.get("lr", 1e-3),
            "reconstruction_loss": float(recon.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu() if epoch == 1 or model_name == "rssm" else val_kl.detach().cpu()),
        }
        history.append(record)
        if record["validation_loss"] < best_val - 1e-5:
            best_val = record["validation_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best.pt")
    torch.save(model.state_dict(), output_dir / "last.pt")
    metadata = {
        "model": model_name,
        "seed": seed,
        "config": config,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_val,
        "training_time": time.perf_counter() - start_time,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return NeuralModelAdapter(model_name, model, train_data.means, train_data.stds, device), history, metadata


def build_torch_model(model_name: str, config: dict[str, Any]) -> Any:
    """Build a V3 torch model from saved metadata config."""
    require_torch()
    if model_name == "gru":
        return GRUWorldModel(hidden_size=config.get("hidden_size", 32), num_layers=config.get("num_layers", 1), dropout=config.get("dropout", 0.1))
    if model_name == "transformer":
        return CausalTransformerWorldModel(d_model=config.get("d_model", 32), n_layers=config.get("n_layers", 2), n_heads=config.get("n_heads", 4), dim_feedforward=config.get("dim_feedforward", 128), dropout=config.get("dropout", 0.1))
    if model_name == "rssm":
        return RSSMWorldModel(deterministic_dim=config.get("deterministic_dim", 32), latent_dim=config.get("latent_dim", 16), hidden_dim=config.get("hidden_dim", 64))
    raise ValueError(model_name)


def load_trained_adapter(model_name: str, checkpoint_dir: str | Path, means: dict[str, float], stds: dict[str, float], checkpoint_name: str = "best.pt") -> NeuralModelAdapter:
    """Load a V3 trained model adapter from checkpoint metadata."""
    require_torch()
    directory = Path(checkpoint_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = build_torch_model(model_name, metadata["config"])
    state = torch.load(directory / checkpoint_name, map_location="cpu")
    model.load_state_dict(state)
    return NeuralModelAdapter(model_name, model, means, stds, "cpu")


def overfit_smoke_test(model_name: str, train_data: TensorDatasetV3, config: dict[str, Any], seed: int) -> dict[str, Any]:
    small = TensorDatasetV3(train_data.inputs[:16], train_data.targets[:16], train_data.target_masks[:16], train_data.means, train_data.stds)
    cfg = dict(config)
    cfg.update({"max_epochs": 15, "patience": 15})
    _, history, _ = train_model(model_name, small, small, cfg, seed, Path("checkpoints/v3_neural/overfit") / model_name / str(seed))
    return {
        "model": model_name,
        "seed": seed,
        "initial_loss": history[0]["train_loss"],
        "final_loss": history[-1]["train_loss"],
        "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
    }
