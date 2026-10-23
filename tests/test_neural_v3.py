import pytest

torch = pytest.importorskip("torch")

from trace.models.neural_v3 import CausalTransformerWorldModel, RSSMWorldModel


def test_transformer_causal_mask_blocks_future_tokens() -> None:
    torch.manual_seed(1)
    model = CausalTransformerWorldModel(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64, dropout=0.0)
    model.eval()
    x = torch.randn(2, 6, 16)
    y1, _ = model(x)
    x2 = x.clone()
    x2[:, 4:, :] += 1000.0
    y2, _ = model(x2)
    assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5)


def test_rssm_prior_rollout_does_not_require_targets() -> None:
    torch.manual_seed(1)
    model = RSSMWorldModel(deterministic_dim=16, latent_dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 16)
    mu, logvar, kl = model(x, targets=None)
    assert mu.shape == (2, 5, 7)
    assert logvar.shape == (2, 5, 7)
    assert kl.shape == (2, 5)

