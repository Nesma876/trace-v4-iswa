"""Action-free reuse of existing neural families; native masked ridge AR."""
import numpy as np


def build_model(name, state_dim, **kwargs):
    if state_dim < 1:
        raise ValueError("Positive state dimension required")
    if name == "linear_ar":
        return LinearAR(state_dim, **kwargs)
    from trace.models.neural_v3 import GRUWorldModel, CausalTransformerWorldModel, RSSMWorldModel
    cls = {"gru": GRUWorldModel, "transformer": CausalTransformerWorldModel, "rssm": RSSMWorldModel}[name]
    return cls(state_dim=state_dim, action_dim=0, auxiliary_dim=state_dim * 2, **kwargs)


class LinearAR:
    """Ridge one-step AR using value, mask and elapsed; per-target observed rows only."""
    action_dim = 0

    def __init__(self, state_dim, ridge=.05):
        self.state_dim = state_dim
        self.ridge = ridge

    def fit(self, inputs, targets, masks, *, source, split):
        if (source, split) != ("A", "train"):
            raise ValueError("AR fitting is restricted to Set A training")
        x = np.concatenate([np.ones((len(inputs), 1)), np.asarray(inputs)], axis=1)
        if x.shape[1] != 1 + 3 * self.state_dim:
            raise ValueError("Expected action-free value/mask/elapsed inputs")
        self.weights = np.empty((x.shape[1], self.state_dim))
        self.variance = np.empty(self.state_dim)
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0
        for j in range(self.state_dim):
            observed = np.asarray(masks)[:, j].astype(bool)
            if not observed.any():
                raise ValueError("Unobserved training target")
            a, y = x[observed], np.asarray(targets)[observed, j]
            self.weights[:, j] = np.linalg.lstsq(a.T @ a + penalty, a.T @ y, rcond=None)[0]
            self.variance[j] = max(np.mean((y - a @ self.weights[:, j])**2), 1e-6)
        return self

    def predict(self, inputs):
        x = np.asarray(inputs)
        if x.shape[-1] != 3 * self.state_dim:
            raise ValueError("Action-free dimension mismatch")
        mu = np.concatenate([np.ones(x.shape[:-1] + (1,)), x], axis=-1) @ self.weights
        return mu, np.broadcast_to(np.log(self.variance), mu.shape)
