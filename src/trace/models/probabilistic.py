"""Probabilistic prediction utilities."""

def interval_from_mean_variance(mean: float, variance: float, z_value: float) -> tuple[float, float]:
    """Return a symmetric Gaussian interval."""
    half_width = z_value * (variance ** 0.5)
    return mean - half_width, mean + half_width

