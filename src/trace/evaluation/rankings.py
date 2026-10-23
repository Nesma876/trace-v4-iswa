"""Ranking instability utilities."""

from __future__ import annotations


def rank_models(metric_by_model: dict[str, float], lower_is_better: bool = True) -> list[tuple[str, int]]:
    """Return model ranks for a single metric."""
    ordered = sorted(metric_by_model, key=metric_by_model.get, reverse=not lower_is_better)
    return [(name, idx + 1) for idx, name in enumerate(ordered)]

