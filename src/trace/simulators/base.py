"""Simulator interface for controlled counterfactual evaluation."""

from __future__ import annotations

from typing import Protocol


class Simulator(Protocol):
    """Branchable simulator protocol."""

    def reset(self, patient_state: dict[str, float]) -> None:
        ...

    def step(self, action: dict[str, float]) -> dict[str, float]:
        ...

    def rollout(self, action_sequence: list[dict[str, float]], horizon: int) -> list[dict[str, float]]:
        ...

    def branch(self, actions: list[list[dict[str, float]]]) -> list[list[dict[str, float]]]:
        ...

