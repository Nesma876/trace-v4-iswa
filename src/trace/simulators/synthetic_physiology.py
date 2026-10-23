"""Deterministic branchable synthetic physiology simulator."""

from __future__ import annotations

from copy import deepcopy


class SyntheticPhysiologySimulator:
    """Small deterministic simulator with known intervention effects."""

    def __init__(self) -> None:
        self.state: dict[str, float] = {}

    def reset(self, patient_state: dict[str, float]) -> None:
        self.state = dict(patient_state)

    def step(self, action: dict[str, float]) -> dict[str, float]:
        fluids = action.get("iv_fluids", 0.0)
        pressors = action.get("vasopressors", 0.0)
        self.state["mean_arterial_pressure"] += 0.6 * fluids + 1.8 * pressors + 0.1 * (78 - self.state["mean_arterial_pressure"])
        self.state["lactate"] += -0.04 * fluids - 0.02 * pressors + 0.08 * max(0.0, 65 - self.state["mean_arterial_pressure"])
        self.state["heart_rate"] += 0.1 * (88 - self.state["heart_rate"]) + 0.8 * pressors
        self.state["respiratory_rate"] += 0.1 * (20 - self.state["respiratory_rate"])
        self.state["spo2"] += 0.1 * (96 - self.state["spo2"])
        self.state["temperature"] += 0.1 * (37.2 - self.state["temperature"])
        self.state["creatinine"] += 0.02 * max(0.0, 70 - self.state["mean_arterial_pressure"])
        return dict(self.state)

    def rollout(self, action_sequence: list[dict[str, float]], horizon: int) -> list[dict[str, float]]:
        return [self.step(action_sequence[min(idx, len(action_sequence) - 1)]) for idx in range(horizon)]

    def branch(self, actions: list[list[dict[str, float]]]) -> list[list[dict[str, float]]]:
        origin = deepcopy(self.state)
        trajectories = []
        for sequence in actions:
            self.reset(origin)
            trajectories.append(self.rollout(sequence, len(sequence)))
        self.reset(origin)
        return trajectories

