"""Action semantics for TRACE synthetic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from trace.data.synthetic import ACTION_VARIABLES


RAW_ACTION_LEVELS: dict[str, dict[str, float]] = {
    "none": {"iv_fluids": 0.0, "vasopressors": 0.0},
    "low": {"iv_fluids": 0.5, "vasopressors": 0.15},
    "moderate": {"iv_fluids": 1.0, "vasopressors": 0.4},
    "high": {"iv_fluids": 1.5, "vasopressors": 0.8},
}


@dataclass(frozen=True)
class IdentityActionTransform:
    """Identity transform used by the current dependency-free synthetic pipeline."""

    name: str = "identity_raw_action"

    def transform_action(self, raw_action: dict[str, float]) -> dict[str, float]:
        """Transform raw action values into model input space."""
        return {name: float(raw_action[name]) for name in ACTION_VARIABLES}

    def inverse_transform_action(self, transformed_action: dict[str, float]) -> dict[str, float]:
        """Recover raw action values from model input space."""
        return {name: float(transformed_action[name]) for name in ACTION_VARIABLES}


def raw_no_intervention_action() -> dict[str, float]:
    """Return the raw no-intervention action convention."""
    return dict(RAW_ACTION_LEVELS["none"])


def transform_action(raw_action: dict[str, float]) -> dict[str, float]:
    """Official train/evaluation action transform."""
    return IdentityActionTransform().transform_action(raw_action)


def inverse_transform_action(transformed_action: dict[str, float]) -> dict[str, float]:
    """Official inverse action transform."""
    return IdentityActionTransform().inverse_transform_action(transformed_action)


def action_levels() -> list[tuple[str, dict[str, float]]]:
    """Return frozen controlled intervention levels in ascending dose order."""
    return [(name, dict(values)) for name, values in RAW_ACTION_LEVELS.items()]


def action_semantics_rows() -> list[dict[str, str | float]]:
    """Return an audit table describing action semantics."""
    rows = []
    transform = IdentityActionTransform()
    for level, raw in action_levels():
        transformed = transform.transform_action(raw)
        recovered = transform.inverse_transform_action(transformed)
        for variable in ACTION_VARIABLES:
            rows.append(
                {
                    "level": level,
                    "action_variable": variable,
                    "raw_value": raw[variable],
                    "transformed_value": transformed[variable],
                    "inverse_value": recovered[variable],
                    "transform": transform.name,
                    "zero_means": "raw absence of intervention",
                }
            )
    return rows

