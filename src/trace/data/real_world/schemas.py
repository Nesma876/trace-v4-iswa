"""Canonical schema objects for TRACE V4 real-world ICU data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalVariable:
    name: str
    domain: str
    target_unit: str
    aggregation: str
    included: bool
    reason: str


@dataclass(frozen=True)
class CanonicalSchema:
    state_variables: tuple[str, ...]
    action_variables: tuple[str, ...]
    units: dict[str, str]
    state_suffixes: tuple[str, ...] = ("value", "observation_mask", "time_since_last_observation")

    @property
    def state_dimension(self) -> int:
        return len(self.state_variables) * len(self.state_suffixes)

    @property
    def action_dimension(self) -> int:
        return len(self.action_variables)

    def assert_identity(self, other: "CanonicalSchema") -> None:
        if self.state_variables != other.state_variables:
            raise ValueError("state variable order differs")
        if self.action_variables != other.action_variables:
            raise ValueError("action variable order differs")
        if self.units != other.units:
            raise ValueError("canonical units differ")
        if self.state_suffixes != other.state_suffixes:
            raise ValueError("mask/value semantics differ")


TRACE_COMMON_STATE = (
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "temperature",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "mean_arterial_pressure",
    "lactate",
    "creatinine",
    "bilirubin",
    "platelets",
    "pao2",
    "fio2",
    "urine_output",
)

TRACE_COMMON_ACTIONS = ("iv_fluids_volume_ml_per_4h", "vasopressor_exposure")

TRACE_COMMON_UNITS = {
    "heart_rate": "beats/min",
    "respiratory_rate": "breaths/min",
    "spo2": "percent",
    "temperature": "degC",
    "systolic_blood_pressure": "mmHg",
    "diastolic_blood_pressure": "mmHg",
    "mean_arterial_pressure": "mmHg",
    "lactate": "mmol/L",
    "creatinine": "mg/dL",
    "bilirubin": "mg/dL",
    "platelets": "K/uL",
    "pao2": "mmHg",
    "fio2": "fraction",
    "urine_output": "mL/4h",
}


def frozen_common_schema() -> CanonicalSchema:
    return CanonicalSchema(TRACE_COMMON_STATE, TRACE_COMMON_ACTIONS, dict(TRACE_COMMON_UNITS))


def table_a_rows() -> list[dict[str, Any]]:
    schema = frozen_common_schema()
    rows = []
    for name in schema.state_variables:
        rows.append(
            {
                "canonical_variable": name,
                "mimic_source": "see configs/real_world/canonical_variables.yaml",
                "eicu_source": "see configs/real_world/canonical_variables.yaml",
                "canonical_unit": schema.units[name],
                "aggregation_4h": _aggregation(name),
                "mimic_missingness": "not_available_until_data_configured",
                "eicu_missingness": "not_available_until_data_configured",
                "included": "true",
            }
        )
    return rows


def _aggregation(name: str) -> str:
    if name == "urine_output":
        return "sum"
    if name in {"lactate", "creatinine", "bilirubin", "platelets", "pao2", "fio2"}:
        return "last"
    return "mean"

