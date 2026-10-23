"""Unit conversion helpers for real-world TRACE variables."""

from __future__ import annotations


def fahrenheit_to_celsius(value: float) -> float:
    return (float(value) - 32.0) * 5.0 / 9.0


def celsius_to_celsius(value: float) -> float:
    return float(value)


def percent_to_fraction(value: float) -> float:
    return float(value) / 100.0


def fraction_to_fraction(value: float) -> float:
    return float(value)


def identity(value: float) -> float:
    return float(value)


CONVERSIONS = {
    ("temperature", "F", "degC"): fahrenheit_to_celsius,
    ("temperature", "degF", "degC"): fahrenheit_to_celsius,
    ("temperature", "C", "degC"): celsius_to_celsius,
    ("temperature", "degC", "degC"): celsius_to_celsius,
    ("fio2", "%", "fraction"): percent_to_fraction,
    ("fio2", "percent", "fraction"): percent_to_fraction,
    ("fio2", "fraction", "fraction"): fraction_to_fraction,
}


def convert_value(variable: str, value: float, source_unit: str, target_unit: str) -> float:
    key = (variable, source_unit, target_unit)
    if key in CONVERSIONS:
        return CONVERSIONS[key](value)
    if source_unit == target_unit:
        return identity(value)
    raise ValueError(f"no conversion registered for {variable}: {source_unit} -> {target_unit}")

