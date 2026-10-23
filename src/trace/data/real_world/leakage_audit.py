"""Automated leakage checks for TRACE V4 real-world sequences."""

from __future__ import annotations

from datetime import timedelta
from typing import Any


def check_monotonic_time_bins(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["stay_id"], []).append(row)
    for stay_id, vals in grouped.items():
        indices = [int(v["bin_index"]) for v in vals]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            errors.append(f"{stay_id}: non-monotonic or duplicate bins")
    return errors


def check_4h_bin_width(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        if row["time_end"] - row["time_start"] != timedelta(hours=4):
            errors.append(f"{row['stay_id']}:{row['bin_index']}: bin width is not 4h")
    return errors


def check_no_future_observations(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        for timestamp in row.get("state_observation_times", []):
            if timestamp >= row["time_end"]:
                errors.append(f"{row['stay_id']}:{row['bin_index']}: future state observation")
    return errors


def check_no_future_actions(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        for timestamp in row.get("action_times", []):
            if timestamp < row["time_start"] or timestamp >= row["time_end"]:
                errors.append(f"{row['stay_id']}:{row['bin_index']}: action outside S_t interval")
    return errors


def check_mask_alignment(row: dict[str, Any], state_variables: tuple[str, ...]) -> list[str]:
    errors = []
    for variable in state_variables:
        value = row.get(f"{variable}_value")
        mask = row.get(f"{variable}_observation_mask")
        if value is None and mask not in {0, False}:
            errors.append(f"{variable}: missing value has observed mask")
    return errors

