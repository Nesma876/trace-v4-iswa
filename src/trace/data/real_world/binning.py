"""Time binning helpers for leakage-safe ICU trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TimeBin:
    bin_index: int
    time_start: datetime
    time_end: datetime


def make_4h_bins(anchor: datetime, n_bins: int, bin_hours: int = 4) -> list[TimeBin]:
    if bin_hours != 4:
        raise ValueError("TRACE V4 primary bins must be 4 hours")
    return [TimeBin(i, anchor + timedelta(hours=4 * i), anchor + timedelta(hours=4 * (i + 1))) for i in range(n_bins)]


def assign_bin(anchor: datetime, timestamp: datetime, bin_hours: int = 4) -> int:
    if timestamp < anchor:
        raise ValueError("timestamp precedes ICU anchor")
    elapsed = timestamp - anchor
    return int(elapsed.total_seconds() // (bin_hours * 3600))


def assert_no_future_state_observation(observation_time: datetime, state_bin: TimeBin) -> None:
    if observation_time >= state_bin.time_end:
        raise ValueError("state observation occurs at or after bin endpoint")


def assert_action_within_current_bin(action_time: datetime, action_bin: TimeBin) -> None:
    if not (action_bin.time_start <= action_time < action_bin.time_end):
        raise ValueError("action is outside the current S_t/A_t interval")

