from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trace.data.real_world.binning import TimeBin, assert_action_within_current_bin, assert_no_future_state_observation, make_4h_bins
from trace.data.real_world.governance import assert_publication_safe_rows, raw_root_from_env
from trace.data.real_world.inventory import synthetic_inventory
from trace.data.real_world.leakage_audit import check_4h_bin_width, check_mask_alignment, check_monotonic_time_bins, check_no_future_actions, check_no_future_observations
from trace.data.real_world.preprocessing import fit_training_only_stats
from trace.data.real_world.schemas import frozen_common_schema
from trace.data.real_world.splits import assert_no_overlapping_patient_ids
from trace.data.real_world.units import convert_value


def test_patient_split_leakage_detection() -> None:
    with pytest.raises(ValueError):
        assert_no_overlapping_patient_ids(
            [
                {"patient_id": "p1", "split": "train"},
                {"patient_id": "p1", "split": "validation"},
            ]
        )


def test_4h_bins_and_monotonicity() -> None:
    anchor = datetime(2020, 1, 1)
    bins = make_4h_bins(anchor, 2)
    assert bins[0].time_end - bins[0].time_start == timedelta(hours=4)
    rows = [{"stay_id": "s1", "bin_index": b.bin_index, "time_start": b.time_start, "time_end": b.time_end} for b in bins]
    assert check_monotonic_time_bins(rows) == []
    assert check_4h_bin_width(rows) == []


def test_no_future_state_or_action_leakage() -> None:
    bin0 = TimeBin(0, datetime(2020, 1, 1), datetime(2020, 1, 1, 4))
    assert_no_future_state_observation(datetime(2020, 1, 1, 3, 59), bin0)
    assert_action_within_current_bin(datetime(2020, 1, 1, 1), bin0)
    rows = [
        {
            "stay_id": "s1",
            "bin_index": 0,
            "time_start": bin0.time_start,
            "time_end": bin0.time_end,
            "state_observation_times": [datetime(2020, 1, 1, 4)],
            "action_times": [datetime(2020, 1, 1, 5)],
        }
    ]
    assert check_no_future_observations(rows)
    assert check_no_future_actions(rows)


def test_unit_conversion_and_training_only_fit() -> None:
    assert round(convert_value("temperature", 98.6, "F", "degC"), 1) == 37.0
    assert convert_value("fio2", 40.0, "%", "fraction") == 0.4
    stats = fit_training_only_stats([{"heart_rate": 80.0}, {"heart_rate": 100.0}], ["heart_rate"], "mimic_train")
    assert stats.means["heart_rate"] == 90.0
    with pytest.raises(ValueError):
        fit_training_only_stats([{"heart_rate": 80.0}], ["heart_rate"], "mimic_validation")


def test_schema_identity_and_mask_alignment() -> None:
    mimic = frozen_common_schema()
    eicu = frozen_common_schema()
    mimic.assert_identity(eicu)
    row = {"heart_rate_value": None, "heart_rate_observation_mask": 1}
    assert check_mask_alignment(row, ("heart_rate",)) == ["heart_rate: missing value has observed mask"]


def test_environment_and_publication_safeguards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_MIMIC_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        raw_root_from_env("mimic")
    with pytest.raises(ValueError):
        assert_publication_safe_rows([{"subject_id": "1", "n": 2}], "publication")


def test_dry_run_inventory_redacts_identifier_columns() -> None:
    inventory = synthetic_inventory()
    text = str(inventory)
    assert "subject_id" not in text
    assert "hadm_id" not in text
    assert "stay_id" not in text
    assert "patientunitstayid" not in text
    assert "IDENTIFIER_REDACTED" in text
