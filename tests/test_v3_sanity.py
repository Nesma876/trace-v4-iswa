from scripts.run_v3_sanity import _horizon_indexing_toy


def test_horizon_indexing_toy_selects_expected_targets() -> None:
    rows = _horizon_indexing_toy()
    checks = [row for row in rows if row["horizon_steps"] != "alignment"]
    assert all(row["indexing_correct"] for row in checks)

