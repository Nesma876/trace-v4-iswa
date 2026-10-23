from trace.evaluation.audit_v2 import handcrafted_rule_test_rows, stress_scenarios


def test_handcrafted_rule_cases() -> None:
    rows = {row["case"]: row["total_violations"] for row in handcrafted_rule_test_rows()}
    assert rows["valid"] == 0
    assert rows["single_rule_violation"] == 1
    assert rows["multi_rule_violation"] == 2
    assert rows["boundary"] == 0


def test_hypotension_severity_inputs_are_monotonic() -> None:
    scenarios = {item["name"]: item for item in stress_scenarios()}
    assert scenarios["hypotension_mild"]["state"]["mean_arterial_pressure"] > scenarios["hypotension_moderate"]["state"]["mean_arterial_pressure"]
    assert scenarios["hypotension_moderate"]["state"]["mean_arterial_pressure"] > scenarios["hypotension_severe"]["state"]["mean_arterial_pressure"]

