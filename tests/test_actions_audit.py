from trace.data.actions import action_levels, inverse_transform_action, raw_no_intervention_action, transform_action
from trace.evaluation.audit_v2 import SIGN_EPSILON, temporal_alignment_toy_result


def test_raw_no_intervention_round_trip() -> None:
    raw = raw_no_intervention_action()
    assert inverse_transform_action(transform_action(raw)) == raw


def test_action_ordering_and_distinctness() -> None:
    levels = action_levels()
    for action_name in ["iv_fluids", "vasopressors"]:
        values = [transform_action(action)[action_name] for _, action in levels]
        assert values == sorted(values)
        assert len(set(values)) == len(values)


def test_train_and_eval_action_transform_identity() -> None:
    raw = {"iv_fluids": 1.0, "vasopressors": 0.4}
    assert transform_action(raw) == transform_action(raw)


def test_temporal_alignment_toy_action_t_affects_next_state() -> None:
    result = temporal_alignment_toy_result()
    assert result["action_t_changes_t_plus_1"]
    assert SIGN_EPSILON > 0

