from trace.evaluation.statistics_v2 import bootstrap_ci


def test_bootstrap_ci_nonzero_for_variable_values() -> None:
    low, high = bootstrap_ci([1, 2, 3, 4, 5], n_bootstrap=500, seed=1)
    assert high > low


def test_bootstrap_ci_zero_for_constant_values() -> None:
    low, high = bootstrap_ci([3, 3, 3, 3, 3], n_bootstrap=500, seed=1)
    assert low == high == 3

