import json
from pathlib import Path

import numpy as np
import pytest

from trace.physionet2019.policy import canonicalize, fio2_audit_view, POLICY, load_frozen_preprocessing
from trace.physionet2019.pipeline import sha
from trace.physionet2019.preprocessing import transform_prefix, forecasting_window
from trace.physionet2019.schema import HEADER
from trace.physionet2019.scoring import score_forecasts, masked_training_nll, common_eligible_variables


def stats(features=None):
    features = ["HR", "FiO2"] if features is None else features
    return {"features": features, "means": [0.] * len(features), "stds": [1.] * len(features),
            "fit_source": "A", "fit_split": "train", "range_policy_hash": sha(POLICY)}


@pytest.mark.parametrize("raw,expected,flag", [(.21,.21,1),(.8,.8,1),(1.,1.,1),(.2,.2,2),(.02,.02,2),
    (21.,None,3),(40.,None,3),(100.,None,3),(10.,None,3),(-.1,None,3),(0.,None,3),(1000.,None,3),(np.nan,None,0)])
def test_fio2_fraction_only_policy(raw, expected, flag):
    view = fio2_audit_view([raw])
    assert view["conversion_flag"][0] == flag
    if expected is None:
        assert np.isnan(view["canonical_FiO2"][0])
    else:
        assert view["canonical_FiO2"][0] == expected == view["raw_FiO2"][0]
    if np.isfinite(raw): assert view["raw_FiO2"][0] == raw


def test_keep_policy_does_not_clip_statistical_extremes():
    original = np.array([[1e8, -999.], [10., 100.]])
    result, flags = canonicalize(original, ["AST", "BaseExcess"])
    np.testing.assert_array_equal(result, original)
    assert (flags == 1).all()


def test_post_policy_inputs_targets_and_labels():
    raw = np.ones((10, len(HEADER)))
    raw[:, HEADER.index("FiO2")] = [.5, 10, .8, .9, 0, .4, .5, .6, .7, .8]
    raw[:, -1] = np.arange(10)  # labels cannot affect selected physiology
    unchanged = raw.copy()
    x = transform_prefix(raw, stats())
    assert x[1, 1] == .5 and x[1, 3] == 0 and x[1, 5] == 1
    _, target, mask = forecasting_window(raw, 3, 4, stats())
    assert target[0, 1] == 0 and not mask[0, 1]
    modified = raw.copy()
    modified[4:, :34] += 1000
    modified[:, -1] = -999
    np.testing.assert_array_equal(transform_prefix(modified, stats())[:4], x[:4])
    np.testing.assert_array_equal(raw, unchanged)
    old_stats = {k:v for k,v in stats().items() if k != "range_policy_hash"}
    with pytest.raises(ValueError, match="historical"):
        transform_prefix(raw, old_stats)


def test_masked_loss_ignores_imputed_nan_and_extreme_targets():
    import torch
    mu = torch.tensor([[1., 2.]], requires_grad=True)
    lv = torch.zeros_like(mu)
    mask = torch.tensor([[True, False]])
    first = masked_training_nll(mu, lv, torch.tensor([[0., float("nan")]]), mask)
    second = masked_training_nll(mu, lv, torch.tensor([[0., 1e30]]), mask)
    torch.testing.assert_close(first, second)
    first.backward()
    assert mu.grad[0, 1] == 0
    zero = masked_training_nll(mu, lv, torch.full_like(mu, float("nan")), torch.zeros_like(mask))
    assert zero.item() == 0


def test_macro_and_micro_are_distinct_and_missingness_only_drives_support():
    y = np.zeros((1000, 2))
    pred = np.column_stack([np.ones(1000), np.full(1000, 10.)])
    mask = np.ones_like(y, dtype=bool)
    mask[100:, 1] = False
    r = score_forecasts(y, pred, mask, ["HR","O2Sat"], stats(["HR","O2Sat"]), model="fixture", horizon=4, cohort="full_available")
    assert r["aggregate"]["MacroMAE"] == 5.5
    assert r["aggregate"]["MacroRMSE"] == 5.5
    assert r["aggregate"]["MicroMAE"] == pytest.approx(2000 / 1100)
    assert r["aggregate"]["MicroRMSE"] == pytest.approx(np.sqrt(11000 / 1100))
    assert r["per_variable"][1]["N_observed_targets"] == 100
    pred[100:, 1] = np.nan
    y[100:, 1] = 1e30
    again = score_forecasts(y, pred, mask, ["HR","O2Sat"], stats(["HR","O2Sat"]), model="fixture", horizon=4, cohort="full_available")
    assert again == r


def test_sparse_variable_is_explicit_and_scaling_is_training_only():
    y = np.zeros((100, 2)); pred = np.ones_like(y); mask = np.ones_like(y, dtype=bool)
    mask[99, 1] = False
    fitted = stats(["HR", "O2Sat"]); fitted["stds"] = [2., 10.]
    r = score_forecasts(y, pred, mask, fitted["features"], fitted, model="fixture", horizon=48, cohort="common_48h")
    assert r["aggregate"]["MacroMAE"] == .5
    assert r["per_variable"][1]["status"] == "INSUFFICIENT_TARGET_OBSERVATIONS"
    assert r["per_variable"][1]["MAE"] is None
    assert r["aggregate"]["eligible_variables"] == ["HR"]
    bad = {**fitted, "fit_source": "B"}
    with pytest.raises(ValueError): score_forecasts(y,pred,mask,fitted["features"],bad,model="fixture",horizon=48,cohort="common_48h")
    pred[0,0] = np.nan
    with pytest.raises(ValueError, match="predictions"):
        score_forecasts(y,pred,mask,fitted["features"],fitted,model="fixture",horizon=48,cohort="common_48h")


def test_common_horizon_variable_intersection():
    counts = {h: [100, 110, 50] for h in [4,12,24,48]}
    counts[48][1] = 99
    assert common_eligible_variables(counts, ["HR", "O2Sat", "Temp"]) == ["HR"]


def test_frozen_source_configs_exclude_actions_context_and_label():
    directory = POLICY.parent
    schema = json.loads((directory / "FROZEN_STATE_SCHEMA.yaml").read_text())
    assert len(schema["dynamic_variables"]) == 33
    assert not set(schema["dynamic_variables"]) & {"EtCO2", "SepsisLabel", "Age", "ICULOS"}
    assert schema["action_dim"] == 0 and schema["context_model_inputs"] == []
