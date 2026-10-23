"""Synthetic-only JBI reporting audit: never load patients or fitted checkpoints."""
import json
from statistics import NormalDist

import numpy as np
import pytest
import torch

from trace.physionet2019.internal import make_model, rollout
from trace.physionet2019.internal_evaluation import F, FIELDS, add_statistics, summarize
from trace.physionet2019.preprocessing import forecasting_window
from trace.physionet2019.policy import POLICY
from trace.physionet2019.pipeline import sha
from trace.physionet2019.schema import DYNAMIC, HEADER


@pytest.mark.parametrize("name", ["gru", "transformer", "rssm"])
@pytest.mark.parametrize("perturbation", ["values", "mask", "label"])
def test_future_changes_cannot_change_forecast(name, perturbation):
    torch.manual_seed(17)
    raw = np.ones((10, len(HEADER)))
    raw[:, HEADER.index("FiO2")] = .5
    changed = raw.copy()
    if perturbation == "values":
        changed[4:, :34] = 10000
    elif perturbation == "mask":
        changed[4:, :34] = np.nan
    else:
        changed[:, -1] = 1 - raw[:, -1]
    features = [v for v in DYNAMIC if v != "EtCO2"]
    stats = dict(features=features, means=[0.]*33, stds=[1.]*33,
                 fit_source="A", fit_split="train", range_policy_hash=sha(POLICY))
    model = make_model(name)
    forecasts = []
    for source in (raw, changed):
        context, target, mask = forecasting_window(source, 3, 4, stats)
        forecasts.append(rollout(model, torch.from_numpy(context[None]), 4))
        # Scoring-only future tensors are deliberately not part of the API.
        target[:] = -123456
        mask[:] = ~mask
        again = rollout(model, torch.from_numpy(context[None]), 4)
        for a, b in zip(forecasts[-1], again):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(*forecasts):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_recursive_feedback_and_never_observed_sentinel():
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            self.seen.append(x.clone())
            return x[..., :33] + 2, torch.zeros_like(x[..., :33])

    model = Probe()
    x = torch.zeros(1, 4, 99)
    x[..., 66] = -1
    x[..., 67:] = 3
    mean, variance = rollout(model, x, 4)
    torch.testing.assert_close(mean[0, :, 0], torch.tensor([2., 4., 6., 8.]))
    for k, seen in enumerate(model.seen):
        assert seen[0, -1, 66] == -1
        assert torch.all(seen[0, -1, 67:] == 3+k)
        if k:
            assert not seen[0, -1, 33:66].any()
            torch.testing.assert_close(seen[:, -1, :33], mean[:, k-1])
    assert torch.all(variance == 1)


@pytest.mark.parametrize("name", ["gru", "transformer", "rssm"])
def test_neural_variance_and_deterministic_prior(name):
    model = make_model(name)
    x = torch.randn(2, 4, 99)
    a = rollout(model, x, 4)
    torch.manual_seed(999)
    b = rollout(model, x, 4)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.isfinite(a[1]).all()
    assert (a[1] >= np.exp(-6)-1e-8).all()
    assert (a[1] <= np.exp(3)+1e-5).all()


def accumulate(y, pred, mask=None, sd=None, owners=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mask = np.ones_like(y, bool) if mask is None else mask
    sd = np.ones_like(y) if sd is None else sd
    owners = np.zeros(len(y), int) if owners is None else owners
    dest = np.zeros((owners.max()+1, y.shape[-1], len(FIELDS)))
    add_statistics(dest, owners, y, pred, sd, mask, y[:, 0], mask[:, 0])
    return dest


def test_analytic_interval_quantiles_and_observed_coverage():
    # Errors 0, 1, 1.5, 2; last value is masked out at the endpoint.
    y = np.array([0., 1., 1.5, 2.]).reshape(4, 1, 1)
    mask = np.ones_like(y, bool)
    mask[-1] = False
    stats = accumulate(y, np.zeros_like(y), mask).sum(0)
    per, _, _ = summarize(stats, minimum=1)
    for p, expected in ((50, 1/3), (80, 2/3), (95, 1.)):
        assert per[f"coverage{p}"][0] == pytest.approx(expected)
        assert per[f"width{p}"][0] == pytest.approx(2*NormalDist().inv_cdf((1+p/100)/2))
    assert per["width95"] >= per["width80"] >= per["width50"]
    assert stats[0, F["n"]] == 3


def test_hand_calculated_dynamic_mask_and_macro():
    # Valid increments: var 0 only 1->2 (error 1); var 1 all (errors 0,2,1).
    y = np.array([[[0., 0.], [1., 1.], [4., 3.], [10., 6.]]])
    p = np.array([[[0., 0.], [2., 1.], [5., 5.], [9., 7.]]])
    mask = np.ones_like(y, bool)
    mask[:, 2, 0] = False
    stats = accumulate(y, p, mask).sum(0)
    per, agg, _ = summarize(stats, minimum=1)
    np.testing.assert_equal(stats[:, F["dn"]], [1, 3])
    np.testing.assert_allclose(per["dynamic_mae"], [1, 1])
    assert agg["dynamic_mae"] == pytest.approx(1)
    changed = y.copy()
    changed[~mask] = 1e9
    np.testing.assert_allclose(accumulate(changed, p, mask).sum(0), stats)
    # The baseline-to-step-1 difference is never in dn: h=4 has at most 3 pairs.
    assert stats[1, F["dn"]] == 3


def test_population_change_variance_and_zero_denominator():
    y = np.array([0., 1., 3., 7.]).reshape(1, 4, 1)
    p = np.array([0., 2., 6., 14.]).reshape(1, 4, 1)
    per, agg, _ = summarize(accumulate(y, p).sum(0), minimum=1)
    assert per["change_variance_ratio"][0] == pytest.approx(4)
    assert agg["change_variance_ratio"] == pytest.approx(4)
    constant = np.arange(4.).reshape(1, 4, 1)
    per, agg, _ = summarize(accumulate(constant, p).sum(0), minimum=1)
    assert np.isnan(per["change_variance_ratio"][0])
    assert np.isnan(agg["change_variance_ratio"])


def test_production_bootstrap_keeps_patient_clusters_and_paired_differences(tmp_path, monkeypatch):
    from trace.physionet2019 import internal_reporting as report
    # Three patients, with two origins pooled for patient 0; counts differ from
    # error totals, so averaging origins/patients as independent units will fail.
    output = tmp_path / "audit"
    for sub in ("analysis", "tables", "local"):
        (output/sub).mkdir(parents=True)
    (output/"analysis/common_variable_eligibility.json").write_text(json.dumps({"eligible": [True]}))
    names = ["linear_ar", "gru", "transformer", "rssm"]
    cfg = {"models": dict.fromkeys(names), "seeds": [17,23,31,43,59], "bootstrap": {"seed":4317}}
    arrays = {}
    for mi, name in enumerate(names):
        for si, seed in enumerate(cfg["seeds"]):
            a = np.zeros((2, 5, 3, 1, len(FIELDS)))
            for ci in range(2):
                for hi in range(5):
                    a[ci, hi, :, 0, F["n"]] = [200,100,100]
                    a[ci, hi, :, 0, F["dn"]] = [200,100,100]
                    # Patient 0 aggregates 2 origins, errors 1 and 9, 100 each.
                    a[ci, hi, :, 0, F["abs"]] = np.array([1000,200,800])*(mi+1)+(si*10)
                    a[ci, hi, :, 0, F["da"]] = a[ci, hi, :, 0, F["abs"]]
                    a[ci, hi, :, 0, F["nll"]] = a[ci, hi, :, 0, F["abs"]]
                    a[ci, hi, :, 0, F["c95"]] = [190,95,95]
            arrays[(name,seed)] = a
            np.savez(output/"local"/f"{name}_{seed}_patient_stats.npz", statistics=a)
    monkeypatch.setattr(report, "OUT", output)
    monkeypatch.setattr(report, "config", lambda: cfg)
    report.bootstrap(replicates=25)
    actual = np.load(output/"local/bootstrap_replicates.npz")["values"]
    rng = np.random.default_rng(4317)
    expected = []
    for _ in range(25):
        si = rng.integers(0,5,5)
        pi = rng.integers(0,3,3)
        weights = np.bincount(pi,minlength=3)
        # The same patient multiplicity multiplies both origins of patient 0.
        expected.append([np.mean([np.dot(weights, arrays[(name,cfg["seeds"][s])][1,4,:,0,F["abs"]]) /
                                      np.dot(weights, [200,100,100]) for s in si]) for name in names])
    np.testing.assert_allclose(actual[:,:,1], expected)
    import csv
    rows = list(csv.DictReader((output/"tables/paired_comparisons.csv").open()))
    row = next(r for r in rows if r["dimension"] == "48h_state" and r["model_a"] == "linear_ar" and r["model_b"] == "gru")
    lo, hi = np.quantile(actual[:,0,1]-actual[:,1,1], [.025,.975])
    assert float(row["ci_low"]) == pytest.approx(lo)
    assert float(row["ci_high"]) == pytest.approx(hi)
