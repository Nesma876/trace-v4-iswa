import numpy as np
import pytest
import torch

from trace.physionet2019.internal import (make_model, posterior_forward, generated_row,
                                         rollout, windows, load_patients)
from trace.physionet2019.scoring import masked_training_nll


def test_context_windows_do_not_cross_patients():
    patients = [(np.full((7, 99), i, np.float32), np.full((7, 33), i, np.float32),
                 np.ones((7, 33), bool), False) for i in (1, 9)]
    x, y, m = windows(patients)
    assert x.shape == (6, 4, 99)
    assert torch.all(x[:3] == 1) and torch.all(x[3:] == 9)
    assert torch.all(y[:3] == 1) and torch.all(m)


def test_generated_rows_are_not_observations():
    previous = torch.ones(2, 99)
    previous[:, 66] = -1
    result = generated_row(torch.zeros(2, 33), previous)
    assert not result[:, 33:66].any()
    assert torch.all(result[:, 66] == -1)
    assert torch.all(result[:, 67:] == 2)


def test_transformer_causal_mask_real_dimensions():
    torch.manual_seed(17)
    model = make_model("transformer").eval()
    x = torch.randn(2, 4, 99)
    altered = x.clone()
    altered[:, 2:] += 100
    with torch.no_grad():
        for a, b in zip(model(x), model(altered)):
            torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-6)


def test_rssm_missing_posterior_targets_cannot_influence_output():
    torch.manual_seed(17)
    model = make_model("rssm")
    x = torch.randn(2, 4, 99)
    y = torch.randn(2, 4, 33)
    mask = torch.rand_like(y) > .6
    modified = y.clone()
    modified[~mask] = float("nan")
    torch.manual_seed(23)
    a = posterior_forward(model, x, y, mask)
    torch.manual_seed(23)
    b = posterior_forward(model, x, modified, mask)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
        assert torch.isfinite(left).all()


def test_rssm_rollout_never_calls_posterior():
    model = make_model("rssm")
    def forbidden(*args):
        raise AssertionError("Posterior accessed during forecast")
    model.posterior.forward = forbidden
    mu, var = rollout(model, torch.zeros(2, 4, 99), 4)
    assert mu.shape == var.shape == (2, 4, 33)
    assert torch.isfinite(mu).all() and torch.all(var > 0)


def test_target_mask_loss_has_zero_missing_gradient():
    mu = torch.randn(2, 33, requires_grad=True)
    lv = torch.zeros_like(mu, requires_grad=True)
    target = torch.full_like(mu, float("nan"))
    mask = torch.zeros_like(mu, dtype=torch.bool)
    mask[:, :3] = True
    target[:, :3] = 1
    loss = masked_training_nll(mu, lv, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and not mu.grad[~mask].any() and not lv.grad[~mask].any()


def test_external_source_not_supported_by_internal_loader():
    with pytest.raises(ValueError, match="Only Set A"):
        load_patients("B")


def test_new_configs_allowed_but_frozen_config_edits_rejected(tmp_path,monkeypatch):
    from trace.physionet2019 import pipeline, policy
    from trace.physionet2019.pipeline import sha, write_json
    audit=tmp_path/"audit"
    audit.mkdir()
    split=tmp_path/"splits/setA_patient_split.csv"
    split.parent.mkdir()
    split.write_text("file_id,split\np000001.psv,train\n")
    frozen={"original.yaml":"original-hash"}
    stats={"config_hashes":frozen,"range_policy_hash":sha(policy.POLICY),"split_hash":sha(split)}
    write_json(audit/"frozen_training_stats.json",stats)
    write_json(audit/"final_freeze_manifest.json",{"preprocessing_statistics_hash":sha(audit/"frozen_training_stats.json")})
    monkeypatch.setattr(pipeline,"config_hashes",lambda:{**frozen,"new_model_config.json":"new-hash"})
    assert policy.load_frozen_preprocessing(tmp_path)==stats
    monkeypatch.setattr(pipeline,"config_hashes",lambda:{"original.yaml":"modified","new_model_config.json":"new-hash"})
    with pytest.raises(ValueError,match="Frozen configurations changed"):
        policy.load_frozen_preprocessing(tmp_path)


def test_patient_statistics_masked_and_endpoint_variance():
    from trace.physionet2019.internal_evaluation import add_statistics, summarize, F, FIELDS
    truth = np.broadcast_to(np.arange(4.)[None,:,None], (120,4,33)).copy()
    truth += np.arange(120)[:,None,None]*np.arange(4)[None,:,None]/100
    pred = truth + .25
    mask = np.ones_like(truth, dtype=bool)
    mask[:30,:,0] = False
    dest = np.zeros((3,33,len(FIELDS)))
    owners = np.arange(120)%3
    add_statistics(dest,owners,truth,pred,np.ones_like(truth),mask,truth[:,0],mask[:,0])
    per, agg, eligible = summarize(dest.sum(0))
    assert not eligible[0] and eligible[1:].all()
    assert np.isclose(agg["mae"],.25)
    assert np.isclose(agg["dynamic_mae"],0)
    assert np.isclose(agg["change_variance_ratio"],1)
    assert np.isclose(agg["variance_ratio"],1)
    assert np.isclose(agg["trajectory_correlation"],1)
    assert np.isclose(agg["change_from_baseline_error"],.25)
    assert dest[:,0,F["n"]].sum() == 90
    assert np.isclose(agg["coverage50"],1)
    assert np.isclose(agg["coverage80"],1)


def test_patient_statistics_never_score_unobserved_truth():
    from trace.physionet2019.internal_evaluation import add_statistics, FIELDS
    rng = np.random.default_rng(17)
    y=rng.normal(size=(4,4,33))
    m=rng.uniform(size=y.shape)>.5
    changed=y.copy()
    changed[~m]=1e6
    a,b=np.zeros((2,33,len(FIELDS))),np.zeros((2,33,len(FIELDS)))
    owners=np.arange(4)%2
    for dest,truth in ((a,y),(b,changed)):
        add_statistics(dest,owners,truth,np.zeros_like(y),np.ones_like(y),m,y[:,0],m[:,0])
    np.testing.assert_allclose(a,b)


def test_reporting_pipeline_on_synthetic_sufficient_statistics(tmp_path,monkeypatch):
    import json
    from trace.physionet2019 import internal_reporting as report
    from trace.physionet2019.internal_evaluation import EVAL_HORIZONS, FIELDS, add_statistics
    from trace.physionet2019.schema import DYNAMIC
    from trace.physionet2019.pipeline import write_json, write_csv
    output=tmp_path/"output"
    checkpoints=tmp_path/"checkpoints"
    for folder in (output/"tables",output/"analysis",output/"local",output/"figures"):
        folder.mkdir(parents=True)
    audit=tmp_path/"v4_physionet2019/audit"
    audit.mkdir(parents=True)
    cfg={**report.config(),"bootstrap":{"replicates":12,"seed":17}}
    features=[v for v in DYNAMIC if v!="EtCO2"]
    monkeypatch.setattr(report,"OUT",output)
    monkeypatch.setattr(report,"CHECKPOINTS",checkpoints)
    monkeypatch.setattr(report,"config",lambda:cfg)
    monkeypatch.setattr(report,"verify",lambda:{"features":features,"stds":[1.]*33})
    write_json(output/"analysis/common_variable_eligibility.json",{"eligible":[True]*33})
    rng=np.random.default_rng(17)
    truth=rng.normal(size=(300,48,33))
    mask=np.ones_like(truth,dtype=bool)
    owners=np.arange(300)%3
    missingness=[]
    for index,v in enumerate(features):
        missingness.append({"variable":v,"missingness_percentage":10+index*2.8})
    write_csv(audit/"trainA_missingness.csv",missingness)
    a=np.zeros((2,len(EVAL_HORIZONS),3,33,len(FIELDS)))
    for ci in range(2):
        for hi,h in enumerate(EVAL_HORIZONS):
            add_statistics(a[ci,hi],owners,truth[:,:h],truth[:,:h]+.2,np.ones_like(truth[:,:h]),mask[:,:h],truth[:,0],mask[:,0])
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            np.savez_compressed(output/"local"/f"{name}_{seed}_patient_stats.npz",statistics=a,subgroup=[True,False,False],inference_time=1.)
            write_json(checkpoints/name/str(seed)/"metadata.json",{"model":name,"seed":seed,"parameters":1,"convergence_status":"CONVERGED","validation_nll":1+seed/10000})
            write_csv(output/"tables"/f"selective_{name}_{seed}.csv",[{"model":name,"seed":seed,"cohort":"common_48h","horizon":48,"retained_fraction":p,"MacroMAE":.2,"NLL":1.} for p in [1.,.9,.8,.7,.5]])
    summaries=report.tables()
    report.bootstrap(replicates=12)
    consistency=report.jbi_complementary_outputs(summaries)
    assert {r["dimension"] for r in consistency} >= {"validation_nll","rollout_4h","rollout_48h","latent_state_fidelity"}
    conclusions=report.conclusions(summaries)
    assert conclusions["H2"]=="not tested"
    for row in report.read_table("paired_comparisons"):
        assert abs(float(row["mean_difference_a_minus_b"]))<1e-10
    report.figures(summaries)
    assert len(list((output/"figures").glob("*.png")))==18
    write_json(output/"analysis/bootstrap_provenance.json",{"replicates":2000,"row_level_resampling":False})
    write_json(output/"analysis/jbi_reproducibility_checks.json",{"all_checks":True})
    assert report.assert_jbi_output_contract()
    preflight=report.finisher_jbi_preflight()
    assert preflight["status"]=="READY" and preflight["held_out_data_opened"] is False


def test_jbi_kendall_and_ties():
    from trace.physionet2019.internal_reporting import _kendall_tau, _mean_rank
    ranks,order=_mean_rank([("a",1.),("b",2.),("c",3.)])
    assert order==["a","b","c"] and ranks["a"]==1
    assert _kendall_tau(ranks,ranks)==1
    assert _kendall_tau(ranks,{"a":3,"b":2,"c":1})==-1
