import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from trace.physionet2019.data import discover, patient_split, read_psv, split_for
from trace.physionet2019.schema import DYNAMIC, HEADER
from trace.physionet2019.preprocessing import transform_prefix, forecasting_window, eligible_origins, mask_history, fit_preprocessing
from trace.physionet2019.evaluation import external_evaluation, trajectory_metrics, metric_change


def raw_data(n=60):
    raw = np.ones((n, len(HEADER)))
    raw[:, :len(DYNAMIC)] = np.arange(n)[:, None] + np.arange(len(DYNAMIC))[None, :] + 1
    raw[:, -2] = np.arange(1, n + 1)
    raw[:, -1] = 0
    return raw


def write_fixture(path, raw=None, header=HEADER):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, raw_data() if raw is None else raw, delimiter="|", header="|".join(header), comments="")
    return path


def stats():
    return {"features": ["HR", "O2Sat"], "means": [10., 20.], "stds": [2., 4.], "fit_source": "A", "fit_split": "train"}


def test_parser_and_nested_discovery(tmp_path):
    p = write_fixture(tmp_path / "training_setA" / "training" / "p000001.psv")
    assert read_psv(p).shape == (60, 41)
    assert discover(tmp_path, "A") == [p]
    write_fixture(tmp_path / "training_setA" / "duplicate" / p.name)
    with pytest.raises(ValueError, match="Duplicate"):
        discover(tmp_path, "A")


@pytest.mark.parametrize("kind", ["hour", "gap", "label", "infinity", "header"])
def test_invalid_psv(tmp_path, kind):
    raw = raw_data()
    header = HEADER
    if kind == "hour": raw[2, -2] = 2
    if kind == "gap": raw[2:, -2] += 1
    if kind == "label": raw[4, -1] = 1
    if kind == "infinity": raw[2, 0] = np.inf
    if kind == "header": header = ("BAD",) + HEADER[1:]
    with pytest.raises(ValueError):
        read_psv(write_fixture(tmp_path / "bad.psv", raw, header))


def test_split_determinism_and_no_silent_regeneration(tmp_path):
    paths = [Path(f"p{i:06}.psv") for i in range(100)]
    dest = tmp_path / "split.csv"
    a = patient_split(paths, dest)
    assert patient_split(paths[::-1], dest) == a
    assert set(a.values()) == {"train", "validation", "internal_test"}
    assert len(a) == 100
    with pytest.raises(ValueError, match="Frozen"):
        patient_split(paths[:-1], dest)
    with pytest.raises(ValueError):
        patient_split(paths + paths[:1], dest)


def test_causal_masks_imputation_and_future_label_exclusion():
    raw = raw_data()
    raw[:2, 0] = np.nan
    raw[3, 0] = np.nan
    transformed = transform_prefix(raw, stats())
    assert transformed[0, 0] == 0  # training mean, not a future observation
    assert transformed[0, 2] == 0
    assert transformed[0, 4] == -1
    assert transformed[3, 0] == transformed[2, 0]
    assert transformed[3, 4] == 1
    changed = raw.copy()
    changed[6:, :34] += 10000
    changed[:, -1] = 1
    np.testing.assert_array_equal(transform_prefix(changed, stats())[:6], transformed[:6])
    x, y, mask = forecasting_window(raw, 5, 4, stats())
    x2, _, _ = forecasting_window(changed, 5, 4, stats())
    np.testing.assert_array_equal(x, x2)
    assert x.shape == (4, 6) and y.shape == (4, 2)
    assert mask.all()
    illegal = {**stats(), "features": ["HR", "SepsisLabel"]}
    with pytest.raises(ValueError): transform_prefix(raw, illegal)


def test_observed_targets_only_and_mask_stress():
    raw = raw_data()
    raw[4, 0] = np.nan
    _, y, mask = forecasting_window(raw, 3, 4, stats())
    assert not mask[0, 0] and y[0, 0] == 0
    masked = mask_history(raw, .5, 4)
    np.testing.assert_array_equal(masked[:, 34:], raw[:, 34:])
    assert np.isnan(masked[:, :34]).sum() > np.isnan(raw[:, :34]).sum()
    np.testing.assert_array_equal(mask_history(raw, .5, 4), masked)


def test_training_only_fit(tmp_path):
    p = write_fixture(tmp_path / "training_setA" / "p1.psv")
    cfg = {"minimum_observations": 1, "minimum_observed_patients": 1, "minimum_standard_deviation": 1e-8}
    fitted = fit_preprocessing([p], {p.name: "train"}, cfg)
    assert fitted["means"][0] == 30.5
    with pytest.raises(ValueError): fit_preprocessing([p], {p.name: "validation"}, cfg)
    b = write_fixture(tmp_path / "training_setB" / "p1.psv")
    with pytest.raises(ValueError, match="Set A"):
        fit_preprocessing([b], {b.name: "train"}, cfg)
    changed = raw_data()
    changed[:, 0] = 1e6
    # Historical raw FiO2 statistics are deliberately unusable after final freeze.
    with pytest.raises(ValueError, match="historical"):
        transform_prefix(changed, fitted)
    hr_stats = {**fitted, "features": ["HR"], "means": fitted["means"][:1], "stds": fitted["stds"][:1]}
    normalized = transform_prefix(changed, hr_stats)
    assert normalized[0, 0] > 10000  # never refit on the application source


def test_48h_common_cohort():
    assert list(eligible_origins(52, 48)) == [3]
    assert not list(eligible_origins(51, 48))
    assert list(eligible_origins(52, 4, common_48h=True)) == [3]
    assert len(eligible_origins(52, 4)) > 1


@pytest.mark.parametrize("horizon", [4, 12, 24, 48])
def test_all_horizon_boundaries(horizon):
    assert list(eligible_origins(horizon + 4, horizon)) == [3]
    assert not list(eligible_origins(horizon + 3, horizon))
    assert list(eligible_origins(56, horizon, common_48h=True)) == list(range(3, 8))


def test_downloader_resumes_and_validates(tmp_path, monkeypatch):
    import io
    from trace.physionet2019 import download
    fixture = write_fixture(tmp_path / "fixture.psv")
    payload = fixture.read_bytes()
    destination = tmp_path / "output" / "p000001.psv"
    destination.parent.mkdir()
    destination.with_suffix(".psv.part").write_bytes(payload[:100])
    seen = []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 100-{len(payload)-1}/{len(payload)}", "Content-Length": str(len(payload)-100)}

    def fake(url, headers=None):
        seen.append(headers)
        return Response(payload[100:])

    monkeypatch.setattr(download, "open_official", fake)
    download.fetch_file(download.BASE + "p000001.psv", destination, hashlib.sha256(payload).hexdigest())
    assert destination.read_bytes() == payload
    assert seen == [{"Range": "bytes=100-"}]
    download.fetch_file(download.BASE + "p000001.psv", destination)
    assert len(seen) == 1  # completed files are parsed, not downloaded again
    with pytest.raises(ValueError, match="checksum"):
        download.fetch_file(download.BASE + "p000001.psv", destination, "incorrect")


def test_downloader_official_only_and_bad_range(tmp_path, monkeypatch):
    import io
    from trace.physionet2019 import download
    with pytest.raises(ValueError): download.open_official("https://example.org/fake.psv")
    path = tmp_path / "p000001.psv"
    path.with_suffix(".psv.part").write_bytes(b"partial")

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 1-4/5"}

    monkeypatch.setattr(download, "open_official", lambda *args: Response(b"data"))
    with pytest.raises(ValueError, match="range"):
        download.fetch_file(download.BASE + "p000001.psv", path)
    assert not path.exists()


def test_external_cli_requires_explicit_unlock():
    script = Path(__file__).resolve().parents[1] / "scripts/physionet2019/evaluate_external.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "locked" in result.stderr
    source = script.read_text(encoding="utf-8")
    assert "completed_event.exists()" in source
    assert source.index("completed_event.exists()") < source.index("from trace.physionet2019.external_evaluation")


def test_inventory_aggregates_invalid_and_duplicate_files(tmp_path, monkeypatch):
    from trace.physionet2019 import pipeline
    monkeypatch.setattr(pipeline, "code_commit", lambda: "fixture-only")
    root, out = tmp_path / "raw", tmp_path / "out"
    write_fixture(root / "training_setA" / "p1.psv")
    write_fixture(root / "training_setA" / "duplicate" / "p1.psv")
    write_fixture(root / "training_setB" / "p2.psv")
    bad = root / "training_setA" / "p3.psv"
    bad.write_text("not a header\n")
    result = pipeline.inventory(root, out)
    assert result["sets"]["A"]["duplicate_patient_filenames"] == 1
    assert result["sets"]["A"]["invalid_files"] == 1
    assert not result["complete"]
    assert result["sets"]["B"]["rows_hours_distribution"] == "withheld: Set B locked"
    assert "p1.psv" not in (out / "data_inventory.json").read_text()


def test_provenance_hash_is_portable_but_raw_bytes_are_not_normalized(tmp_path):
    from trace.physionet2019.pipeline import sha
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"a": 1}\n')
    expected = sha(artifact)
    artifact.write_bytes(b'{"a": 1}\r\n')
    assert sha(artifact) == expected
    raw = tmp_path / "raw.psv"
    raw.write_bytes(b'header\n')
    expected = sha(raw)
    raw.write_bytes(b'header\r\n')
    assert sha(raw) != expected


def test_external_lock_and_once_only(tmp_path):
    calls = []
    with pytest.raises(PermissionError): external_evaluation(lambda: calls.append(1))
    with pytest.raises(PermissionError): external_evaluation(lambda: calls.append(1), unlock_external_test=True)
    assert calls == []
    artifacts = {}
    for kind in ("model", "protocol", "preprocessing", "feature_schema", "split"):
        p = tmp_path / kind
        p.write_text("synthetic test fixture")
        artifacts[kind] = {"file": kind, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    f = tmp_path / "freeze.json"
    f.write_text(json.dumps({"development_source": "A", "calibration_source": "A_validation", "model_selection_complete": True, "artifacts": artifacts}))
    kwargs = {"unlock_external_test": True, "freeze_file": f, "event_file": tmp_path / "event.json"}
    external_evaluation(lambda: calls.append(1), **kwargs)
    with pytest.raises(FileExistsError): external_evaluation(lambda: calls.append(2), **kwargs)
    assert calls == [1]
    (tmp_path / "model").write_text("changed")
    with pytest.raises(PermissionError): external_evaluation(lambda: None, **kwargs)


def test_metrics_perfect_and_missing_adjacent():
    y = np.arange(6.).reshape(3, 2)
    mask = np.ones_like(y, dtype=bool)
    results = trajectory_metrics(y, y.copy(), np.ones_like(y), mask, np.zeros(2), [True, True])
    assert results[0]["MAE"] == results[0]["DynamicMAE"] == 0
    assert results[0]["coverage_90"] == 1
    assert results[0]["variance_ratio"] == pytest.approx(1)
    mask[1, :] = False
    assert trajectory_metrics(y, y, np.ones_like(y), mask, np.zeros(2), [True, True])[0]["DynamicMAE"] is None
    assert metric_change(2, 1) == {"absolute_change": -1, "relative_change": -.5}


def test_linear_action_free_fixture_only():
    from trace.physionet2019.models import build_model
    model = build_model("linear_ar", 2)
    x = np.random.default_rng(42).normal(size=(30, 6))
    # Architecture inference fixture only; no regression fitting in pre-modelling audit.
    model.weights = np.zeros((7, 2))
    model.weights[1, 0] = model.weights[2, 1] = 1
    model.variance = np.ones(2)
    mu, logvar = model.predict(x)
    assert mu.shape == logvar.shape == (30, 2) and model.action_dim == 0


@pytest.mark.parametrize("name", ["gru", "transformer", "rssm"])
def test_neural_action_free_and_causal(name):
    torch = pytest.importorskip("torch")
    from trace.physionet2019.models import build_model
    torch.manual_seed(4)
    model = build_model(name, 3).eval()
    x = torch.randn(2, 8, 9)
    with torch.no_grad():
        before = model(x)[0]
        x[:, 5:] += 100
        after = model(x)[0]
    assert before.shape == (2, 8, 3)
    torch.testing.assert_close(before[:, :5], after[:, :5])
    assert model.action_dim == 0


@pytest.mark.parametrize("name", ["GRUWorldModel", "CausalTransformerWorldModel", "RSSMWorldModel"])
def test_frozen_v31_default_numerical_regression(name):
    torch = pytest.importorskip("torch")
    from trace.models import neural_v3
    # Read the immutable starting commit, not a second implementation maintained in-tree.
    old_source = subprocess.check_output(["git", "show", "5e3ba3da011e5c478651c440b0e35e2fc8cd6c4e:src/trace/models/neural_v3.py"], text=True, timeout=30)
    old = types.ModuleType("trace_v31_frozen_test")
    sys.modules[old.__name__] = old
    exec(compile(old_source, "frozen_v31", "exec"), old.__dict__)
    torch.manual_seed(4317)
    expected = getattr(old, name)().eval()
    torch.manual_seed(4317)
    current = getattr(neural_v3, name)().eval()
    assert expected.state_dict().keys() == current.state_dict().keys()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        for a, b in zip(expected(x), current(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
