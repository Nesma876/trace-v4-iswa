from trace.models.comparative import build_model, transformer_like_features


def _row(time_index: int, hr: float) -> dict:
    return {
        "patient_id": "p",
        "time_index": time_index,
        "iv_fluids": 0.1,
        "vasopressors": 0.2,
        "heart_rate": hr,
        "heart_rate_mask": 1,
        "respiratory_rate": 20.0,
        "respiratory_rate_mask": 1,
        "spo2": 96.0,
        "spo2_mask": 1,
        "temperature": 37.0,
        "temperature_mask": 1,
        "mean_arterial_pressure": 75.0,
        "mean_arterial_pressure_mask": 1,
        "lactate": 2.0,
        "lactate_mask": 1,
        "creatinine": 1.1,
        "creatinine_mask": 1,
    }


def test_transformer_features_are_causal() -> None:
    means = {key: value for key, value in _row(0, 80.0).items() if isinstance(value, float)}
    history = [_row(0, 80.0), _row(1, 82.0)]
    future = history + [_row(2, 140.0)]
    action = {"iv_fluids": 0.1, "vasopressors": 0.2}
    assert transformer_like_features(history, action, means) != transformer_like_features(future, action, means)
    assert transformer_like_features(history, action, means) == transformer_like_features(history, action, means)


def test_all_v1_models_construct() -> None:
    for name in ["linear_ar", "gru", "transformer", "rssm"]:
        assert build_model(name).name == name

