from pathlib import Path

from trace.data.preprocessing import read_patient_sequences
from trace.data.synthetic import generate_synthetic_dataset
from trace.evaluation.temporal import evaluate_rollouts
from trace.models.baseline import PersistenceModel


def test_persistence_rollout_shapes(tmp_path: Path) -> None:
    files = generate_synthetic_dataset(tmp_path, n_patients=16, n_steps=8, seed=5)
    train = read_patient_sequences(files["train"])
    test = read_patient_sequences(files["test"])
    model = PersistenceModel().fit(train)
    metrics = evaluate_rollouts(model, test, [1, 2])
    assert len(metrics) == 2
    assert metrics[0]["mae"] >= 0

