from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.preprocessing import read_patient_sequences
from trace.evaluation.reliability import deterministic_reliability_summary
from trace.evaluation.temporal import evaluate_rollouts
from trace.models.baseline import PersistenceModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.experiment)
    model = PersistenceModel.load(checkpoint)
    data_dir = Path("outputs/data/synthetic")
    horizons = [1, 3, 6, 12]
    metrics = evaluate_rollouts(model, read_patient_sequences(data_dir / "test.csv"), horizons)
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "temporal_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    reliability = deterministic_reliability_summary(metrics[0]["mae"])
    reliability_path = out_dir / "reliability_summary.json"
    reliability_path.write_text(json.dumps(reliability, indent=2), encoding="utf-8")
    print(json.dumps({"temporal_metrics": str(metrics_path), "reliability": str(reliability_path), "first_row": metrics[0]}, indent=2))


if __name__ == "__main__":
    main()

