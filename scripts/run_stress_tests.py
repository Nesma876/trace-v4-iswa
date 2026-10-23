from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.preprocessing import read_patient_sequences
from trace.evaluation.clinical_alignment import evaluate_rules


def main() -> None:
    rules = [
        {"function": "lower_bound", "variable": "mean_arterial_pressure", "threshold": 30.0},
        {"function": "upper_bound", "variable": "spo2", "threshold": 100.0},
    ]
    sequences = read_patient_sequences("outputs/data/synthetic/test.csv")
    rows = [row for patient_rows in sequences.values() for row in patient_rows]
    summary = evaluate_rules(rows, rules)
    out = Path("outputs/tables/clinical_stress_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"clinical_stress_summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()

