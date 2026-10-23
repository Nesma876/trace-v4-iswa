from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.leakage_audit import audit_patient_splits, audit_temporal_order
from trace.data.preprocessing import missingness_report
from trace.data.synthetic import generate_synthetic_dataset
from trace.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    synthetic = config["synthetic"]
    files = generate_synthetic_dataset(
        synthetic["output_dir"],
        n_patients=synthetic["n_patients"],
        n_steps=synthetic["n_steps"],
        seed=config["seed"],
        missing_probability=synthetic["missing_probability"],
        external_shift=synthetic["external_shift"],
    )
    errors = audit_patient_splits(files)
    for split, path in files.items():
        errors.extend(audit_temporal_order(path))
    if errors:
        raise SystemExit("\n".join(errors))
    report = {split: missingness_report(path) for split, path in files.items()}
    out = Path(synthetic["output_dir"]) / "missingness_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files": files, "missingness_report": str(out)}, indent=2))


if __name__ == "__main__":
    main()

