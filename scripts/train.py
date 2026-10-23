from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.preprocessing import read_patient_sequences
from trace.models.baseline import PersistenceModel
from trace.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="persistence")
    parser.add_argument("--config", default="configs/experiments/main.yaml")
    args = parser.parse_args()
    if args.model != "persistence":
        raise SystemExit("Milestone 1 implements only the persistence model.")
    config = load_config(args.config)["experiment"]
    train_path = Path(config["data_dir"]) / "train.csv"
    model = PersistenceModel().fit(read_patient_sequences(train_path))
    checkpoint = Path(config["checkpoint_dir"]) / "persistence_run"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    shutil.copyfile(args.config, checkpoint / "experiment.yaml")
    (checkpoint / "manifest.json").write_text(json.dumps({"model": args.model, "train_path": str(train_path)}, indent=2), encoding="utf-8")
    print(str(checkpoint))


if __name__ == "__main__":
    main()

