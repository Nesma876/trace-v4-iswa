from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    exe = sys.executable
    run([exe, "scripts/prepare_data.py", "--config", "configs/data.yaml"])
    run([exe, "scripts/train.py", "--model", "persistence", "--config", "configs/experiments/main.yaml"])
    run([exe, "scripts/evaluate.py", "--experiment", "outputs/checkpoints/persistence_run"])
    run([exe, "scripts/run_stress_tests.py", "--experiment", "outputs/checkpoints/persistence_run"])
    run([exe, "scripts/run_counterfactuals.py", "--experiment", "outputs/checkpoints/persistence_run"])
    run([exe, "scripts/generate_paper_outputs.py"])


if __name__ == "__main__":
    main()

