from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = {
    "inventory": ["scripts/real_data/inventory.py"],
    "cohort": ["scripts/real_data/build_cohort.py", "--dataset", "mimic"],
    "harmonization": ["scripts/real_data/harmonize.py"],
    "sequences": ["scripts/real_data/build_sequences.py"],
    "audit": ["scripts/real_data/audit.py"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=COMMANDS, required=True)
    parser.add_argument("--dataset", choices=["mimic", "eicu"], help="Dataset override for cohort stage.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, *COMMANDS[args.stage]]
    if args.stage == "cohort" and args.dataset:
        command = [sys.executable, "scripts/real_data/build_cohort.py", "--dataset", args.dataset]
    if args.dry_run:
        command.append("--dry-run")
    raise SystemExit(subprocess.run(command, cwd=Path(__file__).resolve().parents[2]).returncode)


if __name__ == "__main__":
    main()
