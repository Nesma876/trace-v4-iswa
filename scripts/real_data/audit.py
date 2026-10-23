from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.run_v4_real_data import main as run_v4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_world/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_v4(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
