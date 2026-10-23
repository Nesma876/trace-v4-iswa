from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trace.utils.config import load_config
from trace.data.real_world.governance import raw_root_from_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_world/harmonization.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    if args.dry_run:
        print("dry-run harmonization schema validated")
        return
    raw_root_from_env("mimic")
    raw_root_from_env("eicu")
    raise SystemExit("Harmonization requires completed MIMIC/eICU inventory and cohort extraction.")


if __name__ == "__main__":
    main()
