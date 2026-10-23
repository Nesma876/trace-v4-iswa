from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trace.data.real_world.inventory import build_inventory, write_inventory
from trace.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_world/data.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic schemas only; do not inspect credentialed data.")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(str(config.get("output_root", "outputs/v4_real_data")))
    inventory = build_inventory(config, dry_run=args.dry_run)
    missing = [name for name, item in inventory["datasets"].items() if not item["available"]]
    if missing and not args.dry_run:
        raise SystemExit("Missing required environment variables: TRACE_MIMIC_ROOT and/or TRACE_EICU_ROOT")
    write_inventory(output_root / "data_inventory.json", inventory)


if __name__ == "__main__":
    main()
