from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trace.data.real_world.governance import raw_root_from_env
from trace.data.real_world.sepsis3 import eicu_sepsis3_mapping_audit, mimic_sepsis3_plan
from trace.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mimic", "eicu"], required=True)
    parser.add_argument("--config", default="configs/real_world/cohort.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    if args.dry_run:
        plan = mimic_sepsis3_plan() if args.dataset == "mimic" else eicu_sepsis3_mapping_audit()
        print(plan)
        return
    raw_root_from_env(args.dataset)
    raise SystemExit(
        f"{args.dataset} cohort extraction scaffold is ready. The authorized researcher must "
        "connect the local MIT-LCP-derived Sepsis-3 concept outputs before materialization."
    )


if __name__ == "__main__":
    main()
