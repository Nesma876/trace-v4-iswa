"""Training-only pre-modelling audit; never imports a model or fits one."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from trace.physionet2019.final_audit import describe_training, freeze_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data_external/physionet2019"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v4_physionet2019"))
    parser.add_argument("--freeze", action="store_true", help="Apply reviewed policy and freeze train-only scaling/scoring support; requires descriptive audit")
    args = parser.parse_args()
    if args.freeze:
        freeze_training(args.root, args.output)
    else:
        describe_training(args.root, args.output)
