"""inventory stage: no model training or predictive evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from trace.physionet2019.pipeline import inventory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data_external/physionet2019"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v4_physionet2019"))
    args = parser.parse_args()
    print(inventory(args.root, args.output))
