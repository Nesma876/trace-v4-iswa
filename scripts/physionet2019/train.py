"""Set A-only training; sanity gates must pass before full training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if __name__ == "__main__":
    import torch
    from trace.physionet2019.internal_training import sanity, train_all
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.sanity_only:
        sanity()
    else:
        train_all()
