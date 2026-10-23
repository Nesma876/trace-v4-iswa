"""Set A internal evaluation, gated on all trained validation-selected candidates."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))

if __name__ == "__main__":
    import torch
    from trace.physionet2019.internal_evaluation import evaluate_all
    parser=argparse.ArgumentParser()
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate_all()
