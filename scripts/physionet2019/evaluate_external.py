"""Historical one-shot Set B evaluator; completed runs fail closed."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-external-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_external_test:
        parser.error("Set B locked; --unlock-external-test required")
    # TRACE V4 has completed its one-shot external evaluation.  Refuse before
    # importing evaluation code or touching Set B whenever its event exists.
    completed_event = Path(__file__).resolve().parents[2] / "outputs/v4_external/analysis/external_evaluation_event.json"
    if completed_event.exists():
        parser.error("Set B evaluation event already exists; scientific execution is closed and must not be repeated")
    import torch
    from trace.physionet2019.external_evaluation import OUT, run_once
    from trace.physionet2019.external_reporting import report
    torch.set_num_threads(4)
    try:
        run_once()
        classification=report()
        event=OUT/"analysis/external_evaluation_event.json"
        receipt=json.loads(event.read_text()); receipt.update(status="completed",completed_unix=time.time(),classification=classification)
        event.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    except Exception:
        event=Path("outputs/v4_external/analysis/external_evaluation_event.json")
        if event.exists():
            receipt=json.loads(event.read_text()); receipt.update(status="failed; explicit scientific review required",traceback=traceback.format_exc())
            event.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        raise
