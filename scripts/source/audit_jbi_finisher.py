"""Audit finisher coverage without loading validation, internal-test or Set B data."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from trace.physionet2019.internal_reporting import finisher_jbi_preflight

if __name__=="__main__":
    print(json.dumps(finisher_jbi_preflight(),indent=2))
