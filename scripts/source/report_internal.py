"""Generate reports only from completed, local internal evaluations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from trace.physionet2019.internal_reporting import (tables, bootstrap, conclusions, figures,
    jbi_complementary_outputs, jbi_reproducibility_audit)
from trace.physionet2019.internal_reporting import assert_jbi_output_contract

if __name__ == "__main__":
    summary=tables()
    bootstrap(2000)
    jbi_complementary_outputs(summary)
    conclusions(summary)
    figures(summary)
    jbi_reproducibility_audit(regenerate=True,bootstrap_replicates=2000)
    assert_jbi_output_contract()
