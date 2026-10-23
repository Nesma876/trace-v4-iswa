from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.plotting.figures import write_figure_manifest


def main() -> None:
    figure_manifest = write_figure_manifest("outputs/figures")
    table_manifest = Path("outputs/tables/table_manifest.json")
    table_manifest.parent.mkdir(parents=True, exist_ok=True)
    table_manifest.write_text(
        json.dumps(
            {
                "Table 1": "outputs/data/synthetic/missingness_report.json",
                "Table 2": "outputs/tables/temporal_metrics.csv",
                "Table 3": "outputs/tables/reliability_summary.json",
                "Table 5": "outputs/tables/clinical_stress_summary.json",
                "Table 6": "outputs/tables/counterfactual_summary.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"figures": figure_manifest, "tables": str(table_manifest)}, indent=2))


if __name__ == "__main__":
    main()

