"""Publication output placeholders."""

from __future__ import annotations

from pathlib import Path


def write_figure_manifest(output_dir: str | Path) -> str:
    """Write a figure manifest until matplotlib outputs are implemented."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "figure_manifest.txt"
    path.write_text(
        "\n".join(
            [
                "Figure 1: experimental architecture placeholder",
                "Figure 2: prediction error vs horizon from temporal_metrics.csv",
                "Figure 3: calibration placeholder for probabilistic models",
                "Figure 4: MIMIC vs eICU degradation placeholder",
                "Figure 5: clinical stress-test violation rates",
                "Figure 6: controlled counterfactual trajectory comparison",
                "Figure 7: TRACE multidimensional profile placeholder",
                "Figure 8: ranking reversal matrix placeholder",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)

