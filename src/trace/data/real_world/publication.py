"""Publication-output safeguards for TRACE V4 aggregate artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from trace.data.real_world.governance import assert_publication_safe_file, assert_publication_safe_rows


def write_publication_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert_publication_safe_rows(rows, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert_publication_safe_file(path)
