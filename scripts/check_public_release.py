"""Fail closed on common accidental public-release exposures.

This check is intentionally metadata-only. It neither opens clinical data nor executes
the scientific pipeline.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "data_external/",
    "outputs/v4_physionet2019/splits/",
    "outputs/v4_physionet2019/local/",
    "outputs/v4_internal/local/",
    "outputs/v4_external/local/",
    "checkpoints/v4_internal/",
    "logs/v4_internal/",
)
REQUIRED = (
    "README.md",
    "LICENSE",
    "PUBLIC_RELEASE.md",
    "DATA_AVAILABILITY.md",
    "scripts/physionet2019/README.md",
    "reports/JBI_FINAL_METHODS_COMPLETION.md",
    "reports/JBI_FINAL_METHODS_COMPLETION.tex",
    "artifacts/TRACE_JBI_FINAL_REPRODUCIBILITY_MANIFEST.json",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)(?:api[_-]?key|client[_-]?secret|password|passwd)\s*[:=]\s*['\"][^'\"\s]{8,}"),
)
MAX_REVIEW_BYTES = 50 * 1024 * 1024


def public_files() -> list[str]:
    # Include prospective additions so the check is useful before the release commit.
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def main() -> None:
    files = public_files()
    problems: list[str] = []
    for prefix in FORBIDDEN_PREFIXES:
        matches = [name for name in files if name.startswith(prefix)]
        if matches:
            problems.append(f"forbidden tracked prefix {prefix}: {len(matches)} file(s)")
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            problems.append(f"required public file missing: {name}")
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_REVIEW_BYTES:
            problems.append(f"tracked file exceeds 50 MiB: {name} ({size} bytes)")
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                problems.append(f"possible credential/private key in tracked file: {name}")
                break
    if problems:
        raise SystemExit("PUBLIC RELEASE CHECK FAILED\n- " + "\n- ".join(problems))
    print(f"PUBLIC RELEASE CHECK PASSED: {len(files)} public/prospective files; no forbidden data paths, oversized files, or credential patterns.")


if __name__ == "__main__":
    main()
