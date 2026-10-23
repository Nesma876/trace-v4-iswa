# Reproduction commands

Run from repository root with Python 3.12. Commands that need PhysioNet data or ignored checkpoints are listed for reconstruction but were not executed in this audit.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest --basetemp=outputs/test_runtime/public-review -q
.\.venv\Scripts\python.exe scripts/check_public_release.py
```

Dataset acquisition and preparation (network/data-use terms apply; not run automatically):

```powershell
.\.venv\Scripts\python.exe scripts/physionet2019/download.py
.\.venv\Scripts\python.exe scripts/physionet2019/inventory.py
.\.venv\Scripts\python.exe scripts/physionet2019/prepare.py
.\.venv\Scripts\python.exe scripts/physionet2019/audit.py
```

Training and internal evaluation require the regenerated split and protected/local data. Do not run against frozen submission artifacts without creating a new output directory and experiment identifier:

```powershell
.\.venv\Scripts\python.exe scripts/physionet2019/train.py --help
.\.venv\Scripts\python.exe scripts/physionet2019/evaluate_internal.py --help
.\.venv\Scripts\python.exe scripts/physionet2019/report_internal.py --help
```

Set B evaluation was one-shot and is already consumed. Do not rerun `evaluate_external.py --unlock-external-test` for submission verification. Verify the retained reporting layer instead:

```powershell
.\.venv\Scripts\python.exe scripts/physionet2019/audit_jbi_methods.py evidence
.\.venv\Scripts\python.exe scripts/physionet2019/audit_jbi_methods.py manifest
.\.venv\Scripts\python.exe scripts/physionet2019/audit_jbi_methods.py verify
```

The manuscript-specific PDF figure-generation command is presently missing. Add a deterministic script that reads only adjudicated final CSVs, writes to a new build directory, and emits the exact filenames referenced by `TRACE_ISWA_revised.tex`.

