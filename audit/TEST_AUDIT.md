# Automated test audit

Pytest discovered 102 tests.

First command: `.venv/Scripts/python.exe -m pytest -q`. Outcome: 83 passed, 19 setup errors. Every error arose from failure to create pytest's default Windows sandbox temp directory; no test assertion failed.

Audited command: `.venv/Scripts/python.exe -m pytest --basetemp=outputs/test_runtime/iswa-audit -q`. This uses the repository's documented writable-basetemp pattern. Independent outcome on 2026-10-22: **102 passed, 0 failed, 0 skipped, 0 errors in 61.88 seconds**.

The 17 targeted cases are inventoried in `reports/JBI_SYNTHETIC_TEST_INVENTORY.md` and primarily reside in `tests/test_jbi_methods_audit.py`. They cover deterministic mean rollout and neural variance, future-value independence, generated-value reinjection, future-mask exclusion, observed-target score masking, DynamicMAE support and origin-transition exclusion, population variance/zero denominators, and paired patient/seed bootstrap clustering. Additional supporting tests are in `test_physionet2019_internal.py`, `test_physionet2019.py`, `test_rollout.py`, and `test_bootstrap_v2.py`.

Limitations: these are synthetic/unit tests. They do not re-download PhysioNet, recompute the actual split, retrain 20 runs, re-evaluate checkpoints, or reproduce the manuscript's numerical results. Passing them establishes code behavior, not scientific reproduction.

