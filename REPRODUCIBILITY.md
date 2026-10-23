# Reproducibility workflow

## 1. Environment and release integrity

Use Python 3.12 from a clean checkout. Install `requirements-lock.txt`, then run `scripts/verify_release.py`. The verifier reports PASS, FAIL, NOT AVAILABLE, or NOT EXECUTED and checks every included file against `ARTIFACT_MANIFEST.csv`.

## 2. Dataset acquisition

Obtain PhysioNet/Computing in Cardiology Challenge 2019 version 1.0.0 from the official source under its terms (DOI `10.13026/v64v-d857`). Raw PSV records are not redistributed.

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/physionet2019/download.py
.\.venv\Scripts\python.exe scripts/physionet2019/inventory.py
.\.venv\Scripts\python.exe scripts/physionet2019/prepare.py
.\.venv\Scripts\python.exe scripts/physionet2019/audit.py
```

Expected inventory: Set A 20,336 patients and 790,215 rows; Set B 20,000 patients and 761,995 rows. Deterministic Set A split seed is 4317, with 14,198 / 3,011 / 3,127 patients and split hash `2ef72de30dff98ac4c67da1eb4f68ba0e40156953fa76ffaade7371c0eaa9484` under the repository's canonical hashing procedure.

## 3. Preprocessing

Use only the four `FROZEN_*.yaml` files, `INTERNAL_MODELLING.json`, and Set A training statistics. Fit normalization and imputation on Set A training only. SepsisLabel is not a model input, forecast target, or selection metric. Generated recursive states receive mask zero and never ingest future values or masks.

## 4. Checkpoints

The release contains 20 seed-specific checkpoints under `checkpoints/<model>/<seed>/`. Verify their SHA-256 values before use. Linear AR is deterministic and therefore has the same checkpoint hash for all five nominal seeds; each neural run has a distinct hash.

## 5. Evaluation and statistics

The copied implementation is under `scripts/source/trace_physionet2019`. Evaluation uses observed endpoints, training-SD normalization, equal variable macro weighting, and common 48-hour support for primary comparisons. DynamicMAE uses only consecutive genuinely observed future pairs and excludes the origin-to-first-prediction transition.

The included aggregate tables reproduce manuscript values. Bootstrap replicate arrays are included, but patient-level sufficient statistics are not publicly packaged. Authorized full recomputation requires reconstructing them from PhysioNet records with the frozen code. Do not invoke the historical one-shot Set B unlock merely to validate this package.

## 6. Figures

Regenerate all five PDFs without model inference:

```powershell
.\.venv\Scripts\python.exe release/TRACE_V4/scripts/generate_figures.py
```

The generator reads only final internal/external tables. Existing release PDFs are checksummed. Because the manuscript-referenced PDFs were absent before release preparation, numerical and semantic consistency could be verified, but pixel-level comparison with prior artwork was NOT AVAILABLE.

## 7. Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=outputs/test_runtime/trace-v4-release -q
```

The 2026-10-22 acceptance run passed 102/102 tests. These tests validate implementation behavior with synthetic fixtures; they do not independently reproduce the clinical experiments.

## 8. Manuscript build

The final ISWA manuscript was compiled with MiKTeX pdfTeX 1.40.28 using `pdflatex`, `bibtex`, and two final `pdflatex` passes. The 26-page build resolved all citations and references. The submission source, bibliography, figures, highlights, declarations, and supplementary reproducibility note are staged under `submission/ISWA/`. Current journal instructions could not be independently retrieved, so no unverified formatting requirement is claimed.

## 9. Public-safe release boundary

Checkpoint binaries are `REVIEW REQUIRED` pending redistribution-rights confirmation and are excluded from the separately staged public-cleared bundle. Patient-ID split assignments and patient-level sufficient statistics remain excluded. Two historical event receipts containing workstation paths are also excluded. The existing repository history contains checkpoint objects and a historical patient-split CSV, so publication must use a new clean repository after author approval of `PUBLICATION_MANIFEST.md`.

## 10. Tagging and publication (not executed)

After review, commit the release candidate and replace `<RELEASE_COMMIT>` with the resulting commit:

```powershell
git add <PUBLIC_CLEARED_BUNDLE>
git commit -m "Prepare TRACE V4 ISWA scientific release"
git tag -a trace-v4-iswa-submission <RELEASE_COMMIT> -m "TRACE V4 ISWA submission release"
git push origin <RELEASE_BRANCH>
git push origin trace-v4-iswa-submission
```

Publishing or pushing requires explicit approval. Verify the remote, branch, author metadata, manuscript identifiers, data-use terms, and final clean test run first.

