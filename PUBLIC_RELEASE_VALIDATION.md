# Public release validation

Status: local public-staging validation completed; external publication not executed.

## Verified evidence

- Source experiment commit: `fe5110dfd8e42b5e4c4b318339f1da66044ed991`.
- Manuscript values: 59/59 reconciled.
- Selected checkpoints: 20/20 hashes verified in the complete local candidate; binaries excluded from the public-cleared bundle pending rights review.
- Figures: five vector PDFs regenerated from validated aggregate results.
- Tests: 102/102 passed in the audited acceptance run; the final staging rerun is recorded below.

## Safety checks

- Text secret scan found no credentials or private keys.
- Two historical JSON receipts contain workstation paths and are excluded.
- Raw records, patient-ID assignments, and patient-level sufficient statistics are absent.
- Existing Git history contains checkpoint objects and a patient-split CSV; a clean repository is mandatory.
- Checkpoint binaries are `REVIEW REQUIRED` and excluded.

## Commands and actual outcomes

1. `python scripts/verify_release.py` in the clean staged directory: **PASS** for the initial 234-file manifest, all five non-empty PDF figures, valid configuration JSON, and 59 result-verification rows with zero non-matches.
2. `python scripts/generate_figures.py`: the bundled generic Python lacked matplotlib (**FAIL**, environment mismatch). Repeated with the documented project virtual environment: **PASS**, five figures generated.
3. Initial staged test invocation from the parent checkout: **FAIL during collection** because the first staging copy omitted the full `src/` tree. After adding the public source tree and running from the staged root, the next run produced 67 passed / 35 failed because `configs/physionet2019/` had been flattened. The package layout was corrected.
4. `python -m pytest --basetemp=.pytest-runtime-final -q` from the corrected staged root: **PASS, 102 passed in 57.17 s**.
5. LaTeX build using MiKTeX pdfTeX 1.40.28 with `pdflatex`, `bibtex`, and two final `pdflatex` passes: **PASS**, 26 pages, citations and references resolved. All 26 rendered pages were visually inspected; figures, tables, equations, declarations, and references were present and legible.
6. Full repository-history filename review: **FAIL for publishing the existing history**. Historical objects include checkpoint binaries and `outputs/v4_real_data/splits/mimic_patient_split.csv`; therefore the existing repository must remain private and a clean repository must be used.
7. Public classification: 218 original manifest rows `PUBLIC`, 46 checkpoint rows `REVIEW REQUIRED`, and five rows `RESTRICTED` (three patient-level unavailable artifacts plus two private-path receipts). The expanded clean bundle adds the complete source/configuration layout and manuscript assets; its regenerated manifest is verified separately.
8. Final staged-bundle verifier after cleanup: **PASS** for 345 checksummed files, all five figures, valid configuration JSON, and 59/59 result rows. Final scans found no private workstation path, credential pattern, checkpoint binary, patient-split/statistics file, or file larger than 50 MiB in the staged bundle.

## Unresolved limitations

No public GitHub URL, GitHub Release, public tag, Zenodo archive, or DOI exists. Checkpoint redistribution rights remain unresolved. Journal instructions were not independently retrieved. Availability statements therefore make no public-link claim.
