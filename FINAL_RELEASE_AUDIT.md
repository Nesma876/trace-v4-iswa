# Final TRACE V4 release acceptance audit

Date: 2026-10-22. Candidate code commit: `fe5110dfd8e42b5e4c4b318339f1da66044ed991`. Proposed tag: `trace-v4-iswa-submission` (not created).

## Verdict: RELEASE READY WITH DOCUMENTED LIMITATIONS

The release candidate is scientifically coherent and preserves the validated experiment. All 20 checkpoints match the frozen checkpoint manifest, all 59 manuscript values match, the 102-test suite passes, and all five figures were regenerated from adjudicated tables and visually inspected. It is not yet a public release: no commit/tag has been created, URLs are unverified, patient-level resources remain excluded, and original training hardware metadata are incomplete.

| Criterion | Status | Evidence |
|---|---|---|
| Numerical reconciliation | PASS | `RESULTS_VERIFICATION.csv`: 59/59 MATCH with source-artifact SHA-256 values. |
| Automated tests | PASS | 102 passed, 0 failed, 0 skipped, 0 errors using writable `--basetemp`. |
| Figure reproduction | PASS | Five vector PDFs generated from final V4 tables; rendered visual QA passed. Exact pixel comparison with absent prior PDFs was NOT AVAILABLE. |
| Artifact completeness | FAIL | Patient-ID split manifest and 3.1 GB patient-level statistics are not packaged pending redistribution/privacy clearance. They are reconstructible from authorized data. |
| Checkpoint provenance | PASS | 20/20 release checkpoints match frozen manifest SHA-256; all neural seed hashes are distinct. |
| Dataset reconstruction | PASS | Official acquisition route, version/DOI, deterministic seed, counts, frozen configs, and canonical split hash are documented and tested. Raw data were not re-downloaded. |
| Environment reproducibility | FAIL | Python/package versions are available; original OS/GPU/CUDA/cuDNN details are incomplete. |
| RSSM documentation | PASS | Configuration, per-seed NLL, epoch-1 selection, hashes, and scientific qualification are retained. |
| Scientific traceability | PASS | Candidate commit, configs, checkpoints, results, statistics, figures, scripts, tests, audit, and hashes are associated in the manifest. |
| Public release readiness | FAIL | Publication approval, final commit/tag, repository/archive URLs, data-use review, and citation metadata remain outstanding. |

## Acceptance details

- No training or model inference was run.
- No original checkpoint, result, configuration, or manuscript file was modified.
- Figure generation uses only retained aggregate V4 tables.
- Secret-pattern scan found no credentials or private keys. Historical local-path test receipt was excluded, and the copied audit path was sanitized.
- Repository public-release checker passed.
- The release verifier must be rerun after the final manifest is generated and whenever any release file changes.

## Remaining release actions

1. Review checkpoint redistribution and PhysioNet-derived artifact policy with the authors/institution.
2. Decide whether the split manifest and patient sufficient statistics can be shared under controlled or public access; otherwise retain the documented reconstruction path.
3. Commit the candidate, rerun verification from a clean checkout, then create the proposed annotated tag after explicit approval.
4. Insert verified repository/archive URLs and final citation metadata into the manuscript.
5. Preserve OS/GPU/CUDA/cuDNN metadata for any future rerun; do not retroactively claim it for the frozen training runs.

