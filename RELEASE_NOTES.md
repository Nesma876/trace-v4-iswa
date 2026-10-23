# TRACE V4 release notes

Status: release candidate prepared 2026-10-22; publication not performed.

## Included

- Frozen PhysioNet 2019 modeling and scoring configurations.
- All five seeds for Linear AR, GRU, Transformer, and RSSM, including metadata and SHA-256-verifiable checkpoints.
- Per-seed neural training histories.
- Final internal Set A and external Set B aggregate tables and provenance.
- Internal and external bootstrap replicate arrays (no direct patient identifiers).
- Five regenerated manuscript figures and their deterministic source script.
- Test suite, scientific audit, verification script, and numerical reconciliation.

## Scientific history retained

- Internal reporting uses 2,000 hierarchical paired bootstrap draws; one Transformer-RSSM DynamicMAE contrast retained 1,999 finite draws.
- Set B integrity classification remains E2 potential unblinding: an early preflight reached data loading and aborted before inference or result exposure.
- External density reporting was corrected post hoc from immutable sufficient statistics so point estimates and intervals use the same estimand.
- RSSM checkpoint selection at epoch 1 and extreme validation NLL are preserved without reinterpretation.

## Excluded

Raw PhysioNet data, patient-ID split assignments, patient-level sufficient statistics, local caches, credentials, and private machine paths are not included. Exclusion is explicit in `ARTIFACT_MANIFEST.csv`.

