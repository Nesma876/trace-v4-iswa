# Proposed public publication manifest — approval required

## Target

- Owner inferred from configured remote: `Nesma876`.
- Proposed new clean repository: `Nesma876/trace-v4-iswa`.
- Do **not** change visibility of `Nesma876/Trace-paper`.
- Proposed annotated tag: `trace-v4-iswa-submission`.
- Source experiment commit: `fe5110dfd8e42b5e4c4b318339f1da66044ed991`.
- Publication commit: not created; it will be the root/history-clean commit of the approved repository.

## Proposed contents

Only the staged public-cleared bundle: source/evaluation code, frozen configurations, environment files, aggregate results, bootstrap replicate arrays, training histories, five vector figures, tests, data-access and split-reconstruction instructions, audit/provenance documents, manuscript assets, checksums, and file-level classification.

## Exclusions

- Model checkpoint and frozen-candidate binaries: `REVIEW REQUIRED` pending redistribution-rights confirmation.
- Patient-ID splits, raw PhysioNet records, and patient-level sufficient statistics: `RESTRICTED`.
- Two historical receipts containing workstation paths: excluded.
- Existing Git history, LFS objects, Actions artifacts, and unreviewed historical outputs: excluded by using a clean repository.

## Remaining concerns

- GitHub creation/authentication and unauthenticated accessibility are untested.
- Without checkpoint binaries, inference cannot be reproduced from supplied weights.
- MIT covers repository code; PhysioNet terms remain separate and checkpoint licensing is not asserted.
- No Zenodo action or DOI minting is authorized here.

No push, release upload, visibility change, tag publication, or archive deposit has occurred. Explicit approval of this target and content is required first.
