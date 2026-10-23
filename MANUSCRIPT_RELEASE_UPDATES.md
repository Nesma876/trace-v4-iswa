# Publication-ready manuscript updates

## Code availability

“A versioned release candidate containing the analysis code, frozen configurations, validation-selected model checkpoints, aggregate result tables, bootstrap replicate arrays, automated tests, figure-generation code, and cryptographic artifact manifest was prepared for this study. At the time of submission, public repository and archival URLs had not yet been assigned. These resources should therefore be described as prepared for release, not publicly available, until the final URLs and immutable release tag have been verified.”

## Data availability

“PhysioNet/Computing in Cardiology Challenge 2019 version 1.0.0 is available from PhysioNet under its applicable terms (DOI 10.13026/v64v-d857). Patient-level records are not redistributed with this work. The release provides acquisition and deterministic split-reconstruction instructions, expected cohort counts, and a canonical split hash. The patient-ID split manifest and patient-level sufficient statistics remain excluded pending data-use and privacy review.”

## Reproducibility

“The release verification workflow checks file presence, SHA-256 checksums, configuration validity, checkpoint availability, numerical reconciliation, and figure availability. All 59 prespecified manuscript values matched the retained canonical artifacts at the reported precision. The synthetic/unit regression suite contained 102 tests and passed in the audited environment; these tests validate implementation behavior but do not constitute independent reproduction of the clinical experiments.”

## Experimental provenance and Set B qualification

“The reported results correspond to code commit `fe5110dfd8e42b5e4c4b318339f1da66044ed991` and the frozen V4 checkpoint manifest. The Set B integrity classification remains E2 (potential unblinding): an early automated preflight accessed Set B during evaluator implementation but stopped during data loading, before model inference or exposure of predictions, metrics, or rankings. No evidence of result-informed adaptation was identified, but this incident prevents characterization of the external evaluation as a pristine untouched lock.”

## RSSM interpretation

“The RSSM findings apply to the fixed evaluated configuration. Its validation NLL used the shared observed-target Gaussian loss and excluded the KL component; five independent checkpoints nevertheless selected epoch 1 and showed consistently poor likelihood. Lower 48-hour DynamicMAE coexisted with marked temporal-variance contraction and poorer conditional interval coverage. The result therefore does not establish a general advantage of RSSM architectures, and the present experiments do not isolate the cause of the likelihood failure.”

## Limitations

“Recursive uncertainty intervals are conditional on deterministic predictive-mean feedback and are not samples from a fully propagated joint trajectory distribution. One internal DynamicMAE contrast retained 1,999 finite bootstrap draws because one resample had no estimable dynamic support. External density reporting was corrected post hoc from immutable sufficient statistics to align point estimates and confidence intervals to the same frozen estimand; no model inference was rerun. Exact OS/GPU/CUDA metadata for the original training environment were not fully preserved.”

## Figure provenance

“Figures 1–5 were regenerated as vector PDFs from the frozen protocol and adjudicated internal/Set B tables without retraining or model inference. Their input artifacts and generation script are included in the release manifest.”

