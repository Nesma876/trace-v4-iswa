# TRACE ISWA scientific reproducibility audit

Audit date: 2026-10-22. Manuscript: `TRACE_ISWA_revised.tex` (author-supplied audit input). Repository HEAD: `fe5110dfd8e42b5e4c4b318339f1da66044ed991` on `v4-physionet2019`; the tree was clean before this audit.

## Final decision: SCIENTIFIC REVISION REQUIRED

The committed aggregate results support most numerical statements and the central, deliberately limited conclusion that model choice depends on the evaluation axis. They do not support calling the submission independently reproducible. The public/repository-tracked layer omits the Set A patient split, all V4 checkpoints and training logs, and internal/Set B patient-level sufficient statistics. The manuscript's ten result-figure PDFs plus `pipeline.pdf` are also absent. Local ignored copies of the restricted data, checkpoints, logs, and sufficient statistics exist, but these are machine-local evidence rather than shareable repository artifacts.

The headline values in the manuscript match the canonical CSV artifacts at the published precision. The five GRU, Transformer, and RSSM seeds have distinct checkpoint hashes; Linear AR is deterministically identical across seeds, appropriately yielding the same hash. RSSM validation NLL uses the same masked Gaussian NLL as the other neural families and excludes KL at validation. All RSSM checkpoints were genuinely selected at epoch 1, but the evidence does not identify whether the extreme likelihood is an architectural limitation or an optimization/configuration failure.

## Critical blockers

1. No immutable public release/tag or manuscript commit citation exists; the repository is on a feature branch and has no tags.
2. V4 checkpoints, per-seed logs, split manifest, and patient sufficient statistics are gitignored. Their hashes are recorded, but an independent reviewer cannot recompute training, evaluation, or bootstrap results from the tracked repository.
3. Every PDF referenced by Figures 1–5 is missing. Existing PNGs under `outputs/v4_internal/figures` and `outputs/v4_external/figures` use different names and are not the manuscript files.
4. The manuscript says the pipeline “passed 102/102” and the historical record supports that, but this audit's first run produced 83 passes and 19 setup errors solely because the default sandbox temp directory was unwritable. The documented writable-basetemp rerun is recorded in `TEST_AUDIT.md`.
5. The frozen model config still says 1,000 bootstrap replicates, while executed reporting code and provenance use 2,000. The final result provenance is coherent, but the configuration contract is internally inconsistent.

## High-priority weaknesses

- The original per-origin predictions are not retained; aggregate sufficient statistics prevent a fully independent metric reimplementation.
- Set B has a documented E2 potential-unblinding incident: a preflight accessed Set B during evaluator development and aborted during loading. No result was exposed, but the strong “rules unchanged” wording needs this disclosure.
- One internal Transformer–RSSM DynamicMAE interval uses 1,999 finite draws because one draw had no estimable support. This is disclosed and statistically reasonable, but must remain explicit.
- Density tables required a post-hoc reporting correction from immutable sufficient statistics. The correction appears estimand-preserving, but the final manuscript should cite only the corrected tables.
- The checked-in preparation manifest is historical (`models_trained=false`, Set B locked) and should not be mistaken for the final experiment manifest.

## Verified strengths

- Dataset inventory records 20,336 Set A and 20,000 Set B patients; Set A split counts and row counts match the manuscript.
- Split seed 4317, train-only normalization, causal imputation, observed-target scoring, generated-mask zeroing, and SepsisLabel exclusion are implemented and synthetically tested.
- Model sizes and fixed hyperparameters match the manuscript: 3,333 / 65,090 / 82,050 / 58,818 parameters.
- Neural log-variance is clamped to [-6, 3]; validation uses the common observed-target Gaussian NLL.
- Patient/seed paired bootstrap provenance records 2,000 draws, seed 4317, clustered patient resampling, and common draws across models.
- All headline internal and Set B point estimates and intervals reconcile to stored canonical tables within rounding tolerance.

## Scientific claim assessment

| Claim | Verdict | Scope assessment |
|---|---|---|
| One-step selection does not establish recursive suitability | SUPPORTED | Supported for these four fixed configurations and datasets. |
| Preference depends on evaluation axis | SUPPORTED | Directly supported; not a universal architecture ranking. |
| Transformer and RSSM unresolved for 48-h state fidelity | SUPPORTED | Paired CIs include zero internally and on Set B; not evidence of equivalence. |
| RSSM lower DynamicMAE but stronger variance contraction | SUPPORTED | Both datasets; configuration-specific. |
| Transformer intervals closer to nominal | SUPPORTED | Conditional mean-path intervals only; not full trajectory uncertainty. |
| Density strata change relative behavior | SUPPORTED | Descriptive/effect-modification result; not causal. |
| Multi-axis pattern persists across hospital systems | PARTIALLY SUPPORTED | Qualitative pattern persists, but rankings differ and Set B integrity is E2. |
| Observational simulation, not causal validity | SUPPORTED | Correctly scoped in the manuscript. |

## Five actions before submission

1. Create and cite an immutable release/tag containing the exact manuscript, canonical tables, configs, code, hashes, and generated figure PDFs.
2. Publish a privacy-safe reproducibility bundle with split-regeneration verification, V4 checkpoint files (or an accessible archive), per-seed training histories, and bootstrap replicate hashes; explain any item that cannot be shared.
3. Generate Figures 1–5 from the canonical final tables, commit the exact PDFs referenced by LaTeX, and verify captions/axes against the final estimands.
4. Correct the bootstrap configuration inconsistency and explicitly disclose the Set B E2 incident, the 1,999-finite-draw exception, and the post-hoc density reporting correction.
5. Run the documented clean-environment test/release checks and archive the machine-readable outputs with Python/CUDA/platform metadata.

