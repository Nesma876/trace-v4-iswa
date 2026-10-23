# Missing or non-public artifacts

- Exact manuscript figure files: `pipeline.pdf`, `figure2a_rollout_internal.pdf`, `figure2b_rollout_external.pdf`, `figure3a_dynamic_internal.pdf`, `figure3b_dynamic_external.pdf`, `figure3c_variance_ratio.pdf`, `figure4a_reliability_internal.pdf`, `figure4b_reliability_external.pdf`, `figure5a_density_internal.pdf`, and `figure5b_density_external.pdf`.
- Tracked Set A patient-ID split manifest. A hash exists, but the file is gitignored.
- Tracked V4 checkpoints and complete per-seed training logs. Local ignored directories exist; public reconstruction cannot use them.
- Tracked internal and Set B patient-level sufficient-statistic NPZ files and bootstrap replicate arrays.
- Per-origin prediction artifacts sufficient for an independent metric implementation.
- Immutable tagged release and experiment-to-release mapping; current repository has no tags.
- Final manuscript DOI, repository URL/access terms, author/citation metadata.
- Clean-environment machine-readable test log tied to the submission commit.
- Fully locked environment beyond the short Python package lock: OS, CUDA/cuDNN, GPU model/driver, and deterministic-kernel settings.

Raw PhysioNet patient data are appropriately not redistributed. The reproducibility package should instead provide acquisition instructions, verified content hashes, deterministic split regeneration plus a privacy-safe split hash check, shareable checkpoints where permitted, aggregate/per-patient non-identifying statistics where permitted, and exact commands.

