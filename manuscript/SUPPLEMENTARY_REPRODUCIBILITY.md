# Supplementary reproducibility documentation

The complete local release candidate is stored under `release/TRACE_V4/`. It includes frozen configurations, 20 validated checkpoints, aggregate internal and external results, bootstrap replicate arrays, source and tests, figure-generation code, SHA-256 manifests, and the independent audit.

Validated invariants: 20/20 checkpoint hashes match; 59/59 manuscript values reconcile; 102/102 automated tests pass; five vector figures regenerate from adjudicated result tables. Patient-ID splits and patient-level sufficient statistics are deliberately excluded pending redistribution review and are reconstructible from authorized PhysioNet records using seed 4317 and the frozen pipeline.

RSSM evidence is configuration-specific: five distinct checkpoints selected epoch 1, stopped after epoch 21, and produced validation NLL $57.0183 \pm 2.3563$. Validation excludes the KL training regularizer and uses the shared observed-target Gaussian NLL. Lower 48-hour DynamicMAE coexists with marked temporal variance contraction and poor conditional interval reliability.
