# TRACE V4 scientific release candidate

TRACE is a multi-criteria framework for evaluating whether clinical time-series models selected by one-step predictive fit remain suitable for recursive simulation. The redistribution-cleared package is public at `https://github.com/Nesma876/trace-v4-iswa`, with the immutable ISWA release at `https://github.com/Nesma876/trace-v4-iswa/releases/tag/trace-v4-iswa-submission`. No DOI has been minted.

## Scientific objective and design

Four fixed configurations—Linear AR, GRU, Transformer, and RSSM—are compared across predictive fit, recursive state fidelity, temporal-change error, temporal variability, conditional interval reliability, and observation-density sensitivity. Seeds are 17, 23, 31, 43, and 59. This is an evaluation framework, not a new architecture or clinical-deployment claim.

## Dataset access

The work uses PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0 (DOI `10.13026/v64v-d857`). Obtain data from PhysioNet under its terms. Raw records, patient identifiers, patient-level sufficient statistics, and the patient-ID split file are not redistributed. The deterministic split uses seed 4317; see `splits/README.md` for counts and the canonical hash.

## Models, metrics, and provenance

Model definitions and frozen policies are under `scripts/source/` and `configs/`. Metrics include observed-target Gaussian NLL, normalized MacroMAE, DynamicMAE, first-difference variance ratios, conditional interval coverage, and density-stratified comparisons. The candidate preserves source commit `fe5110dfd8e42b5e4c4b318339f1da66044ed991`; it does not rerun training or inference.

## Reproduction workflow

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/verify_release.py
```

Linux/macOS users can replace the interpreter paths with `.venv/bin/python`. Original GPU/CUDA and training-host details were not preserved sufficiently for a defensible hardware claim; aggregate verification and figure generation are CPU-feasible.

## Results verification

`RESULTS_VERIFICATION.csv` records 59/59 reconciled manuscript values. The complete local candidate has 20/20 selected checkpoint hashes, five vector figures, and a 102-test synthetic/unit suite. Tests validate implementation behavior, not independent clinical replication.

## Publication boundary

Checkpoint binaries remain outside the public-cleared bundle until redistribution rights are confirmed. Existing Git history contains historical checkpoints and a patient-split CSV, so it must not be made public. See `PUBLIC_ARTIFACT_MANIFEST.csv` and `PUBLICATION_MANIFEST.md`.

## Limitations

Only four fixed configurations were compared without architecture-specific optimization. RSSM findings are configuration-specific. Intervals are conditional on deterministic mean rollout. Set B is a second hospital system within the same benchmark, not an unrelated external dataset. No causal, counterfactual, treatment-response, or bedside-deployment validity is established.

## Citation and license

Use the manuscript metadata in `zenodo_metadata.json` and cite the tagged GitHub Release for the exact package. Do not cite a DOI unless one is subsequently minted and verified. The MIT `LICENSE` covers repository code. PhysioNet terms remain separate, and checkpoint redistribution is not asserted.

