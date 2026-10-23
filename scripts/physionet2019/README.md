# TRACE implementation for PhysioNet/Computing in Cardiology Challenge 2019

This directory contains the executable entry points for the manuscript's real-world arm.
The pipeline models multivariate physiological trajectories; it does not train a sepsis
classifier. `SepsisLabel` is excluded from inputs and targets and is used only for data
integrity and a secondary retrospective subgroup description.

## Method-to-code map

| Manuscript component | Implementation | Frozen evidence |
|---|---|---|
| Official data parsing and validation | `src/trace/physionet2019/data.py` | `outputs/v4_physionet2019/data_inventory.json` |
| Patient split | `data.py::split_for`, `data.py::patient_split` | split hash in final reproducibility manifest |
| Range policy and canonicalization | `policy.py::canonicalize` | `configs/physionet2019/FROZEN_RANGE_POLICY.yaml` |
| Training-only normalization | `preprocessing.py::fit_preprocessing` | `outputs/v4_physionet2019/audit/frozen_training_stats.json` |
| Value/mask/elapsed representation | `preprocessing.py::transform_prefix` | frozen state and preprocessing configs |
| Architectures and Linear AR | `models.py`; `src/trace/models/neural_v3.py` | modelling config and capacity table |
| Training and checkpoint selection | `internal_training.py`; `internal_evaluation.py::freeze_candidates` | checkpoint manifests and metadata hashes |
| Recursive mean rollout | `internal.py::generated_row,rollout` | rollout-semantics audit and synthetic tests |
| Endpoint, dynamics and reliability metrics | `internal_evaluation.py::add_statistics,summarize` | internal/external aggregate tables |
| Paired patient/seed bootstrap | `internal_reporting.py::bootstrap` | bootstrap provenance and saved-array hashes |
| One-shot Set B evaluation | `external_evaluation.py::run_once` | external event, addendum and final read-only manifest |

## Reviewer-safe commands

From the repository root, the following commands use synthetic data or public aggregate
artifacts and do not execute clinical inference:

```bash
python -m pytest tests/test_physionet2019.py \
  tests/test_physionet2019_internal.py \
  tests/test_physionet2019_final_audit.py \
  tests/test_jbi_methods_audit.py \
  --basetemp=outputs/test_runtime/physionet-review -q
python scripts/check_public_release.py
```

The final methods report and generated CSV/LaTeX supplements under `reports/` and
`artifacts/` are the preferred reviewer entry points.

## Full reconstruction stages

For an independently obtained dataset, the historical stages were:

1. `download.py`, `inventory.py`, `prepare.py`, `premodelling_audit.py` and `audit.py`;
2. `train.py --sanity-only`, followed by `train.py`;
3. `evaluate_internal.py` and `report_internal.py`;
4. a separately authorized one-shot `evaluate_external.py --unlock-external-test`;
5. read-only reporting/adjudication scripts using saved sufficient statistics.

Stages 2--4 are computationally expensive and are already complete for the manuscript.
The public snapshot should not invoke them to regenerate frozen tables. In particular,
the Set B entry point is deliberately fail-closed after an evaluation event.

## Frozen semantics that reviewers commonly ask about

- Input dimension is 99: 33 normalized values, 33 observation indicators and 33
  hours-since-last-observation channels.
- Four hourly context rows predict the next state. Recursive rollout inserts predicted
  means, sets generated observation indicators to zero and increments elapsed time.
- No future physiological value, future measurement schedule or SepsisLabel enters a
  forecast input.
- Targets are never imputed. Metrics use only genuinely observed post-policy targets.
- Neural heads output means and log variances with log variance clipped to `[-6,3]`.
- DynamicMAE uses predicted-versus-observed differences for future steps 1→2 through
  47→48; the origin→step-1 transition is excluded.
- The change-variance ratio is calculated per variable with population moments and then
  aggregated by an unweighted arithmetic macro-mean, followed by an equal-seed mean.
- Bootstrap draws sample patient clusters and modeling seeds with replacement and share
  both draws across compared models.

See `reports/JBI_FINAL_METHODS_COMPLETION.md` for equations, line-level references and
the few support quantities that were not retained in public artifacts.
