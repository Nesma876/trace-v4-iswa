# Reproducibility matrix

| Manuscript item / claim | Code and configuration | Inputs / provenance | Canonical outputs | Assessment |
|---|---|---|---|---|
| Cohort and split | `src/trace/physionet2019/data.py`, `pipeline.py`, `configs/physionet2019/FROZEN_PREPROCESSING.yaml` | Raw PhysioNet 1.0.0; split seed 4317; split hash in `outputs/v4_physionet2019/manifest.json` | `data_inventory.json`, `cohort_summary.csv`, `sequence_length_summary.csv` | Counts verified from aggregates; patient-ID manifest is ignored and unavailable to public reviewers. |
| Causal preprocessing | `policy.py`, `preprocessing.py` | Set A train statistics in `audit/frozen_training_stats.json` | `leakage_audit.json`, `final_verification.json` | Code and synthetic tests support prefix-only behavior; raw-data recomputation not performed. |
| Models/training | `models.py`, `models/neural_v3.py`, `internal_training.py`, `INTERNAL_MODELLING.json` | 20 model/seed runs; hashes in `frozen_seed_checkpoint_manifest.json` | `jbi_core_training_status.csv`, `model_capacity.csv` | Architecture and hyperparameters verified; checkpoint/log files are ignored. |
| One-step Table 1 | shared masked NLL in `scoring.py:79-93` | Validation and common-48h test support | `jbi_one_step_performance.csv` internal/external | Values match. |
| Recursive fidelity / Figure 2 | `internal.py:133-142`, `internal_evaluation.py`, reporting modules | common-48h sufficient statistics | `bootstrap_intervals.csv`, `paired_comparisons.csv` | Values match; manuscript PDFs absent. |
| DynamicMAE / Figure 3a-b | `internal_evaluation.py`, `scoring.py` | Adjacent observed future pairs; origin-to-first excluded | `table_3_dynamic_fidelity.csv`, `jbi_48h_dynamic_paired_ci.csv` | Values match; one internal comparison retains 1,999 finite draws. |
| Change variance / Figure 3c | population increment moments in `internal_evaluation.py` | Same observed-transition support | `table_3_dynamic_fidelity.csv` | Values match; descriptive seed aggregation, not a trajectory-distribution sample. |
| Conditional coverage / Figure 4 | Gaussian head plus deterministic mean rollout | Observed endpoints; per-step conditional variance | `table_4_reliability.csv`, `calibration.csv` | Values match; interpretation correctly conditional, not fully propagated. |
| Density / Figure 5 | `internal_reporting.py`, `external_reporting.py`; frozen density definition | Set A training missingness groups | `jbi_density_paired_ci.csv`, external `E6_density_paired_ci.csv` | Final corrected estimands support claims; manuscript PDFs absent. |
| Hierarchical paired inference | `internal_reporting.py:116-183` | patient statistics and seed dimension | `bootstrap_provenance.json`, replicate NPZ (ignored), paired tables | Design verified in code/tests; public recomputation blocked by ignored NPZ. |
| RSSM interpretation | `neural_v3.py:150-204`, `internal_training.py` | five distinct checkpoint hashes/logs | `JBI_RSSM_SANITY_AUDIT.md`, training status | Extreme NLL and epoch-1 selection verified; causal diagnosis unresolved. |
| Figure 1 pipeline | no canonical generator identified for `figures/pipeline.pdf` | manuscript design | none with exact referenced name | NOT REPRODUCIBLE. |
| Tables 1–2 | manuscript Table 1 is numeric; Table 2 is qualitative synthesis | canonical CSVs above | LaTeX manuscript | Table 1 matches; Table 2 wording is supported with qualifications. |

The canonical source of truth is the final `outputs/v4_internal` / `outputs/v4_external` table set plus their `analysis/*manifest*` and bootstrap provenance, not older `pre_*` files or the preparation-stage `outputs/v4_physionet2019/manifest.json` status flags.

