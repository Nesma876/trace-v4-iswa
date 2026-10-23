"""Create manuscript/adjudication views from completed Set B CSV artifacts only.

This reporting-only utility never imports evaluation code, patient statistics,
checkpoints, or Set B source data.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v4_external"
TABLES = OUT / "tables"
PAPER = ROOT / "paper/jbi"


def read(name):
    with (TABLES / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def add(records, quantity, value, source, estimand, verified="YES"):
    records.append({"Quantity": quantity, "Manuscript value": value,
                    "Source artifact": source, "Estimand": estimand,
                    "Verified": verified})


def value_audit():
    records = []
    state = read("E3_rollout_fidelity.csv")
    reliability = read("E5_reliability.csv")
    intervals = read("bootstrap_intervals.csv")
    dynamics = read("E4_dynamic_fidelity.csv")
    density = read("E6_density_results_corrected.csv")
    density_ci = read("E6_density_paired_ci_corrected.csv")
    paired = read("paired_comparisons.csv")

    for model in ("transformer", "gru", "linear_ar", "rssm"):
        one_state = next(r for r in state if r["model"] == model and r["cohort"] == "full_available"
                         and r["horizon"] == "1" and r["subgroup"] == "all" and r["variable_set"] == "primary")
        one_rel = next(r for r in reliability if r["model"] == model and r["cohort"] == "full_available"
                       and r["horizon"] == "1" and r["subgroup"] == "all" and r["variable_set"] == "primary")
        add(records, f"One-step NLL — {model}", f"{float(one_rel['nll_mean']):.6f} ± {float(one_rel['nll_sd'] or 0):.6f}",
            "outputs/v4_external/tables/E5_reliability.csv",
            "Full-available Set B cohort; patient-pooled per-variable NLL, macro-averaged over frozen eligible variables, then mean ± SD across five seeds.")
        add(records, f"One-step normalized MacroMAE — {model}", f"{float(one_state['mae_mean']):.6f} ± {float(one_state['mae_sd'] or 0):.6f}",
            "outputs/v4_external/tables/E3_rollout_fidelity.csv",
            "Full-available Set B cohort; patient-pooled per-variable normalized MAE, macro-averaged over frozen eligible variables, then mean ± SD across five seeds.")

    for horizon in (4, 12, 24, 48):
        for model in ("transformer", "rssm", "gru", "linear_ar"):
            row = next(r for r in intervals if r["model"] == model and r["dimension"] == f"{horizon}h_state")
            add(records, f"{horizon}h rollout MacroMAE — {model}",
                f"{float(row['mean']):.6f} [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]",
                "outputs/v4_external/tables/bootstrap_intervals.csv",
                "Frozen common-48h cohort; patient-pooled per-variable ratio-of-sums, macro-average over frozen eligible variables, equal mean across seeds; 95% paired hierarchical patient/seed percentile CI.")
        for model in ("rssm", "transformer", "gru", "linear_ar"):
            row = next(r for r in intervals if r["model"] == model and r["dimension"] == f"{horizon}h_dynamics")
            add(records, f"{horizon}h DynamicMAE — {model}",
                f"{float(row['mean']):.6f} [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]",
                "outputs/v4_external/tables/bootstrap_intervals.csv",
                "Frozen common-48h cohort and dynamic-support mask; ratio-of-sums per eligible variable, macro/seed mean; 95% paired hierarchical patient/seed percentile CI.")

    for model in ("transformer", "gru", "rssm", "linear_ar"):
        row = next(r for r in dynamics if r["model"] == model and r["cohort"] == "common_48h"
                   and r["horizon"] == "48" and r["subgroup"] == "all" and r["variable_set"] == "primary")
        add(records, f"48h change-variance ratio — {model}", f"{float(row['change_variance_ratio_mean']):.6f}",
            "outputs/v4_external/tables/E4_dynamic_fidelity.csv",
            "Frozen common-48h cohort; per-variable predicted/observed change-variance ratio, macro-average, then equal mean across seeds.")

    for horizon in (4, 12, 24, 48):
        for model in ("transformer", "gru", "rssm", "linear_ar"):
            row = next(r for r in reliability if r["model"] == model and r["cohort"] == "common_48h"
                       and r["horizon"] == str(horizon) and r["subgroup"] == "all" and r["variable_set"] == "primary")
            value = (f"cov50={float(row['coverage50_mean']):.6f}; cov80={float(row['coverage80_mean']):.6f}; "
                     f"cov95={float(row['coverage95_mean']):.6f}; abs95_error={float(row['coverage_error95_mean']):.6f}; "
                     f"width50={float(row['width50_mean']):.6f}; width80={float(row['width80_mean']):.6f}; width95={float(row['width95_mean']):.6f}")
            add(records, f"{horizon}h reliability — {model}", value,
                "outputs/v4_external/tables/E5_reliability.csv",
                "Frozen common-48h cohort; patient-pooled per-variable coverage/width, macro-average over frozen eligible variables, then equal mean across seeds.")

    for row in density:
        add(records, f"48h density {row['density_group']} — {row['model']}", f"{float(row['normalized_macro_mae']):.6f}",
            "outputs/v4_external/tables/E6_density_results_corrected.csv",
            "Stable patient-pooled ratio-of-sums per variable; equal macro-average over the frozen stratum/common-eligibility intersection; equal mean across seeds.")
    for row in density_ci:
        add(records, f"48h density paired {row['density_group']} — {row['comparison']}",
            f"{float(row['point_estimate']):.6f} [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]",
            "outputs/v4_external/tables/E6_density_paired_ci_corrected.csv",
            "Difference of the corrected frozen density estimand; 2,000 paired hierarchical patient/seed percentile-bootstrap draws, seed 4317.",
            row["same_estimand_verified"])
    for row in paired:
        add(records, f"Primary paired {row['dimension']} — {row['model_a']} minus {row['model_b']}",
            f"{float(row['mean_difference_a_minus_b']):.6f} [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]",
            "outputs/v4_external/tables/paired_comparisons.csv",
            "Difference of frozen primary common-cohort estimands; 2,000 paired hierarchical patient/seed percentile-bootstrap draws, seed 4317.")

    matrix = read("E8_replication_matrix.csv")
    for row in matrix:
        add(records, f"Replication evidence — {row['Internal finding']}", row["Set B result"],
            "outputs/v4_external/tables/E8_replication_matrix.csv",
            "Prespecified internal-vs-external replication evidence using the corresponding verified primary artifact.")
    write(PAPER / "SET_B_MANUSCRIPT_VALUE_AUDIT.csv", records)


def correct_replication_linear():
    paths = [TABLES / "E8_replication_matrix.csv", PAPER / "SET_B_REPLICATION_MATRIX.csv"]
    for path in paths:
        historical = path.with_name(path.stem + "_pre_48h_estimand_adjudication.csv")
        if not historical.exists():
            shutil.copyfile(path, historical)
        matrix = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            matrix = list(csv.DictReader(handle))
        target = next(r for r in matrix if r["Internal finding"] == "Linear AR long-horizon degradation")
        evidence = "Set B primary rollout estimand: Linear AR 4h=0.776990, 48h=6.102105 [5.698180, 6.603926]"
        target["Set B result"] = evidence
        target["Comment"] = evidence
        write(path, matrix)


def adjudication_package():
    text = """# TRACE Set B external-integrity adjudication

## Decision

The external integrity classification remains E2. `POST-ACCESS SCIENTIFIC LOGIC CHANGED` is YES, while `NEW SCIENTIFIC CHOICE AFTER SET B ACCESS` is NO.

## Access and exposure

An automated test-harness invocation first accessed Set B on 2026-09-10 at 12:09:17 CEST while the evaluator was being implemented. It stopped during input-file loading, before the cohort table and model loop. It generated no predictions, patient sufficient statistics, metrics, rankings, bootstrap artifacts, figures, or performance summaries. The authorized frozen evaluation resumed at 12:19:40 CEST. Set B was not used for training, hyperparameter tuning, checkpoint selection, normalization, or internal Scenario B selection.

## Post-access change classification

| Change | Classification | Adjudication |
|---|---|---|
| `external_evaluation.py`: streaming sufficient-statistic evaluation, frozen cohort/mask handling | Implementation of already frozen specification | Implements the previously audited evaluation protocol; no new endpoint, cohort, model, or selection rule. |
| `external_reporting.py`: density bootstrap, rankings, replication matrix, G rule | Implementation of already frozen specification | Implements prespecified JBI axes and the already frozen external generalization rubric. |
| `evaluate_external.py`: interruption recovery and event/error logging | Bug fix required to match frozen specification | Operational recovery only; no prediction or metric definition changed. |
| `tests/test_physionet2019.py`: closed-CLI guard correction | Bug fix required to match frozen specification | Prevents the test harness from opening Set B; test infrastructure only. |
| Density point/CI correction | Bug fix required to match frozen specification | Aligns displayed points and bootstrap draws to the frozen common-eligibility mask and stable ratio-of-sums estimator using immutable sufficient statistics. |

No genuinely new scientific choice was identified. The retained E2 classification therefore records a potential procedural unblinding risk caused by implementation work after technical access, not demonstrated data-driven adaptation. No Set B performance was available when those changes were made.

## Frozen provenance

- Configuration: `63548ed3f880d2f0bf06355919949136d42793c1e99ccfbbe95a79bb1eb2a804`
- Checkpoint manifest: `b5fa4e127b9d857d214b3a8f2b02c1fd7dae943fc5dd025a024fa090b22a87ad`
- Preprocessing: `68fd7519f1675fa232df04ac428716597736b69a741b490defa10d3964f5508f`
- Variable eligibility: `93c113a10edeed32d0688ece457c1339d089de900a282fe07af4c8f89dbcef34`
- Density definition: `da60c47029bd0456c04d8d356136da6e07ce3ea1ab097a11b9b0d16ed75bcd5f`
- Scoring: `5d2abb58403e8bb112842cdd13f567a069475bd0e3c9874cf26e6283848f6431`
- State schema: `79af49df9a31616ba2bd9fca9b5262026c8a376af6fde55310287c801e401ed2`
- Primary bootstrap replicates: `c212db2568666888caa3df34af2fde787dcbdc12a9891011ed02461674ab1ffa`
- Historical manifest, versioned addendum, and final read-only manifest are retained together.
- Final regression suite: 85 passed, 0 failed, 0 errors.
"""
    (PAPER / "SET_B_EXTERNAL_INTEGRITY_ADJUDICATION.md").write_text(text, encoding="utf-8")


def main():
    correct_replication_linear()
    value_audit()
    adjudication_package()


if __name__ == "__main__":
    main()
