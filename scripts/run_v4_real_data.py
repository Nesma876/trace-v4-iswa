from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.data.real_world.governance import write_safe_json
from trace.data.real_world.inventory import build_inventory, write_inventory
from trace.data.real_world.publication import write_publication_csv
from trace.data.real_world.schemas import TRACE_COMMON_ACTIONS, TRACE_COMMON_STATE, table_a_rows
from trace.data.real_world.sepsis3 import eicu_sepsis3_mapping_audit, mimic_sepsis3_plan
from trace.utils.config import load_config

OUT = Path("outputs/v4_real_data")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SPLITS = OUT / "splits"
DOCS = Path("docs")


def main(dry_run: bool = False) -> None:
    for directory in [OUT, TABLES, FIGURES, SPLITS, DOCS]:
        directory.mkdir(parents=True, exist_ok=True)
    data_config = load_config("configs/real_world/data.yaml")
    inventory = build_inventory(data_config, dry_run=dry_run)
    write_inventory(OUT / "data_inventory.json", inventory)
    blockers = _blockers(inventory)
    if dry_run:
        blockers.insert(0, "dry_run_not_real_data")
    data_available = all(d["available"] for d in inventory["datasets"].values()) and not dry_run
    decision = "NO-GO" if not data_available else "CONDITIONAL GO"

    _write_csv(TABLES / "table_A_harmonization.csv", table_a_rows())
    _write_csv(TABLES / "mimic_cohort_flow.csv", _blocked_flow("mimic", blockers))
    _write_csv(TABLES / "eicu_cohort_flow.csv", _blocked_flow("eicu", blockers))
    _write_csv(TABLES / "cohort_characteristics.csv", _cohort_characteristics())
    _write_csv(TABLES / "missingness_by_dataset.csv", _missingness_rows())
    _write_csv(TABLES / "action_distribution.csv", _action_rows())
    _write_csv(TABLES / "unit_audit.csv", _unit_rows())
    _write_csv(TABLES / "value_range_audit.csv", _range_rows())
    _write_csv(TABLES / "mimic_eicu_shift_audit.csv", _shift_rows())
    _write_csv(TABLES / "sepsis3_eicu_mapping_audit.csv", eicu_sepsis3_mapping_audit())
    _write_csv(SPLITS / "mimic_patient_split.csv", [{"status": "not_generated", "reason": "authorized MIMIC cohort unavailable"}])
    _write_figures()
    manifest = _manifest(decision, blockers)
    write_safe_json(OUT / "manifest.json", manifest)
    _write_docs(inventory, decision, blockers)
    print(json.dumps({"decision": decision, "blockers": blockers, "outputs": str(OUT)}, indent=2))


def _blockers(inventory: dict[str, Any]) -> list[str]:
    blockers = []
    for name, dataset in inventory["datasets"].items():
        if not dataset["available"]:
            blockers.append(f"{name}_root_not_configured_or_not_found")
        elif dataset["missing_expected_tables"]:
            blockers.append(f"{name}_missing_expected_tables:{','.join(dataset['missing_expected_tables'])}")
    blockers.append("sepsis_cohort_definition_not_selected_or_materialized")
    blockers.append("no_real_patient_counts_or_4h_windows_available")
    return blockers


def _blocked_flow(dataset: str, blockers: list[str]) -> list[dict[str, Any]]:
    return [
        {"step": 1, "criterion": "authorized local data available", "n_stays_before": "not_available", "n_stays_excluded": "not_available", "n_stays_remaining": "not_available"},
        {"step": 2, "criterion": f"blocked: {'; '.join(b for b in blockers if b.startswith(dataset) or b.startswith('sepsis'))}", "n_stays_before": "not_available", "n_stays_excluded": "not_available", "n_stays_remaining": "not_available"},
    ]


def _cohort_characteristics() -> list[dict[str, Any]]:
    rows = []
    for label in ["MIMIC train", "MIMIC validation", "MIMIC internal test", "eICU external"]:
        rows.append({"split": label, "patients": "not_available", "icu_stays": "not_available", "median_sequence_length": "not_available", "n_4h_bins": "not_available"})
    return rows


def _missingness_rows() -> list[dict[str, Any]]:
    return [{"dataset": dataset, "variable": variable, "missing_fraction": "not_available", "measurement_frequency_per_4h": "not_available"} for dataset in ["mimic", "eicu"] for variable in TRACE_COMMON_STATE]


def _action_rows() -> list[dict[str, Any]]:
    return [{"dataset": dataset, "action": action, "prevalence": "not_available", "mean_intensity": "not_available", "representation": _action_representation(action)} for dataset in ["mimic", "eicu"] for action in TRACE_COMMON_ACTIONS]


def _unit_rows() -> list[dict[str, Any]]:
    return [{"dataset": dataset, "variable": variable, "observed_unit": "not_available", "n_observations": "not_available", "target_unit": "see canonical schema", "conversion": "pending_raw_unit_audit", "inconsistent_or_missing_units": "not_available"} for dataset in ["mimic", "eicu"] for variable in TRACE_COMMON_STATE]


def _range_rows() -> list[dict[str, Any]]:
    keys = ["dataset", "variable", "stage", "N", "missing_pct", "min", "p0_1", "p1", "median", "p99", "p99_9", "max", "below_range", "above_range", "removed", "clipped", "retained"]
    return [dict.fromkeys(keys, "not_available") | {"dataset": dataset, "variable": variable, "stage": "before_cleaning"} for dataset in ["mimic", "eicu"] for variable in TRACE_COMMON_STATE]


def _shift_rows() -> list[dict[str, Any]]:
    return [{"variable": variable, "mimic_median": "not_available", "eicu_median": "not_available", "standardized_mean_difference": "not_available", "missingness_difference": "not_available", "measurement_frequency_difference": "not_available", "wasserstein_distance": "not_available"} for variable in list(TRACE_COMMON_STATE) + list(TRACE_COMMON_ACTIONS)]


def _write_figures() -> None:
    _svg(FIGURES / "figure_A_cohort_flow.svg", "Figure A: Cohort flow blocked until authorized data are configured")
    _svg(FIGURES / "figure_B_missingness_map.svg", "Figure B: Missingness map pending MIMIC/eICU extraction")
    _svg(FIGURES / "figure_C_distribution_shift.svg", "Figure C: Distribution shift pending harmonized data")


def _svg(path: Path, title: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220"><rect width="900" height="220" fill="#f8fafc"/><text x="40" y="105" font-size="24" fill="#111827">{title}</text><text x="40" y="145" font-size="16" fill="#374151">No model training; no fabricated real-data counts.</text></svg>',
        encoding="utf-8",
    )


def _manifest(decision: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "phase": "V4_real_data_engineering",
        "decision": decision,
        "created_at": time.strftime("%Y-%m-%d"),
        "code_commit": _git("rev-parse", "HEAD"),
        "configuration_hash": _sha("configs/real_world/data.yaml") + ":" + _sha("configs/real_world/cohort.yaml") + ":" + _sha("configs/real_world/harmonization.yaml"),
        "split_hash": _sha("outputs/v4_real_data/splits/mimic_patient_split.csv"),
        "variable_dictionary_hash": _sha("configs/real_world/canonical_variables.yaml"),
        "cohort_definition_hash": _sha("configs/real_world/FROZEN_COHORT_PROTOCOL.yaml"),
        "unit_conversion_hash": _sha("src/trace/data/real_world/units.py"),
        "blockers": blockers,
        "raw_paths_recorded": False,
    }


def _write_docs(inventory: dict[str, Any], decision: str, blockers: list[str]) -> None:
    found = []
    for name, dataset in inventory["datasets"].items():
        found.append(f"- {name}: available={dataset['available']}; version={dataset['version']}; missing={dataset['missing_expected_tables']}")
    harmonization = "# MIMIC/eICU Harmonization\n\n" + "\n".join(
        f"- {row['canonical_variable']}: MIMIC={row['mimic_source']}; eICU={row['eicu_source']}; unit={row['canonical_unit']}; aggregation={row['aggregation_4h']}; included={row['included']}."
        for row in table_a_rows()
    )
    (DOCS / "MIMIC_EICU_HARMONIZATION.md").write_text(harmonization + "\n\nVariables excluded/problematic: pao2_fio2_ratio, gcs_total, sofa_total pending semantic/proxy audit.\n", encoding="utf-8")
    (DOCS / "V4_REAL_DATA_PROTOCOL.md").write_text(
        "# V4 Real Data Protocol\n\nMIMIC-IV is the development dataset and eICU is locked external validation. No model may be trained until cohort, schema, units, split, imputation and normalization are frozen. Actions are observational and must not be interpreted causally.\n\n" + "\n".join(found) + "\n",
        encoding="utf-8",
    )
    leakage = "# Real Data Leakage Audit\n\n- patient leakage: blocked until MIMIC patient split exists\n- temporal leakage: automated bin/action tests implemented\n- imputation leakage: fit guard requires mimic_train\n- normalization leakage: fit guard requires mimic_train\n- action leakage: automated current-bin action test implemented\n- cohort leakage: blocked until cohort definition is selected\n"
    (DOCS / "REAL_DATA_LEAKAGE_AUDIT.md").write_text(leakage, encoding="utf-8")
    sepsis = mimic_sepsis3_plan()
    sepsis_doc = """# Sepsis-3 Operationalization

## Suspected Infection
MIMIC uses the MIT-LCP MIMIC concept implementation for suspected infection. TRACE V4 does not invent a new antibiotic/culture rule. The local researcher must materialize or reference the vetted concept output.

## SOFA >= 2
MIMIC uses the MIT-LCP SOFA/Sepsis-3 concept logic. Baseline SOFA assumptions must match that implementation and be recorded before modelling.

## ICU Onset
Onset is anchored to ICU time and must not use future prediction-target information.

## Baseline SOFA Assumption
Use the MIT-LCP concept assumption explicitly; do not silently substitute an unavailable baseline.

## MIMIC Implementation
Required source: MIT-LCP MIMIC Sepsis-3 concept implementation, executed locally by the authorized researcher.

## eICU Implementation
eICU uses a separate mapping layer intended to reproduce the same conceptual components, not exact table semantics.

## Differences Between Datasets
Semantic equivalence is not assumed. The eICU audit must report which suspected-infection, SOFA-component, and ICU-onset elements are reproducible.

## Leakage Warning
SOFA_total remains excluded from TRACE primary predictive inputs after cohort/onset definition unless an explicit leakage analysis later justifies otherwise.
"""
    (DOCS / "SEPSIS3_OPERATIONALIZATION.md").write_text(sepsis_doc + "\n\n```json\n" + json.dumps(sepsis, indent=2) + "\n```\n", encoding="utf-8")
    final = f"""# V4 Real Data Final Report

## Dataset inventory
{chr(10).join(found)}

## Cohort
No reproducible real cohort was materialized because authorized local MIMIC/eICU paths are not configured.

## Canonical state
{", ".join(TRACE_COMMON_STATE)}

## Canonical actions
iv_fluids_volume_ml_per_4h and vasopressor_exposure.

## Harmonization
Mappings are frozen as candidate common schema pending raw table/unit inspection.

## Missingness
Not available until local data are configured.

## Distribution shift
Not available until both cohorts are materialized.

## Leakage audit
Automated structural checks are implemented; data-level checks are blocked pending real sequences.

## External-validation readiness
Not ready: same frozen representation is specified but not yet built from both datasets.

## Limitations
No MIMIC/eICU counts, units, missingness, or cohort feasibility are reported without authorized local data.

## Decision
{decision}: {", ".join(blockers)}
"""
    (DOCS / "V4_REAL_DATA_FINAL_REPORT.md").write_text(final, encoding="utf-8")


def _action_representation(action: str) -> str:
    if action == "iv_fluids_volume_ml_per_4h":
        return "total compatible IV crystalloid volume administered during [t,t+4h)"
    return "audited vasopressor exposure/intensity during [t,t+4h)"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_publication_csv(path, rows)


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-c", f"safe.directory={Path.cwd().as_posix()}", *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
