"""Clinical consistency rule engine."""

from __future__ import annotations

from typing import Any


def evaluate_rules(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate simple lower/upper-bound demonstration rules."""
    checks = 0
    violations = 0
    for row in rows:
        for rule in rules:
            variable = rule.get("variable")
            if variable not in row or row[variable] is None:
                continue
            checks += 1
            if rule.get("function") == "lower_bound" and row[variable] < float(rule["threshold"]):
                violations += 1
            if rule.get("function") == "upper_bound" and row[variable] > float(rule["threshold"]):
                violations += 1
    return {
        "checks": float(checks),
        "violations": float(violations),
        "physiological_violation_rate": violations / checks if checks else 0.0,
    }

