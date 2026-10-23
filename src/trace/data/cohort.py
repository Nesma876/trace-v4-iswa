"""Cohort configuration hooks."""

def describe_configured_cohort(config: dict) -> str:
    """Return a readable cohort description without inventing clinical criteria."""
    cohort = config.get("cohort", {})
    return f"{cohort.get('name', 'unspecified')} from {cohort.get('source', 'unspecified')}"

