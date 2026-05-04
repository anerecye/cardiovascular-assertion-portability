REQUIRED_ROUTING_FIELDS = [
    "assertion_id",
    "domain",
    "variation_id",
    "gene",
    "baseline_environment",
    "direct_deterministic_use_allowed",
    "routing_primary_action",
]


def validate_rows(rows):
    errors = []
    warnings = []
    for i, row in enumerate(rows):
        for field in REQUIRED_ROUTING_FIELDS:
            if field not in row:
                errors.append({"row": i, "field": field, "error": "missing_required_field"})
        if str(row.get("direct_deterministic_use_allowed", "")).lower() not in {"true", "false"}:
            warnings.append({"row": i, "field": "direct_deterministic_use_allowed", "warning": "not_boolean_string"})
    return errors, warnings
