REQUIRED_FIELDS = [
 "assertion_id","domain","variation_id","gene","input_condition_label",
 "normalized_condition_label","baseline_environment","classification",
 "review_status","submitter_count","cab_portability_regime","cab_portability_score",
 "cab_mode","direct_deterministic_use_allowed","routing_primary_action",
 "routing_secondary_flags","unsupported_reuse_risk","cross_environment_drift_risk",
 "condition_label_drift_risk","warning_codes","explanation_short","evidence_fields_used",
 "limitations"
]
CAB_MODES = {"ClinVar-label-only", "CAB-Strict", "CAB-Balanced"}
ACTIONS = {"direct_deterministic_use","contextual_repair","disease_specific_review","population_penetrance_review","no_deterministic_reuse"}

def validate_rows(rows):
    errors, warnings = [], []
    for i, row in enumerate(rows):
        miss = [f for f in REQUIRED_FIELDS if f not in row]
        if miss: errors.append({"row": i, "error": "missing_required_fields", "fields": miss})
        if row.get("cab_mode") not in CAB_MODES: errors.append({"row": i, "error": "invalid_cab_mode", "value": row.get("cab_mode")})
        if row.get("routing_primary_action") not in ACTIONS: errors.append({"row": i, "error": "invalid_routing_primary_action", "value": row.get("routing_primary_action")})
        if not str(row.get("limitations", "")).strip(): warnings.append({"row": i, "warning": "limitations_empty"})
    return errors, warnings
