# CAB Routing Output Schema

CAB routing output is a row-level table for assertion portability simulation.

Required fields: `assertion_id`, `domain`, `variation_id`, `gene`,
`input_condition_label`, `normalized_condition_label`, `baseline_environment`,
`classification`, `review_status`, `submitter_count`, `cab_portability_regime`,
`cab_portability_score`, `cab_mode`, `direct_deterministic_use_allowed`,
`routing_primary_action`, `routing_secondary_flags`, `unsupported_reuse_risk`,
`cross_environment_drift_risk`, `condition_label_drift_risk`, `warning_codes`,
`explanation_short`, `evidence_fields_used`, `limitations`.

Allowed modes: `ClinVar-label-only`, `CAB-Strict`, `CAB-Balanced`.

CAB is not a diagnostic tool, does not reclassify variants, and does not replace
ACMG/AMP interpretation or expert curation.
