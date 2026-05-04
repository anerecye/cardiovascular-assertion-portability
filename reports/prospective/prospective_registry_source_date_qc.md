# Prospective Registry Source-Date QC

A strict post-cutoff filter using ClinVar `LastEvaluated > 2026-04-01` returned zero candidate rows.

Interpretation:
`LastEvaluated` is not a reliable source-date or snapshot-novelty field for defining a prospective CAB registry cohort.

The first broad registry run should be treated as a dry run, not a locked prospective registry.

Final lock rule:
Use snapshot-diff inclusion between archived ClinVar releases:
- new_in_current_snapshot
- materially_updated_since_prior_snapshot

Do not claim prospective validation.
Do not claim the dry run is a locked registry.
Do not use follow-up fields in baseline predictions.
