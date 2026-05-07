# External ClinVar Join Truth Audit

## Existing join audit

- rows in existing join file: 26,725
- explicit `external_clinvar_match == True`: 0
- rows with any external ClinVar fields populated: 0
- rows with numeric `variation_id` in existing join file: 0
- interpretation: `left_join_only_not_true_match`

A 26,725-row left join is not the same as 26,725 true ClinVar matches. True matching requires a positive match flag or populated external ClinVar fields.

## Repair attempt

- repair status: `completed`
- benchmark rows: 26,725
- rows with numeric mapping: 25,783
- true external ClinVar matches: 25,783
- true external match rate: 0.9648
- mapping sources: data\processed\cancer_assertion_master.csv; data\processed\cardiomyopathy_baseline_only_regimes.csv

## Claim boundary

ClinVar can support public assertion-source comparability and traceability. It does not clinically validate CAB, does not validate patient outcomes, and does not validate prospective deployment.
