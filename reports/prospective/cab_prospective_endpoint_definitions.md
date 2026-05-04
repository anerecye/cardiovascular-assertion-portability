# CAB Prospective Endpoint Definitions

## Baseline

- baseline date: 2026-05-05
- planned 12-month follow-up: 2027-05-05
- optional 6-month interim: exploratory only

## Primary endpoint

### future_cross_environment_disease_model_drift at 12 months

A baseline assertion is endpoint-positive if, at follow-up, the public assertion has moved into or acquired a disease label/environment outside its baseline disease-model environment such that deterministic reuse in the baseline environment would be unsupported.

## Secondary endpoints

### future_condition_label_drift

Any materially changed ClinVar condition label or phenotype list relative to baseline.

### future_any_meaning_drift

Any condition-label drift, cross-environment disease-model drift, phenotype-domain discordance, underresolved-to-specific shift, specific-to-broad shift, or review/routing-relevant meaning change.

### self_loop_stability

The assertion remains within a concordant baseline disease environment and remains meaning-portable within the same disease loop.

### unsupported deterministic reuse under ClinVar-label-only

A future-drift-positive assertion that ClinVar-label-only baseline would have allowed as direct deterministic reuse.

### classification downgrade / P_LP_to_VUS_or_lower

A P/LP assertion at baseline becomes VUS, conflicting, likely benign, benign, or otherwise no longer P/LP at follow-up.

### review status change

ClinVar review status changes from baseline to follow-up.

### conflict emergence

A conflict or conflicting interpretation emerges at follow-up.

### disease-specific review emergence

A disease-specific expert panel, VCEP/CSpec-like review, or disease-specific review signal appears after baseline.

## Endpoint blinding rule

No endpoint label may be added to the locked baseline registry. Follow-up endpoints must be stored in a separate follow-up table and joined only during the follow-up analysis.
