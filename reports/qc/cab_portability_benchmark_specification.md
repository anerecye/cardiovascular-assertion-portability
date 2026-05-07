# CAB Portability Benchmark Specification

Technical benchmark specification; not manuscript prose.

## Benchmark components
1. Baseline assertion table.
2. Follow-up assertion table.
3. Domain-specific disease-model environment ontology.
4. Baseline-only portability regimes.
5. Temporal endpoints.
6. Model tasks.
7. Decision-routing tasks.
8. Leakage rules.
9. Claim-strength labels.

## Baseline assertion table
Domain-specific P/LP assertion table derived from the January 2023 parsed ClinVar snapshot.

## Follow-up assertion table
Domain-specific P/LP assertion table derived from the April 2026 parsed ClinVar snapshot.

## Environment ontology
A reproducible domain-specific mapping from raw condition labels to disease-model environments. Failed/ambiguous mappings are preserved as other/unknown.

## Baseline-only portability regimes
Regimes use only baseline gene, condition label/environment, review status, submitter count, classification, and baseline architecture flags.

## Temporal endpoints
- classification_change
- condition_label_change
- cross_environment_drift
- within_environment_label_drift
- self_loop_stable
- any_meaning_drift

## Model tasks
- gene-only
- regime-only
- metadata-only
- gene+regime
- gene+regime+metadata

## Decision-routing tasks
- direct_single_model_reuse_allowed
- cross_environment_reuse_allowed
- contextual_repair_required
- disease_specific_expert_review_required
- population_or_penetrance_review_required
- high_future_meaning_drift_risk
- high_future_cross_environment_drift_risk

## Leakage rules
- no follow-up labels/environments in predictors
- no endpoint labels in predictors
- no follow-up review status or submitter count in predictors
- deprecated/leaky outputs remain quarantined

## Claim-strength labels
Claim strength is assigned as discovery-domain supported, external cardiovascular replication supported, non-cardiovascular replication supported, three-domain evidence, routing-only actionability support, external constraint only, or limitation.