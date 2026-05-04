# CAB Prospective Analysis Plan

## Purpose

Test whether locked CAB baseline-only predictions identify future assertion portability failures and disease-model drift in newly submitted or materially updated public P/LP assertions.

## Comparators

- ClinVar-label-only baseline
- gene-only model
- metadata-only model
- classification-support proxy
- AlphaMissense-only where matched
- random review queue

## Primary endpoint

future_cross_environment_disease_model_drift at 12 months.

## Secondary endpoints

- future_condition_label_drift
- future_any_meaning_drift
- self_loop_stability
- unsupported deterministic reuse under ClinVar-label-only
- classification downgrade / P_LP_to_VUS_or_lower
- review status change
- conflict emergence
- disease-specific review emergence

## Metrics

- AUROC
- AUPRC
- precision@top 5%, 10%, 20%
- recall@top 5%, 10%, 20%
- calibration
- Brier score
- enrichment over random
- workload required to capture 50% of future cross-environment drift
- net reduction in unsupported reuse

## Primary comparison

CAB predicted cross-environment drift risk versus future_cross_environment_disease_model_drift.

## Routing comparison

CAB-Balanced and CAB-Strict direct-use decisions will be evaluated for unsupported deterministic reuse reduction relative to ClinVar-label-only direct reuse.

## Review queue comparison

Predicted review priority rank will be evaluated using precision/recall at top 5%, 10%, and 20% of the review queue.

## Calibration

Calibration will be evaluated by grouping predicted risk into deciles or quantiles, subject to endpoint N.

## SADS/sudden-death stratum

The SADS/sudden-death stratum is exploratory. It cannot be used for individual risk prediction or clinical outcome claims. Its purpose is to test whether genotype-first / absent-phenotype categories appear in new public assertions and whether CAB routes them to PRF-needed / no deterministic reuse.

## Prohibited claims

- CAB prospectively predicted future drift before follow-up.
- CAB is clinically validated.
- CAB predicts patient outcomes.
- CAB predicts individual SADS risk.
- CAB proves therapeutic utility.
