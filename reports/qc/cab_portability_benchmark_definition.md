# CAB Portability Benchmark Definition

Technical benchmark definition; not manuscript prose.

## Baseline snapshot
ClinVar January 2023 parsed snapshot.

## Follow-up snapshot
ClinVar April 2026 parsed snapshot.

## Assertion universe
Domain-specific germline P/LP assertions temporally aligned between baseline and follow-up snapshots.

## Domains
- inherited arrhythmia
- cardiomyopathy
- hereditary cancer predisposition

## Domain-specific environment ontology
Each domain maps condition labels to disease-model environments using reproducible script-level rules. Failed/ambiguous mappings are preserved as other/unknown.

## Baseline-only portability regimes
Regimes are derived from baseline gene, baseline condition label/environment, baseline review status, baseline submitter count, baseline classification, and baseline consequence/HGVS when available.

## Endpoints
- classification_change
- condition_label_change
- cross_environment_drift
- within_environment_label_drift
- self_loop_stable
- any_meaning_drift

## Models
- gene-only
- regime-only
- metadata-only
- gene+regime
- gene+regime+metadata

## Leakage rules
- no follow-up condition labels in predictors
- no follow-up environments in predictors
- no endpoint labels in predictors
- no follow-up review status or submitter count in predictors
- old leakage-contaminated outputs remain quarantined

## Claim-strength rules
- Tier 1 requires cross-domain table support and leakage-clean predictors.
- Tier 2 supports mechanism/actionability/comparator claims only within tested scope.
- Tier 3 records explicit limitations and blocked overclaims.