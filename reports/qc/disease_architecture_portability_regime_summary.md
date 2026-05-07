# Disease-Architecture Portability Regime Summary

## Core result

CAB identifies recurrent disease-architecture regimes that determine how pathogenic meaning travels. The result is not simply that domains differ. Instead, recurring regimes appear across domains and govern whether source-matched pathogenic assertions travel within a self-loop, require contextual repair, or fail across disease-model boundaries.

## Summary interpretation

### Phenotype-anchored regimes

Phenotype-anchored monogenic regimes show the strongest expectation of self-loop portability and direct or balanced reuse when the disease model remains concordant.

- self-loop stable rate: 0.8816
- dominant routing: direct_source_match_only

### Collision / syndrome-organ / genotype-first regimes

Pleiotropic collision, syndrome-organ boundary, and genotype-first absent phenotype regimes have weaker deterministic portability and require review/repair routing when assertions cross disease-model boundaries.

- syndrome-organ cross-environment drift rate: 0.1978
- syndrome-organ dominant routing: source_identity_accepted; contextual_repair_or_disease_specific_review

### Modifier/penetrance regimes

Modifier and penetrance-boundary regimes travel as conditional risk rather than deterministic disease assignment. They require PRF-style framing and should not be treated as direct deterministic disease reuse.

- modifier/penetrance condition-label drift rate: 0.3631
- modifier/penetrance dominant routing: population/penetrance review; PRF-needed

### Nonspecific/underresolved regimes

Nonspecific and underresolved regimes require contextual repair because broad or unknown phenotype labels are insufficient for deterministic disease-model reuse.

- nonspecific condition-label drift rate: 0.6379
- nonspecific dominant routing: contextual_repair_or_disease_specific_review

## Publication-safe claim

We identify recurrent disease-architecture regimes that determine pathogenic meaning travel. Phenotype-anchored monogenic assertions tend to travel within self-loops, whereas collision, syndrome-organ, genotype-first, penetrance/modifier, and underresolved regimes require contextual repair, disease-specific review, or PRF-style conditional-risk framing.

## Claim boundary

This table is a disease-architecture synthesis layer. It does not reclassify variants, invalidate ClinVar records, claim clinical outcome validation, or replace expert disease-specific curation.
