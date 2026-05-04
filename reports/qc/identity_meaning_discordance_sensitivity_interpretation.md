# Identity-vs-Meaning Discordance Sensitivity Interpretation

## Subsets

- full benchmark: 26,725
- excluding meaning-rejected phenotype-domain discordant rows: 26,421
- meaning-rejected phenotype-domain discordant rows only: 304

## Stability check

Condition-label drift rate was 0.3692 in the full benchmark and 0.3694 after excluding the 304 meaning-rejected rows. The difference is small, so the 304 rows do not drive or break the main benchmark-level conclusion.

CAB-Balanced temporal unsupported reuse rate was 0.0746 in the full benchmark and 0.0754 after excluding the 304 rows. This supports the interpretation that the main CAB-Balanced result is stable to removal of the identity-vs-meaning discordance class.

## High-risk portability class check

Within the 304 meaning-rejected rows, condition-label drift rate was 0.3586 and CAB-Balanced temporal unsupported reuse was 0.0066. Review/repair routing rate for the meaning-rejected class was 1.0000, compared with 0.8134 in the full benchmark.

## Interpretation

The 304 source-matched but meaning-rejected rows should be retained as a QC/security layer and interpreted as a high-risk portability class. They are not source match failures, invalid ClinVar records, or variant reclassifications. They are external source matches for which deterministic disease-meaning reuse is blocked by phenotype-domain discordance and routed to contextual repair or disease-specific review.

## Claim boundary

Allowed: the 304 rows demonstrate that identity/source concordance is necessary but insufficient for deterministic assertion reuse.

Forbidden: CAB invalidates ClinVar records, reclassifies variants, or diagnoses phenotype mismatch.

## Enrichment summary

- condition_label_drift: meaning-rejected rate 0.3586, meaning-accepted rate 0.3694, OR 0.956, depleted_in_meaning_rejected.
- cross_environment_drift: meaning-rejected rate 0.0000, meaning-accepted rate 0.1462, OR 0.010, depleted_in_meaning_rejected.
- any_meaning_drift: meaning-rejected rate 0.3586, meaning-accepted rate 0.3865, OR 0.889, depleted_in_meaning_rejected.
- CAB_Balanced_direct_use_allowed: meaning-rejected rate 0.3947, meaning-accepted rate 0.2717, OR 1.750, enriched_in_meaning_rejected.
