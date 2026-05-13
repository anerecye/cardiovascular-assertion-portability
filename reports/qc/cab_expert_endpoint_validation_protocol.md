# CAB Expert Endpoint Validation Protocol

This run replaces ClinVar temporal drift as the primary validation endpoint once real expert verdicts are collected.
ClinVar drift remains an analyst-only proxy and must not be shown to reviewers.

## Primary Expert Endpoint

Binary verdict: can this P/LP assertion be reused in the specified target context without additional interpretation?

Allowed values: yes, no, not_adjudicable.

## Blinding

- Reviewer packets exclude follow-up condition labels and all temporal endpoint columns.
- Reviewer packets also exclude CAB regime, CAB routing action, sample bucket, and model reason codes.
- Analyst endpoint and CAB prediction keys are written separately and are not reviewer inputs.

## Reviewer Design

- Core casebook cases: 400.
- Domain counts: {'hereditary_cancer': 296, 'inherited_arrhythmia': 66, 'cardiomyopathy': 38}.
- Reviewer assignments: 1332 rows.
- Assignment counts by reviewer: {'ARR_R1': 66, 'ARR_R2': 66, 'ARR_R3': 66, 'ARR_R4': 66, 'ARR_R5': 66, 'CM_R1': 38, 'CM_R2': 38, 'CM_R3': 38, 'HC_R1': 296, 'HC_R2': 296, 'HC_R3': 296}.
- SADS/CPVT priority addendum cases: 50.
- SADS/CPVT priority reviewer assignments: 250 rows.

## Consensus Rule

Use simple majority among completed yes/no verdicts. Ties, not-adjudicable majorities, or median confidence <3 go to panel resolution.

## Manuscript Upgrade Logic

1. Validate endpoint: compare CAB routing/scores against expert non-portability consensus and against ClinVar drift.
2. Calibrate regimes: estimate expert non-portability rates for structural-functional overlap, trigger-dependent latent, PRF-needed, and syndrome-organ-boundary regimes.
3. Convert SADS from future-work language to adjudicated evidence if the SADS/CPVT addendum has at least 40 completed expert-consensus cases.

## Claim Boundary

Do not report expert-endpoint performance until real specialist verdicts are entered. The current artifact is an executable run packet, not completed external validation.
