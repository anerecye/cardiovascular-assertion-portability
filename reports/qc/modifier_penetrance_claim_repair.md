# Modifier/Penetrance Claim Repair

## Status

The old modifier/penetrance-boundary OR is quarantined in QC, not removed.

- Old routing-defined OR: 39870.69
- Modifier/penetrance-boundary N: 16,125
- Label: routing-defined endpoint; quasi-separation; not independent biological effect size
- Publication use: QC only

## Replacement Evidence

Primary independent endpoint: `future_cross_environment_drift`.

Three-domain benchmark result, baseline 2023-01 to follow-up 2026-04:

- Modifier endpoint rate: 0.1456
- Non-modifier endpoint rate: 0.1430
- Crude Haldane-Anscombe OR: 1.0206 (0.9518 to 1.0944)
- Penalized logistic fallback OR: 1.0193
- Gene-cluster bootstrap CI for OR: 0.752672 to 1.54815

Matched anti-circularity test within domain-gene-origin blocks:

- Matched strata: 18
- Matched rate difference: -0.0720
- CMH OR: 0.5052
- Within-block permutation p-value: 0.00049975

Incremental prediction:

- Rolling-origin package gene-only to gene + all regimes delta AUROC: 0.1383 (0.1083 to 0.1627, paired by origin)
- Benchmark row-level gene-only to gene + modifier flag delta AUROC: 0.0042 (-0.0020 to 0.0194, gene-cluster bootstrap)

## Boundary

The rolling-origin predictor repair package provides aggregate held-out origin metrics for gene-only, regime-only, gene + regime, and full-baseline predictors. It does not include row-level modifier feature matrices, so `gene + modifier flag` is evaluated in the three-domain benchmark snapshot with cross-validated row-level predictions and gene-cluster uncertainty, not as an origin-paired package metric.

## Publication-Safe Wording

Modifier/penetrance-boundary is a conditional-liability portability class supported by its N, routing behavior, matched within-gene/domain temporal tests, and independent temporal prediction contrasts. It is not supported by the magnitude of the routing-defined OR.

## Forbidden Wording

- Do not state that OR = 39,870.69 is a biological effect size.
- Do not use population/penetrance review, PRF-needed routing, or primary routing action as the primary endpoint.
- Do not present the quarantined OR in the main figure as biological enrichment.
