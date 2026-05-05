# CAB 10-year predictor repair package

Package:
reports/packages/cab_10yr_predictor_repair_package.zip

Purpose:
Repair/package outputs for the CAB 10-year rolling-origin historical prospective emulation.

Held-out endpoint:
future_cross_environment_drift

Testable origins:
11

Held-out AUROC summary:
- random: ~0.4924
- gene-only: ~0.5781
- regime-only: ~0.6998
- gene+regime: ~0.7164
- all-baseline predictor: ~0.7419

Incremental value:
- gene-only vs gene+regime mean delta AUROC: ~+0.1383
- paired bootstrap CI across held-out origins: 0.1066 to 0.1627

Top-10% review queue enrichment:
- random: ~0.984
- gene-only: ~1.412
- regime-only: ~2.290
- gene+regime: ~2.735
- full baseline: ~2.608

Claim boundary:
This is a 10-year rolling-origin historical prospective emulation / temporal backtest, not true prospective validation. It does not claim patient outcome prediction, clinical validation, cause-of-death inference, or patient-risk prediction.
