# CLI Final Decision Column Sync Report

Benchmark baseline exports have been synced with final operating-frontier decision columns.

## Decision columns injected

- `cab_strict_direct_use_allowed`
- `cab_balanced_direct_use_allowed`
- `direct_single_model_reuse_allowed`
- `contextual_repair_required`
- `disease_specific_expert_review_required`
- `population_or_penetrance_review_required`
- `final_decision_column_source`

## Source priority

1. `data/processed/cab_routing_operating_modes_final.csv`
2. fallback: `data/processed/cab_decision_challenge_tasks.csv`

## QC

              domain status  rows  matched_decision_rows  matched_decision_rate  balanced_direct_use_rate  strict_direct_use_rate  balanced_source_ready_for_cli
inherited_arrhythmia synced   942                    942                    1.0                  0.421444                0.676221                           True
      cardiomyopathy synced  4918                   4918                    1.0                  0.122814                0.122814                           True
   hereditary_cancer synced 20865                  20865                    1.0                  0.301845                0.044141                           True

## Leakage boundary

Injected columns are baseline-time routing decisions used by the final operating-frontier benchmark. Future endpoint labels remain only in `temporal_endpoints.csv`.

## Limitation

This is a research workflow simulation. CAB is not diagnostic, not clinical deployment, and not variant reclassification.
