# CLI Decision Column Lock Report

The CAB CLI workflow simulation has been locked to final operating-frontier decision columns where available.

## Decision sources

- `ClinVar-label-only`: assumes direct deterministic use.
- `CAB-Strict`: uses explicit `cab_strict_direct_use_allowed` if present; otherwise derives the final strict rule from gene + baseline disease-model regime/failure topology.
- `CAB-Balanced`: uses `direct_single_model_reuse_allowed` when present. This is the final operating-frontier direct-use decision column.

## Endpoint source

The benchmark command now prefers `temporal_endpoints.csv` from the same benchmark domain directory. If not available, it reconstructs endpoints from baseline/follow-up replay files.

## Leakage boundary

Baseline routing files remain baseline-only. Temporal endpoints are read only during benchmark evaluation, not during routing.

## Limitation

This is a research workflow simulation. CAB is not a diagnostic tool, does not reclassify variants, and does not replace ACMG/AMP or expert curation.

