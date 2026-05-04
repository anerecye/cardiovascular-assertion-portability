# ARR Identifier Resolution for External ClinVar Join

## Problem

The repaired external ClinVar join matched cardiomyopathy and hereditary cancer rows, but left 942 inherited-arrhythmia rows unmatched because the benchmark uses local identifiers such as `ARR_1325231`.

## Method

For unmatched inherited-arrhythmia rows, this script extracted the numeric suffix from `ARR_*` identifiers and treated it as a candidate ClinVar `VariationID`. Candidate IDs were looked up in the downloaded ClinVar `variant_summary.txt.gz`. Resolutions were accepted when the candidate was found and the local gene was concordant, or when local gene was missing/unavailable.

## Results

- rows total: 26,725
- matched before ARR resolution: 25,783
- ARR rows attempted: 942
- ARR numeric suffix candidates: 942
- ARR candidates found in downloaded ClinVar: 942
- ARR gene-concordant rows: 942
- ARR resolutions accepted: 942
- matched after ARR resolution: 26,725
- unmatched after ARR resolution: 0
- final match rate: 1.0000

## Claim boundary

This resolves public source identifiers and improves external ClinVar assertion-source traceability. It does not validate CAB clinically, does not validate patient outcomes, and does not validate prospective deployment.
