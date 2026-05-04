# ARR Identifier Resolution for External ClinVar Join

## Problem

The repaired external ClinVar join matched cardiomyopathy and hereditary cancer rows, but left 942 inherited-arrhythmia rows unmatched because the benchmark uses local identifiers such as `ARR_1325231`.

## Method

For unmatched inherited-arrhythmia rows, this script extracted the numeric suffix from `ARR_*` identifiers and treated it as a candidate ClinVar `VariationID`. Candidate IDs were looked up in the downloaded ClinVar `variant_summary.txt.gz`. Resolutions were accepted when the candidate was found and tokenized gene matching confirmed concordance.

## Source-vs-meaning QC

Rows can be valid external ClinVar source matches while failing disease/phenotype-domain meaning portability. Such rows are retained, not deleted, with:

- `external_clinvar_match=True`
- `source_match_accepted=True`
- `phenotype_domain_concordant=False`
- `phenotype_domain_discordance_flag=True`
- `meaning_match_accepted=False`
- `routing_implication=contextual_repair_or_disease_specific_review`

## Results

- rows total: 26,725
- matched before ARR resolution: 25,783
- ARR rows attempted: 942
- ARR numeric suffix candidates: 942
- ARR candidates found in downloaded ClinVar: 942
- ARR gene-concordant rows: 942
- ARR resolutions accepted: 942
- phenotype-domain discordance flagged: 304
- meaning matches rejected: 304
- matched after ARR resolution: 26,725
- unmatched after ARR resolution: 0
- final match rate: 1.0000

## Claim boundary

This resolves public source identifiers and improves external ClinVar assertion-source traceability. It does not validate CAB clinically, does not validate patient outcomes, and does not validate prospective deployment.
