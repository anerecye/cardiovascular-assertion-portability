# Identity vs Meaning Positioning

## Key distinction

ClinVar / VRS / source matching establishes variant or record identity. CAB evaluates disease-meaning portability.

A gene-concordant source match can still be phenotype-domain discordant. Therefore identity resolution is necessary but insufficient for deterministic assertion reuse.

## Current result

All 26,725 benchmark assertions were externally matched to ClinVar source records. Of these, 26,421 were meaning accepted and 304 were source-accepted but disease-meaning rejected because phenotype-domain concordance failed.

## Allowed claim

All 26,725 benchmark assertions were externally matched to ClinVar source records, but 304 ARR records were source-accepted and gene-concordant while disease-meaning rejected because phenotype-domain concordance failed.

## Forbidden claims

- ClinVar is wrong.
- These records are invalid.
- CAB reclassifies these variants.
- Source match failure.

These rows are source matches with meaning-portability failure. They are retained with discordance flags and routed to contextual repair or disease-specific review.

## Example

ARR_977320 resolves to ClinVar VariationID 977320 and is gene-concordant for KCNQ1. Its ClinVar phenotype label is Silver-Russell syndrome 1, so CAB retains the source match but rejects deterministic inherited-arrhythmia meaning reuse.
