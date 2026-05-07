# External Download Analysis Limitations

This is the real download/analysis layer for open external resources.

## Downloaded/analyzed

- ClinVar bulk `variant_summary.txt.gz`
- ClinVar `gene_condition_source_id`
- PhysioNet ECG-arrhythmia metadata: `RECORDS`, `ConditionNames_SNOMED-CT.csv`, license/checksum files

## Not automatically downloaded

- eMERGE row-level genotype/EHR data: not available as no-application bulk public data for this validation task.
- DiscovEHR/Geisinger row-level genotype/EHR data: not available as no-application bulk public data for this validation task.
- PGP participant-level files: open-consent but identifiable; use manual profile selection and minimum necessary extraction.
- LOVD/GPCards: use manual or per-terms download/sampling; do not scrape aggressively.

## Claim boundary

This analysis provides external public comparator analysis and phenotype-side feasibility analysis. It does not provide patient-outcome validation, prospective clinical deployment, or clinical validation of CAB/PRF.
