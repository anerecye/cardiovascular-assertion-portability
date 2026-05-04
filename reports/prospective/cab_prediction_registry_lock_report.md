# CAB Prospective Prediction Registry Lock Report

## Registry status

LOCKED BASELINE PREDICTION REGISTRY.

This registry contains baseline-only predictions for future assertion portability. It must not be modified after lock except by creating a new versioned registry.

## Snapshot

- baseline date: 2026-05-05
- ClinVar source: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
- downloaded current raw file: `data\prospective\raw\clinvar_variant_summary_baseline_2026-05-05.txt.gz`
- current raw file SHA256: `d92f9bde56463cece9c65dfb4487934d9da8117e88082a0d5d90e570efc23d2f`
- inclusion rule: snapshot-diff against prior ClinVar release when `--prior-clinvar` is supplied; otherwise broad dry run is not lockable
- cohort file: `data\prospective\cab_prospective_cohort_baseline_2026.csv`
- cohort SHA256: `7bd317cbbc04c682b93b031de055a3176524dada3c1525b80770f0f646407876`
- prediction registry: `data\prospective\cab_prospective_prediction_registry_2026_locked.csv`
- prediction registry SHA256: `92e4c18fc3151e06968b6b8079b992bc674f4564240685021cf0152b953b4e75`
- registry lock timestamp UTC: 2026-05-04T20:46:09+00:00
- Git commit hash at lock: `ff879b8028106852a5cb222d59e793db29987bc7`
- Python: 3.13.13
- platform: Windows-11-10.0.26200-SP0

## Cohort

- cohort N: 6,259
- SADS/sudden-death exploratory stratum N: 273
- domain counts: {'hereditary_cancer': 4190, 'cardiomyopathy': 1900, 'inherited_arrhythmia': 169}

## Model versions

- CAB regime-only: `cab_regime_only_v2026_locked`
- gene + CAB regime: `gene_plus_regime_v2026_locked`
- CAB-Balanced routing: `CAB_Balanced_v2026_locked`
- CAB-Strict routing: `CAB_Strict_v2026_locked`
- metadata-only baseline: `metadata_only_v2026_locked`
- gene-only baseline: `gene_only_v2026_locked`

## Baseline-only feature list

- ClinVar VariationID
- gene
- condition label
- classification
- review status
- submitter count
- baseline environment
- domain
- source date
- new_or_updated_status
- baseline disease-architecture regime
- baseline disease-label / phenotype text

## Prohibited follow-up fields

The locked registry must not include future ClinVar classifications, future review status, future condition labels, future submission counts, future conflicts, future disease-specific reviews, or any endpoint labels derived after baseline.

## Routing counts

{'contextual_repair_or_disease_specific_review': 2539, 'source_identity_accepted_contextual_repair_or_disease_specific_review': 2012, 'domain_repair_disease_specific_expert_review': 1224, 'population_penetrance_review_PRF_needed': 292, 'direct_deterministic_use_with_concordant_context': 185, 'contextual_repair_trigger_phenotype_context_review': 7}

## Regime counts

{'nonspecific_underresolved': 2539, 'syndrome_organ_boundary': 2012, 'structural_functional_overlap': 1224, 'modifier_penetrance_boundary': 292, 'phenotype_anchored_monogenic': 185, 'trigger_dependent_latent': 7}

## Planned follow-up

- primary follow-up date: 2027-05-05
- optional interim: 6 months exploratory only

## Primary endpoint

`future_cross_environment_disease_model_drift` at 12 months.

## Secondary endpoints

- future_condition_label_drift
- future_any_meaning_drift
- self_loop_stability
- unsupported deterministic reuse under ClinVar-label-only
- classification downgrade / P_LP_to_VUS_or_lower
- review status change
- conflict emergence
- disease-specific review emergence

## Analysis plan summary

The follow-up analysis will compare locked CAB baseline-only predictions against ClinVar-label-only, gene-only, metadata-only, classification-support proxy, AlphaMissense-only where matched, and random review queue baselines.

## Claim boundary

This registry is not prospective validation yet. It is a locked prospective-style prediction registry. No clinical outcome, patient risk, therapy, or clinical validation claim is permitted until endpoint follow-up is performed, and even then only assertion-portability claims are in scope.
