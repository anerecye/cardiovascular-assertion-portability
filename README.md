# Contextual Assertion Biology (CAB)

Disease-architecture-aware measurement of pathogenic assertion portability across Mendelian disease-model environments.

## One-Paragraph Summary

Contextual Assertion Biology (CAB) measures assertion portability, not variant pathogenicity. CAB asks whether public pathogenic/likely pathogenic (P/LP) assertions retain disease-model meaning when transferred into a new inference environment. The current benchmark contains 26,725 temporally aligned ClinVar assertions from inherited arrhythmia, cardiomyopathy, and hereditary cancer, comparing January 2023 and April 2026 snapshots. CAB separates source identity, disease meaning, and phenotype realization so that a stable source match or P/LP label is not mistaken for deterministic disease-meaning reuse. CAB does not reclassify variants, predict penetrance, predict sudden death, or provide clinical decision support.

## Conceptual Overview

### What problem does CAB address?

Pathogenicity labels are often reused as if disease meaning were invariant across contexts. CAB treats that reuse as a portability problem: can this assertion be deterministically reused in this new disease-model environment?

CAB does not ask whether the variant is damaging. CAB does not ask whether the patient will develop disease. CAB asks whether an existing public assertion keeps the same disease-model meaning after context transfer.

## Key Concepts

- assertion portability: whether an existing assertion can be reused with the same disease-model meaning in a target context.
- disease-model environment: the clinical or biological inference environment in which an assertion is being interpreted.
- self-loop stability: stable reuse inside the same disease-model environment.
- cross-environment drift: movement from one disease-model environment into another.
- contextual repair: additional mapping, curation, or context needed before deterministic reuse.
- disease-specific review: expert or domain-specific review required before transfer.
- PRF-needed: population/penetrance/risk framing is required; deterministic disease reuse is not supported.
- unsupported deterministic reuse: direct reuse of an assertion when CAB routing says the target context is not portable.
- disease architecture regime: biological architecture class governing whether meaning is stable, conditional, underresolved, or boundary-crossing.

## Main Benchmark

### Benchmark summary

Source snapshots: January 2023 versus April 2026 ClinVar.

| domain | assertions | condition-label drift | cross-environment drift |
|---|---:|---:|---:|
| inherited arrhythmia | 942 | 38.75% | 15.50% |
| cardiomyopathy | 4,918 | 38.65% | 9.86% |
| hereditary cancer | 20,865 | 36.43% | 16.19% |

Primary source tables:

- `reports/tables/three_domain_portability_summary.csv`
- `reports/tables/final/Table1_benchmark_cohort.csv`
- `data/processed/cab_decision_challenge_tasks.csv`
- `benchmark/{inherited_arrhythmia,cardiomyopathy,hereditary_cancer}/`

## Main Findings

- Classification stability does not equal meaning stability: disease labels and environments drift even when classification change is low.
- Portability follows disease architecture, not only gene identity.
- Anchored architectures stabilize meaning inside concordant self-loops.
- Modifier/penetrance-boundary architectures dominate conditional-risk and PRF-needed space.
- Source identity does not equal meaning portability: 26,725 rows were source matched, but 304 were meaning rejected because phenotype-domain concordance failed.
- CAB routing reduces unsupported deterministic reuse: in the temporal condition-label benchmark, ClinVar-label-only unsupported reuse was 36.92%, CAB-Strict was 2.42%, and CAB-Balanced was 7.46%.
- CAB-Balanced preserves more direct-use capacity than CAB-Strict: 27.31% versus 8.09% direct-use allowance, and 31.48% versus 8.99% true-portable allowance.
- Disease-model regimes add information beyond gene identity in rolling-origin temporal emulation: AUROC 0.5781 for gene-only, 0.6998 for regime-only, 0.7164 for gene+regime, and 0.7419 for the full baseline predictor.

## Disease Architecture Framing

### Disease architecture governs pathogenic meaning mobility

Portable meaning depends on the disease architecture regime. Phenotype-anchored assertions tend to be stable in self-loops. Modifier-dependent or penetrance-boundary assertions often require PRF-style conditional-risk framing. Underresolved assertions require contextual repair. Structural-overlap assertions often require disease-specific review. Syndrome-organ boundaries separate source identity from disease meaning.

The current benchmark includes anchored, modifier-dependent, underresolved, structural-overlap, syndrome-organ, trigger-dependent, pleiotropic-collision, and genotype-first categories. Trigger-dependent and genotype-first categories are underpowered or unsampled in the current benchmark and should not be presented as empirically validated regimes.

## Repository Structure

```text
data/
  processed/                         locked benchmark and routing inputs
  prospective/                       locked prospective-style registry outputs
benchmark/
  inherited_arrhythmia/              CLI benchmark replay inputs
  cardiomyopathy/
  hereditary_cancer/
reports/
  tables/                            locked analysis tables
  tables/final/                      manuscript Tables 1-6
  figure_source_tables/              source data for each main and supplementary figure
  figures/final/                     final SVG/PDF manuscript figures
  figure_captions/                   standalone captions
  workflow_simulation/               routing replay outputs
  packages/                          rolling-origin validation package
scripts/
  figures/                           figure/table generation script
  build_*.py                         analysis-layer builders
cab_portability/configs/             domain configuration YAMLs
```

Reviewer entry points:

- Final tables: `reports/tables/final/`
- Figure source data: `reports/figure_source_tables/`
- Figure captions: `reports/figure_captions/`
- Final figures: `reports/figures/final/`
- Routing outputs: `reports/workflow_simulation/`
- Prospective registry: `data/prospective/cab_prospective_prediction_registry_2026_locked.csv`
- Rolling-origin package: `reports/packages/cab_10yr_predictor_repair_package.zip`

## Reproducibility

### Reproducing the benchmark

Environment setup:

```bash
python -m pip install -e .
```

Materialize full three-domain benchmark exports:

```bash
python scripts/materialize_full_three_domain_benchmark_exports.py
python scripts/sync_benchmark_exports_to_final_decision_columns.py
```

Replay routing/benchmark outputs:

```bash
python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia
python -m cab_portability.cli benchmark --baseline benchmark/cardiomyopathy/baseline_assertions.csv --followup benchmark/cardiomyopathy/followup_assertions.csv --domain cardiomyopathy --output-dir reports/workflow_simulation/cardiomyopathy
python -m cab_portability.cli benchmark --baseline benchmark/hereditary_cancer/baseline_assertions.csv --followup benchmark/hereditary_cancer/followup_assertions.csv --domain hereditary_cancer --output-dir reports/workflow_simulation/hereditary_cancer
```

Generate figures, captions, source tables, and manuscript tables:

```bash
python scripts/figures/generate_cab_publication_figures.py
```

Predictive modeling and rolling-origin historical prospective emulation:

```bash
python scripts/build_cab_10yr_rolling_origin_validation.py --inventory-only
python scripts/build_cab_10yr_rolling_origin_validation.py --download-missing
```

Locked prospective-style registry reproduction, using the archived raw files already present in this repository:

```bash
python scripts/build_cab_prospective_portability_prediction_registry.py --baseline-date 2026-05-05 --raw-clinvar data/prospective/raw/clinvar_variant_summary_baseline_2026-05-05.txt.gz --prior-clinvar data/prospective/raw/clinvar_variant_summary_prior_2026-04.txt.gz
```

## Routing Modes

### Routing modes

| mode | unsupported reuse | direct-use allowance | intended use case |
|---|---:|---:|---|
| ClinVar-label-only | 36.92% | 100.00% | counterfactual baseline: reuse all P/LP assertions directly |
| CAB-Strict | 2.42% | 8.09% | high-stringency minimization of unsupported deterministic reuse |
| CAB-Balanced | 7.46% | 27.31% | operating point that preserves more direct-use capacity while reducing unsupported reuse |

The frontier concept is central: CAB-Strict and CAB-Balanced are operating points, not universally optimal classifiers. Lower unsupported reuse usually comes with higher overrestriction or lower direct-use allowance.

## Predictive Modeling

### Temporal portability forecasting

The rolling-origin package implements historical prospective emulation: baseline-only information from earlier ClinVar snapshots is evaluated against later snapshot endpoints. It is temporal backtesting, not prospective clinical validation.

Locked AUROCs for future cross-environment drift:

- random: 0.4924
- gene-only: 0.5781
- regime-only: 0.6998
- gene+regime: 0.7164
- full baseline predictor: 0.7419

Gene+regime improved over gene-only by +0.1383 AUROC with paired bootstrap CI 0.1066-0.1627. Source: `reports/packages/cab_10yr_predictor_repair_package_manifest.md` and `reports/tables/final/Table5_predictive_modeling.csv`.

## Identity Versus Meaning

### Source identity is necessary but insufficient

CAB accepts source matching and then separately tests disease-meaning portability. All 26,725 benchmark rows were source matched. Of these, 26,421 were meaning accepted and 304 were meaning rejected due to phenotype-domain discordance.

ARR_977320 illustrates the distinction: the KCNQ1 source/gene match is accepted, but the ClinVar phenotype label is Silver-Russell syndrome 1, so deterministic inherited-arrhythmia meaning reuse is rejected and routed to contextual repair or disease-specific review.

## SADS/Postmortem Positioning

SADS is treated as a portability stress-test and use case. CAB can audit whether an inherited-arrhythmia assertion is being transferred into postmortem causal interpretation, family-risk interpretation, genotype-first phenotype-negative relatives, trigger-dependent contexts, or disease-specific channelopathy review.

CAB does not infer cause of death, predict sudden death, or predict family-member risk.

## Limitations

- The benchmark is retrospective and includes historical prospective emulation, not true prospective clinical validation.
- CAB has no patient outcome validation.
- Trigger-dependent and genotype-first regimes are underpowered or unsampled.
- CAB depends on public assertion data and public snapshot structure.
- CAB does not predict penetrance.
- CAB is not clinically deployed and is not clinical decision support.

## What CAB Is NOT

CAB does NOT:

- reclassify variants
- replace ACMG/AMP
- replace ClinGen/VCEP
- predict penetrance
- predict sudden death
- diagnose patients
- provide clinical decision support
- validate SADS risk

## Citation

```text
Contextual Assertion Biology (CAB): Disease-architecture-aware measurement of pathogenic assertion portability across Mendelian disease-model environments.
Manuscript in preparation. Citation details to be added at publication.
```

## License

Recommended license split:

- Code: MIT License
- Derived benchmark tables, documentation, figure source tables, and captions: CC-BY-4.0
