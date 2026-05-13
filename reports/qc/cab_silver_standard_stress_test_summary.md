# CAB Silver-Standard Stress-Test Package

This package implements proxy adjudication and robustness analyses requested for reviewer-hardening.

## Key Claim Boundary

The proxy adjudication layer uses locally available condition identifiers, environment mappings, ClinGen/VCEP coverage artifacts, and string/ontology heuristics. It is stronger than raw ClinVar label drift, but it is not a substitute for real blinded expert adjudication or full MONDO/HPO graph traversal with archived ontology releases.

## Generated artifacts

- reports/tables/cab_silver_standard_analysis_frame.csv
- reports/tables/cab_proxy_adjudication_layer.csv
- reports/tables/cab_proxy_adjudication_summary.csv
- reports/tables/cab_proxy_adjudicated_main_claim_models.csv
- reports/tables/cab_proxy_adjudicated_main_claim_deltas.csv
- reports/tables/cab_leave_gene_family_environment_domain_out_validation.csv
- reports/tables/cab_ontology_only_baseline_comparator.csv
- reports/tables/cab_baseline_only_ontology_forecasting_comparator.csv
- reports/tables/cab_baseline_only_ontology_incremental_cab_deltas.csv
- reports/tables/cab_negative_control_stable_domain_audit.csv
- reports/tables/cab_calibration_metrics.csv
- reports/tables/cab_risk_decile_calibration.csv
- reports/figures/cab_risk_decile_calibration.svg
- reports/tables/cab_submitter_stratified_models.csv
- reports/tables/cab_submitter_matched_drift_nondrift_analysis.csv
- reports/tables/cab_external_curated_subset_agreement.csv
- reports/tables/cab_sads_molecular_autopsy_special_validation_set.csv
- reports/tables/cab_component_ablation_study.csv
- reports/tables/cab_baseline_only_component_ablation_study.csv
- reports/tables/cab_auprc_enrichment_review_utility.csv
- reports/tables/cab_strict_endpoint_hierarchy_models.csv
- reports/tables/cab_direct_use_safety_analysis.csv
- reports/tables/cab_overrestriction_and_repair_audit.csv
- reports/tables/cab_rule_selected_case_studies.csv
- reports/tables/cab_prospective_temporal_hardening_inventory.csv

## Main endpoint hierarchy

- E1: crude condition-label drift.
- E2: normalized condition-label drift after proxy synonym/submitter-noise repair.
- E3: cross-environment drift.
- E4: proxy-adjudicated true disease-model shift.

## Run highlights

- Proxy adjudication split drift into true environment shift, ontology synonym/parent-child drift, submitter noise, uncertain, and stable rows across all 26,725 assertions.
- In inherited arrhythmia, the corrected row-level endpoint recovered 140 proxy true-environment shifts, 135 submitter-noise rows, 80 ontology synonym/parent-child rows, 10 uncertain rows, and 577 stable rows.
- On the strict E4 proxy endpoint, gene+regime improved AUROC over gene-only by 0.0100 (95% CI 0.0076-0.0129).
- Against a baseline-only ontology comparator with no follow-up label features, gene+baseline ontology+CAB improved AUROC by 0.0026 on E3 and 0.0164 on true+uncertain.
- Calibration is now reported without follow-up ontology features: gene+baseline-ontology+CAB AUROC 0.8796, AUPRC 0.5785, Brier 0.0879, ECE 0.0086 for E3.
- At a top-10% review budget, gene+baseline-ontology+CAB captured E4 proxy true shifts with precision 0.6105, lift 4.08, and 61.1 expected unsupported reuses per 100 reviewed.
