# CAB Table Index

| Table | Role | Source |
|---|---|---|
| cab_hardcore_evidence_upgrade_index.csv | generated artifact index | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_endpoint_triangulation_matrix.csv | endpoint triangulation | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_falsification_negative_controls.csv | negative controls | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_random_routing_null_comparison.csv | random routing controls | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_domain_balanced_metrics.csv | macro/micro domain robustness | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_domain_downsample_stability.csv | downsample stability | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_leave_one_domain_out_metrics.csv | leave-one-domain-out robustness | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_curator_review_budget_utility.csv | curator utility | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_workload_capture_curves.csv | workload capture curves | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_continuous_operating_frontier.csv | threshold-free operating frontier | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_cost_sensitive_frontier_recommendations.csv | cost-sensitive recommendations | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_positive_claims_supported_by_new_analyses.csv | main positive claim map | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_claim_boundaries_quarantined.csv | claim-boundary quarantine | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_reviewer_evidence_map.csv | reviewer issue response map | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_clingen_vcep_comparator_mapping.csv | external comparator mapping | scripts/build_cab_hardcore_evidence_upgrade.py |
| cab_vcep_disease_specific_review_enrichment.csv | VCEP/review enrichment | scripts/build_cab_hardcore_evidence_upgrade.py |
| emerge_quantitative_proxy_metrics.csv | eMERGE quantitative proxy | scripts/build_cab_hardcore_evidence_upgrade.py |
| genotype_first_quantitative_proxy_metrics.csv | genotype-first proxy metrics | scripts/build_cab_hardcore_evidence_upgrade.py |
| lovd_gpcards_external_label_sample.csv | external label sample schema | scripts/build_cab_hardcore_evidence_upgrade.py |
## Expert Endpoint Adjudication Run Tables

| Table | Role | Source |
|---|---|---|
| cab_expert_endpoint_run_packet_blinded.csv | true-blind reviewer case packet | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_verdict_template.csv | reviewer assignment and verdict-entry template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_consensus_template.csv | consensus endpoint aggregation template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_temporal_endpoint_key.csv | analyst-only ClinVar drift endpoint key | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_priority_cases.csv | SADS/CPVT high-value adjudication addendum | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_verdict_template.csv | SADS/CPVT five-reviewer verdict template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_prediction_key.csv | analyst-only SADS/CPVT CAB prediction key | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_validation_analysis_contract.csv | endpoint-validation analysis contract | scripts/build_cab_expert_endpoint_adjudication_run.py |
## Reviewer Methodological Repair Tables

| Table | Role | Source |
|---|---|---|
| structural_functional_overlap_disease_specific_review_2x2.csv | 2x2 table for structural-functional overlap x disease-specific review | scripts/build_reviewer_response_analyses.py |
| structural_functional_overlap_disease_specific_review_ci.csv | Fisher exact p and Haldane-Anscombe/Woolf CI for OR=82.43 | scripts/build_reviewer_response_analyses.py |
| cab_alphamissense_selection_bias_audit.csv | AlphaMissense matched/unmatched observability and selection-bias audit | scripts/build_reviewer_response_analyses.py |
| cab_alphamissense_selection_bias_functional_class_proxy.csv | ClinVar submitter-count proxy comparison for AM-feasible missense rows | scripts/build_reviewer_response_analyses.py |
| clinvar_label_drift_decomposition.csv | ClinVar label drift decomposition into environment shift, submitter change, and rename/term-change classes | scripts/build_reviewer_response_analyses.py |
| clinvar_label_drift_decomposition_cases.csv | row-level inherited-arrhythmia label-drift decomposition cases | scripts/build_reviewer_response_analyses.py |
| cab_domain_split_operating_frontier.csv | domain-split continuous operating frontier source table | scripts/build_reviewer_response_analyses.py |
| cab_domain_split_operating_frontier_shape_comparison.csv | hereditary cancer vs cardiomyopathy+arrhythmia frontier shape comparison | scripts/build_reviewer_response_analyses.py |

## Silver-Standard Stress-Test Tables

| Table | Role | Source |
|---|---|---|
| cab_proxy_adjudication_layer.csv | proxy adjudication of drifted and stable rows using condition IDs/environment mappings | scripts/build_cab_silver_standard_stress_tests.py |
| cab_proxy_adjudicated_main_claim_models.csv | main-claim models recomputed on proxy strict endpoints | scripts/build_cab_silver_standard_stress_tests.py |
| cab_leave_gene_family_environment_domain_out_validation.csv | leave-one-gene/family/environment/domain-out validation | scripts/build_cab_silver_standard_stress_tests.py |
| cab_ontology_only_baseline_comparator.csv | ontology-only baseline comparator | scripts/build_cab_silver_standard_stress_tests.py |
| cab_negative_control_stable_domain_audit.csv | false-alarm audit in high-confidence stable strata | scripts/build_cab_silver_standard_stress_tests.py |
| cab_calibration_metrics.csv | Brier, ECE, calibration slope/intercept | scripts/build_cab_silver_standard_stress_tests.py |
| cab_risk_decile_calibration.csv | risk decile calibration source table | scripts/build_cab_silver_standard_stress_tests.py |
| cab_submitter_stratified_models.csv | submitter-stratified model robustness | scripts/build_cab_silver_standard_stress_tests.py |
| cab_external_curated_subset_agreement.csv | partial external curated-subset agreement proxy | scripts/build_cab_silver_standard_stress_tests.py |
| cab_sads_molecular_autopsy_special_validation_set.csv | SADS/molecular-autopsy special validation set | scripts/build_cab_silver_standard_stress_tests.py |
| cab_component_ablation_study.csv | ablation of gene, domain, ontology, metadata, and CAB regime components | scripts/build_cab_silver_standard_stress_tests.py |
| cab_auprc_enrichment_review_utility.csv | AUPRC and review-budget enrichment across endpoint hierarchy | scripts/build_cab_silver_standard_stress_tests.py |
| cab_direct_use_safety_analysis.csv | direct-use safety analysis for ClinVar-label-only, CAB-Strict, and CAB-Balanced | scripts/build_cab_silver_standard_stress_tests.py |
| cab_overrestriction_and_repair_audit.csv | overrestriction and contextual-repair rescue audit | scripts/build_cab_silver_standard_stress_tests.py |
| cab_rule_selected_case_studies.csv | rule-selected case studies, not cherry-picked | scripts/build_cab_silver_standard_stress_tests.py |

## Domain-Calibrated Balanced v2 Tables

| Table | Role | Source |
|---|---|---|
| cab_domain_calibrated_balanced_v2_metrics.csv | CAB-Balanced-global vs domain-calibrated vs SADS-high-stringency metrics | scripts/build_domain_calibrated_balanced_v2.py |
| cab_syndrome_organ_repair_simulation_summary.csv | syndrome-organ repair rescue summary | scripts/build_domain_calibrated_balanced_v2.py |
| cab_syndrome_organ_repair_simulation_cases.csv | row-level repair simulation cases | scripts/build_domain_calibrated_balanced_v2.py |
