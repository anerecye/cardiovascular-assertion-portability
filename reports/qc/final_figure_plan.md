# Final Figure Plan

Technical figure plan; not manuscript prose.

## Figure 1 — Conceptual model
Goal: distinguish a ClinVar P/LP classification label from an assertion portability object.
Panel requirements:
- P/LP label layer.
- assertion-use context layer.
- disease-model environment layer.
- routing layer: allow / contextual repair / expert review.
Do not add decorative pathway fluff.

## Figure 2 — Arrhythmia temporal meaning drift and transition network
Inputs:
- reports/tables/condition_environment_transition_edges.csv
- reports/tables/transition_network_enrichment_tests.csv
- data/processed/cab_cross_environment_drift.csv
Show:
- disease-model self-loops vs cross-environment edges.
- enrichment for disease_model_collision / low portability.
- canonical self-loop enrichment.
Blocked:
- do not show leaked CPI AUCs.

## Figure 3 — Gene+CAB decomposition in arrhythmia
Inputs:
- reports/tables/gene_vs_cab_model_comparison.csv
- reports/tables/mixed_effects_gene_variance_decomposition.csv
- reports/tables/cab_gene_archetypes.csv
Show:
- gene-only vs CAB-only vs gene+CAB vs gene+CAB+metadata.
- residual gene variance reduction.
- sentinel gene archetypes: SCN5A, RYR2/CASQ2/TRDN, KCNQ1/KCNH2, CACNA1C, HCN4/ANK2.

## Figure 4 — Cardiomyopathy replication
Inputs:
- reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv
- reports/tables/cardiomyopathy_model_comparison_baseline_only.csv
- reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv
Show:
- endpoint rates.
- baseline-only model comparison.
- low baseline portability enrichment.
Blocked:
- do not show v1 cross-environment AUROC=0.9742 except as deprecated/quarantined if needed.

## Figure 5 — Cross-domain portability grammar
Inputs:
- reports/tables/cab_cross_domain_replication_summary.csv
- reports/tables/domain_specific_portability_grammar.csv
Show:
- inherited arrhythmia vs cardiomyopathy side-by-side.
- stable architecture, unstable architecture, gene role, external constraint status.

## Figure 6 — Comparator/actionability
Inputs:
- reports/tables/cab_alphamissense_model_comparison.csv
- reports/tables/cab_alphamissense_hg38_join_qc.csv
- reports/tables/cab_counterfactual_task_metrics.csv
Show:
- AlphaMissense-only vs CAB-only vs CAB+AlphaMissense in missense subset.
- counterfactual unsupported deterministic reuse reduction across five tasks.
Guardrail:
- AlphaMissense is missense-only sensitivity, not full universe.
- counterfactual correctness is rule-adjudicated, not external expert adjudication.
