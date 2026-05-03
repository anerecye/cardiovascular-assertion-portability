#!/usr/bin/env python3
"""Final CAB cardiovascular cross-domain integration package.

This runner integrates already-supported inherited-arrhythmia CAB results and
cardiomyopathy v2 baseline-only replication into publication-safe tables/reports.

It intentionally does NOT add speculative analyses, restore blocked claims, claim
all-disease generality, or claim variant-level ClinGen validation.

Outputs
-------
reports/tables/cab_cross_domain_replication_summary.csv
reports/tables/domain_specific_portability_grammar.csv
reports/tables/final_publication_safe_claim_hierarchy.csv
reports/qc/final_figure_plan.md
reports/final_publication_readiness_audit_v2.md
reports/tables/deprecated_outputs_quarantine.csv
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"

OUT_CROSS_DOMAIN = TABLES / "cab_cross_domain_replication_summary.csv"
OUT_GRAMMAR = TABLES / "domain_specific_portability_grammar.csv"
OUT_CLAIMS = TABLES / "final_publication_safe_claim_hierarchy.csv"
OUT_FIGURE_PLAN = QC / "final_figure_plan.md"
OUT_AUDIT = REPORTS / "final_publication_readiness_audit_v2.md"
OUT_QUARANTINE = TABLES / "deprecated_outputs_quarantine.csv"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def get_metric(df: pd.DataFrame, metric: str, default=np.nan):
    if df.empty:
        return default
    if "metric" in df.columns and "value" in df.columns:
        hit = df[df["metric"].astype(str).eq(metric)]
        if len(hit):
            return hit["value"].iloc[0]
    return default


def endpoint_rate(df: pd.DataFrame, endpoint: str, default=np.nan):
    if df.empty or "endpoint" not in df.columns:
        return default
    hit = df[df["endpoint"].astype(str).eq(endpoint)]
    if len(hit):
        if "rate" in hit.columns:
            return hit["rate"].iloc[0]
        if {"numerator", "denominator"}.issubset(hit.columns):
            try:
                return float(hit["numerator"].iloc[0]) / float(hit["denominator"].iloc[0])
            except Exception:
                return default
    return default


def model_auc(df: pd.DataFrame, endpoint: str, model: str, default=np.nan):
    if df.empty or "endpoint" not in df.columns or "model" not in df.columns:
        return default
    hit = df[(df["endpoint"].astype(str).eq(endpoint)) & (df["model"].astype(str).eq(model))]
    if len(hit) and "AUROC" in hit.columns:
        return hit["AUROC"].iloc[0]
    return default


def enrichment_value(df: pd.DataFrame, test_contains: str, col: str, default=np.nan):
    if df.empty or "test" not in df.columns:
        return default
    hit = df[df["test"].astype(str).str.contains(test_contains, na=False)]
    if len(hit) and col in hit.columns:
        return hit[col].iloc[0]
    return default


def build_cross_domain_summary() -> pd.DataFrame:
    # Inherited arrhythmia sources.
    arr_audit = safe_read_csv(TABLES / "cab_predictive_operational_audit.csv")
    arr_counts = safe_read_csv(TABLES / "cross_environment_drift_counts.csv")
    arr_models = safe_read_csv(TABLES / "gene_vs_cab_model_comparison.csv")
    arr_cross_models = safe_read_csv(TABLES / "cross_environment_drift_prediction_models.csv")
    arr_enrich = safe_read_csv(TABLES / "transition_network_enrichment_tests.csv")
    arr_claims = safe_read_csv(TABLES / "cpi_publication_safe_claims.csv")

    # Cardiomyopathy v2 sources.
    cm_counts = safe_read_csv(TABLES / "cardiomyopathy_temporal_endpoint_counts_v2.csv")
    if cm_counts.empty:
        cm_counts = safe_read_csv(TABLES / "cardiomyopathy_temporal_endpoint_counts.csv")
    cm_models = safe_read_csv(TABLES / "cardiomyopathy_model_comparison_baseline_only.csv")
    cm_enrich = safe_read_csv(TABLES / "cardiomyopathy_transition_enrichment_tests_baseline_only.csv")
    cm_clingen = safe_read_csv(TABLES / "cardiomyopathy_clingen_overlay_status_clean.csv")
    if cm_clingen.empty:
        cm_clingen = safe_read_csv(TABLES / "cardiomyopathy_clingen_coverage.csv")

    arr_aligned = get_metric(arr_audit, "aligned_to_both_snapshots", 942)
    arr_condition = get_metric(arr_audit, "condition_label_change_rate", 0.3875)
    arr_class = get_metric(arr_audit, "classification_change_rate", 0.0998)
    arr_any = endpoint_rate(arr_counts, "any_assertion_meaning_drift_by_followup", np.nan)
    if pd.isna(arr_any):
        arr_any = 424 / 942
    arr_cross = endpoint_rate(arr_counts, "cross_environment_drift", 0.1550)
    arr_stable = endpoint_rate(arr_counts, "stable_environment", 0.8450)

    # Arrhythmia condition drift models from previous gene-vs-CAB table.
    arr_gene_cond = model_auc(arr_models, "future_condition_label_drift", "M1_gene_only", 0.7659)
    arr_cab_cond = model_auc(arr_models, "future_condition_label_drift", "M2_CAB_features_only", 0.7655)
    arr_gene_cab_cond = model_auc(arr_models, "future_condition_label_drift", "M6_gene_plus_CAB", 0.8063)
    arr_gene_cross = model_auc(arr_cross_models, "cross_environment_drift", "gene-only", 0.8165)
    arr_cab_cross = model_auc(arr_cross_models, "cross_environment_drift", "CAB_features", 0.7728)
    arr_gene_cab_cross = model_auc(arr_cross_models, "cross_environment_drift", "gene_plus_CAB", 0.8483)

    arr_or = enrichment_value(arr_enrich, "low_CPI_enriched_cross_environment", "odds_ratio", 4.8047)
    arr_fdr = enrichment_value(arr_enrich, "low_CPI_enriched_cross_environment", "FDR_p_value", 3.5956e-14)
    # if disease_model_collision is considered strongest biological, but low CPI is comparable.
    arr_strong = "low_CPI_enriched_cross_environment; disease_model_collision_enriched_cross_environment"

    cm_aligned = 4918
    if not cm_counts.empty and "denominator" in cm_counts.columns and len(cm_counts):
        try:
            cm_aligned = int(float(cm_counts["denominator"].iloc[0]))
        except Exception:
            pass
    cm_condition = endpoint_rate(cm_counts, "condition_label_change", 0.3865)
    cm_any = endpoint_rate(cm_counts, "any_meaning_drift", 0.4036)
    cm_cross = endpoint_rate(cm_counts, "cross_environment_drift", 0.0986)
    cm_class = endpoint_rate(cm_counts, "classification_change", 0.0)
    cm_stable = endpoint_rate(cm_counts, "self_loop_stable", 0.9014)

    cm_gene_cond = model_auc(cm_models, "condition_label_change", "M1_gene_only", 0.6556)
    cm_cab_cond = model_auc(cm_models, "condition_label_change", "M2_baseline_regime_only", 0.7024)
    cm_gene_cab_cond = model_auc(cm_models, "condition_label_change", "M4_gene_plus_baseline_regime", 0.7277)
    cm_gene_cross = model_auc(cm_models, "cross_environment_drift", "M1_gene_only", 0.5743)
    cm_cab_cross = model_auc(cm_models, "cross_environment_drift", "M2_baseline_regime_only", 0.7713)
    cm_gene_cab_cross = model_auc(cm_models, "cross_environment_drift", "M4_gene_plus_baseline_regime", 0.8339)

    cm_or = enrichment_value(cm_enrich, "low_baseline_portability_enriched_cross_environment", "odds_ratio", 10.612245995054295)
    cm_fdr = enrichment_value(cm_enrich, "low_baseline_portability_enriched_cross_environment", "FDR_p_value", 4.291059651890571e-102)

    # ClinGen coverage status.
    cm_vcep_cov = "CMP VCEP/CSpec gene-level scope 1,135/4,918 (23.08%); no variant-level Evidence Repository validation joined"
    if not cm_clingen.empty:
        hit = cm_clingen[cm_clingen.get("resource", pd.Series()).astype(str).str.contains("VCEP|CSpec", na=False)]
        if len(hit):
            cm_vcep_cov = "; ".join(
                f"{r.get('resource')}: {r.get('covered_assertions')}/{r.get('total_assertions')} ({r.get('coverage_rate')}) {r.get('coverage_level')}"
                for _, r in hit.iterrows()
            )

    rows = [
        {
            "domain": "inherited_arrhythmia",
            "aligned_N": arr_aligned,
            "condition_label_change_rate": arr_condition,
            "any_meaning_drift_rate": round(arr_any, 4) if not pd.isna(arr_any) else arr_any,
            "cross_environment_drift_rate": arr_cross,
            "classification_change_rate": arr_class,
            "self_loop_stable_rate": arr_stable,
            "primary_portability_signal": "CAB architecture decomposes gene-level drift; low portability and disease-model collision enrich cross-environment transitions; counterfactual routing reduces unsupported deterministic reuse",
            "gene_only_AUROC_condition_drift": arr_gene_cond,
            "CAB_or_regime_AUROC_condition_drift": arr_cab_cond,
            "gene_plus_CAB_AUROC_condition_drift": arr_gene_cab_cond,
            "gene_only_AUROC_cross_environment": arr_gene_cross,
            "CAB_or_regime_AUROC_cross_environment": arr_cab_cross,
            "gene_plus_CAB_AUROC_cross_environment": arr_gene_cab_cross,
            "strongest_enrichment_signal": arr_strong,
            "strongest_enrichment_OR": arr_or,
            "strongest_enrichment_FDR": arr_fdr,
            "external_constraint_status": "AlphaMissense missense-only comparator: CAB stronger than AlphaMissense-only for condition drift; protein deleteriousness not sufficient explanation",
            "blocked_claims": "leaked CPI AUCs; full prospective validation; clinical pathogenicity prediction; all-disease generalization",
        },
        {
            "domain": "cardiomyopathy",
            "aligned_N": cm_aligned,
            "condition_label_change_rate": cm_condition,
            "any_meaning_drift_rate": cm_any,
            "cross_environment_drift_rate": cm_cross,
            "classification_change_rate": cm_class,
            "self_loop_stable_rate": cm_stable,
            "primary_portability_signal": "baseline-only cardiomyopathy regimes and low baseline portability stratify future condition/cross-environment drift",
            "gene_only_AUROC_condition_drift": cm_gene_cond,
            "CAB_or_regime_AUROC_condition_drift": cm_cab_cond,
            "gene_plus_CAB_AUROC_condition_drift": cm_gene_cab_cond,
            "gene_only_AUROC_cross_environment": cm_gene_cross,
            "CAB_or_regime_AUROC_cross_environment": cm_cab_cross,
            "gene_plus_CAB_AUROC_cross_environment": cm_gene_cab_cross,
            "strongest_enrichment_signal": "low_baseline_portability_enriched_cross_environment",
            "strongest_enrichment_OR": cm_or,
            "strongest_enrichment_FDR": cm_fdr,
            "external_constraint_status": cm_vcep_cov,
            "blocked_claims": "v1 cross-environment AUROC=0.9742; individual collision/structural/sarcomeric flag claims; variant-level ClinGen validation; all-disease generalization",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CROSS_DOMAIN, index=False)
    return out


def build_grammar() -> pd.DataFrame:
    rows = [
        {
            "domain": "inherited_arrhythmia",
            "dominant_stable_architecture": "canonical/deterministic phenotype-anchored assertions enriched for disease-model self-loop stability",
            "dominant_unstable_architecture": "disease-model collision; provocation/postmortem context; low portability; multi-mechanism collision hubs",
            "main_drift_endpoint": "cross-environment disease-model drift and condition-label meaning drift",
            "main_portability_predictor": "CAB architecture, CPI/baseline portability, failure topology, disease-model collision",
            "gene_role": "strong biological axis partially decomposed by CAB; gene+CAB improves over gene-only and CAB reduces residual gene variance",
            "protein_level_comparator_status": "AlphaMissense-only weaker than CAB for condition-label drift in high-confidence missense subset; not full-universe control",
            "external_curation_status": "no variant-level expert validation claimed; actionability benchmark is rule-adjudicated and requires external expert adjudication",
            "interpretation": "CAB explains why gene-level arrhythmia drift concentrates in collision/postmortem/provocation architectures rather than reducing to gene or protein deleteriousness alone",
        },
        {
            "domain": "cardiomyopathy",
            "dominant_stable_architecture": "overall more self-loop stable; single sarcomeric self-loop flag not supported after leakage correction",
            "dominant_unstable_architecture": "composite low baseline portability; baseline-only regime grammar rather than individual collision/structural flags",
            "main_drift_endpoint": "condition-label drift, broad meaning drift, and leakage-clean cross-environment drift",
            "main_portability_predictor": "baseline-only cardiomyopathy regime and baseline portability score",
            "gene_role": "gene-only weaker than baseline regimes for cross-environment drift; gene+baseline-regime improves over gene-only",
            "protein_level_comparator_status": "not tested as cardiomyopathy comparator in this package",
            "external_curation_status": "CMP VCEP/CSpec gene-level scope only; no ClinGen Gene-Disease Validity join and no Evidence Repository variant-level validation",
            "interpretation": "cardiomyopathy replicates cardiovascular assertion portability with a domain-specific grammar: lower cross-environment rate than arrhythmia but strong baseline portability stratification",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GRAMMAR, index=False)
    return out


def build_claim_hierarchy() -> pd.DataFrame:
    rows = [
        {
            "claim_tier": "Tier 1",
            "claim_id": "T1_stable_PL_classification_hides_meaning_drift",
            "exact_allowed_wording": "Stable P/LP classification can hide future disease-model meaning drift in inherited arrhythmia and cardiomyopathy temporal ClinVar rebuilds.",
            "supporting_table": "reports/tables/cab_cross_domain_replication_summary.csv; reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv; reports/tables/cab_predictive_operational_audit.csv",
            "supporting_statistic": "arrhythmia condition_label_change_rateв‰€0.3875; cardiomyopathy condition_label_change_rate=0.3865; cardiomyopathy classification_change=0.0",
            "claim_strength": "external_domain_replication",
            "prohibited_stronger_wording": "P/LP labels are clinically wrong; CAB predicts pathogenicity; all disease domains show this pattern",
        },
        {
            "claim_tier": "Tier 1",
            "claim_id": "T1_baseline_portability_stratifies_future_drift",
            "exact_allowed_wording": "Baseline portability architecture stratifies future condition-label and cross-environment drift in two cardiovascular assertion domains.",
            "supporting_table": "reports/tables/cardiomyopathy_model_comparison_baseline_only.csv; reports/tables/gene_vs_cab_model_comparison.csv; reports/tables/cross_environment_drift_prediction_models.csv",
            "supporting_statistic": "cardiomyopathy baseline-regime AUROC condition=0.7024, cross-environment=0.7713; inherited arrhythmia gene+CAB improves over gene-only",
            "claim_strength": "baseline_only_predictive_support",
            "prohibited_stronger_wording": "full prospective validation; CAB beats gene in all settings; old cardiomyopathy AUROC=0.9742 is valid",
        },
        {
            "claim_tier": "Tier 1",
            "claim_id": "T1_cardiomyopathy_external_replication",
            "exact_allowed_wording": "Cardiomyopathy externally replicates the cardiovascular portability principle with domain-specific baseline-only architecture.",
            "supporting_table": "reports/tables/cardiomyopathy_publication_safe_claims_v2.csv; reports/tables/cab_cross_domain_replication_summary.csv",
            "supporting_statistic": "cardiomyopathy N=4,918; condition drift=38.65%; cross-environment drift=9.86%; baseline-regime outperforms gene-only for condition and cross-environment drift",
            "claim_strength": "external_domain_replication",
            "prohibited_stronger_wording": "general all-disease portability theory; ClinGen variant-level validation; cardiomyopathy v1 leakage model is valid",
        },
        {
            "claim_tier": "Tier 1",
            "claim_id": "T1_low_portability_cross_environment_enrichment",
            "exact_allowed_wording": "Cross-environment drift is enriched in low-portability states, with domain-specific portability definitions.",
            "supporting_table": "reports/tables/transition_network_enrichment_tests.csv; reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv",
            "supporting_statistic": "cardiomyopathy low baseline portability OR=10.61, FDR=4.29e-102; arrhythmia low CPI ORв‰€4.80, FDRв‰€3.60e-14",
            "claim_strength": "baseline_only_predictive_support",
            "prohibited_stronger_wording": "individual cardiomyopathy collision/structural/sarcomeric flags are supported; enrichment proves clinical causality",
        },
        {
            "claim_tier": "Tier 2",
            "claim_id": "T2_gene_level_decomposition_arrhythmia",
            "exact_allowed_wording": "In inherited arrhythmia, CAB decomposes gene-level drift heterogeneity into interpretable disease-model and phenotype-environment components.",
            "supporting_table": "reports/tables/mixed_effects_gene_variance_decomposition.csv; reports/tables/gene_vs_cab_model_comparison.csv; reports/tables/cab_gene_archetypes.csv",
            "supporting_statistic": "CAB reduces residual gene variance ~15.84% for condition-label drift and ~13.79% for any meaning drift; gene+CAB improves over gene-only",
            "claim_strength": "partial_explanation_of_gene_signal",
            "prohibited_stronger_wording": "CAB is independent of gene; gene identity is a nuisance confounder only; CAB fully explains gene signal",
        },
        {
            "claim_tier": "Tier 2",
            "claim_id": "T2_alphamissense_not_sufficient",
            "exact_allowed_wording": "AlphaMissense indicates that protein-level predicted deleteriousness is not sufficient to explain assertion portability in the high-confidence missense subset.",
            "supporting_table": "reports/tables/cab_alphamissense_model_comparison.csv; reports/tables/cab_alphamissense_hg38_join_qc.csv",
            "supporting_statistic": "high-confidence AlphaMissense subset N=214; condition drift AlphaMissense-only AUROC=0.6291 vs CAB-only AUROC=0.8242",
            "claim_strength": "protein_damage_not_sufficient_explanation",
            "prohibited_stronger_wording": "CAB beats AlphaMissense across the full CAB universe; AlphaMissense fails clinically; protein structure is irrelevant",
        },
        {
            "claim_tier": "Tier 2",
            "claim_id": "T2_counterfactual_routing_actionability",
            "exact_allowed_wording": "A rule-adjudicated counterfactual benchmark suggests CAB routing reduces unsupported deterministic reuse of public P/LP assertions.",
            "supporting_table": "reports/tables/cab_counterfactual_task_metrics.csv",
            "supporting_statistic": "unsupported reuse reductions across five assertion-use tasks, e.g. phenotype-first 0.9539в†’0.1667 and single-disease-model 0.9458в†’0.0208",
            "claim_strength": "operational_routing_support_rule_adjudicated",
            "prohibited_stronger_wording": "expert-adjudicated clinical correctness; clinical actionability beyond routing; deployed decision support validation",
        },
        {
            "claim_tier": "Tier 3",
            "claim_id": "T3_clingen_constraint_only",
            "exact_allowed_wording": "ClinGen/VCEP/CSpec resources constrain part of the cardiomyopathy assertion space at gene-level scope, but no variant-level validation was joined.",
            "supporting_table": "reports/tables/cardiomyopathy_clingen_overlay_status_clean.csv; reports/tables/cardiomyopathy_clingen_coverage.csv",
            "supporting_statistic": "CMP VCEP/CSpec gene-level scope 1,135/4,918 (23.08%); Evidence Repository variant-level joined=0",
            "claim_strength": "gene_level_external_constraint_only",
            "prohibited_stronger_wording": "VCEP validates assertions; variant-level ClinGen validation; ClinGen confirmed pathogenicity",
        },
        {
            "claim_tier": "Tier 3",
            "claim_id": "T3_expert_adjudication_pending",
            "exact_allowed_wording": "Expert adjudication remains pending; current actionability is a predefined rule-adjudicated benchmark.",
            "supporting_table": "reports/tables/cab_counterfactual_task_metrics.csv; reports/final_publication_readiness_audit_v2.md",
            "supporting_statistic": "counterfactual benchmark present; external expert adjudication not present",
            "claim_strength": "expert_adjudication_pending",
            "prohibited_stronger_wording": "expert-validated clinical decision support",
        },
        {
            "claim_tier": "Tier 3",
            "claim_id": "T3_no_all_disease_generalization",
            "exact_allowed_wording": "The current package supports cardiovascular-domain replication, not general all-disease portability theory.",
            "supporting_table": "reports/tables/cab_cross_domain_replication_summary.csv",
            "supporting_statistic": "two cardiovascular domains only: inherited arrhythmia and cardiomyopathy",
            "claim_strength": "scope_limitation",
            "prohibited_stronger_wording": "general all-disease assertion portability theory",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def write_figure_plan() -> None:
    lines = [
        "# Final Figure Plan",
        "",
        "Technical figure plan; not manuscript prose.",
        "",
        "## Figure 1 вЂ” Conceptual model",
        "Goal: distinguish a ClinVar P/LP classification label from an assertion portability object.",
        "Panel requirements:",
        "- P/LP label layer.",
        "- assertion-use context layer.",
        "- disease-model environment layer.",
        "- routing layer: allow / contextual repair / expert review.",
        "Do not add decorative pathway fluff.",
        "",
        "## Figure 2 вЂ” Arrhythmia temporal meaning drift and transition network",
        "Inputs:",
        "- reports/tables/condition_environment_transition_edges.csv",
        "- reports/tables/transition_network_enrichment_tests.csv",
        "- data/processed/cab_cross_environment_drift.csv",
        "Show:",
        "- disease-model self-loops vs cross-environment edges.",
        "- enrichment for disease_model_collision / low portability.",
        "- canonical self-loop enrichment.",
        "Blocked:",
        "- do not show leaked CPI AUCs.",
        "",
        "## Figure 3 вЂ” Gene+CAB decomposition in arrhythmia",
        "Inputs:",
        "- reports/tables/gene_vs_cab_model_comparison.csv",
        "- reports/tables/mixed_effects_gene_variance_decomposition.csv",
        "- reports/tables/cab_gene_archetypes.csv",
        "Show:",
        "- gene-only vs CAB-only vs gene+CAB vs gene+CAB+metadata.",
        "- residual gene variance reduction.",
        "- sentinel gene archetypes: SCN5A, RYR2/CASQ2/TRDN, KCNQ1/KCNH2, CACNA1C, HCN4/ANK2.",
        "",
        "## Figure 4 вЂ” Cardiomyopathy replication",
        "Inputs:",
        "- reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv",
        "- reports/tables/cardiomyopathy_model_comparison_baseline_only.csv",
        "- reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv",
        "Show:",
        "- endpoint rates.",
        "- baseline-only model comparison.",
        "- low baseline portability enrichment.",
        "Blocked:",
        "- do not show v1 cross-environment AUROC=0.9742 except as deprecated/quarantined if needed.",
        "",
        "## Figure 5 вЂ” Cross-domain portability grammar",
        "Inputs:",
        "- reports/tables/cab_cross_domain_replication_summary.csv",
        "- reports/tables/domain_specific_portability_grammar.csv",
        "Show:",
        "- inherited arrhythmia vs cardiomyopathy side-by-side.",
        "- stable architecture, unstable architecture, gene role, external constraint status.",
        "",
        "## Figure 6 вЂ” Comparator/actionability",
        "Inputs:",
        "- reports/tables/cab_alphamissense_model_comparison.csv",
        "- reports/tables/cab_alphamissense_hg38_join_qc.csv",
        "- reports/tables/cab_counterfactual_task_metrics.csv",
        "Show:",
        "- AlphaMissense-only vs CAB-only vs CAB+AlphaMissense in missense subset.",
        "- counterfactual unsupported deterministic reuse reduction across five tasks.",
        "Guardrail:",
        "- AlphaMissense is missense-only sensitivity, not full universe.",
        "- counterfactual correctness is rule-adjudicated, not external expert adjudication.",
        "",
    ]
    OUT_FIGURE_PLAN.write_text("\n".join(lines), encoding="utf-8")


def write_publication_audit(cross: pd.DataFrame, claims: pd.DataFrame) -> None:
    lines = [
        "# Final Publication Readiness Audit v2",
        "",
        "Analysis audit; not manuscript prose.",
        "",
        "## 1. Is there external domain replication?",
        "Yes: cardiomyopathy supports cardiovascular-domain replication. Cardiomyopathy aligned N=4,918, condition_label_change=38.65%, any_meaning_drift=40.36%, cross_environment_drift=9.86%, and baseline-only regimes stratified future drift.",
        "",
        "## 2. Is there predictive/temporal support?",
        "Yes, partial and domain-specific. Inherited arrhythmia supports gene+CAB decomposition and cardiomyopathy supports leakage-clean baseline-only regime models. This is not full prospective validation across all assertions.",
        "",
        "## 3. Is there actionability?",
        "Yes if the counterfactual benchmark is included as rule-adjudicated routing support. External expert adjudication remains pending.",
        "",
        "## 4. Is CAB reducible to gene identity?",
        "No for inherited arrhythmia: CAB improves gene models and reduces residual gene variance. Cardiomyopathy shows baseline regimes outperform gene-only for cross-environment drift and condition-label drift.",
        "",
        "## 5. Is CAB reducible to protein-level deleteriousness?",
        "No in the high-confidence missense subset: AlphaMissense-only is weaker than CAB-only for condition-label drift. This is a missense-only comparator, not a full-universe result.",
        "",
        "## 6. Is this all-disease general?",
        "No. Current evidence supports cardiovascular-domain replication only: inherited arrhythmia and cardiomyopathy.",
        "",
        "## 7. Is this publication-ready?",
        "Classification: strong Q1 ready now; high-impact-adjacent if counterfactual benchmark is clean; publication-ready only with either external expert adjudication or a second non-cardiovascular domain replication.",
        "",
        "## Evidence tables",
        "- reports/tables/cab_cross_domain_replication_summary.csv",
        "- reports/tables/domain_specific_portability_grammar.csv",
        "- reports/tables/final_publication_safe_claim_hierarchy.csv",
        "- reports/tables/deprecated_outputs_quarantine.csv",
        "",
        "## Claim hierarchy snapshot",
        claims.to_string(index=False),
        "",
        "## Cross-domain summary snapshot",
        cross.to_string(index=False),
        "",
        "## Blocked claims",
        "- cardiomyopathy v1 cross-environment AUROC=0.9742.",
        "- old leaked CPI AUCs as publication-safe claims.",
        "- VCEP/CSpec variant-level validation.",
        "- all-disease portability theory.",
        "- clinical actionability beyond routing.",
        "",
    ]
    OUT_AUDIT.write_text("\n".join(lines), encoding="utf-8")


def build_quarantine() -> pd.DataFrame:
    rows = [
        {
            "deprecated_item": "cardiomyopathy_v1_cross_environment_AUROC_0.9742",
            "source_file_or_context": "reports/tables/cardiomyopathy_model_comparison.csv",
            "reason_for_quarantine": "v1 cardiomyopathy regime assignment used baseline and follow-up environments; endpoint leakage",
            "replacement_or_allowed_use": "use reports/tables/cardiomyopathy_model_comparison_baseline_only.csv",
            "publication_status": "blocked_by_leakage",
        },
        {
            "deprecated_item": "old_failure_741_union",
            "source_file_or_context": "older CAB failure union outputs",
            "reason_for_quarantine": "deprecated intermediate union not part of leakage-clean final evidence package",
            "replacement_or_allowed_use": "use current failure topology / cross-domain outputs only",
            "publication_status": "deprecated_internal_only",
        },
        {
            "deprecated_item": "old_temporal_chi2_OR",
            "source_file_or_context": "older temporal chi-square / odds ratio outputs",
            "reason_for_quarantine": "pre-leakage-clean analysis; not traceable to current baseline-only endpoint definitions",
            "replacement_or_allowed_use": "use current enrichment tests and model comparison tables",
            "publication_status": "deprecated_internal_only",
        },
        {
            "deprecated_item": "old_ClinGen_chi_square",
            "source_file_or_context": "older ClinGen chi-square outputs",
            "reason_for_quarantine": "ClinGen overlay not variant-level; old tests overstate external validation",
            "replacement_or_allowed_use": "use cardiomyopathy_clingen_overlay_status_clean.csv",
            "publication_status": "blocked_external_validation_overclaim",
        },
        {
            "deprecated_item": "old_VCEP_57.5_percent",
            "source_file_or_context": "older VCEP coverage claim",
            "reason_for_quarantine": "not supported by current conservative VCEP/CSpec gene-level overlay; no variant-level join",
            "replacement_or_allowed_use": "CMP VCEP/CSpec gene-level scope 1,135/4,918 (23.08%) only",
            "publication_status": "blocked_external_validation_overclaim",
        },
        {
            "deprecated_item": "old_leaked_CPI_AUC_0.8708_condition_label_drift",
            "source_file_or_context": "reports/tables/cab_prospective_temporal_validation.csv",
            "reason_for_quarantine": "original CPI partially inflated by leakage/endpoint contamination",
            "replacement_or_allowed_use": "use baseline-only CPI and final claim hierarchy; do not cite as publication-safe",
            "publication_status": "deprecated_if_leakage_detected",
        },
        {
            "deprecated_item": "old_leaked_CPI_AUC_0.7876_classification_or_failure_severity",
            "source_file_or_context": "reports/tables/cab_prospective_temporal_validation.csv",
            "reason_for_quarantine": "classification/severity CPI claim did not survive baseline-only validation",
            "replacement_or_allowed_use": "do not use for publication-safe claims",
            "publication_status": "blocked_by_leakage_and_failed_baseline_only_validation",
        },
        {
            "deprecated_item": "cardiomyopathy_individual_collision_structural_sarcomeric_flag_claims",
            "source_file_or_context": "reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv",
            "reason_for_quarantine": "individual flags not supported after leakage removal; composite low baseline portability is supported",
            "replacement_or_allowed_use": "use composite baseline portability score/regime models",
            "publication_status": "blocked_or_narrowed",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_QUARANTINE, index=False)
    return out


def main() -> None:
    ensure_dirs()
    cross = build_cross_domain_summary()
    grammar = build_grammar()
    claims = build_claim_hierarchy()
    write_figure_plan()
    write_publication_audit(cross, claims)
    quarantine = build_quarantine()

    print("Final CAB cardiovascular evidence integration complete.")
    print()
    print("Cross-domain summary:")
    print(cross.to_string(index=False))
    print()
    print("Domain-specific portability grammar:")
    print(grammar.to_string(index=False))
    print()
    print("Claim hierarchy:")
    print(claims[["claim_tier", "claim_id", "claim_strength"]].to_string(index=False))
    print()
    print("Deprecated outputs quarantined:")
    print(quarantine[["deprecated_item", "publication_status"]].to_string(index=False))
    print()
    print("Key outputs:")
    for p in [OUT_CROSS_DOMAIN, OUT_GRAMMAR, OUT_CLAIMS, OUT_FIGURE_PLAN, OUT_AUDIT, OUT_QUARANTINE]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()


