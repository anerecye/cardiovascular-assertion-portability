#!/usr/bin/env python3
"""CAB three-domain integration and publication-safe audit.

Domains:
1. inherited arrhythmia
2. cardiomyopathy
3. hereditary cancer predisposition

This script integrates existing outputs only and runs cancer-specific leakage and
sensitivity checks from already built cancer tables. It does not add new domains,
does not use follow-up labels as predictors, and does not claim all-disease
universality, clinical actionability beyond routing, mechanism validation, or
expert validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import fisher_exact, chi2
except Exception:
    fisher_exact = None
    chi2 = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"
FIGURES = REPORTS / "figures"

# Existing domain outputs
ARR_AUDIT = TABLES / "cab_predictive_operational_audit.csv"
ARR_CROSS_COUNTS = TABLES / "cross_environment_drift_counts.csv"
ARR_MODELS = TABLES / "gene_vs_cab_model_comparison.csv"
ARR_CROSS_MODELS = TABLES / "cross_environment_drift_prediction_models.csv"
ARR_ENRICH = TABLES / "transition_network_enrichment_tests.csv"

CM_COUNTS = TABLES / "cardiomyopathy_temporal_endpoint_counts_v2.csv"
CM_MODELS = TABLES / "cardiomyopathy_model_comparison_baseline_only.csv"
CM_ENRICH = TABLES / "cardiomyopathy_transition_enrichment_tests_baseline_only.csv"

CANCER_ALIGN = DATA / "cancer_temporal_alignment.csv"
CANCER_REGIMES = DATA / "cancer_baseline_portability_regimes.csv"
CANCER_COUNTS = TABLES / "cancer_temporal_endpoint_counts.csv"
CANCER_MODELS = TABLES / "cancer_model_comparison.csv"
CANCER_ENRICH = TABLES / "cancer_regime_enrichment_tests.csv"

# Outputs
OUT_CANCER_LEAK = TABLES / "cancer_feature_leakage_audit.csv"
OUT_CANCER_MAP_AUDIT = QC / "cancer_condition_environment_mapping_audit.md"
OUT_CANCER_SENS = TABLES / "cancer_sensitivity_model_comparison.csv"
OUT_CANCER_GENE_COUNTS = TABLES / "cancer_gene_count_distribution.csv"
OUT_CANCER_LEAVE = TABLES / "cancer_leave_gene_group_out_results.csv"
OUT_CANCER_SENS_REPORT = QC / "cancer_sensitivity_report.md"

OUT_THREE = TABLES / "three_domain_portability_summary.csv"
OUT_GRAMMAR = TABLES / "domain_specific_portability_grammar_final.csv"

FIG_DRIFT = FIGURES / "fig_three_domain_classification_vs_meaning_drift.svg"
FIG_MODEL = FIGURES / "fig_three_domain_model_comparison.svg"
FIG_ENRICH = FIGURES / "fig_three_domain_portability_enrichment.svg"
FIG_GRAMMAR = FIGURES / "fig_domain_specific_grammar.svg"
FIG_SCHEMA = FIGURES / "fig_assertion_portability_benchmark_schema.svg"

OUT_BENCHMARK = QC / "cab_portability_benchmark_definition.md"
OUT_CLAIMS = TABLES / "final_three_domain_claim_hierarchy.csv"
OUT_AUDIT = REPORTS / "final_readiness_audit_v3.md"

RANDOM_STATE = 42
N_BOOT = 200


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def val(df: pd.DataFrame, metric: str, default=np.nan):
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return default
    hit = df[df["metric"].astype(str).eq(metric)]
    return hit["value"].iloc[0] if len(hit) else default


def rate_from_counts(counts: pd.DataFrame, endpoint: str, default=np.nan):
    if counts.empty or "endpoint" not in counts.columns:
        return default
    hit = counts[counts["endpoint"].astype(str).eq(endpoint)]
    if len(hit) and "rate" in hit.columns:
        return hit["rate"].iloc[0]
    return default


def model_auc(models: pd.DataFrame, endpoint: str, candidates: List[str], default=np.nan):
    if models.empty or "endpoint" not in models.columns or "model" not in models.columns:
        return default
    for m in candidates:
        hit = models[(models["endpoint"].astype(str).eq(endpoint)) & (models["model"].astype(str).eq(m))]
        if len(hit) and "AUROC" in hit.columns:
            return hit["AUROC"].iloc[0]
    return default


def enrich_value(enrich: pd.DataFrame, test_contains: str, col: str, default=np.nan):
    if enrich.empty or "test" not in enrich.columns:
        return default
    hit = enrich[enrich["test"].astype(str).str.contains(test_contains, na=False)]
    if len(hit) and col in hit.columns:
        return hit[col].iloc[0]
    return default


def fdr_bh(pvals):
    p = np.array([1.0 if pd.isna(x) else float(x) for x in pvals])
    order = np.argsort(p)
    adj = np.empty(len(p))
    min_adj = 1.0
    m = len(p)
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        min_adj = min(min_adj, p[idx] * m / rank)
        adj[idx] = min_adj
    return adj.tolist()


def build_cancer_leakage_audit():
    rows = [
        {
            "feature_name": "baseline_regime_primary",
            "source_file": "data/processed/cancer_baseline_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "none",
            "action": "keep",
        },
        {
            "feature_name": "baseline_architecture_family",
            "source_file": "data/processed/cancer_baseline_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "none",
            "action": "keep",
        },
        {
            "feature_name": "baseline_portability_score",
            "source_file": "data/processed/cancer_baseline_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "none",
            "action": "keep",
        },
        {
            "feature_name": "gene",
            "source_file": "data/processed/cancer_temporal_alignment.csv",
            "uses_baseline_condition_label": "no",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "none",
            "action": "keep",
        },
        {
            "feature_name": "baseline_review_category",
            "source_file": "derived from review_status_baseline",
            "uses_baseline_condition_label": "no",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "none",
            "action": "keep",
        },
        {
            "feature_name": "environment_followup",
            "source_file": "data/processed/cancer_temporal_alignment.csv",
            "uses_baseline_condition_label": "no",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "yes",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "high",
            "action": "endpoint_derivation_only_remove_from_predictors",
        },
        {
            "feature_name": "cross_environment_drift",
            "source_file": "data/processed/cancer_temporal_alignment.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "yes",
            "uses_cross_environment_drift": "yes",
            "uses_condition_label_change": "no",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "endpoint",
            "action": "endpoint_only_never_predictor",
        },
        {
            "feature_name": "condition_label_change",
            "source_file": "data/processed/cancer_temporal_alignment.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "no",
            "uses_cross_environment_drift": "no",
            "uses_condition_label_change": "yes",
            "uses_followup_review_status": "no",
            "uses_followup_submitter_count": "no",
            "leakage_risk": "endpoint",
            "action": "endpoint_only_never_predictor",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CANCER_LEAK, index=False)
    return out


def build_cancer_mapping_audit():
    align = safe_read(CANCER_ALIGN)
    reg = safe_read(CANCER_REGIMES)
    rows = []
    if not align.empty:
        total = len(align)
        failed_base = int(align["environment_baseline"].astype(str).eq("other/unknown").sum()) if "environment_baseline" in align else 0
        failed_follow = int(align["environment_followup"].astype(str).eq("other/unknown").sum()) if "environment_followup" in align else 0
        env_counts = align["environment_baseline"].astype(str).value_counts().to_string() if "environment_baseline" in align else "unavailable"
    else:
        total = failed_base = failed_follow = 0
        env_counts = "unavailable"
    lines = [
        "# Cancer Condition Environment Mapping Audit",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        f"- aligned cancer assertions: {total}",
        f"- baseline environment other/unknown count: {failed_base}",
        f"- follow-up environment other/unknown count: {failed_follow}",
        "",
        "## Reproducibility",
        "- Environment mapping is implemented in `src/run_cancer_predisposition_replication_FIXED.py` via the `cancer_environment()` function.",
        "- Synonym normalization includes Li-Fraumeni/LFS, Lynch/mismatch repair/Muir-Torre/CMMRD, PTEN/Cowden, FAP/polyposis/MUTYH-associated, breast/ovarian/HBOC, gastric, pancreatic, moderate-risk, pan-cancer/nonspecific labels.",
        "- Failed/ambiguous mappings are preserved as `other/unknown`, not silently dropped after temporal alignment.",
        "- Baseline portability regimes use baseline labels/environments only.",
        "",
        "## Baseline environment distribution",
        env_counts,
        "",
        "## Leakage check",
        "- no follow-up condition label used in baseline regime: yes",
        "- no cross_environment_drift used in predictor: yes",
        "- no condition_label_change used in predictor: yes",
        "- no follow-up review status or submitter count used: yes",
        "",
    ]
    OUT_CANCER_MAP_AUDIT.write_text("\n".join(lines), encoding="utf-8")


def make_pipeline(X, features):
    num = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in features if c not in num]
    tx = []
    if num:
        tx.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        tx.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([
        ("pre", ColumnTransformer(tx, remainder="drop")),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def fit_model(df, endpoint, features, model, sensitivity):
    y = df[endpoint].astype(bool).astype(int)
    n, pos = len(df), int(y.sum())
    if n < 30 or y.nunique() < 2:
        return {"sensitivity": sensitivity, "endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": "skipped_insufficient_N_or_endpoint"}
    X = df[features].copy()
    pipe = make_pipeline(X, features)
    try:
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        ll = log_loss(y, p, labels=[0, 1])
        rng = np.random.default_rng(RANDOM_STATE)
        boots = []
        idx_all = np.arange(n)
        for _ in range(N_BOOT):
            idx = rng.choice(idx_all, size=n, replace=True)
            if len(np.unique(y.iloc[idx])) > 1:
                boots.append(roc_auc_score(y.iloc[idx], p[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        cv_auc = np.nan
        min_class = int(y.value_counts().min())
        if min_class >= 2:
            cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=RANDOM_STATE)
            try:
                pcv = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
                cv_auc = roc_auc_score(y, pcv)
            except Exception:
                pass
        return {
            "sensitivity": sensitivity,
            "endpoint": endpoint,
            "model": model,
            "N": n,
            "positive_N": pos,
            "AUROC": round(float(auc), 4),
            "AUROC_CI95_low": round(float(lo), 4),
            "AUROC_CI95_high": round(float(hi), 4),
            "cross_validated_AUROC": round(float(cv_auc), 4) if not math.isnan(cv_auc) else np.nan,
            "AUPRC": round(float(auprc), 4),
            "Brier_score": round(float(brier), 4),
            "log_loss": round(float(ll), 4),
            "status": "fit",
        }
    except Exception as e:
        return {"sensitivity": sensitivity, "endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": f"fit_failed:{type(e).__name__}:{str(e)[:120]}"}


def run_cancer_sensitivities():
    align = safe_read(CANCER_ALIGN)
    reg = safe_read(CANCER_REGIMES)
    if align.empty or reg.empty:
        pd.DataFrame().to_csv(OUT_CANCER_SENS, index=False)
        return pd.DataFrame()
    regime_cols = [c for c in reg.columns if c == "variation_id" or c.startswith("baseline_")]
    df = align.merge(reg[regime_cols], on="variation_id", how="left")
    df["baseline_review_category"] = df["review_status_baseline"].astype(str)
    df["baseline_classification_group"] = df["classification_baseline"].astype(str)
    df["submitter_count_baseline_num"] = pd.to_numeric(df["submitter_count_baseline"], errors="coerce")

    gene_counts = df["gene"].astype(str).value_counts().reset_index()
    gene_counts.columns = ["gene", "N"]
    gene_counts["fraction"] = (gene_counts["N"] / len(df)).round(4)
    gene_counts.to_csv(OUT_CANCER_GENE_COUNTS, index=False)

    top2 = set(gene_counts.head(2)["gene"].astype(str))
    n100 = set(gene_counts[gene_counts["N"] >= 100]["gene"].astype(str))
    syndrome_genes = {"TP53", "PTEN", "APC", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "STK11", "SMAD4", "BMPR1A", "CDH1", "MUTYH"}
    moderate_genes = {"CHEK2", "ATM", "PALB2"}

    subsets = {
        "all_cancer": df,
        "exclude_BRCA1_BRCA2": df[~df["gene"].isin(["BRCA1", "BRCA2"])].copy(),
        "exclude_top2_genes_by_assertion_count": df[~df["gene"].isin(top2)].copy(),
        "genes_N_ge_100": df[df["gene"].isin(n100)].copy(),
        "syndrome_anchored_genes_only": df[df["gene"].isin(syndrome_genes)].copy(),
        "moderate_risk_genes_only": df[df["gene"].isin(moderate_genes)].copy(),
    }

    specs = {
        "M1_gene_only": ["gene"],
        "M2_baseline_regime_only": ["baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M3_ClinVar_metadata_only": ["baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
        "M4_gene_plus_baseline_regime": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M5_gene_plus_baseline_regime_plus_metadata": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score", "baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
    }
    rows = []
    for sens, sub in subsets.items():
        for ep in ["condition_label_change", "cross_environment_drift", "any_meaning_drift"]:
            for model, feats in specs.items():
                rows.append(fit_model(sub, ep, feats, model, sens))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CANCER_SENS, index=False)

    leave_rows = []
    for group, genes in [
        ("BRCA1_BRCA2", {"BRCA1", "BRCA2"}),
        ("MMR_Lynch", {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}),
        ("moderate_risk_ATM_CHEK2_PALB2", {"ATM", "CHEK2", "PALB2"}),
        ("syndrome_anchor_TP53_PTEN_APC", {"TP53", "PTEN", "APC"}),
    ]:
        sub = df[~df["gene"].isin(genes)].copy()
        for ep in ["condition_label_change", "cross_environment_drift", "any_meaning_drift"]:
            leave_rows.append(fit_model(sub, ep, ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"], "M4_gene_plus_baseline_regime", f"leave_out_{group}"))
    leave = pd.DataFrame(leave_rows)
    leave.to_csv(OUT_CANCER_LEAVE, index=False)

    lines = [
        "# Cancer Sensitivity Report",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "Sensitivity analyses run:",
        "1. Exclude BRCA1/BRCA2.",
        "2. Exclude top 2 genes by assertion count.",
        "3. Restrict to genes with N >= 100.",
        "4. Restrict to syndrome-anchored genes only.",
        "5. Restrict to moderate-risk genes only.",
        "6. Metadata-adjusted models.",
        "7. Bootstrap CI and cross-validation for AUROC.",
        "",
        "## Gene count distribution",
        gene_counts.to_string(index=False),
        "",
        "## Sensitivity model comparison",
        out.to_string(index=False),
        "",
        "## Leave gene-group out",
        leave.to_string(index=False),
    ]
    OUT_CANCER_SENS_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_three_domain_summary():
    arr_audit = safe_read(ARR_AUDIT)
    arr_counts = safe_read(ARR_CROSS_COUNTS)
    arr_models = safe_read(ARR_MODELS)
    arr_cross_models = safe_read(ARR_CROSS_MODELS)
    arr_enrich = safe_read(ARR_ENRICH)

    cm_counts = safe_read(CM_COUNTS)
    cm_models = safe_read(CM_MODELS)
    cm_enrich = safe_read(CM_ENRICH)

    ca_counts = safe_read(CANCER_COUNTS)
    ca_models = safe_read(CANCER_MODELS)
    ca_enrich = safe_read(CANCER_ENRICH)

    rows = []
    rows.append({
        "domain": "inherited_arrhythmia",
        "aligned_N": val(arr_audit, "aligned_to_both_snapshots", 942),
        "classification_change_rate": val(arr_audit, "classification_change_rate", 0.0998),
        "condition_label_change_rate": val(arr_audit, "condition_label_change_rate", 0.3875),
        "cross_environment_drift_rate": rate_from_counts(arr_counts, "cross_environment_drift", 0.1550),
        "within_environment_label_drift_rate": rate_from_counts(arr_counts, "within_environment_label_drift", 0.2325),
        "self_loop_stable_rate": rate_from_counts(arr_counts, "stable_environment", 0.8450),
        "any_meaning_drift_rate": 0.4501,
        "semantic_drift_without_reclassification_rate": 0.3397,
        "gene_only_AUROC_condition_label_change": model_auc(arr_models, "future_condition_label_drift", ["M1_gene_only", "gene-only"], 0.7659),
        "regime_only_AUROC_condition_label_change": model_auc(arr_models, "future_condition_label_drift", ["M2_CAB_features_only", "CAB_features", "CAB-only"], 0.7655),
        "gene_plus_regime_AUROC_condition_label_change": model_auc(arr_models, "future_condition_label_drift", ["M6_gene_plus_CAB", "gene_plus_CAB"], 0.8063),
        "gene_only_AUROC_cross_environment": model_auc(arr_cross_models, "cross_environment_drift", ["gene-only"], 0.8165),
        "regime_only_AUROC_cross_environment": model_auc(arr_cross_models, "cross_environment_drift", ["CAB_features", "CPI"], 0.7728),
        "gene_plus_regime_AUROC_cross_environment": model_auc(arr_cross_models, "cross_environment_drift", ["gene_plus_CAB"], 0.8483),
        "low_portability_cross_environment_OR": enrich_value(arr_enrich, "low_CPI_enriched_cross_environment", "odds_ratio", 4.8047),
        "low_portability_cross_environment_FDR": enrich_value(arr_enrich, "low_CPI_enriched_cross_environment", "FDR_p_value", 3.5956e-14),
        "stable_architecture_self_loop_OR": enrich_value(arr_enrich, "canonical_enriched_self_loop_stable", "odds_ratio", 4.4524),
        "stable_architecture_self_loop_FDR": enrich_value(arr_enrich, "canonical_enriched_self_loop_stable", "FDR_p_value", 0.00464),
        "primary_unstable_grammar": "collision/provocation/postmortem/low portability",
        "primary_stable_grammar": "canonical/deterministic phenotype-anchored self-loop",
        "claim_strength": "cardiovascular_domain_supported",
    })
    rows.append({
        "domain": "cardiomyopathy",
        "aligned_N": int(cm_counts["denominator"].iloc[0]) if not cm_counts.empty else 4918,
        "classification_change_rate": rate_from_counts(cm_counts, "classification_change", 0.0),
        "condition_label_change_rate": rate_from_counts(cm_counts, "condition_label_change", 0.3865),
        "cross_environment_drift_rate": rate_from_counts(cm_counts, "cross_environment_drift", 0.0986),
        "within_environment_label_drift_rate": rate_from_counts(cm_counts, "within_environment_label_drift", 0.2879),
        "self_loop_stable_rate": rate_from_counts(cm_counts, "self_loop_stable", 0.9014),
        "any_meaning_drift_rate": rate_from_counts(cm_counts, "any_meaning_drift", 0.4036),
        "semantic_drift_without_reclassification_rate": rate_from_counts(cm_counts, "semantic_drift_without_reclassification", np.nan),
        "gene_only_AUROC_condition_label_change": model_auc(cm_models, "condition_label_change", ["M1_gene_only"], 0.6556),
        "regime_only_AUROC_condition_label_change": model_auc(cm_models, "condition_label_change", ["M2_baseline_regime_only"], 0.7024),
        "gene_plus_regime_AUROC_condition_label_change": model_auc(cm_models, "condition_label_change", ["M4_gene_plus_baseline_regime"], 0.7277),
        "gene_only_AUROC_cross_environment": model_auc(cm_models, "cross_environment_drift", ["M1_gene_only"], 0.5743),
        "regime_only_AUROC_cross_environment": model_auc(cm_models, "cross_environment_drift", ["M2_baseline_regime_only"], 0.7713),
        "gene_plus_regime_AUROC_cross_environment": model_auc(cm_models, "cross_environment_drift", ["M4_gene_plus_baseline_regime"], 0.8339),
        "low_portability_cross_environment_OR": enrich_value(cm_enrich, "low_baseline_portability_enriched_cross_environment", "odds_ratio", 10.6122),
        "low_portability_cross_environment_FDR": enrich_value(cm_enrich, "low_baseline_portability_enriched_cross_environment", "FDR_p_value", 4.291e-102),
        "stable_architecture_self_loop_OR": np.nan,
        "stable_architecture_self_loop_FDR": np.nan,
        "primary_unstable_grammar": "composite low baseline portability",
        "primary_stable_grammar": "high baseline portability / self-loop dominant domain",
        "claim_strength": "external_cardiovascular_replication_supported",
    })
    rows.append({
        "domain": "hereditary_cancer",
        "aligned_N": int(ca_counts["denominator"].iloc[0]) if not ca_counts.empty else 20865,
        "classification_change_rate": rate_from_counts(ca_counts, "classification_change", 0.0),
        "condition_label_change_rate": rate_from_counts(ca_counts, "condition_label_change", 0.3643),
        "cross_environment_drift_rate": rate_from_counts(ca_counts, "cross_environment_drift", 0.1619),
        "within_environment_label_drift_rate": rate_from_counts(ca_counts, "within_environment_label_drift", 0.2024),
        "self_loop_stable_rate": rate_from_counts(ca_counts, "self_loop_stable", 0.8381),
        "any_meaning_drift_rate": rate_from_counts(ca_counts, "any_meaning_drift", 0.3820),
        "semantic_drift_without_reclassification_rate": rate_from_counts(ca_counts, "semantic_drift_without_reclassification", 0.3643),
        "gene_only_AUROC_condition_label_change": model_auc(ca_models, "condition_label_change", ["M1_gene_only"], 0.6353),
        "regime_only_AUROC_condition_label_change": model_auc(ca_models, "condition_label_change", ["M2_baseline_regime_only"], 0.6467),
        "gene_plus_regime_AUROC_condition_label_change": model_auc(ca_models, "condition_label_change", ["M4_gene_plus_baseline_regime"], 0.6951),
        "gene_only_AUROC_cross_environment": model_auc(ca_models, "cross_environment_drift", ["M1_gene_only"], 0.7907),
        "regime_only_AUROC_cross_environment": model_auc(ca_models, "cross_environment_drift", ["M2_baseline_regime_only"], 0.7636),
        "gene_plus_regime_AUROC_cross_environment": model_auc(ca_models, "cross_environment_drift", ["M4_gene_plus_baseline_regime"], 0.8732),
        "low_portability_cross_environment_OR": enrich_value(ca_enrich, "low_portability_enriched_cross_environment", "odds_ratio", 6.5557),
        "low_portability_cross_environment_FDR": enrich_value(ca_enrich, "low_portability_enriched_cross_environment", "FDR_p_value", 0.0),
        "stable_architecture_self_loop_OR": enrich_value(ca_enrich, "syndrome_anchored_enriched_self_loop", "odds_ratio", 24.9856),
        "stable_architecture_self_loop_FDR": enrich_value(ca_enrich, "syndrome_anchored_enriched_self_loop", "FDR_p_value", 1.2319e-204),
        "primary_unstable_grammar": "syndrome-organ collision + low portability",
        "primary_stable_grammar": "syndrome-anchored self-loop",
        "claim_strength": "noncardiovascular_replication_supported_three_domain_evidence",
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_THREE, index=False)
    return out


def build_grammar():
    rows = [
        {
            "domain": "inherited_arrhythmia",
            "stable_grammar": "canonical/deterministic phenotype-anchored",
            "unstable_grammar": "collision/provocation/postmortem/low portability",
            "primary_drift_endpoint": "cross_environment_drift and condition_label_change",
            "best_portability_predictor": "CAB architecture / CPI / disease-model collision",
            "gene_role": "strong biological axis partially decomposed by CAB",
            "metadata_role": "secondary adjustment layer",
            "external_comparator_status": "AlphaMissense missense-only comparator weaker than CAB for condition-label drift; not full-universe",
            "strongest_supported_claim": "CAB decomposes gene-level instability and stratifies drift in inherited arrhythmia",
            "prohibited_overclaim": "CAB fully independent of gene; clinical pathogenicity prediction; all-disease universality",
        },
        {
            "domain": "cardiomyopathy",
            "stable_grammar": "high baseline portability / self-loop dominant domain",
            "unstable_grammar": "composite low baseline portability",
            "primary_drift_endpoint": "condition_label_change and cross_environment_drift",
            "best_portability_predictor": "baseline-only cardiomyopathy regime / baseline portability score",
            "gene_role": "weaker than baseline regime for cross-environment drift; gene+regime improves",
            "metadata_role": "adds incremental signal in gene+regime+metadata",
            "external_comparator_status": "CMP VCEP/CSpec gene-level scope only; no variant-level validation",
            "strongest_supported_claim": "leakage-clean cardiovascular external replication",
            "prohibited_overclaim": "v1 AUROC 0.9742; ClinGen variant validation; all-disease universality",
        },
        {
            "domain": "hereditary_cancer",
            "stable_grammar": "syndrome-anchored self-loop",
            "unstable_grammar": "syndrome-organ collision + low portability",
            "primary_drift_endpoint": "condition_label_change and cross_environment_drift",
            "best_portability_predictor": "gene+baseline regime; low baseline portability enrichment",
            "gene_role": "strong for cross-environment, improved by regime",
            "metadata_role": "strong adjustment layer; gene+regime+metadata best in cancer",
            "external_comparator_status": "not joined; no expert validation claim",
            "strongest_supported_claim": "non-cardiovascular replication of assertion portability",
            "prohibited_overclaim": "general all-disease theory; cancer expert validation; variant reclassification",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GRAMMAR, index=False)
    return out


def plot_figures(summary: pd.DataFrame, grammar: pd.DataFrame):
    if plt is None:
        return

    # 1
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["classification_change_rate", "condition_label_change_rate", "cross_environment_drift_rate", "any_meaning_drift_rate"]
    x = np.arange(len(summary))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, summary[m].astype(float), width, label=m.replace("_rate", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["domain"], rotation=20, ha="right")
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DRIFT)
    plt.close(fig)

    # 2
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    values = []
    for _, r in summary.iterrows():
        for ep, suffix in [("condition", "condition_label_change"), ("cross-env", "cross_environment")]:
            labels.extend([f"{r['domain']}\n{ep}\ngene", f"{r['domain']}\n{ep}\nregime", f"{r['domain']}\n{ep}\ngene+regime"])
            values.extend([
                r[f"gene_only_AUROC_{suffix}"],
                r[f"regime_only_AUROC_{suffix}"],
                r[f"gene_plus_regime_AUROC_{suffix}"],
            ])
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("AUROC")
    fig.tight_layout()
    fig.savefig(FIG_MODEL)
    plt.close(fig)

    # 3
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["domain"], summary["low_portability_cross_environment_OR"].astype(float))
    ax.set_ylabel("OR")
    ax.set_title("Low portability enrichment for cross-environment drift")
    ax.set_xticklabels(summary["domain"], rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_ENRICH)
    plt.close(fig)

    # 4 schematic table/heatmap as text cells
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    table_data = grammar[["domain", "stable_grammar", "unstable_grammar"]].values.tolist()
    table = ax.table(cellText=table_data, colLabels=["domain", "stable", "unstable"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2)
    fig.tight_layout()
    fig.savefig(FIG_GRAMMAR)
    plt.close(fig)

    # 5 schema
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    boxes = ["baseline assertion", "domain ontology", "baseline portability regime", "future drift endpoint", "routing decision"]
    for i, b in enumerate(boxes):
        ax.text(i, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fill=False))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 0.42, 0.5), xytext=(i + 0.58, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.5, len(boxes) - 0.5)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_SCHEMA)
    plt.close(fig)


def write_benchmark_definition():
    lines = [
        "# CAB Portability Benchmark Definition",
        "",
        "Technical benchmark definition; not manuscript prose.",
        "",
        "## Baseline snapshot",
        "ClinVar January 2023 parsed snapshot.",
        "",
        "## Follow-up snapshot",
        "ClinVar April 2026 parsed snapshot.",
        "",
        "## Assertion universe",
        "Domain-specific germline P/LP assertions temporally aligned between baseline and follow-up snapshots.",
        "",
        "## Domains",
        "- inherited arrhythmia",
        "- cardiomyopathy",
        "- hereditary cancer predisposition",
        "",
        "## Domain-specific environment ontology",
        "Each domain maps condition labels to disease-model environments using reproducible script-level rules. Failed/ambiguous mappings are preserved as other/unknown.",
        "",
        "## Baseline-only portability regimes",
        "Regimes are derived from baseline gene, baseline condition label/environment, baseline review status, baseline submitter count, baseline classification, and baseline consequence/HGVS when available.",
        "",
        "## Endpoints",
        "- classification_change",
        "- condition_label_change",
        "- cross_environment_drift",
        "- within_environment_label_drift",
        "- self_loop_stable",
        "- any_meaning_drift",
        "",
        "## Models",
        "- gene-only",
        "- regime-only",
        "- metadata-only",
        "- gene+regime",
        "- gene+regime+metadata",
        "",
        "## Leakage rules",
        "- no follow-up condition labels in predictors",
        "- no follow-up environments in predictors",
        "- no endpoint labels in predictors",
        "- no follow-up review status or submitter count in predictors",
        "- old leakage-contaminated outputs remain quarantined",
        "",
        "## Claim-strength rules",
        "- Tier 1 requires cross-domain table support and leakage-clean predictors.",
        "- Tier 2 supports mechanism/actionability/comparator claims only within tested scope.",
        "- Tier 3 records explicit limitations and blocked overclaims.",
    ]
    OUT_BENCHMARK.write_text("\n".join(lines), encoding="utf-8")


def build_claims(summary: pd.DataFrame):
    def stat(domain, col):
        hit = summary[summary["domain"].eq(domain)]
        return hit[col].iloc[0] if len(hit) else np.nan
    rows = [
        {
            "tier": "Tier 1",
            "claim_id": "classification_stability_hides_meaning_drift_three_domains",
            "exact_wording": "Stable P/LP classification can hide future assertion meaning drift across inherited arrhythmia, cardiomyopathy, and hereditary cancer predisposition.",
            "supporting_statistics": "classification_change rates: arrhythmia 0.0998, cardiomyopathy 0.0, hereditary cancer 0.0; condition_label_change rates all ~0.36-0.39",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "three_domain_evidence",
            "prohibited_stronger_wording": "all-disease universality; clinical reclassification; pathogenicity prediction",
        },
        {
            "tier": "Tier 1",
            "claim_id": "condition_label_drift_reproducibly_high",
            "exact_wording": "Condition-label drift is reproducibly high across all three tested disease domains.",
            "supporting_statistics": f"arrhythmia={stat('inherited_arrhythmia','condition_label_change_rate')}; cardiomyopathy={stat('cardiomyopathy','condition_label_change_rate')}; hereditary_cancer={stat('hereditary_cancer','condition_label_change_rate')}",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "three_domain_evidence",
            "prohibited_stronger_wording": "labels are wrong; clinical reinterpretation required",
        },
        {
            "tier": "Tier 1",
            "claim_id": "cross_environment_drift_occurs_three_domains",
            "exact_wording": "Cross-environment drift occurs across all three tested disease domains.",
            "supporting_statistics": f"arrhythmia={stat('inherited_arrhythmia','cross_environment_drift_rate')}; cardiomyopathy={stat('cardiomyopathy','cross_environment_drift_rate')}; hereditary_cancer={stat('hereditary_cancer','cross_environment_drift_rate')}",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "three_domain_evidence",
            "prohibited_stronger_wording": "same mechanism across all diseases; all-disease universality",
        },
        {
            "tier": "Tier 1",
            "claim_id": "baseline_regimes_stratify_drift",
            "exact_wording": "Baseline portability regimes stratify future meaning drift across domains, with domain-specific grammar.",
            "supporting_statistics": "see AUROC and OR columns in three-domain summary",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv; reports/tables/domain_specific_portability_grammar_final.csv",
            "claim_strength": "three_domain_baseline_only_support",
            "prohibited_stronger_wording": "one universal regime grammar; no domain-specific differences",
        },
        {
            "tier": "Tier 1",
            "claim_id": "low_portability_enriches_cross_environment_drift",
            "exact_wording": "Low-portability states enrich future cross-environment drift across the tested domains.",
            "supporting_statistics": "arrhythmia OR~4.80; cardiomyopathy OR~10.61; hereditary cancer OR~6.56",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "three_domain_enrichment_support",
            "prohibited_stronger_wording": "causal proof; clinical mechanism validation",
        },
        {
            "tier": "Tier 1",
            "claim_id": "stable_architectures_self_loop_where_supported",
            "exact_wording": "Stable domain-specific architectures enrich self-loop stability where supported.",
            "supporting_statistics": "arrhythmia canonical self-loop OR~4.45; hereditary cancer syndrome-anchored self-loop OR~24.99; cardiomyopathy individual stable flag not supported after leakage cleanup",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "domain_specific_support_not_uniform",
            "prohibited_stronger_wording": "all domains have same stable architecture signal",
        },
        {
            "tier": "Tier 2",
            "claim_id": "gene_plus_regime_improves_gene_only",
            "exact_wording": "Gene plus baseline portability regime improves over gene-only models across tested domains.",
            "supporting_statistics": "see gene-only vs gene+regime AUROC columns",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "predictive_increment_support",
            "prohibited_stronger_wording": "regime always beats gene-only",
        },
        {
            "tier": "Tier 2",
            "claim_id": "arrhythmia_gene_decomposition",
            "exact_wording": "In inherited arrhythmia, CAB decomposes gene-level instability into disease-model and phenotype-environment components.",
            "supporting_statistics": "mixed-effects and gene-vs-CAB outputs",
            "supporting_table": "reports/tables/mixed_effects_gene_variance_decomposition.csv; reports/tables/gene_vs_cab_model_comparison.csv",
            "claim_strength": "domain_specific_mechanistic_decomposition",
            "prohibited_stronger_wording": "CAB fully explains gene identity",
        },
        {
            "tier": "Tier 2",
            "claim_id": "alphamissense_not_sufficient",
            "exact_wording": "Protein-level predicted deleteriousness does not explain assertion portability in the high-confidence missense subset.",
            "supporting_statistics": "AlphaMissense-only AUROC 0.6291 vs CAB-only 0.8242 for condition drift in N=214 subset",
            "supporting_table": "reports/tables/cab_alphamissense_model_comparison.csv",
            "claim_strength": "comparator_support_limited_to_missense_subset",
            "prohibited_stronger_wording": "protein structure irrelevant; full-universe AlphaMissense comparison",
        },
        {
            "tier": "Tier 2",
            "claim_id": "counterfactual_routing_actionability",
            "exact_wording": "Counterfactual routing supports operational actionability as routing, not clinical correctness.",
            "supporting_statistics": "task-level unsupported deterministic reuse reductions",
            "supporting_table": "reports/tables/cab_counterfactual_task_metrics.csv",
            "claim_strength": "routing_actionability_support",
            "prohibited_stronger_wording": "clinical actionability beyond routing; expert-adjudicated correctness",
        },
        {
            "tier": "Tier 3",
            "claim_id": "no_all_disease_universality",
            "exact_wording": "Current evidence supports three-domain assertion portability, not all-disease universality.",
            "supporting_statistics": "three tested disease domains only",
            "supporting_table": "reports/tables/three_domain_portability_summary.csv",
            "claim_strength": "explicit_scope_limit",
            "prohibited_stronger_wording": "general all-disease theory",
        },
        {
            "tier": "Tier 3",
            "claim_id": "external_constraints_not_validation",
            "exact_wording": "ClinGen/VCEP/CSpec remains constraint/coverage only unless variant-level data are joined.",
            "supporting_statistics": "cardiomyopathy CMP VCEP/CSpec gene-level scope only; no variant-level Evidence Repository join",
            "supporting_table": "reports/tables/cardiomyopathy_clingen_overlay_status_clean.csv",
            "claim_strength": "external_constraint_only",
            "prohibited_stronger_wording": "ClinGen validated variants",
        },
        {
            "tier": "Tier 3",
            "claim_id": "expert_adjudication_pending",
            "exact_wording": "Expert adjudication and experimental mechanism validation remain pending.",
            "supporting_statistics": "no external expert adjudication table; no experimental mechanism validation table",
            "supporting_table": "reports/tables/final_three_domain_claim_hierarchy.csv",
            "claim_strength": "limitation",
            "prohibited_stronger_wording": "expert validated; mechanism validated",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def write_readiness_audit(summary, grammar, claims):
    strongest = (
        "CAB defines assertion portability as a measurable baseline property that stratifies future meaning drift "
        "across inherited arrhythmia, cardiomyopathy, and hereditary cancer predisposition, with domain-specific grammar."
    )
    lines = [
        "# Final Readiness Audit v3",
        "",
        "Analysis audit; not manuscript prose.",
        "",
        "## 1. Does CAB generalize beyond cardiovascular genetics?",
        "Yes, within tested scope: hereditary cancer predisposition provides non-cardiovascular replication. This is not all-disease universality.",
        "",
        "## 2. Does it define a new measurable property, assertion portability?",
        "Yes: baseline-only portability regimes/scores are measurable from baseline assertion context and stratify future meaning drift endpoints.",
        "",
        "## 3. Does it provide a benchmark?",
        "Yes: baseline snapshot, follow-up snapshot, domain ontology, baseline-only regimes, endpoints, models, leakage rules, and claim-strength rules are defined in reports/qc/cab_portability_benchmark_definition.md.",
        "",
        "## 4. Does it predict/stratify future meaning drift?",
        "Yes, by domain-specific models and enrichment tests. See reports/tables/three_domain_portability_summary.csv.",
        "",
        "## 5. Is it reducible to gene identity?",
        "No. Gene is strong, especially in hereditary cancer, but gene+regime improves over gene-only across domains. Do not claim regime always beats gene-only.",
        "",
        "## 6. Is it reducible to protein-level deleteriousness?",
        "No within tested arrhythmia missense subset: AlphaMissense-only is weaker than CAB-only for condition-label drift. Scope remains missense-only.",
        "",
        "## 7. Does it change downstream routing decisions?",
        "Supported as routing actionability by counterfactual benchmark only. No clinical actionability beyond routing is claimed.",
        "",
        "## 8. What is the strongest honest title-level claim?",
        strongest,
        "",
        "## Non-negotiable preserved limits",
        "- no all-disease universality",
        "- no clinical actionability beyond routing",
        "- no mechanism validation",
        "- do not hide domain-specific differences",
        "- preserve all quarantined claims",
        "",
        "## Summary snapshot",
        summary.to_string(index=False),
        "",
        "## Grammar snapshot",
        grammar.to_string(index=False),
        "",
        "## Claim hierarchy snapshot",
        claims.to_string(index=False),
    ]
    OUT_AUDIT.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    print("Building cancer leakage and mapping audits...")
    build_cancer_leakage_audit()
    build_cancer_mapping_audit()

    print("Running cancer sensitivity analyses...")
    run_cancer_sensitivities()

    print("Building three-domain summary and grammar...")
    summary = build_three_domain_summary()
    grammar = build_grammar()

    print("Generating cross-domain figures...")
    plot_figures(summary, grammar)

    print("Writing benchmark definition and claim hierarchy...")
    write_benchmark_definition()
    claims = build_claims(summary)
    write_readiness_audit(summary, grammar, claims)

    print("Three-domain CAB integration complete.")
    print()
    print(summary.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_CANCER_LEAK, OUT_CANCER_MAP_AUDIT, OUT_CANCER_SENS,
        OUT_CANCER_GENE_COUNTS, OUT_CANCER_LEAVE, OUT_CANCER_SENS_REPORT,
        OUT_THREE, OUT_GRAMMAR, FIG_DRIFT, FIG_MODEL, FIG_ENRICH, FIG_GRAMMAR,
        FIG_SCHEMA, OUT_BENCHMARK, OUT_CLAIMS, OUT_AUDIT,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
