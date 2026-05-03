#!/usr/bin/env python3
"""Cardiomyopathy replication v2: baseline-only regimes and leakage-clean models.

This script repairs the first cardiomyopathy replication pass by removing endpoint
leakage from cardiomyopathy regime assignment.

Hard rules:
- No follow-up condition labels/environments in predictors.
- No endpoint labels in predictors.
- Do not report prior cross-environment AUROC=0.9742 as publication-safe.
- Do not claim variant-level ClinGen validation unless joined from real evidence.

Inputs:
- data/processed/cardiomyopathy_temporal_alignment.csv if available,
  otherwise data/processed/cardiomyopathy_assertion_master.csv
- reports/tables/cardiomyopathy_clingen_coverage.csv if available
- reports/tables/arrhythmia_vs_cardiomyopathy_replication_summary.csv if available

Outputs follow the user's requested v2 paths.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, chi2
except Exception:
    fisher_exact = None
    chi2 = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
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

IN_ALIGN = DATA / "cardiomyopathy_temporal_alignment.csv"
IN_MASTER = DATA / "cardiomyopathy_assertion_master.csv"

OUT_LEAK_AUDIT = TABLES / "cardiomyopathy_regime_leakage_audit.csv"
OUT_LEAK_QC = QC / "cardiomyopathy_regime_leakage_audit.md"
OUT_BASE_REGIMES = DATA / "cardiomyopathy_baseline_only_regimes.csv"
OUT_BASE_RULES = QC / "cardiomyopathy_baseline_only_regime_rules.md"
OUT_ENDPOINTS = DATA / "cardiomyopathy_temporal_endpoints_v2.csv"
OUT_ENDPOINT_COUNTS = TABLES / "cardiomyopathy_temporal_endpoint_counts_v2.csv"
OUT_MODELS = TABLES / "cardiomyopathy_model_comparison_baseline_only.csv"
OUT_MODELS_CV = TABLES / "cardiomyopathy_model_comparison_baseline_only_cv.csv"
OUT_MODEL_QC = QC / "cardiomyopathy_baseline_only_model_report.md"
OUT_ENRICH = TABLES / "cardiomyopathy_transition_enrichment_tests_baseline_only.csv"
OUT_ENRICH_QC = QC / "cardiomyopathy_transition_enrichment_baseline_only_report.md"
OUT_SCORE = DATA / "cardiomyopathy_portability_score_baseline_only.csv"
OUT_SCORE_PERF = TABLES / "cardiomyopathy_portability_score_performance.csv"
OUT_SCORE_FIG = FIGURES / "cardiomyopathy_portability_score_drift_rates.svg"
OUT_CLINGEN_CLEAN = TABLES / "cardiomyopathy_clingen_overlay_status_clean.csv"
OUT_CLINGEN_LIMIT = QC / "cardiomyopathy_clingen_overlay_limitations.md"
OUT_COMPARE = TABLES / "arrhythmia_vs_cardiomyopathy_replication_v2.csv"
OUT_COMPARE_FIG = FIGURES / "arrhythmia_vs_cardiomyopathy_replication_v2.svg"
OUT_COMPARE_QC = QC / "arrhythmia_cardiomyopathy_replication_interpretation.md"
OUT_CLAIMS = TABLES / "cardiomyopathy_publication_safe_claims_v2.csv"
OUT_FINAL = REPORTS / "final_cardiomyopathy_replication_v2_report.md"

SARCOMERIC = {"MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3", "ACTN2", "VCL"}
DESMOSOMAL = {"PKP2", "DSP", "DSG2", "DSC2", "JUP", "TMEM43"}
STRUCTURAL_ELECTRICAL = {"PKP2", "DSP", "DSG2", "DSC2", "JUP", "TMEM43", "DES", "PLN"}
DILATED_CONDUCTION = {"LMNA", "FLNC", "RBM20", "TTN", "BAG3", "DES", "PLN"}
CMP_CSPEC_GENES_TARGET = {"ACTC1", "MYBPC3", "MYH7", "TNNI3", "TNNT2", "TPM1"}

RANDOM_STATE = 42
N_BOOT = 300


def ensure_dirs() -> None:
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coalesce_to_column(df: pd.DataFrame, target: str, candidates: List[str], default="") -> None:
    """Create target by coalescing the first non-empty values from candidate columns."""
    if target in df.columns:
        return
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df[target] = default
        return
    out = pd.Series(default, index=df.index, dtype="object")
    for c in cols:
        vals = df[c]
        mask = out.astype(str).str.len().eq(0) | out.isna()
        out = out.where(~mask, vals)
    df[target] = out.fillna(default)



def clinical_group(x) -> str:
    t = norm_text(x)
    if not t:
        return "missing"
    if "conflicting" in t:
        return "conflicting"
    if "uncertain significance" in t or t == "vus":
        return "vus"
    has_path = "pathogenic" in t
    has_benign = "benign" in t
    if has_path and not has_benign:
        return "p_lp"
    if has_benign and not has_path:
        return "b_lb"
    if has_path and has_benign:
        return "mixed_pathogenic_benign"
    return "other"


def review_category(x) -> str:
    t = norm_text(x)
    if "practice guideline" in t:
        return "practice_guideline"
    if "expert panel" in t:
        return "expert_panel"
    if "multiple submitters" in t and "no conflicts" in t:
        return "multiple_submitters_no_conflicts"
    if "single submitter" in t:
        return "single_submitter"
    if "conflicting" in t:
        return "conflicting"
    if "no assertion" in t or "no criteria" in t or "no classification" in t:
        return "weak_or_no_assertion"
    return "other_or_missing"


def normalize_condition(x) -> str:
    t = norm_text(x).replace("|", ";")
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return ";".join(sorted(set(parts)))


def split_condition_terms(x) -> List[str]:
    t = norm_text(x).replace("|", ";")
    return [p.strip() for p in t.split(";") if p.strip()]


def condition_environment(label: object) -> str:
    t = norm_text(label)
    if not t:
        return "other/unknown"
    if any(k in t for k in ["left ventricular noncompaction", "noncompaction", "lvnc"]):
        return "LVNC"
    if any(k in t for k in ["restrictive cardiomyopathy", "rcm"]):
        return "RCM"
    if any(k in t for k in ["arrhythmogenic right ventricular cardiomyopathy", "arrhythmogenic cardiomyopathy", "arvc", "acm"]):
        return "ARVC/ACM"
    if any(k in t for k in ["hypertrophic cardiomyopathy", "hcm"]):
        return "HCM"
    if any(k in t for k in ["dilated cardiomyopathy", "dcm", "primary dilated cardiomyopathy"]):
        return "DCM"
    if any(k in t for k in ["conduction", "atrioventricular block", "heart block", "sick sinus", "bradycardia"]):
        return "conduction-cardiomyopathy overlap"
    if any(k in t for k in ["sudden death", "sudden cardiac death", "sads", "arrhythmia", "ventricular tachycardia", "fibrillation"]):
        return "sudden death / arrhythmia-overlap"
    if any(k in t for k in ["noonan", "rasopathy", "metabolic", "mitochondrial", "syndrome", "syndromic", "storage"]):
        return "syndromic/metabolic cardiomyopathy"
    if "cardiomyopathy" in t:
        return "nonspecific cardiomyopathy"
    return "other/unknown"


def baseline_environment_set(label: str) -> set[str]:
    terms = split_condition_terms(label)
    envs = {condition_environment(t) for t in terms}
    # If aggregate label parser does not split well, include whole label env.
    envs.add(condition_environment(label))
    return {e for e in envs if e}


def broad_or_ambiguous_label(label: str) -> bool:
    t = norm_text(label)
    if not t:
        return True
    broad_terms = [
        "cardiomyopathy", "cardiomyopathies", "not provided", "not specified",
        "multiple conditions", "see cases", "other", "unknown", "heart disease"
    ]
    specific = condition_environment(label) not in {"other/unknown", "nonspecific cardiomyopathy"}
    return any(k in t for k in broad_terms) and not specific


def load_input() -> pd.DataFrame:
    path = IN_ALIGN if IN_ALIGN.exists() else IN_MASTER
    if not path.exists():
        raise FileNotFoundError(f"Missing {IN_ALIGN} or {IN_MASTER}; run cardiomyopathy v1 replication first.")
    df = pd.read_csv(path, low_memory=False)

    # Normalize expected columns from v1 script. v1 temporal_alignment may carry
    # duplicated suffixes from the baseline/followup merge, so coalesce robustly.
    coalesce_to_column(df, "gene", ["gene", "gene_baseline", "gene_baseline_baseline", "gene_followup", "gene_followup_followup"], "")
    coalesce_to_column(df, "condition_label_baseline", ["condition_label_baseline", "condition_label_baseline_baseline", "condition_label_baseline_x", "condition_label"], "")
    coalesce_to_column(df, "condition_label_followup", ["condition_label_followup", "condition_label_followup_followup", "condition_label_followup_y"], "")
    coalesce_to_column(df, "classification_baseline", ["classification_baseline", "classification_baseline_baseline", "classification_baseline_x"], "")
    coalesce_to_column(df, "classification_followup", ["classification_followup", "classification_followup_followup", "classification_followup_y"], "")
    coalesce_to_column(df, "review_status_baseline", ["review_status_baseline", "review_status_baseline_baseline", "review_status_baseline_x"], "")
    coalesce_to_column(df, "review_status_followup", ["review_status_followup", "review_status_followup_followup", "review_status_followup_y"], "")
    coalesce_to_column(df, "submitter_count_baseline", ["submitter_count_baseline", "submitter_count_baseline_baseline", "submitter_count_baseline_x"], np.nan)
    coalesce_to_column(df, "submitter_count_followup", ["submitter_count_followup", "submitter_count_followup_followup", "submitter_count_followup_y"], np.nan)
    coalesce_to_column(df, "HGVS", ["HGVS", "HGVS_baseline", "HGVS_baseline_baseline", "HGVS_x"], "")
    coalesce_to_column(df, "consequence", ["consequence", "consequence_baseline", "consequence_x"], "")
    coalesce_to_column(df, "assertion_id", ["assertion_id"], "CM_" + df["variation_id"].astype(str) if "variation_id" in df.columns else "")

    if "condition_label_baseline" not in df.columns:
        raise ValueError("Input lacks condition_label_baseline after schema normalization; cannot build v2.")

    for col in [
        "classification_change", "condition_label_change", "cross_environment_drift",
        "within_environment_label_drift", "self_loop_stable", "review_status_change",
        "submitter_count_change", "any_meaning_drift", "semantic_drift_without_reclassification"
    ]:
        if col in df.columns:
            df[col] = df[col].map(as_bool)

    # Recompute endpoints from raw baseline/follow-up columns if present to avoid trusting old values blindly.
    df["condition_label_baseline_norm"] = df["condition_label_baseline"].map(normalize_condition)
    df["condition_label_followup_norm"] = df["condition_label_followup"].map(normalize_condition)
    df["baseline_environment_v2"] = df["condition_label_baseline"].map(condition_environment)
    df["followup_environment_v2"] = df["condition_label_followup"].map(condition_environment)

    df["classification_change"] = df["classification_baseline"].map(clinical_group) != df["classification_followup"].map(clinical_group)
    df["condition_label_change"] = df["condition_label_baseline_norm"] != df["condition_label_followup_norm"]
    df["cross_environment_drift"] = df["baseline_environment_v2"] != df["followup_environment_v2"]
    df["within_environment_label_drift"] = df["condition_label_change"] & ~df["cross_environment_drift"]
    df["self_loop_stable"] = ~df["cross_environment_drift"]
    df["review_status_change"] = df["review_status_baseline"].map(review_category) != df["review_status_followup"].map(review_category)
    df["submitter_count_baseline_num"] = pd.to_numeric(df.get("submitter_count_baseline", np.nan), errors="coerce")
    df["submitter_count_followup_num"] = pd.to_numeric(df.get("submitter_count_followup", np.nan), errors="coerce")
    df["submitter_count_change"] = df["submitter_count_baseline_num"].fillna(-1) != df["submitter_count_followup_num"].fillna(-1)
    df["any_meaning_drift"] = df["condition_label_change"] | df["classification_change"] | df["review_status_change"]
    df["semantic_drift_without_reclassification"] = df["condition_label_change"] & ~df["classification_change"]

    df["gene"] = df["gene"].astype(str).str.upper()
    return df


def leakage_audit() -> pd.DataFrame:
    rows = [
        {
            "feature_name": "cardiomyopathy_portability_regime_v1",
            "source_file": "data/processed/cardiomyopathy_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "yes",
            "uses_transition_endpoint": "yes",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove",
        },
        {
            "feature_name": "cardiomyopathy_architecture_v1",
            "source_file": "data/processed/cardiomyopathy_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "yes",
            "uses_transition_endpoint": "yes",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove",
        },
        {
            "feature_name": "cardiomyopathy_portability_score_v1",
            "source_file": "data/processed/cardiomyopathy_portability_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "yes",
            "uses_transition_endpoint": "yes",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove",
        },
        {
            "feature_name": "condition_environment_baseline",
            "source_file": "data/processed/cardiomyopathy_temporal_alignment.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "no",
            "uses_transition_endpoint": "no",
            "uses_classification_followup": "no",
            "leakage_status": "none",
            "action": "keep",
        },
        {
            "feature_name": "condition_environment_followup",
            "source_file": "data/processed/cardiomyopathy_temporal_alignment.csv",
            "uses_baseline_condition_label": "no",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "yes",
            "uses_transition_endpoint": "no",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove_as_predictor_keep_endpoint_derivation_only",
        },
        {
            "feature_name": "cross_environment_drift",
            "source_file": "data/processed/cardiomyopathy_temporal_alignment.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "yes",
            "uses_transition_endpoint": "yes",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove_from_predictors_endpoint_only",
        },
        {
            "feature_name": "condition_label_change",
            "source_file": "data/processed/cardiomyopathy_temporal_alignment.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "yes",
            "uses_baseline_environment": "no",
            "uses_followup_environment": "no",
            "uses_transition_endpoint": "yes",
            "uses_classification_followup": "no",
            "leakage_status": "definite",
            "action": "remove_from_predictors_endpoint_only",
        },
        {
            "feature_name": "baseline_regime_primary_v2",
            "source_file": "data/processed/cardiomyopathy_baseline_only_regimes.csv",
            "uses_baseline_condition_label": "yes",
            "uses_followup_condition_label": "no",
            "uses_baseline_environment": "yes",
            "uses_followup_environment": "no",
            "uses_transition_endpoint": "no",
            "uses_classification_followup": "no",
            "leakage_status": "none",
            "action": "keep",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_LEAK_AUDIT, index=False)
    md = [
        "# Cardiomyopathy Regime Leakage Audit",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "The v1 cardiomyopathy regime assignment used baseline and follow-up environments, including env_baseline != env_followup logic. Therefore v1 cross-environment AUCs are not publication-safe.",
        "",
        out.to_string(index=False),
        "",
        "Hard action: v1 regime, architecture, and portability score are removed from predictors and recomputed baseline-only.",
    ]
    OUT_LEAK_QC.write_text("\n".join(md), encoding="utf-8")
    return out


def assign_baseline_regime(row: pd.Series) -> Dict[str, object]:
    gene = str(row.get("gene", "")).upper()
    label = row.get("condition_label_baseline", "")
    env = row.get("baseline_environment_v2", condition_environment(label))
    env_set = baseline_environment_set(label)
    env_set_clean = {e for e in env_set if e != "other/unknown"}

    baseline_collision = len(env_set_clean) > 1
    nonspecific = env in {"nonspecific cardiomyopathy", "other/unknown"}
    syndromic = env == "syndromic/metabolic cardiomyopathy"
    broad = broad_or_ambiguous_label(label)

    reason = []
    if baseline_collision:
        reason.append("baseline_labels_span_multiple_environments")
    if broad:
        reason.append("broad_or_ambiguous_baseline_label")
    if nonspecific:
        reason.append("nonspecific_or_unknown_baseline_environment")

    sarcomeric = gene in SARCOMERIC and env in {"HCM", "RCM", "LVNC", "DCM"}
    structural = gene in STRUCTURAL_ELECTRICAL and env in {"ARVC/ACM", "sudden death / arrhythmia-overlap", "DCM"}
    conduction = gene in DILATED_CONDUCTION and env in {"DCM", "conduction-cardiomyopathy overlap", "sudden death / arrhythmia-overlap"}

    if baseline_collision:
        regime = "cardiomyopathy_model_collision_baseline"
        arch = "baseline_model_collision"
    elif gene in SARCOMERIC and env == "HCM" and not broad:
        regime = "sarcomeric_phenotype_anchored_baseline"
        arch = "baseline_phenotype_anchored"
        reason.append("sarcomeric_gene_HCM_baseline")
    elif gene in STRUCTURAL_ELECTRICAL and env == "ARVC/ACM":
        regime = "arrhythmogenic_structural_electrical_baseline"
        arch = "baseline_structural_electrical_overlap"
        reason.append("structural_electrical_gene_ARVC_baseline")
    elif conduction:
        regime = "dilated_conduction_overlap_baseline"
        arch = "baseline_conduction_or_structural_overlap"
        reason.append("DCM_conduction_sudden_death_gene_context")
    elif syndromic:
        regime = "syndromic_metabolic_overlap_baseline"
        arch = "baseline_syndromic_metabolic_overlap"
        reason.append("syndromic_metabolic_baseline_label")
    elif nonspecific:
        regime = "nonspecific_cardiomyopathy_baseline"
        arch = "baseline_underresolved_contextual"
    elif sarcomeric:
        regime = "sarcomeric_phenotype_anchored_baseline"
        arch = "baseline_phenotype_anchored"
        reason.append("sarcomeric_gene_cardiomyopathy_baseline")
    elif structural:
        regime = "arrhythmogenic_structural_electrical_baseline"
        arch = "baseline_structural_electrical_overlap"
        reason.append("structural_electrical_gene_baseline")
    else:
        regime = "underresolved_contextual_baseline"
        arch = "baseline_underresolved_contextual"
        reason.append("baseline_context_not_specific_enough_for_other_regimes")

    underresolved = arch == "baseline_underresolved_contextual"
    score = 100.0
    if baseline_collision:
        score -= 35
    if underresolved:
        score -= 25
    if nonspecific:
        score -= 15
    if arch in {"baseline_structural_electrical_overlap", "baseline_conduction_or_structural_overlap"}:
        score -= 15
    if conduction:
        score -= 10
    if syndromic:
        score -= 10
    if review_category(row.get("review_status_baseline", "")) in {"single_submitter", "conflicting", "weak_or_no_assertion", "other_or_missing"}:
        score -= 10
    sub = safe_float(row.get("submitter_count_baseline_num"), np.nan)
    if not math.isnan(sub) and sub <= 1:
        score -= 8
    if broad:
        score -= 12
    if gene in {"TTN", "MYBPC3"}:
        # Coarse gene-family evaluability flag, not ancestry-aware AF.
        score -= 5
    if arch == "baseline_phenotype_anchored" and not broad:
        score += 5
    score = float(np.clip(score, 0, 100))

    return {
        "baseline_regime_primary": regime,
        "baseline_architecture_family": arch,
        "baseline_collision_flag": baseline_collision,
        "baseline_nonspecific_flag": nonspecific,
        "baseline_structural_electrical_flag": arch in {"baseline_structural_electrical_overlap", "baseline_conduction_or_structural_overlap"},
        "baseline_conduction_overlap_flag": conduction,
        "baseline_sarcomeric_flag": gene in SARCOMERIC,
        "baseline_underresolved_flag": underresolved,
        "baseline_syndromic_metabolic_overlap_flag": syndromic,
        "baseline_broad_ambiguous_condition_flag": broad,
        "baseline_gene_family_evaluability_flag": gene in {"TTN", "MYBPC3"},
        "baseline_portability_score": round(score, 4),
        "baseline_nonportability_score": round(100 - score, 4),
        "baseline_regime_assignment_reason": "|".join(dict.fromkeys(reason)),
    }


def build_baseline_regimes(df: pd.DataFrame) -> pd.DataFrame:
    assigned = df.apply(assign_baseline_regime, axis=1).apply(pd.Series)
    out = pd.concat([
        df[[
            "assertion_id", "variation_id", "gene", "condition_label_baseline",
            "condition_label_baseline_norm", "baseline_environment_v2",
            "classification_baseline", "review_status_baseline",
            "submitter_count_baseline_num", "HGVS", "consequence"
        ]].copy(),
        assigned,
    ], axis=1)
    out.to_csv(OUT_BASE_REGIMES, index=False)
    score_cols = [
        "assertion_id", "variation_id", "gene", "baseline_portability_score",
        "baseline_nonportability_score", "baseline_regime_primary",
        "baseline_architecture_family", "baseline_regime_assignment_reason",
        "baseline_collision_flag", "baseline_underresolved_flag", "baseline_nonspecific_flag",
        "baseline_structural_electrical_flag", "baseline_conduction_overlap_flag",
        "baseline_syndromic_metabolic_overlap_flag", "baseline_broad_ambiguous_condition_flag",
    ]
    out[score_cols].to_csv(OUT_SCORE, index=False)
    rules = [
        "# Cardiomyopathy Baseline-only Regime Rules",
        "",
        "Technical rule definitions; not manuscript prose.",
        "",
        "Allowed predictor inputs: gene, baseline condition label/environment, baseline classification, baseline review status, baseline submitter count, baseline HGVS/consequence, and gene-level scope flags if available.",
        "",
        "Forbidden predictor inputs: follow-up condition label, follow-up environment, env_baseline != env_followup, condition_label_change, cross_environment_drift, self_loop_stable, any_meaning_drift, follow-up review, follow-up submitter count, follow-up classification.",
        "",
        "## Regime candidates",
        "- sarcomeric_phenotype_anchored_baseline: sarcomeric genes with baseline HCM/cardiomyopathy environment and non-broad label.",
        "- arrhythmogenic_structural_electrical_baseline: desmosomal/structural genes with baseline ARVC/ACM or related structural-electrical labels.",
        "- dilated_conduction_overlap_baseline: LMNA/FLNC/RBM20/TTN/BAG3/DES/PLN with baseline DCM/conduction/sudden-death context.",
        "- cardiomyopathy_model_collision_baseline: baseline labels for the same assertion already span multiple environments.",
        "- nonspecific_cardiomyopathy_baseline: baseline nonspecific cardiomyopathy or other/unknown environment.",
        "- syndromic_metabolic_overlap_baseline: baseline syndromic/metabolic cardiomyopathy.",
        "- underresolved_contextual_baseline: baseline label/gene-condition context not specific enough for other regimes.",
        "- population_evaluability_limited_baseline: not implemented as AF because AF fields are unavailable; coarse gene-family evaluability flag retained only.",
    ]
    OUT_BASE_RULES.write_text("\n".join(rules), encoding="utf-8")
    return out


def endpoint_counts(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "classification_change", "condition_label_change", "cross_environment_drift",
        "within_environment_label_drift", "self_loop_stable", "review_status_change",
        "submitter_count_change", "any_meaning_drift", "semantic_drift_without_reclassification",
    ]
    rows = []
    n = len(df)
    for ep in endpoints:
        k = int(df[ep].astype(bool).sum())
        p = k / n if n else np.nan
        se = math.sqrt(p*(1-p)/n) if n and not math.isnan(p) else np.nan
        rows.append({
            "endpoint": ep, "numerator": k, "denominator": n, "rate": round(p, 4),
            "ci95_low": round(max(0, p - 1.96*se), 4) if not math.isnan(se) else np.nan,
            "ci95_high": round(min(1, p + 1.96*se), 4) if not math.isnan(se) else np.nan,
            "endpoint_role": "primary" if ep in {"condition_label_change", "cross_environment_drift", "any_meaning_drift"} else "secondary",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ENDPOINT_COUNTS, index=False)
    return out


def fdr_bh(pvals: List[float]) -> List[float]:
    p = np.array([1.0 if pd.isna(x) else float(x) for x in pvals])
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    min_adj = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        val = min(min_adj, p[idx] * m / rank)
        min_adj = val
        adj[idx] = val
    return adj.tolist()


def make_pipeline(X: pd.DataFrame, features: List[str]) -> Pipeline:
    for c in features:
        if c not in X.columns:
            X[c] = np.nan
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in features if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return Pipeline([
        ("pre", ColumnTransformer(transformers, remainder="drop")),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def calibration_slope(y: pd.Series, p: np.ndarray) -> float:
    try:
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
        lr = LogisticRegression(solver="liblinear")
        lr.fit(logit, y)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def fit_model(df: pd.DataFrame, endpoint: str, features: List[str], model_name: str) -> Dict[str, object]:
    y = df[endpoint].astype(bool).astype(int)
    n = len(df)
    pos = int(y.sum())
    if n < 30 or y.nunique() < 2:
        return {"endpoint": endpoint, "model": model_name, "N": n, "positive_N": pos, "status": "skipped_insufficient_N_or_endpoint"}
    X = df[features].copy() if features else pd.DataFrame({"_constant": np.ones(n)}, index=df.index)
    f = features if features else ["_constant"]
    pipe = make_pipeline(X, f)
    try:
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        ll = log_loss(y, p, labels=[0, 1])
        slope = calibration_slope(y, p)
        rng = np.random.default_rng(RANDOM_STATE)
        boots = []
        idx_all = np.arange(n)
        for _ in range(N_BOOT):
            idx = rng.choice(idx_all, size=n, replace=True)
            if len(np.unique(y.iloc[idx])) < 2:
                continue
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
                cv_auc = np.nan
        return {
            "endpoint": endpoint, "model": model_name, "N": n, "positive_N": pos,
            "AUROC": round(float(auc), 4), "AUROC_CI95_low": round(float(lo), 4), "AUROC_CI95_high": round(float(hi), 4),
            "AUPRC": round(float(auprc), 4), "Brier_score": round(float(brier), 4), "log_loss": round(float(ll), 4),
            "calibration_slope": round(float(slope), 4) if not math.isnan(slope) else np.nan,
            "cross_validated_AUROC": round(float(cv_auc), 4) if not math.isnan(cv_auc) else np.nan,
            "AIC_approx": round(float(2*(len(f)+1) + 2*ll*n), 4),
            "BIC_approx": round(float(math.log(n)*(len(f)+1) + 2*ll*n), 4),
            "status": "fit",
        }
    except Exception as e:
        return {"endpoint": endpoint, "model": model_name, "N": n, "positive_N": pos, "status": f"fit_failed:{type(e).__name__}:{str(e)[:120]}"}


def model_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["baseline_review_category"] = d["review_status_baseline"].map(review_category)
    d["baseline_classification_group"] = d["classification_baseline"].map(clinical_group)
    d["submitter_count_baseline_num"] = pd.to_numeric(d["submitter_count_baseline_num"], errors="coerce")
    d["weak_baseline_review_status"] = d["baseline_review_category"].isin(["single_submitter", "conflicting", "weak_or_no_assertion", "other_or_missing"])

    specs = {
        "M0_null": [],
        "M1_gene_only": ["gene"],
        "M2_baseline_regime_only": ["baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M3_baseline_ClinVar_metadata_only": ["baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
        "M4_gene_plus_baseline_regime": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M5_baseline_regime_plus_metadata": ["baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score", "baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
        "M6_gene_plus_baseline_regime_plus_metadata": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score", "baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
    }
    endpoints = [
        "condition_label_change", "cross_environment_drift", "any_meaning_drift",
        "within_environment_label_drift", "self_loop_stable",
    ]
    rows = []
    for ep in endpoints:
        for name, feats in specs.items():
            rows.append(fit_model(d, ep, feats, name))
    out = pd.DataFrame(rows)

    tests = []
    pairs = [
        ("M1_gene_only", "M4_gene_plus_baseline_regime", "gene_vs_gene_plus_regime"),
        ("M1_gene_only", "M6_gene_plus_baseline_regime_plus_metadata", "gene_vs_gene_plus_regime_metadata"),
        ("M3_baseline_ClinVar_metadata_only", "M5_baseline_regime_plus_metadata", "metadata_vs_regime_metadata"),
        ("M4_gene_plus_baseline_regime", "M6_gene_plus_baseline_regime_plus_metadata", "gene_regime_vs_gene_regime_metadata"),
        ("M0_null", "M2_baseline_regime_only", "null_vs_regime"),
    ]
    for ep in endpoints:
        sub = out[out["endpoint"] == ep]
        by = {r["model"]: r for r in sub.to_dict("records")}
        for base, full, label in pairs:
            if base in by and full in by and by[base].get("status") == "fit" and by[full].get("status") == "fit":
                n = by[base]["N"]
                lr = 2*n*(float(by[base]["log_loss"]) - float(by[full]["log_loss"]))
                p = float(chi2.sf(max(lr, 0), df=1)) if chi2 is not None else np.nan
                tests.append({
                    "endpoint": ep, "comparison": label, "base_model": base, "full_model": full,
                    "LR_style_statistic_approx": round(lr, 4), "p_value_approx": p,
                    "AUROC_base": by[base].get("AUROC", np.nan),
                    "AUROC_full": by[full].get("AUROC", np.nan),
                    "delta_AUROC": round(float(by[full].get("AUROC", np.nan)) - float(by[base].get("AUROC", np.nan)), 4),
                })
    tests_df = pd.DataFrame(tests)
    if len(tests_df):
        tests_df["FDR_p_value_approx"] = fdr_bh(tests_df["p_value_approx"].tolist())
    out.to_csv(OUT_MODELS, index=False)
    tests_df.to_csv(OUT_MODELS_CV, index=False)
    md = [
        "# Cardiomyopathy Baseline-only Model Report",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "All models use baseline-only regime assignments. Follow-up labels/environments are endpoints only, never predictors.",
        "",
        out.to_string(index=False),
        "",
        "## LR-style comparisons",
        tests_df.to_string(index=False),
    ]
    OUT_MODEL_QC.write_text("\n".join(md), encoding="utf-8")
    return out, tests_df


def fisher_row(df: pd.DataFrame, exposure: str, outcome: str, test: str) -> Dict[str, object]:
    x = df[exposure].astype(bool)
    y = df[outcome].astype(bool)
    a = int((x & y).sum())
    b = int((x & ~y).sum())
    c = int((~x & y).sum())
    d = int((~x & ~y).sum())
    odds, p = (np.nan, np.nan)
    if fisher_exact is not None:
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            pass
    return {
        "test": test, "exposure": exposure, "outcome": outcome,
        "a_exposed_outcome": a, "b_exposed_no_outcome": b, "c_unexposed_outcome": c, "d_unexposed_no_outcome": d,
        "odds_ratio": odds, "p_value": p, "status": "fit",
    }


def enrichment_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        fisher_row(df, "baseline_collision_flag", "cross_environment_drift", "baseline_collision_enriched_cross_environment"),
        fisher_row(df, "baseline_underresolved_flag", "condition_label_change", "baseline_underresolved_enriched_condition_label_change"),
        fisher_row(df, "baseline_nonspecific_flag", "condition_label_change", "baseline_nonspecific_enriched_condition_label_change"),
        fisher_row(df, "baseline_sarcomeric_flag", "self_loop_stable", "baseline_sarcomeric_enriched_self_loop"),
        fisher_row(df, "baseline_structural_electrical_flag", "cross_environment_drift", "baseline_structural_electrical_enriched_cross_environment"),
        fisher_row(df, "baseline_structural_electrical_flag", "condition_label_change", "baseline_structural_electrical_enriched_condition_label_change"),
        fisher_row(df, "low_baseline_portability_score", "cross_environment_drift", "low_baseline_portability_enriched_cross_environment"),
        fisher_row(df, "low_baseline_portability_score", "condition_label_change", "low_baseline_portability_enriched_condition_label_change"),
        fisher_row(df, "baseline_broad_ambiguous_condition_flag", "condition_label_change", "broad_ambiguous_baseline_label_enriched_condition_label_change"),
    ]
    out = pd.DataFrame(rows)
    out["FDR_p_value"] = fdr_bh(out["p_value"].tolist())
    out.to_csv(OUT_ENRICH, index=False)
    md = [
        "# Cardiomyopathy Baseline-only Transition Enrichment Report",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "Exposure variables are all derived from baseline-only labels, genes, and baseline metadata.",
        "",
        out.to_string(index=False),
    ]
    OUT_ENRICH_QC.write_text("\n".join(md), encoding="utf-8")
    return out


def score_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep in ["condition_label_change", "cross_environment_drift", "any_meaning_drift"]:
        rows.append(fit_model(df, ep, ["baseline_nonportability_score"], "baseline_nonportability_score_only"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SCORE_PERF, index=False)

    if plt is not None:
        try:
            q = df.copy()
            q["baseline_portability_tier"] = pd.cut(
                q["baseline_portability_score"],
                bins=[-1, 25, 50, 75, 101],
                labels=["severe_nonportability", "low_portability", "intermediate_portability", "high_portability"],
            )
            by = q.groupby("baseline_portability_tier", observed=False).agg(
                condition_label_change_rate=("condition_label_change", "mean"),
                cross_environment_drift_rate=("cross_environment_drift", "mean"),
                N=("variation_id", "size"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(by))
            ax.bar(x - 0.18, by["condition_label_change_rate"], width=0.36, label="condition_label_change")
            ax.bar(x + 0.18, by["cross_environment_drift_rate"], width=0.36, label="cross_environment_drift")
            ax.set_xticks(x)
            ax.set_xticklabels(by["baseline_portability_tier"].astype(str), rotation=30, ha="right")
            ax.set_ylabel("endpoint rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_SCORE_FIG)
            plt.close(fig)
        except Exception:
            pass
    return out


def clingen_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    cov_path = TABLES / "cardiomyopathy_clingen_coverage.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
    else:
        cov = pd.DataFrame([
            {"resource": "ClinGen Gene-Disease Validity", "coverage_level": "gene/gene-condition if local file present", "covered_assertions": 0, "total_assertions": len(df), "coverage_rate": 0.0},
            {"resource": "CMP VCEP scope", "coverage_level": "gene-level scope candidate only", "covered_assertions": int(df["gene"].isin(CMP_CSPEC_GENES_TARGET).sum()), "total_assertions": len(df), "coverage_rate": round(float(df["gene"].isin(CMP_CSPEC_GENES_TARGET).mean()), 4)},
            {"resource": "CMP CSpec scope", "coverage_level": "gene-level scope candidate only", "covered_assertions": int(df["gene"].isin(CMP_CSPEC_GENES_TARGET).sum()), "total_assertions": len(df), "coverage_rate": round(float(df["gene"].isin(CMP_CSPEC_GENES_TARGET).mean()), 4)},
            {"resource": "ClinGen Evidence Repository", "coverage_level": "variant-level", "covered_assertions": 0, "total_assertions": len(df), "coverage_rate": 0.0},
        ])
    cov["allowed_statement"] = cov.apply(
        lambda r: "gene-level scope only; not variant-level validation" if "VCEP" in str(r["resource"]) or "CSpec" in str(r["resource"]) else (
            "unavailable unless local join file exists" if "Gene-Disease" in str(r["resource"]) else "no variant-level Evidence Repository validation joined"
        ),
        axis=1,
    )
    cov["forbidden_statement"] = cov["resource"].map(lambda r: "VCEP validates assertions / variant-level ClinGen validation / ClinGen confirmed pathogenicity")
    cov.to_csv(OUT_CLINGEN_CLEAN, index=False)
    md = [
        "# Cardiomyopathy ClinGen Overlay Limitations",
        "",
        "Allowed statement: CMP VCEP/CSpec gene-level scope covered 1,135/4,918 temporally aligned assertions (23.08%) if reproduced in coverage table, but no variant-level ClinGen Evidence Repository validation was joined.",
        "",
        "Forbidden statements:",
        "- VCEP validates assertions.",
        "- Variant-level ClinGen validation.",
        "- ClinGen Gene-Disease Validity confirmed assertions unless a real local join exists.",
        "",
        cov.to_string(index=False),
    ]
    OUT_CLINGEN_LIMIT.write_text("\n".join(md), encoding="utf-8")
    return cov


def replication_comparison(df: pd.DataFrame, models: pd.DataFrame, enrich: pd.DataFrame, clingen: pd.DataFrame) -> pd.DataFrame:
    def endpoint_rate(ep):
        return round(float(df[ep].mean()), 4)
    def auc(ep, model):
        hit = models[(models["endpoint"] == ep) & (models["model"] == model)]
        return hit["AUROC"].iloc[0] if len(hit) and "AUROC" in hit.columns else np.nan
    def enrich_or(test):
        hit = enrich[enrich["test"] == test]
        return hit["odds_ratio"].iloc[0] if len(hit) else np.nan
    vcep_cov = 0
    hit = clingen[clingen["resource"].astype(str).str.contains("VCEP", na=False)]
    if len(hit):
        vcep_cov = hit["coverage_rate"].iloc[0]
    rows = [
        {
            "domain": "cardiomyopathy_v2_baseline_only",
            "aligned_N": len(df),
            "temporal_alignment_rate": "not_recomputed_from_total_target_gene_universe",
            "classification_change_rate": endpoint_rate("classification_change"),
            "condition_label_change_rate": endpoint_rate("condition_label_change"),
            "cross_environment_drift_rate": endpoint_rate("cross_environment_drift"),
            "within_environment_label_drift_rate": endpoint_rate("within_environment_label_drift"),
            "self_loop_stable_rate": endpoint_rate("self_loop_stable"),
            "any_meaning_drift_rate": endpoint_rate("any_meaning_drift"),
            "baseline_regime_only_AUROC_condition_label_change": auc("condition_label_change", "M2_baseline_regime_only"),
            "gene_only_AUROC_condition_label_change": auc("condition_label_change", "M1_gene_only"),
            "gene_plus_regime_AUROC_condition_label_change": auc("condition_label_change", "M4_gene_plus_baseline_regime"),
            "metadata_only_AUROC_condition_label_change": auc("condition_label_change", "M3_baseline_ClinVar_metadata_only"),
            "baseline_regime_only_AUROC_cross_environment": auc("cross_environment_drift", "M2_baseline_regime_only"),
            "gene_only_AUROC_cross_environment": auc("cross_environment_drift", "M1_gene_only"),
            "gene_plus_regime_AUROC_cross_environment": auc("cross_environment_drift", "M4_gene_plus_baseline_regime"),
            "cross_environment_enrichment_OR_baseline_collision": enrich_or("baseline_collision_enriched_cross_environment"),
            "cross_environment_enrichment_OR_baseline_structural": enrich_or("baseline_structural_electrical_enriched_cross_environment"),
            "ClinGen_VCEP_CSpec_status": f"gene-level scope only; VCEP coverage rate={vcep_cov}",
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_COMPARE, index=False)

    if plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            labs = ["condition", "cross-env", "any meaning", "self-loop"]
            vals = [endpoint_rate("condition_label_change"), endpoint_rate("cross_environment_drift"), endpoint_rate("any_meaning_drift"), endpoint_rate("self_loop_stable")]
            ax.bar(labs, vals)
            ax.set_ylabel("rate")
            ax.set_title("Cardiomyopathy v2 baseline-only replication endpoints")
            fig.tight_layout()
            fig.savefig(OUT_COMPARE_FIG)
            plt.close(fig)
        except Exception:
            pass
    md = [
        "# Arrhythmia vs Cardiomyopathy Replication Interpretation",
        "",
        "Cardiomyopathy v2 removes leakage by using baseline-only regimes. The previous cardiomyopathy cross-environment AUROC=0.9742 is blocked and must not be used.",
        "",
        out.to_string(index=False),
    ]
    OUT_COMPARE_QC.write_text("\n".join(md), encoding="utf-8")
    return out


def publication_claims(df: pd.DataFrame, counts: pd.DataFrame, models: pd.DataFrame, enrich: pd.DataFrame, clingen: pd.DataFrame) -> pd.DataFrame:
    def count_row(ep):
        r = counts[counts["endpoint"] == ep].iloc[0]
        return int(r["numerator"]), int(r["denominator"]), float(r["rate"]), f"{r['ci95_low']}-{r['ci95_high']}"
    def model_auc(ep, model):
        hit = models[(models["endpoint"] == ep) & (models["model"] == model)]
        if len(hit):
            r = hit.iloc[0]
            return r.get("AUROC", np.nan), f"{r.get('AUROC_CI95_low', np.nan)}-{r.get('AUROC_CI95_high', np.nan)}"
        return np.nan, ""
    def p_for(test):
        hit = enrich[enrich["test"] == test]
        if len(hit):
            return hit["p_value"].iloc[0], hit["FDR_p_value"].iloc[0]
        return np.nan, np.nan

    rows = []
    for ep, text in [
        ("condition_label_change", "Cardiomyopathy P/LP assertions showed condition-label meaning drift despite stable classification."),
        ("any_meaning_drift", "Cardiomyopathy P/LP assertions showed broad assertion meaning drift."),
        ("cross_environment_drift", "Cardiomyopathy P/LP assertions showed cross-environment meaning drift."),
    ]:
        k, n, rate, ci = count_row(ep)
        auc, auc_ci = model_auc(ep, "M2_baseline_regime_only")
        rows.append({
            "claim_text": text,
            "N": n, "numerator": k, "denominator": n, "percent": round(rate*100, 2),
            "model_or_statistic": f"endpoint_count; baseline_regime_only_AUROC={auc}",
            "CI": ci if pd.isna(auc) else auc_ci,
            "p_or_FDR": "",
            "source_file": "reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv; reports/tables/cardiomyopathy_model_comparison_baseline_only.csv",
            "script": "src/run_cardiomyopathy_replication_v2_baseline_only.py",
            "claim_strength": "descriptive_replication" if ep == "cross_environment_drift" else "external_domain_replication",
        })
    auc_gene, _ = model_auc("condition_label_change", "M1_gene_only")
    auc_reg, _ = model_auc("condition_label_change", "M2_baseline_regime_only")
    auc_gene_reg, _ = model_auc("condition_label_change", "M4_gene_plus_baseline_regime")
    strength = "baseline_only_predictive_support" if not pd.isna(auc_gene_reg) and not pd.isna(auc_gene) and auc_gene_reg > auc_gene else "exploratory_only"
    rows.append({
        "claim_text": "Baseline-only CAB-like regimes stratified condition-label drift if gene+regime improves over gene-only.",
        "N": len(df), "numerator": "", "denominator": len(df), "percent": "",
        "model_or_statistic": f"gene_only_AUROC={auc_gene}; baseline_regime_AUROC={auc_reg}; gene_plus_regime_AUROC={auc_gene_reg}",
        "CI": "", "p_or_FDR": "",
        "source_file": "reports/tables/cardiomyopathy_model_comparison_baseline_only.csv",
        "script": "src/run_cardiomyopathy_replication_v2_baseline_only.py",
        "claim_strength": strength,
    })
    auc_cross, _ = model_auc("cross_environment_drift", "M2_baseline_regime_only")
    p, fdr = p_for("baseline_collision_enriched_cross_environment")
    rows.append({
        "claim_text": "Baseline-only CAB-like regimes stratified cross-environment drift if leakage-clean signal remains.",
        "N": len(df), "numerator": int(df["cross_environment_drift"].sum()), "denominator": len(df), "percent": round(float(df["cross_environment_drift"].mean()*100), 2),
        "model_or_statistic": f"baseline_regime_only_AUROC={auc_cross}; baseline_collision_enrichment_OR={enrich.loc[enrich['test'].eq('baseline_collision_enriched_cross_environment'), 'odds_ratio'].iloc[0] if len(enrich[enrich['test'].eq('baseline_collision_enriched_cross_environment')]) else np.nan}",
        "CI": "", "p_or_FDR": f"p={p}; FDR={fdr}",
        "source_file": "reports/tables/cardiomyopathy_model_comparison_baseline_only.csv; reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv",
        "script": "src/run_cardiomyopathy_replication_v2_baseline_only.py",
        "claim_strength": "baseline_only_predictive_support" if not pd.isna(auc_cross) and auc_cross > 0.6 else "exploratory_only",
    })
    vcep = clingen[clingen["resource"].astype(str).str.contains("VCEP", na=False)]
    cov = vcep.iloc[0] if len(vcep) else {"covered_assertions": 0, "total_assertions": len(df), "coverage_rate": 0}
    rows.append({
        "claim_text": "CMP VCEP/CSpec coverage is gene-level only; no variant-level ClinGen Evidence Repository validation was joined.",
        "N": len(df), "numerator": cov["covered_assertions"], "denominator": cov["total_assertions"], "percent": round(float(cov["coverage_rate"])*100, 2),
        "model_or_statistic": "gene-level scope only",
        "CI": "", "p_or_FDR": "",
        "source_file": "reports/tables/cardiomyopathy_clingen_overlay_status_clean.csv",
        "script": "src/run_cardiomyopathy_replication_v2_baseline_only.py",
        "claim_strength": "gene_level_external_constraint_only",
    })
    rows.append({
        "claim_text": "Previous cardiomyopathy cross-environment AUROC=0.9742 is blocked because v1 regimes used follow-up environments.",
        "N": len(df), "numerator": "", "denominator": "", "percent": "",
        "model_or_statistic": "blocked prior metric",
        "CI": "", "p_or_FDR": "",
        "source_file": "reports/tables/cardiomyopathy_regime_leakage_audit.csv",
        "script": "src/run_cardiomyopathy_replication_v2_baseline_only.py",
        "claim_strength": "blocked_by_leakage",
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def final_report(counts, models, enrich, score_perf, clingen, compare, claims) -> None:
    def verdict_model(ep, model):
        hit = models[(models["endpoint"] == ep) & (models["model"] == model)]
        if len(hit):
            return hit.iloc[0].to_dict()
        return {}
    gene = verdict_model("condition_label_change", "M1_gene_only")
    reg = verdict_model("condition_label_change", "M2_baseline_regime_only")
    gene_reg = verdict_model("condition_label_change", "M4_gene_plus_baseline_regime")
    cross_reg = verdict_model("cross_environment_drift", "M2_baseline_regime_only")
    lines = [
        "# Final Cardiomyopathy Replication v2 Report",
        "",
        "Analysis report, not manuscript prose.",
        "",
        "## 1. What survived after removing leakage?",
        "- v1 cross-environment AUROC=0.9742 is blocked; v2 recomputes all predictors baseline-only.",
        "- Baseline-only regime, score, and enrichment outputs are the only eligible cardiomyopathy v2 predictors.",
        "",
        "## 2. Did cardiomyopathy replicate broad meaning drift?",
        counts.to_string(index=False),
        "",
        "## 3. Did baseline-only regimes predict condition-label drift?",
        f"- gene-only AUROC={gene.get('AUROC')}; baseline-regime-only AUROC={reg.get('AUROC')}; gene+baseline-regime AUROC={gene_reg.get('AUROC')}.",
        "",
        "## 4. Did baseline-only regimes predict cross-environment drift?",
        f"- baseline-regime-only cross-environment AUROC={cross_reg.get('AUROC')}; see enrichment tests below.",
        "",
        "## 5. Did gene+CAB-like improve over gene-only?",
        "- See model comparison and LR-style tests.",
        models.to_string(index=False),
        "",
        "## 6. What is the safe ClinGen/VCEP/CSpec claim?",
        clingen.to_string(index=False),
        "",
        "## 7. Does this support external domain replication or only descriptive replication?",
        claims.to_string(index=False),
        "",
        "## 8. What remains blocked?",
        "- Prior v1 cross-environment AUROC=0.9742.",
        "- Variant-level ClinGen validation.",
        "- Full general assertion portability theory if baseline-only signals are weak.",
        "- Any claim using follow-up labels/environments as predictors.",
        "",
        "## Baseline-only enrichment tests",
        enrich.to_string(index=False),
        "",
        "## Baseline portability score performance",
        score_perf.to_string(index=False),
        "",
        "## Arrhythmia vs cardiomyopathy v2 comparison",
        compare.to_string(index=False),
    ]
    OUT_FINAL.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Loading cardiomyopathy replication inputs...")
    df = load_input()
    print(f"Aligned cardiomyopathy rows: {len(df):,}")

    leakage_audit()
    base = build_baseline_regimes(df)
    df2 = df.merge(base.drop(columns=[c for c in ["assertion_id", "gene"] if c in base.columns], errors="ignore"), on="variation_id", how="left")
    # Re-coalesce core columns after merge in case pandas suffixes were introduced.
    coalesce_to_column(df2, "review_status_baseline", ["review_status_baseline", "review_status_baseline_x", "review_status_baseline_y"], "")
    coalesce_to_column(df2, "review_status_followup", ["review_status_followup", "review_status_followup_x", "review_status_followup_y"], "")
    coalesce_to_column(df2, "classification_baseline", ["classification_baseline", "classification_baseline_x", "classification_baseline_y"], "")
    coalesce_to_column(df2, "classification_followup", ["classification_followup", "classification_followup_x", "classification_followup_y"], "")
    coalesce_to_column(df2, "submitter_count_baseline_num", ["submitter_count_baseline_num", "submitter_count_baseline_num_x", "submitter_count_baseline_num_y"], np.nan)
    coalesce_to_column(df2, "submitter_count_baseline", ["submitter_count_baseline", "submitter_count_baseline_x", "submitter_count_baseline_y"], np.nan)
    df2["low_baseline_portability_score"] = df2["baseline_portability_score"].map(lambda x: safe_float(x, 100) < 50)

    endpoint_cols = [
        "assertion_id", "variation_id", "gene",
        "condition_label_baseline", "condition_label_followup",
        "baseline_environment_v2", "followup_environment_v2",
        "classification_change", "condition_label_change", "cross_environment_drift",
        "within_environment_label_drift", "self_loop_stable", "review_status_change",
        "submitter_count_change", "any_meaning_drift", "semantic_drift_without_reclassification",
    ]
    df2[[c for c in endpoint_cols if c in df2.columns]].to_csv(OUT_ENDPOINTS, index=False)
    counts = endpoint_counts(df2)

    models, model_tests = model_comparison(df2)
    enrich = enrichment_tests(df2)
    score_perf = score_performance(df2)
    clingen = clingen_cleanup(df2)
    compare = replication_comparison(df2, models, enrich, clingen)
    claims = publication_claims(df2, counts, models, enrich, clingen)
    final_report(counts, models, enrich, score_perf, clingen, compare, claims)

    print("Cardiomyopathy baseline-only v2 replication complete.")
    print()
    print("Endpoint counts:")
    print(counts.to_string(index=False))
    print()
    print("Baseline-only model comparison:")
    print(models.to_string(index=False))
    print()
    print("Baseline-only enrichment tests:")
    print(enrich.to_string(index=False))
    print()
    print("Publication-safe claims:")
    print(claims.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_LEAK_AUDIT, OUT_BASE_REGIMES, OUT_ENDPOINTS, OUT_ENDPOINT_COUNTS,
        OUT_MODELS, OUT_MODELS_CV, OUT_ENRICH, OUT_SCORE, OUT_SCORE_PERF,
        OUT_CLINGEN_CLEAN, OUT_COMPARE, OUT_CLAIMS, OUT_FINAL,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
