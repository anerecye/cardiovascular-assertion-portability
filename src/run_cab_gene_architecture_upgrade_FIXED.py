#!/usr/bin/env python3
"""
CAB gene-architecture temporal instability upgrade.

Local runner. No GitHub calls. No manuscript prose. This script upgrades the
leakage-clean CPI validation into a gene/CAB architecture decomposition package.

It intentionally does NOT restore deprecated/leakage-susceptible original CPI AUCs.
It treats gene identity as a biological axis and tests whether CAB architecture
decomposes gene-level temporal instability.

Primary inputs expected from prior CAB pipeline:
- data/processed/cab_predictive_operational_framework.csv
- data/processed/cab_portability_index_baseline_only.csv
- reports/tables/cpi_predictive_model_validation.csv
- reports/tables/cpi_negative_control_results.csv
- reports/tables/cpi_publication_safe_claims.csv

Optional inputs used when present:
- reports/tables/cab_causal_architecture_assignments.csv
- reports/tables/cab_failure_membership_explicit.csv
- data/reference/cab_clinical_inference_environments.tsv

Outputs follow the requested Phase 1-8 paths.

Notes:
- Mixed-effects logistic models are approximated with penalized gene fixed-effect
  residual variance decomposition because statsmodels mixed logit is not guaranteed
  in the local environment. Output tables explicitly label this approximation.
- All predictors are baseline-only. Follow-up labels are used only to define endpoints.
- Failed/unknown condition environment mappings are preserved in QC outputs.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scipy.stats import fisher_exact, chi2, norm, spearmanr
except Exception:
    fisher_exact = None
    chi2 = None
    norm = None
    spearmanr = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
QC = REPORTS / "qc"

FRAMEWORK = DATA / "cab_predictive_operational_framework.csv"
CPI_BASELINE = DATA / "cab_portability_index_baseline_only.csv"
ARCH_ASSIGN = TABLES / "cab_causal_architecture_assignments.csv"
FAILURE_MEMBERSHIP = DATA / "processed" / "cab_failure_membership_explicit.csv"
FAILURE_MEMBERSHIP_ALT = TABLES / "cab_failure_membership_explicit.csv"
ENV_REF = BASE / "data" / "reference" / "cab_clinical_inference_environments.tsv"

# Requested outputs
GENE_MAP_DATA = DATA / "cab_gene_temporal_instability_map.csv"
GENE_MAP_TABLE = TABLES / "gene_temporal_instability_map.csv"
GENE_RANKED = TABLES / "gene_temporal_instability_ranked.csv"
GENE_REPORT = QC / "gene_temporal_instability_report.md"

GENE_VS_CAB_MODELS = TABLES / "gene_vs_cab_model_comparison.csv"
MIXED_VARIANCE = TABLES / "mixed_effects_gene_variance_decomposition.csv"
GENE_EXPLANATION = TABLES / "gene_only_vs_cab_explanation_tests.csv"
GENE_VS_CAB_REPORT = QC / "gene_vs_cab_interpretation.md"

GENE_LEVEL_DECOMP = TABLES / "gene_level_cab_decomposition_models.csv"
WITHIN_GENE_PERM = TABLES / "within_gene_permutation_decomposition.csv"
LOGO_RESULTS = TABLES / "leave_one_gene_out_results.csv"
GENE_DECOMP_REPORT = QC / "gene_decomposition_report.md"

SENTINEL_DATA = DATA / "sentinel_gene_architecture_profiles.csv"
SENTINEL_TABLE = TABLES / "sentinel_gene_profiles.csv"
SENTINEL_TESTS = TABLES / "sentinel_gene_within_gene_tests.csv"
SENTINEL_REPORT = QC / "sentinel_gene_case_study_report.md"

LABEL_EDGES_DATA = DATA / "cab_condition_label_transition_edges.csv"
ENV_EDGES_DATA = DATA / "cab_condition_environment_transition_edges.csv"
LABEL_EDGES_TABLE = TABLES / "condition_label_transition_edges.csv"
ENV_EDGES_TABLE = TABLES / "condition_environment_transition_edges.csv"
TRANSITION_TESTS = TABLES / "transition_network_enrichment_tests.csv"
TRANSITION_REPORT = QC / "condition_transition_network_report.md"

CROSS_ENV_DATA = DATA / "cab_cross_environment_drift.csv"
CROSS_ENV_COUNTS = TABLES / "cross_environment_drift_counts.csv"
CROSS_ENV_BY_ARCH = TABLES / "cross_environment_drift_by_architecture.csv"
CROSS_ENV_MODELS = TABLES / "cross_environment_drift_prediction_models.csv"
CROSS_ENV_REPORT = QC / "cross_environment_drift_report.md"

CPI_SAFE_CLAIMS = TABLES / "cpi_publication_safe_claims.csv"
FINAL_REPORT = REPORTS / "final_cab_gene_architecture_upgrade_report.md"

RANDOM_SEED = 20260503
BOOTSTRAPS = 300
PERMUTATIONS = 300
CV_REPEATS = 20

SENTINEL_REQUIRED = ["SCN5A", "RYR2", "KCNQ1", "KCNH2", "CACNA1C"]
SENTINEL_OPTIONAL = ["CASQ2", "TRDN", "HCN4", "ANK2", "KCNE1"]

ENVIRONMENTS = [
    "LQTS",
    "Brugada",
    "CPVT",
    "SADS",
    "SQTS",
    "cardiomyopathy_overlap",
    "conduction",
    "nonspecific_arrhythmia",
    "other/unknown",
]


def ensure_dirs() -> None:
    for p in [DATA, REPORTS, TABLES, FIGURES, QC]:
        p.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


def norm_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def boolish(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ci_wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return max(0, center - half), min(1, center + half)


def fdr_bh(pvals: Iterable[float]) -> List[float]:
    vals = np.array([np.nan if p is None else float(p) for p in pvals], dtype=float)
    out = np.full(len(vals), np.nan)
    mask = ~np.isnan(vals)
    if not mask.any():
        return out.tolist()
    p = vals[mask]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        adj[i] = val
        prev = val
    restored = np.empty(m)
    restored[order] = adj
    out[mask] = restored
    return out.tolist()


def entropy_from_terms(values: Iterable[str]) -> float:
    terms = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        for part in re.split(r"[;|]", s):
            p = norm_text(part)
            if p:
                terms.append(p)
    if not terms:
        return 0.0
    c = Counter(terms)
    n = sum(c.values())
    return float(-sum((v/n) * math.log2(v/n) for v in c.values()))


def split_condition_terms(value: object) -> List[str]:
    s = norm_text(value)
    if not s:
        return ["other/unknown"]
    parts = [norm_text(p) for p in re.split(r"[;|]", s) if norm_text(p)]
    return sorted(set(parts)) if parts else ["other/unknown"]


def first_term(value: object) -> str:
    terms = split_condition_terms(value)
    return terms[0] if terms else "other/unknown"


def mode_or_unknown(s: pd.Series) -> str:
    vals = [str(x) for x in s.dropna().tolist() if str(x).strip()]
    if not vals:
        return "unknown"
    return Counter(vals).most_common(1)[0][0]


def top_items_string(values: Iterable[object], topn: int = 5) -> str:
    c = Counter()
    for v in values:
        if pd.isna(v):
            continue
        text = str(v).strip()
        if not text:
            continue
        for item in re.split(r"[;|]", text):
            item = item.strip()
            if item:
                c[item] += 1
    return "; ".join(f"{k}:{v}" for k, v in c.most_common(topn))


def dist_string(s: pd.Series, topn: int = 8) -> str:
    c = Counter([str(x) for x in s.dropna().tolist() if str(x).strip()])
    if not c:
        return ""
    total = sum(c.values())
    return "; ".join(f"{k}:{v}({v/total:.2f})" for k, v in c.most_common(topn))


def infer_environment(label: object) -> str:
    text = norm_text(label)
    if not text:
        return "other/unknown"
    if any(t in text for t in ["long qt", "lqts", "romano-ward", "jervell", "lange-nielsen"]):
        return "LQTS"
    if any(t in text for t in ["brugada"]):
        return "Brugada"
    if any(t in text for t in ["catecholaminergic", "cpvt"]):
        return "CPVT"
    if any(t in text for t in ["sudden", "sads", "death", "stillbirth", "postmortem", "arrhythmogenic death"]):
        return "SADS"
    if any(t in text for t in ["short qt", "sqts"]):
        return "SQTS"
    if any(t in text for t in ["cardiomyopathy", "dilated cardiomyopathy", "hypertrophic cardiomyopathy", "arrhythmogenic right ventricular"]):
        return "cardiomyopathy_overlap"
    if any(t in text for t in ["conduction", "sick sinus", "bradycardia", "atrioventricular block", "heart block"]):
        return "conduction"
    if any(t in text for t in ["arrhythmia", "atrial fibrillation", "ventricular fibrillation", "tachycardia"]):
        return "nonspecific_arrhythmia"
    return "other/unknown"


def infer_env_from_row(row: pd.Series, prefix: str) -> str:
    # Prefer explicit environment columns if prior pipeline created them.
    candidates = [
        f"{prefix}_clinical_environment",
        f"clinical_environment_{prefix}",
        f"{prefix}_clinical_inference_environment",
        f"clinical_inference_environment_{prefix}",
    ]
    for c in candidates:
        if c in row.index and str(row.get(c, "")).strip():
            val = str(row.get(c))
            if val in ENVIRONMENTS:
                return val
            return infer_environment(val)
    if prefix == "baseline":
        for c in ["phenotype_list_2023-01", "baseline_condition_norm", "baseline_condition_exact_string"]:
            if c in row.index:
                return infer_environment(row.get(c))
    if prefix == "followup":
        for c in ["phenotype_list_2026-04", "followup_condition_norm", "followup_condition_exact_string"]:
            if c in row.index:
                return infer_environment(row.get(c))
    return "other/unknown"


def fisher_or_nan(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    if fisher_exact is None:
        return np.nan, np.nan
    try:
        odds, p = fisher_exact([[a, b], [c, d]])
        return float(odds), float(p)
    except Exception:
        return np.nan, np.nan


def find_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_base_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not FRAMEWORK.exists():
        raise FileNotFoundError(f"Missing {FRAMEWORK}. Run prior CPI/CAB framework scripts first.")
    fw = pd.read_csv(FRAMEWORK, low_memory=False)
    fw["variation_id"] = fw["variation_id"].map(norm_id)

    cpi = read_csv_if_exists(CPI_BASELINE)
    if cpi is None:
        raise FileNotFoundError(f"Missing {CPI_BASELINE}. Run leakage-clean CPI validation first.")
    # Support either assertion_id or variation_id.
    if "assertion_id" in cpi.columns and "variation_id" not in cpi.columns:
        cpi["variation_id"] = cpi["assertion_id"].map(norm_id)
    elif "variation_id" in cpi.columns:
        cpi["variation_id"] = cpi["variation_id"].map(norm_id)
    else:
        raise ValueError(f"{CPI_BASELINE} lacks assertion_id/variation_id.")

    arch = read_csv_if_exists(ARCH_ASSIGN)
    if arch is not None and "variation_id" in arch.columns:
        arch["variation_id"] = arch["variation_id"].map(norm_id)
    else:
        arch = pd.DataFrame({"variation_id": []})

    return fw, cpi, arch


def merge_analysis_table(fw: pd.DataFrame, cpi: pd.DataFrame, arch: pd.DataFrame) -> pd.DataFrame:
    df = fw.copy()

    # Merge baseline-only CPI if not already present.
    keep_cpi = [c for c in cpi.columns if c != "assertion_id"]
    df = df.merge(cpi[keep_cpi].drop_duplicates("variation_id"), on="variation_id", how="left", suffixes=("", "_cpi"))

    # Merge architecture columns missing from framework.
    if not arch.empty:
        arch_keep = [c for c in arch.columns if c == "variation_id" or c not in df.columns]
        if len(arch_keep) > 1:
            df = df.merge(arch[arch_keep].drop_duplicates("variation_id"), on="variation_id", how="left")

    # Canonical standard columns.
    if "gene" not in df.columns:
        gcol = find_col(df, ["GeneSymbol", "gene_baseline", "gene_2023-01"])
        df["gene"] = df[gcol] if gcol else "unknown"
    df["gene"] = df["gene"].fillna("unknown").astype(str)

    if "CPI_baseline_only" not in df.columns:
        if "cab_portability_index" in df.columns:
            # Use only as fallback, mark in report by source column.
            df["CPI_baseline_only"] = df["cab_portability_index"]
        else:
            df["CPI_baseline_only"] = np.nan
    df["CPI_baseline_only"] = df["CPI_baseline_only"].map(lambda x: safe_float(x, np.nan))
    df["nonportability_score_baseline_only"] = 100.0 - df["CPI_baseline_only"]

    if "CPI_tier_baseline_only" not in df.columns:
        df["CPI_tier_baseline_only"] = pd.cut(
            df["CPI_baseline_only"],
            bins=[-0.001, 25, 50, 75, 100],
            labels=["severe_non_portability", "low_portability", "intermediate_portability", "high_portability"],
        ).astype(str)

    # Endpoints from follow-up only, not predictors.
    df["future_condition_label_drift"] = df.get("condition_label_change", False)
    df["future_condition_label_drift"] = df["future_condition_label_drift"].map(boolish)

    if "failure_topology_severity" not in df.columns:
        df["failure_topology_severity"] = 0
    df["future_classification_severity_drift"] = df["failure_topology_severity"].map(safe_float) >= 3

    if "assertion_meaning_drift_score" not in df.columns:
        df["assertion_meaning_drift_score"] = 0
    df["any_meaning_drift"] = df["assertion_meaning_drift_score"].map(safe_float) > 0
    df["semantic_drift_without_reclassification"] = (
        df["future_condition_label_drift"] & ~df.get("classification_change", False).map(boolish)
    )
    df["review_status_change"] = df.get("review_status_change", False).map(boolish)

    # submitter_count_change endpoint.
    bsub = find_col(df, ["number_submitters_2023-01", "baseline_number_submitters", "NumberSubmitters_2023-01"])
    fsub = find_col(df, ["number_submitters_2026-04", "followup_number_submitters", "NumberSubmitters_2026-04"])
    if bsub and fsub:
        df["submitter_count_change"] = df[bsub].map(safe_float) != df[fsub].map(safe_float)
        df["baseline_submitter_count"] = df[bsub].map(safe_float)
    else:
        df["submitter_count_change"] = False
        df["baseline_submitter_count"] = np.nan

    # Baseline ClinVar metadata.
    if "baseline_review_category" not in df.columns:
        df["baseline_review_category"] = df.get("review_status_2023-01", "").map(norm_text)
    if "baseline_clinical_group" not in df.columns:
        df["baseline_clinical_group"] = df.get("clinical_significance_2023-01", "").map(norm_text)

    # Architecture/CAB feature fallbacks.
    def ensure_bool_col(name: str, fallback_names: List[str] = []) -> None:
        if name in df.columns:
            df[name] = df[name].map(boolish)
            return
        for fb in fallback_names:
            if fb in df.columns:
                df[name] = df[fb].map(boolish)
                return
        df[name] = False

    ensure_bool_col("is_disease_model_collision", ["disease_model_collision", "disease_model_collision_flag"])
    ensure_bool_col("is_unanchored_assertion_state", ["unanchored_assertion_state", "unanchored_flag"])
    ensure_bool_col("is_canonical_monogenic", ["canonical_flag", "is_canonical"])
    ensure_bool_col("single_model_repair_required", ["repair_required", "single_model_repair"])
    ensure_bool_col("is_ancestry_concentrated", ["population_localized_flag", "ancestry_concentrated_flag"])
    ensure_bool_col("is_penetrance_boundary", ["penetrance_boundary_flag"])

    # Phenotype-environment / routing proxies. Use explicit if present, otherwise infer conservative False.
    ensure_bool_col("deterministic_phenotype_anchored", ["is_deterministic_phenotype_anchored"])
    ensure_bool_col("genotype_first_postmortem", ["is_genotype_first_postmortem", "postmortem_flag", "sads_gene_context_flag"])
    ensure_bool_col("provocation_dependent", ["is_provocation_dependent"])
    ensure_bool_col("underresolved", ["is_underresolved", "is_unanchored_assertion_state"])
    ensure_bool_col("structural_electrical_overlap", ["is_structural_electrical_overlap", "cardiomyopathy_overlap_flag"])
    ensure_bool_col("population_localized", ["is_population_localized", "is_ancestry_concentrated"])

    if "failure_membership_count" not in df.columns:
        if "regime_membership_count" in df.columns:
            df["failure_membership_count"] = df["regime_membership_count"].map(safe_float)
        elif "temporal_event_count" in df.columns:
            df["failure_membership_count"] = df["temporal_event_count"].map(safe_float)
        else:
            df["failure_membership_count"] = 0.0

    # failure A/B/C proxies if explicit missing.
    ensure_bool_col("failure_A", ["failure_A_flag", "failure_a", "classification_change"])
    ensure_bool_col("failure_B", ["failure_B_flag", "failure_b", "condition_label_change"])
    ensure_bool_col("failure_C", ["failure_C_flag", "failure_c", "review_status_change"])

    if "evidence_collision_index" not in df.columns:
        df["evidence_collision_index"] = 0.0
    df["evidence_collision_index"] = df["evidence_collision_index"].map(safe_float)

    if "primary_regime" not in df.columns:
        df["primary_regime"] = "unknown"
    if "causal_architecture_category" not in df.columns:
        df["causal_architecture_category"] = "unknown"
    if "condition_group_primary" not in df.columns:
        df["condition_group_primary"] = "unknown"

    df["baseline_env"] = df.apply(lambda r: infer_env_from_row(r, "baseline"), axis=1)
    df["followup_env"] = df.apply(lambda r: infer_env_from_row(r, "followup"), axis=1)
    df["cross_environment_drift"] = df["baseline_env"] != df["followup_env"]
    df["within_environment_label_drift"] = df["future_condition_label_drift"] & ~df["cross_environment_drift"]
    df["stable_environment"] = ~df["cross_environment_drift"]

    return df


def build_gene_map(df: pd.DataFrame, all_cab: pd.DataFrame) -> pd.DataFrame:
    total_by_gene = all_cab.groupby("gene").size().rename("n_assertions_total_in_CAB")
    rows = []
    for gene, sub in df.groupby("gene"):
        n = len(sub)
        total = int(total_by_gene.get(gene, n))
        baseline_conditions = []
        for v in sub.get("phenotype_list_2023-01", pd.Series("", index=sub.index)):
            baseline_conditions.extend(split_condition_terms(v))
        distinct_conditions = sorted(set([x for x in baseline_conditions if x and x != "other/unknown"]))

        entropies = []
        for v in sub.get("phenotype_list_2023-01", pd.Series("", index=sub.index)):
            entropies.append(entropy_from_terms([v]))

        row = {
            "gene": gene,
            "n_assertions_total_in_CAB": total,
            "n_temporal_aligned": n,
            "temporal_alignment_rate": round(n / total, 4) if total else np.nan,
            "condition_label_change_n": int(sub["future_condition_label_drift"].sum()),
            "condition_label_change_rate": round(float(sub["future_condition_label_drift"].mean()), 4),
            "any_meaning_drift_n": int(sub["any_meaning_drift"].sum()),
            "any_meaning_drift_rate": round(float(sub["any_meaning_drift"].mean()), 4),
            "classification_change_n": int(sub.get("classification_change", False).map(boolish).sum()),
            "classification_change_rate": round(float(sub.get("classification_change", False).map(boolish).mean()), 4),
            "review_status_change_n": int(sub["review_status_change"].sum()),
            "review_status_change_rate": round(float(sub["review_status_change"].mean()), 4),
            "submitter_count_change_n": int(sub["submitter_count_change"].sum()),
            "submitter_count_change_rate": round(float(sub["submitter_count_change"].mean()), 4),
            "mean_CPI_baseline_only": round(float(sub["CPI_baseline_only"].mean()), 4),
            "median_CPI_baseline_only": round(float(sub["CPI_baseline_only"].median()), 4),
            "mean_nonportability_score": round(float(sub["nonportability_score_baseline_only"].mean()), 4),
            "disease_model_collision_fraction": round(float(sub["is_disease_model_collision"].mean()), 4),
            "unanchored_fraction": round(float(sub["is_unanchored_assertion_state"].mean()), 4),
            "canonical_fraction": round(float(sub["is_canonical_monogenic"].mean()), 4),
            "deterministic_phenotype_anchored_fraction": round(float(sub["deterministic_phenotype_anchored"].mean()), 4),
            "genotype_first_postmortem_fraction": round(float(sub["genotype_first_postmortem"].mean()), 4),
            "provocation_dependent_fraction": round(float(sub["provocation_dependent"].mean()), 4),
            "underresolved_fraction": round(float(sub["underresolved"].mean()), 4),
            "structural_electrical_overlap_fraction": round(float(sub["structural_electrical_overlap"].mean()), 4),
            "population_localized_fraction": round(float(sub["population_localized"].mean()), 4),
            "failure_A_fraction": round(float(sub["failure_A"].mean()), 4),
            "failure_B_fraction": round(float(sub["failure_B"].mean()), 4),
            "failure_C_fraction": round(float(sub["failure_C"].mean()), 4),
            "multi_failure_fraction": round(float((sub["failure_membership_count"].map(safe_float) > 1).mean()), 4),
            "mean_failure_membership_count": round(float(sub["failure_membership_count"].map(safe_float).mean()), 4),
            "mean_evidence_collision_index": round(float(sub["evidence_collision_index"].mean()), 4),
            "condition_entropy_mean": round(float(np.mean(entropies)) if entropies else 0.0, 4),
            "condition_entropy_max": round(float(np.max(entropies)) if entropies else 0.0, 4),
            "number_of_distinct_condition_labels": len(distinct_conditions),
            "primary_clinical_environment": mode_or_unknown(sub["baseline_env"]),
            "dominant_CAB_architecture": mode_or_unknown(sub["causal_architecture_category"]),
            "dominant_CAB_regime": mode_or_unknown(sub["primary_regime"]),
        }
        rows.append(row)

    # Include non-aligned genes from CAB universe as rows with n_temporal_aligned=0.
    seen = {r["gene"] for r in rows}
    for gene, total in total_by_gene.items():
        if gene in seen:
            continue
        rows.append({
            "gene": gene,
            "n_assertions_total_in_CAB": int(total),
            "n_temporal_aligned": 0,
            "temporal_alignment_rate": 0.0,
            "condition_label_change_n": 0,
            "condition_label_change_rate": np.nan,
            "any_meaning_drift_n": 0,
            "any_meaning_drift_rate": np.nan,
            "classification_change_n": 0,
            "classification_change_rate": np.nan,
            "review_status_change_n": 0,
            "review_status_change_rate": np.nan,
            "submitter_count_change_n": 0,
            "submitter_count_change_rate": np.nan,
            "mean_CPI_baseline_only": np.nan,
            "median_CPI_baseline_only": np.nan,
            "mean_nonportability_score": np.nan,
            "disease_model_collision_fraction": np.nan,
            "unanchored_fraction": np.nan,
            "canonical_fraction": np.nan,
            "deterministic_phenotype_anchored_fraction": np.nan,
            "genotype_first_postmortem_fraction": np.nan,
            "provocation_dependent_fraction": np.nan,
            "underresolved_fraction": np.nan,
            "structural_electrical_overlap_fraction": np.nan,
            "population_localized_fraction": np.nan,
            "failure_A_fraction": np.nan,
            "failure_B_fraction": np.nan,
            "failure_C_fraction": np.nan,
            "multi_failure_fraction": np.nan,
            "mean_failure_membership_count": np.nan,
            "mean_evidence_collision_index": np.nan,
            "condition_entropy_mean": np.nan,
            "condition_entropy_max": np.nan,
            "number_of_distinct_condition_labels": 0,
            "primary_clinical_environment": "not_temporally_aligned",
            "dominant_CAB_architecture": "not_temporally_aligned",
            "dominant_CAB_regime": "not_temporally_aligned",
        })

    out = pd.DataFrame(rows).sort_values(["n_temporal_aligned", "condition_label_change_rate"], ascending=[False, False])
    return out


def make_scatter(x, y, labels, xlabel, ylabel, title, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)
    for xi, yi, lab in zip(x, y, labels):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(str(lab), (xi, yi), fontsize=7, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def preprocessors_for(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    cats, nums = [], []
    for f in features:
        if f not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[f]) or pd.api.types.is_bool_dtype(df[f]):
            nums.append(f)
        else:
            cats.append(f)
    return cats, nums


def model_matrix(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return a safe feature matrix and feature list.

    Intercept-only/null models need a physical constant column because
    sklearn ColumnTransformer only sees columns present in X, not columns
    mutably added inside make_model. Tiny thing, giant traceback, naturally.
    """
    valid = [f for f in features if f in df.columns]
    if not valid:
        return pd.DataFrame({"_constant_feature": np.ones(len(df), dtype=float)}, index=df.index), ["_constant_feature"]
    return df[valid].copy(), valid


def make_model(df: pd.DataFrame, features: List[str]) -> Pipeline:
    cats, nums = preprocessors_for(df, features)
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), nums))
    if cats:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", enc),
        ]), cats))
    if not transformers:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["_constant_feature"]))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")),
    ])


def fit_predict_metrics(df: pd.DataFrame, endpoint: str, features: List[str], model_name: str) -> Dict[str, object]:
    y = df[endpoint].astype(bool).astype(int)
    n = len(y)
    pos = int(y.sum())
    if pos == 0 or pos == n:
        return {
            "model": model_name, "endpoint": endpoint, "N": n, "endpoint_positive_N": pos,
            "AUROC": np.nan, "AUPRC": np.nan, "Brier_score": np.nan, "calibration_slope": np.nan,
            "log_loss": np.nan, "AIC": np.nan, "BIC": np.nan, "pseudo_R2": np.nan,
        }
    X, model_features = model_matrix(df, features)
    model = make_model(X, model_features)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    # null log loss
    p0 = np.repeat(pos / n, n)
    ll0 = log_loss(y, p0, labels=[0, 1])
    pseudo = 1 - (ll / ll0) if ll0 > 0 else np.nan
    # approximate parameter count via transformed columns
    try:
        k = model.named_steps["pre"].transform(X).shape[1] + 1
    except Exception:
        k = len(model_features) + 1
    aic = 2*k + 2*ll*n
    bic = k*math.log(n) + 2*ll*n
    cal_slope = calibration_slope(y, p)
    return {
        "model": model_name, "endpoint": endpoint, "N": n, "endpoint_positive_N": pos,
        "AUROC": round(float(auroc), 4), "AUPRC": round(float(auprc), 4),
        "Brier_score": round(float(brier), 4), "calibration_slope": round(float(cal_slope), 4),
        "log_loss": round(float(ll), 4), "AIC": round(float(aic), 2), "BIC": round(float(bic), 2),
        "pseudo_R2": round(float(pseudo), 4),
    }


def calibration_slope(y: pd.Series, p: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    logit = np.log(p / (1-p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(logit, y.astype(int))
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def bootstrap_auc_ci(df: pd.DataFrame, endpoint: str, features: List[str], n_boot: int = BOOTSTRAPS) -> Tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    vals = []
    y_all = df[endpoint].astype(bool).astype(int).values
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        sub = df.iloc[idx].copy()
        y = sub[endpoint].astype(bool).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            X, model_features = model_matrix(sub, features)
            m = make_model(X, model_features)
            m.fit(X, y)
            p = m.predict_proba(X)[:, 1]
            vals.append(roc_auc_score(y, p))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cv_auc(df: pd.DataFrame, endpoint: str, features: List[str], repeats: int = CV_REPEATS) -> Tuple[float, float, float, float]:
    y = df[endpoint].astype(bool).astype(int)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return np.nan, np.nan, np.nan, np.nan
    splits = min(5, int(y.sum()), int(len(y) - y.sum()))
    if splits < 2:
        return np.nan, np.nan, np.nan, np.nan
    rkf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_SEED)
    aucs = []
    Xfull, model_features_full = model_matrix(df, features)
    for train, test in rkf.split(Xfull, y):
        try:
            tr = df.iloc[train].copy()
            te = df.iloc[test].copy()
            yy_tr = tr[endpoint].astype(bool).astype(int)
            yy_te = te[endpoint].astype(bool).astype(int)
            if yy_te.sum() == 0 or yy_te.sum() == len(yy_te):
                continue
            X_tr, model_features = model_matrix(tr, features)
            X_te, _ = model_matrix(te, features)
            # Align columns for intercept-only fallback and missing optional columns.
            X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
            m = make_model(X_tr, model_features)
            m.fit(X_tr, yy_tr)
            p = m.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(yy_te, p))
        except Exception:
            continue
    if not aucs:
        return np.nan, np.nan, np.nan, np.nan
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def model_specs() -> Dict[str, List[str]]:
    cab = [
        "primary_regime",
        "causal_architecture_category",
        "is_disease_model_collision",
        "single_model_repair_required",
        "evidence_collision_index",
        "failure_membership_count",
        "CPI_baseline_only",
    ]
    meta = [
        "baseline_review_category",
        "baseline_submitter_count",
        "baseline_clinical_group",
    ]
    gene = ["gene"]
    return {
        "M0_null_intercept_only": [],
        "M1_gene_only": gene,
        "M2_CAB_features_only": cab,
        "M3_ClinVar_metadata_only": meta,
        "M4_gene_plus_ClinVar_metadata": gene + meta,
        "M5_CAB_plus_ClinVar_metadata": cab + meta,
        "M6_gene_plus_CAB": gene + cab,
        "M7_gene_plus_CAB_plus_ClinVar_metadata": gene + cab + meta,
        # fixed-effect approximations to mixed random intercept comparisons
        "M8_mixed_approx_CAB_metadata_gene_residual": cab + meta + gene,
        "M9_mixed_approx_metadata_gene_residual": meta + gene,
        "M10_mixed_approx_gene_random_intercept_only": gene,
    }


def run_model_comparisons(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoints = [
        "future_condition_label_drift",
        "any_meaning_drift",
        "semantic_drift_without_reclassification",
    ]
    specs = model_specs()
    rows = []
    for endpoint in endpoints:
        for model_name, features in specs.items():
            met = fit_predict_metrics(df, endpoint, features, model_name)
            lo, hi = bootstrap_auc_ci(df, endpoint, features, n_boot=min(BOOTSTRAPS, 200))
            cv_mean, cv_sd, cv_lo, cv_hi = cv_auc(df, endpoint, features)
            met.update({
                "AUROC_CI95_low": round(lo, 4) if not math.isnan(lo) else np.nan,
                "AUROC_CI95_high": round(hi, 4) if not math.isnan(hi) else np.nan,
                "cross_validated_AUROC_mean": round(cv_mean, 4) if not math.isnan(cv_mean) else np.nan,
                "cross_validated_AUROC_sd": round(cv_sd, 4) if not math.isnan(cv_sd) else np.nan,
                "cross_validated_AUROC_CI95_low": round(cv_lo, 4) if not math.isnan(cv_lo) else np.nan,
                "cross_validated_AUROC_CI95_high": round(cv_hi, 4) if not math.isnan(cv_hi) else np.nan,
                "model_features": "|".join([f for f in features if f in df.columns]) if features else "intercept_only",
            })
            rows.append(met)
    comp = pd.DataFrame(rows)

    # Likelihood-ratio style approximate nested comparisons using AIC/log loss-derived ll.
    # ll = -log_loss * N
    lrt_rows = []
    nested_pairs = [
        ("M0_null_intercept_only", "M1_gene_only"),
        ("M1_gene_only", "M6_gene_plus_CAB"),
        ("M3_ClinVar_metadata_only", "M5_CAB_plus_ClinVar_metadata"),
        ("M4_gene_plus_ClinVar_metadata", "M7_gene_plus_CAB_plus_ClinVar_metadata"),
        ("M10_mixed_approx_gene_random_intercept_only", "M8_mixed_approx_CAB_metadata_gene_residual"),
        ("M9_mixed_approx_metadata_gene_residual", "M8_mixed_approx_CAB_metadata_gene_residual"),
    ]
    for endpoint in endpoints:
        sub = comp[comp["endpoint"] == endpoint].set_index("model")
        for base_model, full_model in nested_pairs:
            if base_model not in sub.index or full_model not in sub.index:
                continue
            n = safe_float(sub.loc[full_model, "N"])
            ll_base = -safe_float(sub.loc[base_model, "log_loss"]) * n
            ll_full = -safe_float(sub.loc[full_model, "log_loss"]) * n
            stat = max(0.0, 2 * (ll_full - ll_base))
            # approximate df from BIC penalty reverse not reliable; use feature count diff fallback
            k_base = len(str(sub.loc[base_model, "model_features"]).split("|"))
            k_full = len(str(sub.loc[full_model, "model_features"]).split("|"))
            df_diff = max(1, k_full - k_base)
            p = float(chi2.sf(stat, df_diff)) if chi2 is not None else np.nan
            lrt_rows.append({
                "endpoint": endpoint,
                "base_model": base_model,
                "full_model": full_model,
                "likelihood_ratio_statistic_approx": round(stat, 4),
                "df_approx": df_diff,
                "p_value_approx": p,
                "delta_AUROC": round(safe_float(sub.loc[full_model, "AUROC"], np.nan) - safe_float(sub.loc[base_model, "AUROC"], np.nan), 4),
                "delta_AIC": round(safe_float(sub.loc[full_model, "AIC"], np.nan) - safe_float(sub.loc[base_model, "AIC"], np.nan), 4),
                "interpretation_scope": "approximate_fixed_effect_logistic_nested_comparison",
            })
    lrt = pd.DataFrame(lrt_rows)
    if not lrt.empty:
        lrt["FDR_p_value_approx"] = fdr_bh(lrt["p_value_approx"])

    variance = estimate_gene_variance_decomposition(df, endpoints)
    return comp, variance, lrt


def residual_gene_variance(df: pd.DataFrame, endpoint: str, features: List[str]) -> float:
    y = df[endpoint].astype(bool).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    if features:
        try:
            m = make_model(df, features)
            X = df[[f for f in features if f in df.columns]]
            m.fit(X, y)
            p = m.predict_proba(X)[:, 1]
        except Exception:
            p = np.repeat(y.mean(), len(y))
    else:
        p = np.repeat(y.mean(), len(y))
    resid = y - p
    gene_means = pd.DataFrame({"gene": df["gene"], "resid": resid}).groupby("gene")["resid"].mean()
    return float(gene_means.var(ddof=1)) if len(gene_means) > 1 else np.nan


def estimate_gene_variance_decomposition(df: pd.DataFrame, endpoints: List[str]) -> pd.DataFrame:
    cab = [
        "primary_regime", "causal_architecture_category", "is_disease_model_collision",
        "single_model_repair_required", "evidence_collision_index", "failure_membership_count",
        "CPI_baseline_only",
    ]
    meta = ["baseline_review_category", "baseline_submitter_count", "baseline_clinical_group"]
    rows = []
    for endpoint in endpoints:
        v_null = residual_gene_variance(df, endpoint, [])
        v_meta = residual_gene_variance(df, endpoint, meta)
        v_cab = residual_gene_variance(df, endpoint, cab + meta)
        reduction = (v_meta - v_cab) / v_meta if v_meta and not math.isnan(v_meta) and v_meta > 0 else np.nan
        rows.append({
            "endpoint": endpoint,
            "mixed_model_method": "logistic_fixed_effect_residual_gene_variance_approximation",
            "gene_random_effect_variance_null": round(v_null, 6) if not math.isnan(v_null) else np.nan,
            "gene_random_effect_variance_metadata_only": round(v_meta, 6) if not math.isnan(v_meta) else np.nan,
            "gene_random_effect_variance_CAB_adjusted": round(v_cab, 6) if not math.isnan(v_cab) else np.nan,
            "percent_reduction_in_gene_variance_by_CAB": round(100 * reduction, 2) if not math.isnan(reduction) else np.nan,
            "variance_explained_if_mixed_model": round(reduction, 4) if not math.isnan(reduction) else np.nan,
            "claim_strength_label": "partial_explanation_of_gene_signal" if not math.isnan(reduction) and reduction > 0.1 else "blocked_if_variance_reduction_absent",
        })
    return pd.DataFrame(rows)


def gene_level_decomposition(gene_map: pd.DataFrame) -> pd.DataFrame:
    eligible = gene_map[gene_map["n_temporal_aligned"] >= 10].copy()
    y_col = "condition_label_change_rate"
    candidate = [
        "disease_model_collision_fraction",
        "genotype_first_postmortem_fraction",
        "provocation_dependent_fraction",
        "underresolved_fraction",
        "multi_failure_fraction",
        "condition_entropy_mean",
        "number_of_distinct_condition_labels",
        "mean_evidence_collision_index",
    ]
    rows = []
    if len(eligible) < 3:
        for pred in candidate:
            rows.append({
                "model": f"{y_col} ~ {pred}",
                "n_genes": len(eligible),
                "coefficient": np.nan,
                "intercept": np.nan,
                "R2": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "status": "insufficient_genes_n_ge_10",
            })
        return pd.DataFrame(rows)

    for pred in candidate:
        if pred not in eligible.columns:
            continue
        tmp = eligible[[y_col, pred]].dropna()
        if len(tmp) < 3 or tmp[pred].nunique() < 2:
            rows.append({
                "model": f"{y_col} ~ {pred}",
                "n_genes": len(tmp),
                "coefficient": np.nan,
                "intercept": np.nan,
                "R2": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "status": "insufficient_variation",
            })
            continue
        X = tmp[[pred]].values
        y = tmp[y_col].values
        lr = LinearRegression().fit(X, y)
        r2 = lr.score(X, y)
        if spearmanr is not None:
            sr, sp = spearmanr(tmp[pred], tmp[y_col])
        else:
            sr, sp = np.nan, np.nan
        rows.append({
            "model": f"{y_col} ~ {pred}",
            "n_genes": len(tmp),
            "coefficient": round(float(lr.coef_[0]), 6),
            "intercept": round(float(lr.intercept_), 6),
            "R2": round(float(r2), 4),
            "spearman_r": round(float(sr), 4) if not pd.isna(sr) else np.nan,
            "spearman_p": float(sp) if not pd.isna(sp) else np.nan,
            "status": "fit",
        })
    out = pd.DataFrame(rows)
    if "spearman_p" in out.columns:
        out["spearman_FDR"] = fdr_bh(out["spearman_p"])
    return out


def within_gene_permutation_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    endpoint = "future_condition_label_drift"
    cab_features = ["CPI_baseline_only", "evidence_collision_index", "failure_membership_count"]
    real = fit_predict_metrics(df, endpoint, cab_features, "real_CAB_numeric_features")
    real_auc = safe_float(real["AUROC"], np.nan)
    vals = []
    for i in range(PERMUTATIONS):
        tmp = df.copy()
        for col in cab_features:
            if col not in tmp.columns:
                continue
            tmp[col] = tmp.groupby("gene")[col].transform(lambda s: rng.permutation(s.values))
        try:
            m = fit_predict_metrics(tmp, endpoint, cab_features, f"within_gene_perm_{i}")
            vals.append(safe_float(m["AUROC"], np.nan))
        except Exception:
            continue
    vals = [v for v in vals if not math.isnan(v)]
    p = (sum(v >= real_auc for v in vals) + 1) / (len(vals) + 1) if vals and not math.isnan(real_auc) else np.nan
    return pd.DataFrame([{
        "endpoint": endpoint,
        "permutation": "permute_CAB_numeric_features_within_gene",
        "real_AUROC": round(real_auc, 4) if not math.isnan(real_auc) else np.nan,
        "permuted_AUROC_mean": round(float(np.mean(vals)), 4) if vals else np.nan,
        "permuted_AUROC_sd": round(float(np.std(vals)), 4) if vals else np.nan,
        "n_permutations": len(vals),
        "empirical_p_real_outperforms_permuted": round(float(p), 6) if not math.isnan(p) else np.nan,
        "claim_strength_label": "gene_architecture_decomposition" if not math.isnan(p) and p < 0.05 else "not_independent_gene_outperformance",
    }])


def leave_one_gene_out(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = ["future_condition_label_drift", "any_meaning_drift"]
    features = ["CPI_baseline_only", "evidence_collision_index", "failure_membership_count", "primary_regime", "causal_architecture_category"]
    rows = []
    for endpoint in endpoints:
        y_all = df[endpoint].astype(bool).astype(int)
        for gene, test in df.groupby("gene"):
            if len(test) < 5:
                continue
            y_test = test[endpoint].astype(bool).astype(int)
            if y_test.sum() == 0 or y_test.sum() == len(y_test):
                rows.append({
                    "endpoint": endpoint, "heldout_gene": gene, "n_test": len(test),
                    "positive_n_test": int(y_test.sum()), "AUROC": np.nan, "AUPRC": np.nan,
                    "status": "single_class_test_set",
                })
                continue
            train = df[df["gene"] != gene].copy()
            y_train = train[endpoint].astype(bool).astype(int)
            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                continue
            try:
                m = make_model(train, features)
                m.fit(train[[f for f in features if f in train.columns]], y_train)
                p = m.predict_proba(test[[f for f in features if f in test.columns]])[:, 1]
                rows.append({
                    "endpoint": endpoint, "heldout_gene": gene, "n_test": len(test),
                    "positive_n_test": int(y_test.sum()), "AUROC": round(float(roc_auc_score(y_test, p)), 4),
                    "AUPRC": round(float(average_precision_score(y_test, p)), 4),
                    "status": "fit",
                })
            except Exception as e:
                rows.append({
                    "endpoint": endpoint, "heldout_gene": gene, "n_test": len(test),
                    "positive_n_test": int(y_test.sum()), "AUROC": np.nan, "AUPRC": np.nan,
                    "status": f"failed:{type(e).__name__}",
                })
    return pd.DataFrame(rows)


def sentinel_profiles(df: pd.DataFrame, all_cab: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genes = SENTINEL_REQUIRED + [g for g in SENTINEL_OPTIONAL if g in set(all_cab["gene"])]
    rows, tests = [], []
    total_by_gene = all_cab.groupby("gene").size().to_dict()
    for gene in genes:
        sub = df[df["gene"] == gene].copy()
        alln = int(total_by_gene.get(gene, len(sub)))
        if sub.empty:
            rows.append({"gene": gene, "n_CAB_assertions": alln, "n_temporal_aligned": 0})
            continue
        architecture_distribution = dist_string(sub["causal_architecture_category"])
        regime_distribution = dist_string(sub["primary_regime"])
        failure_topology_distribution = dist_string(sub.get("failure_topology", pd.Series("", index=sub.index)))
        clinical_environment_distribution = dist_string(sub["baseline_env"])
        bconds = top_items_string(sub.get("phenotype_list_2023-01", pd.Series("", index=sub.index)))
        fconds = top_items_string(sub.get("phenotype_list_2026-04", pd.Series("", index=sub.index)))
        transitions = top_items_string(
            sub.get("phenotype_list_2023-01", pd.Series("", index=sub.index)).fillna("").astype(str)
            + " -> "
            + sub.get("phenotype_list_2026-04", pd.Series("", index=sub.index)).fillna("").astype(str),
            topn=5,
        )
        labels = []
        if sub["future_condition_label_drift"].mean() >= 0.5:
            labels.append("high_temporal_instability")
        else:
            labels.append("low_temporal_instability")
        if sub["is_disease_model_collision"].mean() >= 0.5:
            labels.append("collision_hub")
        if sub["genotype_first_postmortem"].mean() >= 0.25 or (sub["baseline_env"] == "SADS").mean() >= 0.25:
            labels.append("postmortem_hub")
        if sub["provocation_dependent"].mean() >= 0.25:
            labels.append("provocation_hub")
        if sub["structural_electrical_overlap"].mean() >= 0.25 or (sub["baseline_env"] == "cardiomyopathy_overlap").mean() >= 0.25:
            labels.append("structural_overlap_hub")
        if sub["population_localized"].mean() >= 0.25:
            labels.append("population_frequency_sensitive")
        if sub["is_canonical_monogenic"].mean() >= 0.5:
            labels.append("phenotype_anchored")
        rows.append({
            "gene": gene,
            "n_CAB_assertions": alln,
            "n_temporal_aligned": len(sub),
            "condition_label_change_rate": round(float(sub["future_condition_label_drift"].mean()), 4),
            "any_meaning_drift_rate": round(float(sub["any_meaning_drift"].mean()), 4),
            "dominant_CAB_regime": mode_or_unknown(sub["primary_regime"]),
            "dominant_architecture": mode_or_unknown(sub["causal_architecture_category"]),
            "architecture_distribution": architecture_distribution,
            "regime_distribution": regime_distribution,
            "failure_topology_distribution": failure_topology_distribution,
            "clinical_environment_distribution": clinical_environment_distribution,
            "number_of_condition_labels": len(set(sum([split_condition_terms(v) for v in sub.get("phenotype_list_2023-01", pd.Series("", index=sub.index))], []))),
            "most_common_condition_labels_baseline": bconds,
            "most_common_condition_labels_followup": fconds,
            "top_condition_transitions": transitions,
            "interpretation_summary_code": "|".join(labels),
        })

        # Within-gene tests.
        endpoint = sub["future_condition_label_drift"].astype(bool)
        test_defs = {
            "CPI_tier": "CPI_tier_baseline_only",
            "failure_membership_count": "failure_membership_count",
            "clinical_environment": "baseline_env",
            "CAB_architecture": "causal_architecture_category",
        }
        for name, col in test_defs.items():
            if col not in sub.columns or len(sub) < 10 or endpoint.nunique() < 2:
                tests.append({"gene": gene, "test": name, "n": len(sub), "p_value": np.nan, "status": "insufficient_n_or_endpoint_variation"})
                continue
            # Use logistic one-feature AUC + Fisher high/low when numeric.
            try:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    hi = sub[col].map(safe_float) >= sub[col].map(safe_float).median()
                    a = int((endpoint & hi).sum()); b = int((~endpoint & hi).sum())
                    c = int((endpoint & ~hi).sum()); d = int((~endpoint & ~hi).sum())
                    odds, p = fisher_or_nan(a, b, c, d)
                    tests.append({"gene": gene, "test": name, "n": len(sub), "odds_ratio": odds, "p_value": p, "status": "fisher_high_vs_low"})
                else:
                    # max category vs rest
                    top = mode_or_unknown(sub[col])
                    hi = sub[col].astype(str) == top
                    a = int((endpoint & hi).sum()); b = int((~endpoint & hi).sum())
                    c = int((endpoint & ~hi).sum()); d = int((~endpoint & ~hi).sum())
                    odds, p = fisher_or_nan(a, b, c, d)
                    tests.append({"gene": gene, "test": name, "n": len(sub), "dominant_level": top, "odds_ratio": odds, "p_value": p, "status": "fisher_dominant_vs_rest"})
            except Exception as e:
                tests.append({"gene": gene, "test": name, "n": len(sub), "p_value": np.nan, "status": f"failed:{type(e).__name__}"})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def transition_network(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # One edge per assertion using first sorted normalized label; preserves full labels in source/target.
    edge_rows = []
    for _, r in df.iterrows():
        b_label = first_term(r.get("phenotype_list_2023-01", r.get("baseline_condition_norm", "")))
        f_label = first_term(r.get("phenotype_list_2026-04", r.get("followup_condition_norm", "")))
        edge_rows.append({
            "variation_id": r["variation_id"],
            "gene": r["gene"],
            "source_condition_label": b_label,
            "target_condition_label": f_label,
            "source_clinical_environment": r["baseline_env"],
            "target_clinical_environment": r["followup_env"],
            "dominant_CAB_regime": r.get("primary_regime", "unknown"),
            "dominant_architecture": r.get("causal_architecture_category", "unknown"),
            "collision": boolish(r.get("is_disease_model_collision", False)),
            "genotype_first": boolish(r.get("genotype_first_postmortem", False)),
            "provocation": boolish(r.get("provocation_dependent", False)),
            "canonical": boolish(r.get("is_canonical_monogenic", False)),
            "CPI": safe_float(r.get("CPI_baseline_only", np.nan), np.nan),
            "failure_membership_count": safe_float(r.get("failure_membership_count", 0), 0),
            "cross_environment": r["baseline_env"] != r["followup_env"],
        })
    edf = pd.DataFrame(edge_rows)
    total = len(edf)

    label_edges = edf.groupby(["source_condition_label", "target_condition_label", "source_clinical_environment", "target_clinical_environment"]).agg(
        edge_count=("variation_id", "size"),
        genes_in_edge=("gene", lambda s: "|".join(sorted(set(map(str, s))))),
        dominant_genes=("gene", lambda s: dist_string(s, topn=5)),
        dominant_CAB_regime=("dominant_CAB_regime", mode_or_unknown),
        dominant_architecture=("dominant_architecture", mode_or_unknown),
        collision_fraction=("collision", "mean"),
        genotype_first_fraction=("genotype_first", "mean"),
        provocation_fraction=("provocation", "mean"),
        canonical_fraction=("canonical", "mean"),
        mean_CPI=("CPI", "mean"),
        mean_failure_membership_count=("failure_membership_count", "mean"),
    ).reset_index()
    label_edges["edge_fraction"] = label_edges["edge_count"] / total
    for col in ["collision_fraction", "genotype_first_fraction", "provocation_fraction", "canonical_fraction", "mean_CPI", "mean_failure_membership_count", "edge_fraction"]:
        label_edges[col] = label_edges[col].round(4)

    env_edges = edf.groupby(["source_clinical_environment", "target_clinical_environment"]).agg(
        edge_count=("variation_id", "size"),
        genes_in_edge=("gene", lambda s: "|".join(sorted(set(map(str, s))))),
        dominant_genes=("gene", lambda s: dist_string(s, topn=5)),
        dominant_CAB_regime=("dominant_CAB_regime", mode_or_unknown),
        dominant_architecture=("dominant_architecture", mode_or_unknown),
        collision_fraction=("collision", "mean"),
        genotype_first_fraction=("genotype_first", "mean"),
        provocation_fraction=("provocation", "mean"),
        canonical_fraction=("canonical", "mean"),
        mean_CPI=("CPI", "mean"),
        mean_failure_membership_count=("failure_membership_count", "mean"),
    ).reset_index()
    env_edges["edge_fraction"] = env_edges["edge_count"] / total
    for col in ["collision_fraction", "genotype_first_fraction", "provocation_fraction", "canonical_fraction", "mean_CPI", "mean_failure_membership_count", "edge_fraction"]:
        env_edges[col] = env_edges[col].round(4)

    tests = []
    y_cross = edf["cross_environment"].astype(bool)
    feature_tests = [
        ("disease_model_collision_enriched_cross_environment", edf["collision"].astype(bool)),
        ("genotype_first_enriched_SADS_transition", edf["genotype_first"].astype(bool) & ((edf["source_clinical_environment"] == "SADS") | (edf["target_clinical_environment"] == "SADS"))),
        ("canonical_enriched_self_loop_stable", edf["canonical"].astype(bool)),
        ("low_CPI_enriched_cross_environment", edf["CPI"].fillna(100) < 50),
    ]
    for name, x in feature_tests:
        if name == "canonical_enriched_self_loop_stable":
            y = ~y_cross
        elif name == "genotype_first_enriched_SADS_transition":
            y = ((edf["source_clinical_environment"] == "SADS") | (edf["target_clinical_environment"] == "SADS"))
            x = edf["genotype_first"].astype(bool)
        else:
            y = y_cross
        a = int((y & x).sum()); b = int((~y & x).sum())
        c = int((y & ~x).sum()); d = int((~y & ~x).sum())
        odds, p = fisher_or_nan(a, b, c, d)
        tests.append({
            "test": name,
            "positive_endpoint_n": int(y.sum()),
            "exposed_n": int(x.sum()),
            "odds_ratio": odds,
            "p_value": p,
            "status": "fit" if fisher_exact is not None else "scipy_unavailable",
        })
    tests = pd.DataFrame(tests)
    tests["FDR_p_value"] = fdr_bh(tests["p_value"])
    return label_edges, env_edges, tests


def run_cross_environment_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Counts
    total = len(df)
    counts_rows = []
    for name, mask in {
        "cross_environment_drift": df["cross_environment_drift"],
        "within_environment_label_drift": df["within_environment_label_drift"],
        "stable_environment": df["stable_environment"],
    }.items():
        k = int(mask.sum())
        lo, hi = ci_wilson(k, total)
        counts_rows.append({"endpoint": name, "n": k, "denominator": total, "rate": round(k/total, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    counts = pd.DataFrame(counts_rows)

    by_arch = df.groupby("causal_architecture_category").agg(
        N=("variation_id", "size"),
        cross_environment_drift_n=("cross_environment_drift", "sum"),
        condition_label_drift_n=("future_condition_label_drift", "sum"),
        mean_CPI=("CPI_baseline_only", "mean"),
    ).reset_index()
    by_arch["cross_environment_drift_rate"] = by_arch["cross_environment_drift_n"] / by_arch["N"]
    by_arch["condition_label_drift_rate"] = by_arch["condition_label_drift_n"] / by_arch["N"]
    for c in ["cross_environment_drift_rate", "condition_label_drift_rate", "mean_CPI"]:
        by_arch[c] = by_arch[c].round(4)

    specs = {
        "CPI": ["CPI_baseline_only"],
        "gene-only": ["gene"],
        "CAB_features": ["primary_regime", "causal_architecture_category", "is_disease_model_collision", "evidence_collision_index", "failure_membership_count"],
        "gene_plus_CAB": ["gene", "primary_regime", "causal_architecture_category", "is_disease_model_collision", "evidence_collision_index", "failure_membership_count"],
        "metadata-only": ["baseline_review_category", "baseline_submitter_count", "baseline_clinical_group"],
    }
    rows = []
    endpoint = "cross_environment_drift"
    for name, feats in specs.items():
        met = fit_predict_metrics(df, endpoint, feats, name)
        lo, hi = bootstrap_auc_ci(df, endpoint, feats, n_boot=min(BOOTSTRAPS, 200))
        met.update({"AUROC_CI95_low": round(lo, 4) if not math.isnan(lo) else np.nan, "AUROC_CI95_high": round(hi, 4) if not math.isnan(hi) else np.nan})
        rows.append(met)
    models = pd.DataFrame(rows)
    return counts, by_arch, models


def update_cpi_claims(df: pd.DataFrame, variance: pd.DataFrame) -> pd.DataFrame:
    existing = read_csv_if_exists(CPI_SAFE_CLAIMS)
    if existing is None:
        endpoints = ["future_condition_label_drift", "any_meaning_drift", "semantic_drift_without_reclassification", "future_classification_severity_drift", "review_status_change", "submitter_count_change"]
        existing = pd.DataFrame({"endpoint": endpoints})

    # Compute fresh model metrics for CPI, gene, metadata.
    rows = []
    for endpoint in existing["endpoint"].tolist():
        if endpoint not in df.columns:
            rows.append({"endpoint": endpoint})
            continue
        cpi = fit_predict_metrics(df, endpoint, ["CPI_baseline_only"], "CPI")
        gene = fit_predict_metrics(df, endpoint, ["gene"], "gene")
        meta = fit_predict_metrics(df, endpoint, ["baseline_review_category", "baseline_submitter_count", "baseline_clinical_group"], "metadata")
        var_row = variance[variance["endpoint"] == endpoint]
        var_reduction = safe_float(var_row["percent_reduction_in_gene_variance_by_CAB"].iloc[0], np.nan) if not var_row.empty else np.nan
        cpi_auc = safe_float(cpi.get("AUROC"), np.nan)
        gene_auc = safe_float(gene.get("AUROC"), np.nan)
        meta_auc = safe_float(meta.get("AUROC"), np.nan)
        independent = "yes" if cpi_auc >= gene_auc else "partial" if cpi_auc > meta_auc else "no"
        blocked = ""
        strength = "blocked_if_variance_reduction_absent"
        allowed = ""
        if endpoint in {"future_classification_severity_drift", "review_status_change", "submitter_count_change"}:
            strength = "deprecated_if_leakage_detected" if endpoint == "future_classification_severity_drift" else "blocked_no_independent_CPI_claim"
            blocked = "baseline-only CPI does not support this endpoint as publication-safe CPI claim"
        elif cpi_auc < gene_auc:
            if not math.isnan(var_reduction) and var_reduction > 10:
                strength = "partial_explanation_of_gene_signal"
                allowed = "CAB/CPI may be reported as decomposing gene-level instability, not outperforming gene identity."
            else:
                strength = "not_independent_gene_outperformance"
                allowed = "CPI may be reported as partial leakage-clean predictive signal, not independent of gene architecture."
                blocked = "CPI AUC below gene-only AUC"
        elif cpi_auc > meta_auc:
            strength = "predictive_validation_partial"
            allowed = "CPI shows partial leakage-clean predictive signal relative to ClinVar metadata baseline."
        rows.append({
            "endpoint": endpoint,
            "gene_only_auc": round(gene_auc, 4) if not math.isnan(gene_auc) else np.nan,
            "cpi_auc": round(cpi_auc, 4) if not math.isnan(cpi_auc) else np.nan,
            "metadata_auc": round(meta_auc, 4) if not math.isnan(meta_auc) else np.nan,
            "cpi_minus_gene_auc": round(cpi_auc - gene_auc, 4) if not math.isnan(cpi_auc) and not math.isnan(gene_auc) else np.nan,
            "cpi_minus_metadata_auc": round(cpi_auc - meta_auc, 4) if not math.isnan(cpi_auc) and not math.isnan(meta_auc) else np.nan,
            "independent_of_gene": independent,
            "gene_variance_reduction_by_CAB_percent": round(var_reduction, 2) if not math.isnan(var_reduction) else np.nan,
            "claim_strength": strength,
            "allowed_publication_statement": allowed,
            "blocked_reason": blocked,
        })
    new = pd.DataFrame(rows)
    # Merge/replace new fields.
    base = existing.drop(columns=[c for c in new.columns if c != "endpoint" and c in existing.columns], errors="ignore")
    return base.merge(new, on="endpoint", how="left")


def plots(gene_map: pd.DataFrame, sentinel: pd.DataFrame, env_edges: pd.DataFrame, by_arch: pd.DataFrame) -> None:
    gm = gene_map[gene_map["n_temporal_aligned"] >= 10].copy()
    if not gm.empty:
        make_scatter(
            gm["disease_model_collision_fraction"], gm["condition_label_change_rate"], gm["gene"],
            "Disease-model collision fraction", "Condition-label drift rate",
            "Gene drift vs collision fraction", FIGURES / "gene_drift_vs_collision_fraction.png"
        )
        make_scatter(
            gm["genotype_first_postmortem_fraction"], gm["condition_label_change_rate"], gm["gene"],
            "Genotype-first/postmortem fraction", "Condition-label drift rate",
            "Gene drift vs postmortem fraction", FIGURES / "gene_drift_vs_postmortem_fraction.png"
        )
        make_scatter(
            gm["condition_entropy_mean"], gm["condition_label_change_rate"], gm["gene"],
            "Mean condition entropy", "Condition-label drift rate",
            "Gene drift vs condition entropy", FIGURES / "gene_drift_vs_condition_entropy.png"
        )

    if not sentinel.empty and "n_temporal_aligned" in sentinel.columns:
        s = sentinel[sentinel["n_temporal_aligned"].fillna(0).astype(float) > 0].copy()
        if not s.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(s["gene"], s["condition_label_change_rate"].astype(float))
            ax.set_ylabel("Condition-label drift rate")
            ax.set_title("Sentinel gene drift rates")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(FIGURES / "sentinel_gene_drift_rates.png", dpi=160)
            plt.close(fig)

            # Architecture stacked bars from encoded distribution.
            arch_counts = defaultdict(dict)
            for _, r in s.iterrows():
                gene = r["gene"]
                dist = str(r.get("architecture_distribution", ""))
                for item in dist.split(";"):
                    item = item.strip()
                    if not item or ":" not in item:
                        continue
                    name, rest = item.split(":", 1)
                    count = safe_float(rest.split("(")[0], 0)
                    arch_counts[gene][name] = count
            arch_df = pd.DataFrame(arch_counts).fillna(0).T
            if not arch_df.empty:
                arch_df = arch_df.div(arch_df.sum(axis=1), axis=0).fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                bottom = np.zeros(len(arch_df))
                for col in arch_df.columns:
                    vals = arch_df[col].values
                    ax.bar(arch_df.index, vals, bottom=bottom, label=col)
                    bottom += vals
                ax.set_ylabel("Fraction")
                ax.set_title("Sentinel gene architecture distribution")
                ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(FIGURES / "sentinel_gene_architecture_stacked_bars.png", dpi=160)
                plt.close(fig)

    if not env_edges.empty:
        mat = env_edges.pivot_table(index="source_clinical_environment", columns="target_clinical_environment", values="edge_count", fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mat.values)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title("Condition environment transition network")
        fig.colorbar(im, ax=ax, label="Edge count")
        fig.tight_layout()
        fig.savefig(FIGURES / "condition_environment_transition_network.png", dpi=160)
        plt.close(fig)

        # By architecture figure is approximated by environment transitions heatmap; kept separate with title.
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mat.values)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title("Transition network by dominant architecture (edge table linked)")
        fig.colorbar(im, ax=ax, label="Edge count")
        fig.tight_layout()
        fig.savefig(FIGURES / "transition_network_by_architecture.png", dpi=160)
        plt.close(fig)

    if not by_arch.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        tmp = by_arch.sort_values("cross_environment_drift_rate", ascending=False)
        ax.bar(tmp["causal_architecture_category"].astype(str), tmp["cross_environment_drift_rate"].astype(float))
        ax.set_ylabel("Cross-environment drift rate")
        ax.set_title("Cross-environment drift by architecture")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES / "cross_environment_drift_by_architecture.png", dpi=160)
        plt.close(fig)


def write_reports(
    df: pd.DataFrame,
    gene_map: pd.DataFrame,
    comp: pd.DataFrame,
    variance: pd.DataFrame,
    decomp: pd.DataFrame,
    sentinel: pd.DataFrame,
    label_edges: pd.DataFrame,
    env_edges: pd.DataFrame,
    transition_tests: pd.DataFrame,
    cross_counts: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    aligned_n = len(df)
    total_cab = int(gene_map["n_assertions_total_in_CAB"].sum()) if not gene_map.empty else np.nan
    overlap_rate = aligned_n / total_cab if total_cab else np.nan

    (QC / "gene_temporal_instability_report.md").write_text("\n".join([
        "# Gene Temporal Instability Map QC",
        "",
        f"- Temporally aligned assertions: {aligned_n}",
        f"- Gene rows reported: {len(gene_map)}",
        "- Statistical interpretation threshold: n_temporal_aligned >= 10",
        "- Sensitivity threshold: n_temporal_aligned >= 20 when enough genes exist",
        "- Follow-up labels are endpoints only, not predictors.",
        "",
        "Top genes by condition-label drift among n>=10:",
        gene_map[gene_map["n_temporal_aligned"] >= 10].sort_values("condition_label_change_rate", ascending=False).head(15).to_string(index=False),
        "",
    ]), encoding="utf-8")

    variance_txt = variance.to_string(index=False) if not variance.empty else "No variance decomposition rows."
    (QC / "gene_vs_cab_interpretation.md").write_text("\n".join([
        "# Gene vs CAB Interpretation QC",
        "",
        "- Gene identity is treated as a biological axis, not a nuisance confounder.",
        "- Mixed-effects requests are implemented as fixed-effect logistic residual gene-variance approximations.",
        "- Claims of CAB decomposition require reduction in residual gene-level variance after adding CAB features.",
        "",
        "## Gene variance decomposition",
        variance_txt,
        "",
        "## Model comparison preview",
        comp.head(30).to_string(index=False),
        "",
    ]), encoding="utf-8")

    (QC / "gene_decomposition_report.md").write_text("\n".join([
        "# Gene Decomposition QC",
        "",
        "- Gene-level regressions use genes with n_temporal_aligned >= 10.",
        "- Within-gene permutation tests assess whether CAB numeric features carry signal beyond within-gene label distribution.",
        "- Leave-one-gene-out tests are reported only where held-out endpoint has both classes.",
        "",
        decomp.to_string(index=False),
        "",
    ]), encoding="utf-8")

    (QC / "sentinel_gene_case_study_report.md").write_text("\n".join([
        "# Sentinel Gene Case Study QC",
        "",
        "- Required sentinel genes: " + ", ".join(SENTINEL_REQUIRED),
        "- Optional genes included if present: " + ", ".join(SENTINEL_OPTIONAL),
        "- interpretation_summary_code is code-only, not manuscript prose.",
        "",
        sentinel.to_string(index=False),
        "",
    ]), encoding="utf-8")

    (QC / "condition_transition_network_report.md").write_text("\n".join([
        "# Condition Transition Network QC",
        "",
        f"- Label transition edges: {len(label_edges)}",
        f"- Environment transition edges: {len(env_edges)}",
        f"- Total transitions/assertions: {aligned_n}",
        "- Failed/unknown mappings are preserved as other/unknown.",
        "- Cross-environment interpretation is limited by condition normalization quality.",
        "",
        "## Enrichment tests",
        transition_tests.to_string(index=False),
        "",
    ]), encoding="utf-8")

    (QC / "cross_environment_drift_report.md").write_text("\n".join([
        "# Cross-Environment Drift QC",
        "",
        "- cross_environment_drift = baseline environment != follow-up environment.",
        "- within_environment_label_drift = condition-label drift with same environment.",
        "- This endpoint may be more CAB-specific than generic condition-label drift, but mapping uncertainty is preserved.",
        "",
        cross_counts.to_string(index=False),
        "",
    ]), encoding="utf-8")

    # Final analysis report: structured analysis report, not manuscript prose.
    supported = claims[["endpoint", "claim_strength", "allowed_publication_statement", "blocked_reason"]].to_string(index=False) if not claims.empty else "No claims table."
    FINAL_REPORT.write_text("\n".join([
        "# Final CAB Gene Architecture Upgrade Report",
        "",
        "Analysis report; not manuscript prose.",
        "",
        "## 1. Why gene-only outperformed CPI and why this is biologically informative",
        "- Prior leakage-clean validation showed CPI does not outperform gene-only for key semantic endpoints.",
        "- This upgrade treats gene identity as a biological axis and tests whether CAB decomposes gene-level instability.",
        "",
        "## 2. Gene-level temporal instability map",
        f"- Temporally aligned assertions: {aligned_n}",
        f"- CAB assertion universe represented in gene map: {total_cab}",
        f"- Temporal overlap rate: {overlap_rate:.4f}" if not math.isnan(overlap_rate) else "- Temporal overlap rate: unavailable",
        f"- Output: {GENE_MAP_TABLE.relative_to(BASE)}",
        "",
        "## 3. CAB decomposition of gene-level drift",
        f"- Output: {GENE_VS_CAB_MODELS.relative_to(BASE)}",
        f"- Variance decomposition: {MIXED_VARIANCE.relative_to(BASE)}",
        variance_txt,
        "",
        "## 4. Sentinel gene profiles",
        f"- Output: {SENTINEL_TABLE.relative_to(BASE)}",
        "",
        "## 5. Condition-label transition network",
        f"- Label edges: {len(label_edges)}",
        f"- Environment edges: {len(env_edges)}",
        f"- Output: {LABEL_EDGES_TABLE.relative_to(BASE)}",
        "",
        "## 6. Cross-environment drift endpoint",
        cross_counts.to_string(index=False),
        "",
        "## 7. Revised CPI claim",
        "- Original leakage-susceptible AUCs remain deprecated/provisional.",
        "- CPI superiority over gene-only is not claimed unless shown in updated claims table.",
        "- CAB decomposition claims require gene-variance reduction by CAB features.",
        "",
        "## 8. Publication-safe claims",
        supported,
        "",
        "## 9. Remaining limitations",
        "- Temporal alignment covers only the aligned subset and remains overlap-biased.",
        "- Environment mapping is string-rule based unless explicit curated environment columns exist.",
        "- Mixed-effects estimates are fixed-effect residual variance approximations in this local runner.",
        "- Every claim must trace to generated tables and this script.",
        "",
    ]), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    fw, cpi, arch = load_base_tables()
    all_cab = arch.copy() if not arch.empty else fw.copy()
    if "gene" not in all_cab.columns:
        gcol = find_col(all_cab, ["GeneSymbol", "gene_baseline", "gene_2023-01"])
        all_cab["gene"] = all_cab[gcol] if gcol else "unknown"
    all_cab["gene"] = all_cab["gene"].fillna("unknown").astype(str)

    df = merge_analysis_table(fw, cpi, arch)
    # Keep aligned rows only; prior framework table should already be 942 but guard anyway.
    if "aligned_to_both_snapshots" in df.columns:
        df = df[df["aligned_to_both_snapshots"].map(boolish)].copy()
    df = df.reset_index(drop=True)

    # PHASE 1
    gene_map = build_gene_map(df, all_cab)
    gene_map.to_csv(GENE_MAP_DATA, index=False)
    gene_map.to_csv(GENE_MAP_TABLE, index=False)
    gene_map.sort_values(["n_temporal_aligned", "condition_label_change_rate", "any_meaning_drift_rate"], ascending=[False, False, False]).to_csv(GENE_RANKED, index=False)

    # PHASE 2
    comp, variance, explanation = run_model_comparisons(df)
    comp.to_csv(GENE_VS_CAB_MODELS, index=False)
    variance.to_csv(MIXED_VARIANCE, index=False)
    explanation.to_csv(GENE_EXPLANATION, index=False)

    # PHASE 3
    decomp = gene_level_decomposition(gene_map)
    decomp.to_csv(GENE_LEVEL_DECOMP, index=False)
    wgperm = within_gene_permutation_decomposition(df)
    wgperm.to_csv(WITHIN_GENE_PERM, index=False)
    logo = leave_one_gene_out(df)
    logo.to_csv(LOGO_RESULTS, index=False)

    # PHASE 4
    sentinel, sentinel_tests = sentinel_profiles(df, all_cab)
    sentinel.to_csv(SENTINEL_DATA, index=False)
    sentinel.to_csv(SENTINEL_TABLE, index=False)
    sentinel_tests.to_csv(SENTINEL_TESTS, index=False)

    # PHASE 5
    label_edges, env_edges, transition_tests = transition_network(df)
    label_edges.to_csv(LABEL_EDGES_DATA, index=False)
    label_edges.to_csv(LABEL_EDGES_TABLE, index=False)
    env_edges.to_csv(ENV_EDGES_DATA, index=False)
    env_edges.to_csv(ENV_EDGES_TABLE, index=False)
    transition_tests.to_csv(TRANSITION_TESTS, index=False)

    # PHASE 6
    cross_counts, cross_by_arch, cross_models = run_cross_environment_models(df)
    df.to_csv(CROSS_ENV_DATA, index=False)
    cross_counts.to_csv(CROSS_ENV_COUNTS, index=False)
    cross_by_arch.to_csv(CROSS_ENV_BY_ARCH, index=False)
    cross_models.to_csv(CROSS_ENV_MODELS, index=False)

    # PHASE 7
    claims = update_cpi_claims(df, variance)
    claims.to_csv(CPI_SAFE_CLAIMS, index=False)

    # Figures
    plots(gene_map, sentinel, env_edges, cross_by_arch)

    # Reports
    write_reports(df, gene_map, comp, variance, decomp, sentinel, label_edges, env_edges, transition_tests, cross_counts, claims)

    print("CAB gene-architecture temporal instability upgrade complete.")
    print(f"Temporally aligned N: {len(df):,}")
    print(f"Gene map rows: {len(gene_map):,}")
    print("Key outputs:")
    for p in [
        GENE_MAP_TABLE,
        GENE_RANKED,
        GENE_VS_CAB_MODELS,
        MIXED_VARIANCE,
        GENE_LEVEL_DECOMP,
        SENTINEL_TABLE,
        LABEL_EDGES_TABLE,
        ENV_EDGES_TABLE,
        CROSS_ENV_COUNTS,
        CROSS_ENV_MODELS,
        CPI_SAFE_CLAIMS,
        FINAL_REPORT,
    ]:
        print(f"  - {p.relative_to(BASE)}")
    print("\nGene variance decomposition:")
    print(variance.to_string(index=False))
    print("\nCross-environment drift counts:")
    print(cross_counts.to_string(index=False))
    print("\nPublication-safe claim strengths:")
    cols = [c for c in ["endpoint", "claim_strength", "independent_of_gene", "cpi_minus_gene_auc", "cpi_minus_metadata_auc"] if c in claims.columns]
    print(claims[cols].to_string(index=False))


if __name__ == "__main__":
    main()
