#!/usr/bin/env python3
"""CAB non-cardiovascular replication: hereditary cancer predisposition.

Builds a baseline-only replication test for assertion portability / temporal
meaning drift outside cardiovascular genetics.

No manuscript prose. No future-label predictors. No clinical actionability,
expert validation, variant reclassification, or all-disease generalization claims.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

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

BASELINE = DATA / "clinvar_snapshot_baseline_202301.csv"
FOLLOWUP = DATA / "clinvar_snapshot_followup_202604.csv"

OUT_MASTER = DATA / "cancer_assertion_master.csv"
OUT_ALIGN = DATA / "cancer_temporal_alignment.csv"
OUT_ENV = DATA / "cancer_condition_environment_map.csv"
OUT_REGIMES = DATA / "cancer_baseline_portability_regimes.csv"
OUT_RULES = QC / "cancer_regime_rules.md"
OUT_COUNTS = TABLES / "cancer_temporal_endpoint_counts.csv"
OUT_GENE = TABLES / "cancer_drift_by_gene.csv"
OUT_REGIME_SUM = TABLES / "cancer_drift_by_regime.csv"
OUT_MODELS = TABLES / "cancer_model_comparison.csv"
OUT_LR = TABLES / "cancer_model_comparison_lr_tests.csv"
OUT_ENRICH = TABLES / "cancer_regime_enrichment_tests.csv"
OUT_THREE = TABLES / "three_domain_portability_summary.csv"
OUT_FINAL = REPORTS / "final_noncardio_replication_report.md"

TARGET_GENES = [
    "BRCA1", "BRCA2", "TP53", "PTEN", "APC", "MLH1", "MSH2", "MSH6",
    "PMS2", "EPCAM", "PALB2", "ATM", "CHEK2", "CDH1", "STK11",
    "SMAD4", "BMPR1A", "MUTYH",
]
MMR = {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}
MODERATE_RISK = {"CHEK2", "ATM", "PALB2"}
SYNDROME_ENV = {
    "TP53": "Li-Fraumeni syndrome",
    "PTEN": "PTEN hamartoma tumor syndrome / Cowden",
    "APC": "colorectal cancer / polyposis",
    "STK11": "syndromic cancer predisposition",
    "SMAD4": "colorectal cancer / polyposis",
    "BMPR1A": "colorectal cancer / polyposis",
    "CDH1": "gastric cancer predisposition",
    "MUTYH": "colorectal cancer / polyposis",
    "MLH1": "Lynch syndrome / mismatch repair cancer predisposition",
    "MSH2": "Lynch syndrome / mismatch repair cancer predisposition",
    "MSH6": "Lynch syndrome / mismatch repair cancer predisposition",
    "PMS2": "Lynch syndrome / mismatch repair cancer predisposition",
    "EPCAM": "Lynch syndrome / mismatch repair cancer predisposition",
}
RANDOM_STATE = 42
N_BOOT = 300


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC]:
        p.mkdir(parents=True, exist_ok=True)


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def first_nonempty(vals: Iterable[object]) -> str:
    for v in vals:
        if not pd.isna(v) and str(v).strip():
            return str(v).strip()
    return ""


def coalesce_to_column(df: pd.DataFrame, target: str, candidates: List[str], default="") -> None:
    """Create target from first non-empty candidate column, robust to pandas merge suffixes."""
    if target in df.columns:
        return
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df[target] = default
        return
    out = pd.Series(default, index=df.index, dtype="object")
    for c in cols:
        vals = df[c]
        mask = out.isna() | out.astype(str).eq("")
        out = out.where(~mask, vals)
    df[target] = out.fillna(default)


def normalize_core_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize core columns after merges that may create _x/_y columns."""
    coalesce_to_column(df, "gene", ["gene", "gene_x", "gene_y", "gene_baseline", "gene_followup"], "")
    coalesce_to_column(df, "condition_label_baseline", ["condition_label_baseline", "condition_label_baseline_x", "condition_label_baseline_y"], "")
    coalesce_to_column(df, "condition_label_followup", ["condition_label_followup", "condition_label_followup_x", "condition_label_followup_y"], "")
    coalesce_to_column(df, "environment_baseline", ["environment_baseline", "environment_baseline_x", "environment_baseline_y"], "")
    coalesce_to_column(df, "environment_followup", ["environment_followup", "environment_followup_x", "environment_followup_y"], "")
    coalesce_to_column(df, "classification_baseline", ["classification_baseline", "classification_baseline_x", "classification_baseline_y"], "")
    coalesce_to_column(df, "classification_followup", ["classification_followup", "classification_followup_x", "classification_followup_y"], "")
    coalesce_to_column(df, "review_status_baseline", ["review_status_baseline", "review_status_baseline_x", "review_status_baseline_y"], "")
    coalesce_to_column(df, "review_status_followup", ["review_status_followup", "review_status_followup_x", "review_status_followup_y"], "")
    coalesce_to_column(df, "submitter_count_baseline", ["submitter_count_baseline", "submitter_count_baseline_x", "submitter_count_baseline_y"], np.nan)
    coalesce_to_column(df, "submitter_count_followup", ["submitter_count_followup", "submitter_count_followup_x", "submitter_count_followup_y"], np.nan)
    coalesce_to_column(df, "HGVS", ["HGVS", "HGVS_x", "HGVS_y", "HGVS_baseline", "HGVS_followup"], "")
    coalesce_to_column(df, "consequence", ["consequence", "consequence_x", "consequence_y"], "")
    return df



def join_unique(vals: Iterable[object]) -> str:
    out, seen = [], set()
    for v in vals:
        if pd.isna(v):
            continue
        for part in re.split(r"[|;]", str(v)):
            p = part.strip()
            if p and p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
    return ";".join(out)


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
    parts = [p.strip() for p in norm_text(x).replace("|", ";").split(";") if p.strip()]
    return ";".join(sorted(set(parts)))


def split_terms(x) -> List[str]:
    return [p.strip() for p in norm_text(x).replace("|", ";").split(";") if p.strip()]


def cancer_environment(label) -> str:
    t = norm_text(label)
    if not t:
        return "other/unknown"
    if re.search(r"somatic|therapeutic response|drug response|oncology response", t):
        return "other/unknown"
    if re.search(r"li[- ]fraumeni|\blfs\b", t):
        return "Li-Fraumeni syndrome"
    if re.search(r"lynch|mismatch repair|muir[- ]torre|constitutional mismatch repair|cmmrd", t):
        return "Lynch syndrome / mismatch repair cancer predisposition"
    if re.search(r"pten hamartoma|cowden|bannayan|proteus", t):
        return "PTEN hamartoma tumor syndrome / Cowden"
    if re.search(r"familial adenomatous polyposis|\bfap\b|polyposis|juvenile polyposis|mutyh[- ]associated|colorectal", t):
        return "colorectal cancer / polyposis"
    if re.search(r"hereditary diffuse gastric|gastric cancer|stomach cancer", t):
        return "gastric cancer predisposition"
    if re.search(r"peutz[- ]jeghers", t):
        return "syndromic cancer predisposition"
    if re.search(r"breast[- ]ovarian|breast and ovarian|hereditary breast ovarian|hereditary breast and ovarian|\bhboc\b", t):
        return "hereditary breast/ovarian cancer"
    if "ovarian" in t:
        return "ovarian cancer predisposition"
    if "breast" in t:
        return "breast cancer predisposition"
    if "pancreatic" in t:
        return "pancreatic cancer predisposition"
    if re.search(r"moderate[- ]risk|cancer susceptibility|predisposition to cancer", t):
        return "moderate-risk cancer susceptibility"
    if re.search(r"hereditary cancer|cancer predisposition|malignant tumor|neoplasm|cancer", t):
        return "pan-cancer / nonspecific cancer predisposition"
    if "syndrome" in t or "syndromic" in t:
        return "syndromic cancer predisposition"
    return "other/unknown"


def env_set(label) -> set[str]:
    envs = {cancer_environment(label)}
    for term in split_terms(label):
        envs.add(cancer_environment(term))
    return {e for e in envs if e}


def is_relevant_condition(label, gene: str) -> bool:
    t = norm_text(label)
    if re.search(r"somatic|therapeutic response|drug response", t):
        return False
    if cancer_environment(label) != "other/unknown":
        return True
    return gene in TARGET_GENES and re.search(r"cancer|tumou?r|neoplasm|predisposition|susceptibility|syndrome|polyposis", t) is not None


def broad_or_ambiguous(label) -> bool:
    t = norm_text(label)
    env = cancer_environment(label)
    return any(k in t for k in ["cancer predisposition", "cancer susceptibility", "hereditary cancer", "neoplasm", "not provided", "unknown"]) and env in {
        "pan-cancer / nonspecific cancer predisposition",
        "moderate-risk cancer susceptibility",
        "other/unknown",
    }


def consequence_from_hgvs(hgvs) -> str:
    s = str(hgvs)
    if re.search(r"\(p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\)", s):
        return "missense"
    if "fs" in s or "Ter" in s or "*" in s:
        return "protein_truncating_or_frameshift"
    if "del" in s or "dup" in s or "ins" in s:
        return "indel_or_copy_change_like"
    if "(p." in s:
        return "protein_change_other"
    return "unavailable"


def load_snapshot(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in [
        "VariationID", "AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance",
        "ReviewStatus", "PhenotypeList", "PhenotypeIDS", "NumberSubmitters",
        "ReferenceAllele", "AlternateAllele", "LastEvaluated", "Origin", "OriginSimple"
    ] if c in header]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=250_000):
        chunk["gene"] = chunk["GeneSymbol"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["gene"].isin(TARGET_GENES)].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["ClinicalSignificance"].map(clinical_group).eq("p_lp")].copy()
        if chunk.empty:
            continue
        origin = chunk.get("Origin", pd.Series("", index=chunk.index)).fillna("").astype(str).str.lower()
        origin_simple = chunk.get("OriginSimple", pd.Series("", index=chunk.index)).fillna("").astype(str).str.lower()
        pheno = chunk.get("PhenotypeList", pd.Series("", index=chunk.index)).fillna("").astype(str).str.lower()
        somatic = origin.str.contains("somatic", na=False) | origin_simple.str.contains("somatic", na=False) | pheno.str.contains("somatic|therapeutic response|drug response", regex=True, na=False)
        chunk = chunk[~somatic].copy()
        if chunk.empty:
            continue
        chunk["condition_relevant"] = chunk.apply(lambda r: is_relevant_condition(r.get("PhenotypeList", ""), r["gene"]), axis=1)
        chunk = chunk[chunk["condition_relevant"]].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame({"variation_id": []})
    df = pd.concat(chunks, ignore_index=True)
    df["variation_id"] = df["VariationID"].map(norm_id)
    df = df[df["variation_id"] != ""].copy()
    agg = df.groupby("variation_id", as_index=False).agg(
        allele_id=("AlleleID", first_nonempty) if "AlleleID" in df.columns else ("VariationID", first_nonempty),
        gene=("gene", first_nonempty),
        variant_type=("Type", first_nonempty) if "Type" in df.columns else ("VariationID", first_nonempty),
        HGVS=("Name", first_nonempty) if "Name" in df.columns else ("VariationID", first_nonempty),
        condition_label=("PhenotypeList", join_unique) if "PhenotypeList" in df.columns else ("VariationID", first_nonempty),
        phenotype_ids=("PhenotypeIDS", join_unique) if "PhenotypeIDS" in df.columns else ("VariationID", first_nonempty),
        classification=("ClinicalSignificance", first_nonempty),
        review_status=("ReviewStatus", first_nonempty),
        submitter_count=("NumberSubmitters", "max") if "NumberSubmitters" in df.columns else ("VariationID", "size"),
        reference_allele=("ReferenceAllele", first_nonempty) if "ReferenceAllele" in df.columns else ("VariationID", first_nonempty),
        alternate_allele=("AlternateAllele", first_nonempty) if "AlternateAllele" in df.columns else ("VariationID", first_nonempty),
        last_evaluated=("LastEvaluated", first_nonempty) if "LastEvaluated" in df.columns else ("VariationID", first_nonempty),
        n_snapshot_rows=("VariationID", "size"),
    )
    agg["environment"] = agg["condition_label"].map(cancer_environment)
    return agg.rename(columns={c: f"{c}_{suffix}" for c in agg.columns if c != "variation_id"})


def build_cohort() -> pd.DataFrame:
    baseline = load_snapshot(BASELINE, "baseline")
    followup = load_snapshot(FOLLOWUP, "followup")
    merged = baseline.merge(followup, on="variation_id", how="outer", indicator=True)
    df = merged[merged["_merge"].eq("both")].copy()
    df["assertion_id"] = "CANCER_" + df["variation_id"].astype(str)
    df["gene"] = df.get("gene_baseline", "").fillna(df.get("gene_followup", ""))
    df["condition_label_baseline"] = df.get("condition_label_baseline", "")
    df["condition_label_followup"] = df.get("condition_label_followup", "")
    df["environment_baseline"] = df["condition_label_baseline"].map(cancer_environment)
    df["environment_followup"] = df["condition_label_followup"].map(cancer_environment)
    df["classification_baseline"] = df.get("classification_baseline", "")
    df["classification_followup"] = df.get("classification_followup", "")
    df["review_status_baseline"] = df.get("review_status_baseline", "")
    df["review_status_followup"] = df.get("review_status_followup", "")
    df["submitter_count_baseline"] = pd.to_numeric(df.get("submitter_count_baseline", np.nan), errors="coerce")
    df["submitter_count_followup"] = pd.to_numeric(df.get("submitter_count_followup", np.nan), errors="coerce")
    df["HGVS"] = df.get("HGVS_baseline", df.get("HGVS_followup", ""))
    df["consequence"] = df["HGVS"].map(consequence_from_hgvs)
    df["date_baseline"] = "2023-01"
    df["date_followup"] = "2026-04"
    df["condition_label_baseline_norm"] = df["condition_label_baseline"].map(normalize_condition)
    df["condition_label_followup_norm"] = df["condition_label_followup"].map(normalize_condition)
    df["classification_change"] = df["classification_baseline"].map(clinical_group) != df["classification_followup"].map(clinical_group)
    df["condition_label_change"] = df["condition_label_baseline_norm"] != df["condition_label_followup_norm"]
    df["cross_environment_drift"] = df["environment_baseline"] != df["environment_followup"]
    df["within_environment_label_drift"] = df["condition_label_change"] & ~df["cross_environment_drift"]
    df["self_loop_stable"] = ~df["cross_environment_drift"]
    df["review_status_change"] = df["review_status_baseline"].map(review_category) != df["review_status_followup"].map(review_category)
    df["submitter_count_change"] = df["submitter_count_baseline"].fillna(-1) != df["submitter_count_followup"].fillna(-1)
    df["any_meaning_drift"] = df["condition_label_change"] | df["classification_change"] | df["review_status_change"]
    df["semantic_drift_without_reclassification"] = df["condition_label_change"] & ~df["classification_change"]
    keep = [
        "assertion_id", "variation_id", "gene", "condition_label_baseline", "condition_label_followup",
        "environment_baseline", "environment_followup", "classification_baseline", "classification_followup",
        "review_status_baseline", "review_status_followup", "submitter_count_baseline", "submitter_count_followup",
        "HGVS", "consequence", "date_baseline", "date_followup", "classification_change", "condition_label_change",
        "cross_environment_drift", "within_environment_label_drift", "self_loop_stable", "review_status_change",
        "submitter_count_change", "any_meaning_drift", "semantic_drift_without_reclassification"
    ]
    df[keep].to_csv(OUT_MASTER, index=False)
    df.to_csv(OUT_ALIGN, index=False)
    df[["assertion_id", "variation_id", "gene", "condition_label_baseline", "condition_label_followup", "environment_baseline", "environment_followup", "cross_environment_drift"]].to_csv(OUT_ENV, index=False)
    return df


def assign_regime(row: pd.Series) -> dict:
    gene = str(row["gene"]).upper()
    label = row["condition_label_baseline"]
    env = row["environment_baseline"]
    envs = {e for e in env_set(label) if e != "other/unknown"}
    baseline_collision = len(envs) > 1
    broad = broad_or_ambiguous(label)
    nonspecific = env in {"pan-cancer / nonspecific cancer predisposition", "other/unknown"}
    syndrome_anchored = gene in SYNDROME_ENV and env == SYNDROME_ENV[gene]
    organ_specific = env in {"breast cancer predisposition", "ovarian cancer predisposition", "colorectal cancer / polyposis", "gastric cancer predisposition", "pancreatic cancer predisposition"}
    moderate = gene in MODERATE_RISK or env == "moderate-risk cancer susceptibility"
    recessive = gene == "MUTYH" or "biallelic" in norm_text(label) or "recessive" in norm_text(label)
    tumor_spectrum = baseline_collision or gene in {"TP53", "PTEN", "STK11", "CDH1", "BRCA1", "BRCA2", "PALB2"}
    if baseline_collision and any(e in envs for e in ["Li-Fraumeni syndrome", "Lynch syndrome / mismatch repair cancer predisposition", "PTEN hamartoma tumor syndrome / Cowden", "syndromic cancer predisposition"]):
        regime, arch = "syndrome_vs_organ_collision", "baseline_syndrome_organ_collision"
    elif syndrome_anchored:
        regime, arch = "syndrome_anchored_high_validity", "baseline_syndrome_anchored"
    elif recessive:
        regime, arch = "recessive_or_biallelic_context", "baseline_recessive_context"
    elif moderate:
        regime, arch = "moderate_risk_penetrance_boundary", "baseline_penetrance_boundary"
    elif tumor_spectrum:
        regime, arch = "tumor_spectrum_expansion_state", "baseline_tumor_spectrum_expansion"
    elif organ_specific:
        regime, arch = "organ_specific_label_state", "baseline_organ_specific"
    elif nonspecific:
        regime, arch = "nonspecific_cancer_susceptibility", "baseline_underresolved_contextual"
    else:
        regime, arch = "underresolved_contextual", "baseline_underresolved_contextual"
    score = 100.0
    if regime == "syndrome_vs_organ_collision": score -= 35
    if regime == "tumor_spectrum_expansion_state": score -= 25
    if regime == "moderate_risk_penetrance_boundary": score -= 25
    if regime == "recessive_or_biallelic_context": score -= 18
    if nonspecific: score -= 20
    if broad: score -= 12
    if arch == "baseline_underresolved_contextual": score -= 20
    if review_category(row.get("review_status_baseline", "")) in {"single_submitter", "conflicting", "weak_or_no_assertion", "other_or_missing"}: score -= 8
    if pd.notna(row.get("submitter_count_baseline")) and float(row.get("submitter_count_baseline")) <= 1: score -= 6
    if syndrome_anchored and not broad: score += 5
    score = float(np.clip(score, 0, 100))
    return {
        "baseline_regime_primary": regime,
        "baseline_architecture_family": arch,
        "baseline_collision_flag": regime == "syndrome_vs_organ_collision",
        "baseline_tumor_spectrum_flag": regime == "tumor_spectrum_expansion_state",
        "baseline_moderate_risk_flag": regime == "moderate_risk_penetrance_boundary",
        "baseline_syndrome_anchored_flag": regime == "syndrome_anchored_high_validity",
        "baseline_organ_specific_flag": regime == "organ_specific_label_state",
        "baseline_recessive_or_biallelic_flag": regime == "recessive_or_biallelic_context",
        "baseline_nonspecific_flag": regime == "nonspecific_cancer_susceptibility",
        "baseline_underresolved_flag": regime == "underresolved_contextual",
        "baseline_broad_ambiguous_label_flag": broad,
        "baseline_portability_score": round(score, 4),
        "baseline_nonportability_score": round(100-score, 4),
        "baseline_regime_assignment_reason": "baseline_only_gene_condition_environment_mapping",
    }


def build_regimes(df):
    assigned = df.apply(assign_regime, axis=1).apply(pd.Series)
    out = pd.concat([df[["assertion_id", "variation_id", "gene", "condition_label_baseline", "environment_baseline", "classification_baseline", "review_status_baseline", "submitter_count_baseline", "HGVS", "consequence"]].copy(), assigned], axis=1)
    out.to_csv(OUT_REGIMES, index=False)
    OUT_RULES.write_text(
        "# Hereditary Cancer Baseline-only Portability Regime Rules\n\n"
        "Technical rule definitions; not manuscript prose.\n\n"
        "Allowed predictor fields: gene, baseline condition label/environment, baseline classification, baseline review status, baseline submitter count, baseline HGVS/consequence.\n\n"
        "Forbidden predictor fields: follow-up labels/environments, endpoint labels, follow-up review/status/submitter/classification.\n\n"
        "Regimes: syndrome_anchored_high_validity, organ_specific_label_state, syndrome_vs_organ_collision, moderate_risk_penetrance_boundary, tumor_spectrum_expansion_state, recessive_or_biallelic_context, nonspecific_cancer_susceptibility, underresolved_contextual.\n",
        encoding="utf-8",
    )
    return out


def endpoint_counts(df):
    endpoints = ["classification_change", "condition_label_change", "cross_environment_drift", "within_environment_label_drift", "self_loop_stable", "review_status_change", "submitter_count_change", "any_meaning_drift", "semantic_drift_without_reclassification"]
    rows, n = [], len(df)
    for ep in endpoints:
        k = int(df[ep].astype(bool).sum())
        p = k/n if n else np.nan
        se = math.sqrt(p*(1-p)/n) if n and not math.isnan(p) else np.nan
        rows.append({"endpoint": ep, "numerator": k, "denominator": n, "rate": round(p, 4), "ci95_low": round(max(0, p-1.96*se), 4) if not math.isnan(se) else np.nan, "ci95_high": round(min(1, p+1.96*se), 4) if not math.isnan(se) else np.nan, "endpoint_role": "primary" if ep in {"condition_label_change", "cross_environment_drift"} else "secondary"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_COUNTS, index=False)
    return out


def group_summary(df, by, path):
    out = df.groupby(by, dropna=False).agg(
        N=("variation_id", "size"),
        classification_change_n=("classification_change", "sum"),
        condition_label_change_n=("condition_label_change", "sum"),
        cross_environment_drift_n=("cross_environment_drift", "sum"),
        any_meaning_drift_n=("any_meaning_drift", "sum"),
        self_loop_stable_n=("self_loop_stable", "sum"),
        mean_baseline_portability_score=("baseline_portability_score", "mean"),
    ).reset_index()
    for ep in ["classification_change", "condition_label_change", "cross_environment_drift", "any_meaning_drift", "self_loop_stable"]:
        out[f"{ep}_rate"] = (out[f"{ep}_n"] / out["N"]).round(4)
    out["mean_baseline_portability_score"] = out["mean_baseline_portability_score"].round(4)
    out.to_csv(path, index=False)
    return out


def fdr_bh(pvals):
    p = np.array([1.0 if pd.isna(x) else float(x) for x in pvals])
    order = np.argsort(p); adj = np.empty(len(p)); min_adj = 1.0; m = len(p)
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        min_adj = min(min_adj, p[idx] * m / rank)
        adj[idx] = min_adj
    return adj.tolist()


def fisher_row(df, exposure, outcome, test):
    x = df[exposure].astype(bool); y = df[outcome].astype(bool)
    a, b, c, d = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())
    odds, p = np.nan, np.nan
    if fisher_exact is not None:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"test": test, "exposure": exposure, "outcome": outcome, "a_exposed_outcome": a, "b_exposed_no_outcome": b, "c_unexposed_outcome": c, "d_unexposed_no_outcome": d, "odds_ratio": odds, "p_value": p, "status": "fit"}


def enrichment_tests(df):
    d = df.copy()
    d["low_baseline_portability_score"] = d["baseline_portability_score"] < 50
    rows = [
        fisher_row(d, "baseline_collision_flag", "cross_environment_drift", "syndrome_organ_collision_enriched_cross_environment"),
        fisher_row(d, "baseline_tumor_spectrum_flag", "cross_environment_drift", "tumor_spectrum_enriched_cross_environment"),
        fisher_row(d, "baseline_moderate_risk_flag", "condition_label_change", "moderate_risk_enriched_condition_label_change"),
        fisher_row(d, "baseline_syndrome_anchored_flag", "self_loop_stable", "syndrome_anchored_enriched_self_loop"),
        fisher_row(d, "baseline_nonspecific_flag", "condition_label_change", "nonspecific_enriched_condition_label_change"),
        fisher_row(d, "baseline_broad_ambiguous_label_flag", "condition_label_change", "broad_ambiguous_enriched_condition_label_change"),
        fisher_row(d, "low_baseline_portability_score", "cross_environment_drift", "low_portability_enriched_cross_environment"),
        fisher_row(d, "low_baseline_portability_score", "condition_label_change", "low_portability_enriched_condition_label_change"),
    ]
    out = pd.DataFrame(rows)
    out["FDR_p_value"] = fdr_bh(out["p_value"].tolist())
    out.to_csv(OUT_ENRICH, index=False)
    return out


def make_pipeline(X, features):
    num = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in features if c not in num]
    tx = []
    if num: tx.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat: tx.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("pre", ColumnTransformer(tx, remainder="drop")), ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))])


def calibration_slope(y, p):
    try:
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
        lr = LogisticRegression(solver="liblinear").fit(z, y)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def fit_model(df, endpoint, features, model):
    y = df[endpoint].astype(bool).astype(int)
    n, pos = len(df), int(y.sum())
    if n < 30 or y.nunique() < 2:
        return {"endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": "skipped_insufficient_N_or_endpoint"}
    X = df[features].copy()
    pipe = make_pipeline(X, features)
    try:
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p); auprc = average_precision_score(y, p); brier = brier_score_loss(y, p); ll = log_loss(y, p, labels=[0,1])
        rng = np.random.default_rng(RANDOM_STATE); boots=[]; idx_all=np.arange(n)
        for _ in range(N_BOOT):
            idx = rng.choice(idx_all, size=n, replace=True)
            if len(np.unique(y.iloc[idx])) > 1:
                boots.append(roc_auc_score(y.iloc[idx], p[idx]))
        lo, hi = np.percentile(boots, [2.5,97.5]) if boots else (np.nan, np.nan)
        cv_auc = np.nan
        if int(y.value_counts().min()) >= 2:
            cv = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_STATE)
            try:
                pcv = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
                cv_auc = roc_auc_score(y, pcv)
            except Exception:
                pass
        return {"endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "AUROC": round(float(auc),4), "AUROC_CI95_low": round(float(lo),4), "AUROC_CI95_high": round(float(hi),4), "AUPRC": round(float(auprc),4), "Brier_score": round(float(brier),4), "log_loss": round(float(ll),4), "calibration_slope": round(float(calibration_slope(y,p)),4), "cross_validated_AUROC": round(float(cv_auc),4) if not math.isnan(cv_auc) else np.nan, "status": "fit"}
    except Exception as e:
        return {"endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": f"fit_failed:{type(e).__name__}:{str(e)[:120]}"}


def model_comparison(df):
    d = normalize_core_schema(df.copy())
    d["baseline_review_category"] = d["review_status_baseline"].map(review_category)
    d["baseline_classification_group"] = d["classification_baseline"].map(clinical_group)
    d["submitter_count_baseline_num"] = pd.to_numeric(d["submitter_count_baseline"], errors="coerce")
    specs = {
        "M1_gene_only": ["gene"],
        "M2_baseline_regime_only": ["baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M3_ClinVar_metadata_only": ["baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
        "M4_gene_plus_baseline_regime": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"],
        "M5_gene_plus_baseline_regime_plus_metadata": ["gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score", "baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
    }
    rows=[]
    for ep in ["condition_label_change", "cross_environment_drift", "any_meaning_drift"]:
        for name, feats in specs.items():
            rows.append(fit_model(d, ep, feats, name))
    out = pd.DataFrame(rows); out.to_csv(OUT_MODELS, index=False)
    tests=[]
    pairs=[("M1_gene_only","M4_gene_plus_baseline_regime","gene_vs_gene_plus_regime"),("M1_gene_only","M5_gene_plus_baseline_regime_plus_metadata","gene_vs_gene_regime_metadata"),("M3_ClinVar_metadata_only","M5_gene_plus_baseline_regime_plus_metadata","metadata_vs_gene_regime_metadata")]
    for ep in ["condition_label_change", "cross_environment_drift", "any_meaning_drift"]:
        by={r["model"]:r for r in out[out["endpoint"].eq(ep)].to_dict("records")}
        for base, full, label in pairs:
            if base in by and full in by and by[base].get("status")=="fit" and by[full].get("status")=="fit":
                n=by[base]["N"]; lr=2*n*(float(by[base]["log_loss"])-float(by[full]["log_loss"])); p=float(chi2.sf(max(lr,0),1)) if chi2 is not None else np.nan
                tests.append({"endpoint":ep,"comparison":label,"base_model":base,"full_model":full,"LR_style_statistic_approx":round(lr,4),"p_value_approx":p,"AUROC_base":by[base].get("AUROC"),"AUROC_full":by[full].get("AUROC"),"delta_AUROC":round(float(by[full].get("AUROC"))-float(by[base].get("AUROC")),4)})
    lrdf=pd.DataFrame(tests)
    if len(lrdf): lrdf["FDR_p_value_approx"]=fdr_bh(lrdf["p_value_approx"].tolist())
    lrdf.to_csv(OUT_LR,index=False)
    return out, lrdf


def safe_read(path):
    try: return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def get_rate(counts, ep):
    h=counts[counts["endpoint"].eq(ep)]
    return h["rate"].iloc[0] if len(h) else np.nan


def get_auc(models, ep, m):
    h=models[(models["endpoint"].eq(ep)) & (models["model"].eq(m))]
    return h["AUROC"].iloc[0] if len(h) and "AUROC" in h.columns else np.nan


def three_domain_summary(counts, models, enrich, n):
    rows=[]
    existing=safe_read(TABLES/"cab_cross_domain_replication_summary.csv")
    if not existing.empty:
        for _,r in existing.iterrows():
            rows.append({"domain":r.get("domain"),"aligned_N":r.get("aligned_N"),"condition_label_change_rate":r.get("condition_label_change_rate"),"cross_environment_drift_rate":r.get("cross_environment_drift_rate"),"classification_change_rate":r.get("classification_change_rate"),"any_meaning_drift_rate":r.get("any_meaning_drift_rate"),"self_loop_stable_rate":r.get("self_loop_stable_rate"),"gene_only_AUROC":r.get("gene_only_AUROC_condition_drift"),"regime_only_AUROC":r.get("CAB_or_regime_AUROC_condition_drift"),"gene_plus_regime_AUROC":r.get("gene_plus_CAB_AUROC_condition_drift"),"primary_unstable_grammar":r.get("primary_portability_signal"),"primary_stable_grammar":"domain-specific self-loop architecture","external_constraint_available":r.get("external_constraint_status"),"claim_strength":"previously_supported_domain"})
    hit=enrich[enrich["test"].astype(str).str.contains("low_portability_enriched_cross_environment",na=False)] if not enrich.empty else pd.DataFrame()
    rows.append({"domain":"hereditary_cancer_predisposition","aligned_N":n,"condition_label_change_rate":get_rate(counts,"condition_label_change"),"cross_environment_drift_rate":get_rate(counts,"cross_environment_drift"),"classification_change_rate":get_rate(counts,"classification_change"),"any_meaning_drift_rate":get_rate(counts,"any_meaning_drift"),"self_loop_stable_rate":get_rate(counts,"self_loop_stable"),"gene_only_AUROC":get_auc(models,"condition_label_change","M1_gene_only"),"regime_only_AUROC":get_auc(models,"condition_label_change","M2_baseline_regime_only"),"gene_plus_regime_AUROC":get_auc(models,"condition_label_change","M4_gene_plus_baseline_regime"),"primary_unstable_grammar":f"syndrome/organ/tumor-spectrum/moderate-risk grammar; low portability OR={hit['odds_ratio'].iloc[0] if len(hit) else np.nan}, FDR={hit['FDR_p_value'].iloc[0] if len(hit) else np.nan}","primary_stable_grammar":"syndrome-anchored or organ-specific self-loop states if supported","external_constraint_available":"not joined; no expert validation claim","claim_strength":"noncardiovascular_replication_candidate"})
    out=pd.DataFrame(rows); out.to_csv(OUT_THREE,index=False); return out


def final_report(counts, models, enrich, three):
    def auc(ep,m): return get_auc(models,ep,m)
    lines=[
        "# Final Non-Cardiovascular Replication Report",
        "",
        "Analysis report; not manuscript prose.",
        "",
        "## Inputs",
        "- data/processed/clinvar_snapshot_baseline_202301.csv",
        "- data/processed/clinvar_snapshot_followup_202604.csv",
        "",
        "## 1. Does hereditary cancer show meaning drift despite classification stability?",
        counts.to_string(index=False),
        "",
        "## 2. Are drift patterns structured by baseline portability regimes?",
        enrich.to_string(index=False),
        "",
        "## 3. Does regime-only or gene+regime improve over gene-only?",
        f"- condition_label_change: gene-only AUROC={auc('condition_label_change','M1_gene_only')}; regime-only AUROC={auc('condition_label_change','M2_baseline_regime_only')}; gene+regime AUROC={auc('condition_label_change','M4_gene_plus_baseline_regime')}.",
        f"- cross_environment_drift: gene-only AUROC={auc('cross_environment_drift','M1_gene_only')}; regime-only AUROC={auc('cross_environment_drift','M2_baseline_regime_only')}; gene+regime AUROC={auc('cross_environment_drift','M4_gene_plus_baseline_regime')}.",
        "",
        "## 4. Does the portability principle replicate outside cardiovascular genetics?",
        "- Supported only if hereditary cancer shows nonzero meaning drift plus baseline-only regime stratification in the tables above.",
        "",
        "## 5. Is this enough for general assertion portability theory?",
        "- No. If hereditary cancer supports replication, this is three-domain evidence, not all-disease generalization.",
        "",
        "## Blocked claims",
        "- no future-label leakage",
        "- no all-disease claim",
        "- no clinical actionability claim",
        "- no variant reclassification claim",
        "- no expert validation claim",
        "",
        "## Three-domain summary",
        three.to_string(index=False),
    ]
    OUT_FINAL.write_text("\n".join(lines),encoding="utf-8")


def main():
    ensure_dirs()
    print("Building hereditary cancer ClinVar P/LP temporal cohort...")
    cohort=build_cohort()
    print(f"Aligned hereditary cancer assertions: {len(cohort):,}")
    regimes=build_regimes(cohort)
    # Keep only regime-derived columns for merge to avoid pandas _x/_y suffix damage
    # to baseline metadata columns used by the prediction models.
    regime_feature_cols = [
        "variation_id",
        "baseline_regime_primary", "baseline_architecture_family",
        "baseline_collision_flag", "baseline_tumor_spectrum_flag",
        "baseline_moderate_risk_flag", "baseline_syndrome_anchored_flag",
        "baseline_organ_specific_flag", "baseline_recessive_or_biallelic_flag",
        "baseline_nonspecific_flag", "baseline_underresolved_flag",
        "baseline_broad_ambiguous_label_flag",
        "baseline_portability_score", "baseline_nonportability_score",
        "baseline_regime_assignment_reason",
    ]
    df=cohort.merge(regimes[[c for c in regime_feature_cols if c in regimes.columns]], on="variation_id", how="left")
    df = normalize_core_schema(df)
    counts=endpoint_counts(df)
    group_summary(df,"gene",OUT_GENE)
    group_summary(df,"baseline_regime_primary",OUT_REGIME_SUM)
    enrich=enrichment_tests(df)
    models, lr=model_comparison(df)
    three=three_domain_summary(counts,models,enrich,len(df))
    final_report(counts,models,enrich,three)
    print("Hereditary cancer non-cardiovascular CAB replication complete.")
    print(counts.to_string(index=False))
    print(models.to_string(index=False))
    print(enrich.to_string(index=False))
    print("Key outputs:")
    for p in [OUT_MASTER,OUT_ALIGN,OUT_ENV,OUT_REGIMES,OUT_RULES,OUT_COUNTS,OUT_GENE,OUT_REGIME_SUM,OUT_MODELS,OUT_LR,OUT_ENRICH,OUT_THREE,OUT_FINAL]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
