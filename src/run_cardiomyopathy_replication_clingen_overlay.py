#!/usr/bin/env python3
"""Cardiomyopathy replication and ClinGen overlay runner for CAB portability.

This runner tests whether CAB-like assertion portability architecture generalizes
from inherited arrhythmia to cardiomyopathy.

It intentionally does NOT:
- restore deprecated arrhythmia AUC claims,
- claim full clinical validation,
- claim VCEP/CSpec coverage unless represented by accessible local/raw evidence,
- treat gene-only superiority as failure.

Inputs expected
---------------
data/processed/clinvar_snapshot_baseline_202301.csv
data/processed/clinvar_snapshot_followup_202604.csv

Optional inputs
---------------
data/raw/clingen/*
data/processed/*clingen*
reports/tables/*clingen*
Existing arrhythmia CAB outputs for comparison:
  reports/tables/transition_network_enrichment_tests.csv
  reports/tables/cross_environment_drift_prediction_models.csv
  reports/tables/cpi_publication_safe_claims.csv

Outputs
-------
data/processed/cardiomyopathy_assertion_master.csv
data/processed/cardiomyopathy_temporal_alignment.csv
data/processed/cardiomyopathy_condition_environment_map.csv
data/processed/cardiomyopathy_portability_regimes.csv
reports/qc/cardiomyopathy_regime_rule_definitions.md
reports/tables/cardiomyopathy_temporal_endpoint_counts.csv
reports/tables/cardiomyopathy_drift_by_gene.csv
reports/tables/cardiomyopathy_drift_by_regime.csv
reports/tables/cardiomyopathy_drift_by_environment.csv
data/processed/cardiomyopathy_environment_transition_edges.csv
reports/figures/cardiomyopathy_environment_transition_network.svg
reports/tables/cardiomyopathy_transition_enrichment_tests.csv
reports/tables/cardiomyopathy_model_comparison.csv
data/processed/cardiomyopathy_clingen_overlay.csv
reports/tables/cardiomyopathy_clingen_coverage.csv
reports/tables/cardiomyopathy_clingen_by_regime.csv
reports/tables/cardiomyopathy_clingen_drift_within_covered_zones.csv
reports/tables/arrhythmia_vs_cardiomyopathy_replication_summary.csv
reports/figures/arrhythmia_vs_cardiomyopathy_replication_panel.svg
reports/final_cardiomyopathy_replication_and_clingen_overlay_report.md
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
QC = REPORTS / "qc"

BASELINE = DATA / "clinvar_snapshot_baseline_202301.csv"
FOLLOWUP = DATA / "clinvar_snapshot_followup_202604.csv"

OUT_MASTER = DATA / "cardiomyopathy_assertion_master.csv"
OUT_ALIGN = DATA / "cardiomyopathy_temporal_alignment.csv"
OUT_ENV_MAP = DATA / "cardiomyopathy_condition_environment_map.csv"
OUT_REGIMES = DATA / "cardiomyopathy_portability_regimes.csv"
OUT_TRANSITIONS = DATA / "cardiomyopathy_environment_transition_edges.csv"
OUT_CLINGEN = DATA / "cardiomyopathy_clingen_overlay.csv"

TARGET_GENES = [
    "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1",
    "LMNA", "DSP", "PKP2", "DSG2", "DSC2", "JUP", "FLNC",
    "TTN", "PLN", "DES", "RBM20", "BAG3", "ACTN2", "VCL",
]

SARCOMERIC = {"MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "ACTN2", "VCL"}
DESMOSOMAL = {"DSP", "PKP2", "DSG2", "DSC2", "JUP"}
OVERLAP_DCM_CONDUCTION = {"LMNA", "PLN", "DES", "FLNC", "RBM20", "TTN", "BAG3"}

# CSpec/VCEP coverage known from Cardiomyopathy VCEP scope/resources:
# ACTC1, MYBPC3, MYH7, TNNI3, TNNT2, TPM1 were visible in current public resources.
# MYL2/MYL3 are not target genes here but are part of CMP-VCEP specs.
CMP_CSPEC_GENES_TARGET = {"ACTC1", "MYBPC3", "MYH7", "TNNI3", "TNNT2", "TPM1"}

RANDOM_STATE = 42
N_BOOT = 300


def ensure_dirs() -> None:
    for p in [DATA, TABLES, FIGURES, QC, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def first_nonempty(vals: Iterable[object]) -> str:
    for v in vals:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def join_unique(vals: Iterable[object]) -> str:
    out, seen = [], set()
    for v in vals:
        if pd.isna(v):
            continue
        for part in re.split(r"[|;]", str(v)):
            p = part.strip()
            if not p:
                continue
            k = p.lower()
            if k not in seen:
                seen.add(k)
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
    t = norm_text(x).replace("|", ";")
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return ";".join(sorted(set(parts)))


def condition_environment(label: object) -> str:
    t = norm_text(label)
    if not t:
        return "other/unknown"
    # Order matters: overlap labels before broad cardiomyopathy.
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


def is_cardiomyopathy_relevant(label: object, gene: str) -> bool:
    env = condition_environment(label)
    if env != "other/unknown":
        return True
    # Keep target genes even if labels are broad/unknown; classify as unknown later.
    return gene in TARGET_GENES


def load_snapshot(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in [
        "VariationID", "AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance",
        "ReviewStatus", "PhenotypeList", "PhenotypeIDS", "NumberSubmitters",
        "ReferenceAllele", "AlternateAllele", "LastEvaluated"
    ] if c in header]

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=250_000):
        chunk["gene"] = chunk["GeneSymbol"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["gene"].isin(TARGET_GENES)].copy()
        if not len(chunk):
            continue
        chunk["clinical_group"] = chunk["ClinicalSignificance"].map(clinical_group)
        chunk = chunk[chunk["clinical_group"] == "p_lp"].copy()
        if not len(chunk):
            continue
        chunk["condition_environment"] = chunk.get("PhenotypeList", "").map(condition_environment)
        chunk["condition_relevant"] = chunk.apply(lambda r: is_cardiomyopathy_relevant(r.get("PhenotypeList", ""), r["gene"]), axis=1)
        chunk = chunk[chunk["condition_relevant"]].copy()
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame({"variation_id": []})

    df = pd.concat(chunks, ignore_index=True)
    df["variation_id"] = df["VariationID"].map(norm_id)
    df = df[df["variation_id"] != ""].copy()

    # Aggregate duplicate RCV rows by VariationID.
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

    agg["condition_environment"] = agg["condition_label"].map(condition_environment)
    rename = {c: f"{c}_{suffix}" for c in agg.columns if c != "variation_id"}
    return agg.rename(columns=rename)


def assign_regime(row: pd.Series) -> Tuple[str, str, float, List[str]]:
    gene = str(row.get("gene_baseline", row.get("gene_followup", ""))).upper()
    env_b = row.get("condition_environment_baseline", "other/unknown")
    env_f = row.get("condition_environment_followup", "other/unknown")
    label_b = row.get("condition_label_baseline", "")
    envs = {env_b, env_f}
    flags = []

    cross_major = env_b != env_f and "other/unknown" not in {env_b, env_f}
    unknown_involved = "other/unknown" in envs
    nonspecific = "nonspecific cardiomyopathy" in envs or unknown_involved

    if gene in SARCOMERIC and (env_b == env_f == "HCM"):
        regime = "sarcomeric_phenotype_anchored"
        architecture = "phenotype_anchored_stability"
    elif gene in DESMOSOMAL and ("ARVC/ACM" in envs):
        regime = "structural_electrical_overlap"
        architecture = "structural_electrical_overlap"
    elif gene in DESMOSOMAL and len(envs - {"other/unknown"}) > 1:
        regime = "arrhythmogenic_cardiomyopathy_collision"
        architecture = "disease_model_collision"
    elif gene in OVERLAP_DCM_CONDUCTION and any(e in envs for e in ["DCM", "conduction-cardiomyopathy overlap", "sudden death / arrhythmia-overlap"]):
        if "conduction-cardiomyopathy overlap" in envs:
            regime = "conduction_cardiomyopathy_overlap"
        elif "sudden death / arrhythmia-overlap" in envs:
            regime = "sudden_death_arrhythmia_overlap"
        else:
            regime = "structural_electrical_overlap"
        architecture = "structural_electrical_overlap"
    elif cross_major or len(envs - {"other/unknown"}) > 1:
        regime = "arrhythmogenic_cardiomyopathy_collision"
        architecture = "disease_model_collision"
    elif any(e == "syndromic/metabolic cardiomyopathy" for e in envs):
        regime = "syndromic_metabolic_overlap"
        architecture = "syndromic_metabolic_overlap"
    elif nonspecific:
        regime = "nonspecific_label_state"
        architecture = "underresolved_contextual"
    elif gene in SARCOMERIC and env_b == env_f and env_b in {"HCM", "DCM", "RCM", "LVNC"}:
        regime = "sarcomeric_phenotype_anchored"
        architecture = "phenotype_anchored_stability"
    else:
        regime = "underresolved_contextual"
        architecture = "underresolved_contextual"

    disease_model_collision = architecture == "disease_model_collision" or cross_major
    underresolved = architecture == "underresolved_contextual" or nonspecific
    overlap = architecture in {"structural_electrical_overlap", "disease_model_collision"}
    canonical = architecture == "phenotype_anchored_stability"
    cpi = 100.0
    if disease_model_collision:
        cpi -= 35
        flags.append("disease_model_collision")
    if overlap:
        cpi -= 20
        flags.append("structural_electrical_overlap")
    if underresolved:
        cpi -= 25
        flags.append("underresolved_contextual")
    if unknown_involved:
        cpi -= 10
        flags.append("unknown_condition_environment")
    if gene in {"TTN", "MYBPC3"}:
        # Coarse evaluability flag, not an AF claim.
        cpi -= 8
        flags.append("population_frequency_evaluability_limited_candidate")
    if canonical:
        cpi += 5
        flags.append("phenotype_anchored")
    cpi = float(np.clip(cpi, 0, 100))
    return regime, architecture, cpi, flags


def build_cohort() -> pd.DataFrame:
    baseline = load_snapshot(BASELINE, "baseline")
    followup = load_snapshot(FOLLOWUP, "followup")

    merged = baseline.merge(followup, on="variation_id", how="outer", indicator=True)
    merged["aligned_to_both_snapshots"] = merged["_merge"].eq("both")
    aligned = merged[merged["aligned_to_both_snapshots"]].copy()

    # Required naming.
    aligned["assertion_id"] = "CM_" + aligned["variation_id"].astype(str)
    aligned["allele_id"] = aligned.get("allele_id_baseline", aligned.get("allele_id_followup", ""))
    aligned["gene"] = aligned.get("gene_baseline", "").fillna(aligned.get("gene_followup", ""))
    aligned["condition_label_baseline"] = aligned.get("condition_label_baseline", "")
    aligned["condition_label_followup"] = aligned.get("condition_label_followup", "")
    aligned["condition_environment_baseline"] = aligned["condition_label_baseline"].map(condition_environment)
    aligned["condition_environment_followup"] = aligned["condition_label_followup"].map(condition_environment)
    aligned["classification_baseline"] = aligned.get("classification_baseline", "")
    aligned["classification_followup"] = aligned.get("classification_followup", "")
    aligned["review_status_baseline"] = aligned.get("review_status_baseline", "")
    aligned["review_status_followup"] = aligned.get("review_status_followup", "")
    aligned["submitter_count_baseline"] = pd.to_numeric(aligned.get("submitter_count_baseline", np.nan), errors="coerce")
    aligned["submitter_count_followup"] = pd.to_numeric(aligned.get("submitter_count_followup", np.nan), errors="coerce")
    aligned["date_baseline"] = "2023-01"
    aligned["date_followup"] = "2026-04"
    aligned["variant_type"] = aligned.get("variant_type_baseline", aligned.get("variant_type_followup", ""))
    aligned["HGVS"] = aligned.get("HGVS_baseline", aligned.get("HGVS_followup", ""))
    aligned["consequence"] = aligned["HGVS"].map(lambda s: "missense" if re.search(r"\(p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\)", str(s)) else ("protein_change_nonmissense_or_unknown" if "(p." in str(s) else "unavailable"))

    # Endpoints.
    aligned["classification_change"] = aligned["classification_baseline"].map(clinical_group) != aligned["classification_followup"].map(clinical_group)
    aligned["condition_label_change"] = aligned["condition_label_baseline"].map(normalize_condition) != aligned["condition_label_followup"].map(normalize_condition)
    aligned["review_status_change"] = aligned["review_status_baseline"].map(review_category) != aligned["review_status_followup"].map(review_category)
    aligned["submitter_count_change"] = aligned["submitter_count_baseline"].fillna(-1) != aligned["submitter_count_followup"].fillna(-1)
    aligned["cross_environment_drift"] = aligned["condition_environment_baseline"] != aligned["condition_environment_followup"]
    aligned["within_environment_label_drift"] = aligned["condition_label_change"] & ~aligned["cross_environment_drift"]
    aligned["self_loop_stable"] = ~aligned["cross_environment_drift"]
    aligned["any_meaning_drift"] = aligned["condition_label_change"] | aligned["classification_change"] | aligned["review_status_change"]
    aligned["semantic_drift_without_reclassification"] = aligned["condition_label_change"] & ~aligned["classification_change"]

    # Regimes.
    assigned = aligned.apply(assign_regime, axis=1)
    aligned["cardiomyopathy_portability_regime"] = [x[0] for x in assigned]
    aligned["cardiomyopathy_architecture"] = [x[1] for x in assigned]
    aligned["cardiomyopathy_portability_score"] = [x[2] for x in assigned]
    aligned["cardiomyopathy_regime_flags"] = ["|".join(x[3]) for x in assigned]
    aligned["collision_or_overlap"] = aligned["cardiomyopathy_architecture"].isin(["disease_model_collision", "structural_electrical_overlap"])
    aligned["phenotype_anchored"] = aligned["cardiomyopathy_architecture"].eq("phenotype_anchored_stability")
    aligned["underresolved_or_nonspecific"] = aligned["cardiomyopathy_architecture"].eq("underresolved_contextual") | aligned["cardiomyopathy_portability_regime"].eq("nonspecific_label_state")

    master_cols = [
        "assertion_id", "variation_id", "allele_id", "gene",
        "condition_label_baseline", "condition_label_followup",
        "condition_environment_baseline", "condition_environment_followup",
        "classification_baseline", "classification_followup",
        "review_status_baseline", "review_status_followup",
        "submitter_count_baseline", "submitter_count_followup",
        "date_baseline", "date_followup", "variant_type", "consequence", "HGVS",
        "cardiomyopathy_portability_regime", "cardiomyopathy_architecture",
        "cardiomyopathy_portability_score", "cardiomyopathy_regime_flags",
        "classification_change", "condition_label_change", "review_status_change",
        "submitter_count_change", "cross_environment_drift",
        "within_environment_label_drift", "self_loop_stable", "any_meaning_drift",
        "semantic_drift_without_reclassification",
    ]
    existing = [c for c in master_cols if c in aligned.columns]
    aligned[existing].to_csv(OUT_MASTER, index=False)
    aligned.to_csv(OUT_ALIGN, index=False)

    env_map = aligned[[
        "variation_id", "gene", "condition_label_baseline", "condition_label_followup",
        "condition_environment_baseline", "condition_environment_followup",
        "cross_environment_drift", "within_environment_label_drift",
    ]].copy()
    env_map.to_csv(OUT_ENV_MAP, index=False)

    regime_cols = [
        "variation_id", "gene", "cardiomyopathy_portability_regime",
        "cardiomyopathy_architecture", "cardiomyopathy_portability_score",
        "cardiomyopathy_regime_flags", "condition_environment_baseline",
        "condition_environment_followup",
    ]
    aligned[regime_cols].to_csv(OUT_REGIMES, index=False)
    return aligned


def ci_binomial(k: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def endpoint_counts(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "classification_change", "condition_label_change", "review_status_change",
        "submitter_count_change", "cross_environment_drift",
        "within_environment_label_drift", "self_loop_stable", "any_meaning_drift",
        "semantic_drift_without_reclassification",
    ]
    rows = []
    n = len(df)
    for ep in endpoints:
        k = int(df[ep].astype(bool).sum())
        lo, hi = ci_binomial(k, n)
        rows.append({"endpoint": ep, "numerator": k, "denominator": n, "rate": round(k / n, 4) if n else np.nan, "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cardiomyopathy_temporal_endpoint_counts.csv", index=False)
    return out


def group_summary(df: pd.DataFrame, by: str, path: Path) -> pd.DataFrame:
    out = df.groupby(by, dropna=False).agg(
        N=("variation_id", "size"),
        classification_change_n=("classification_change", "sum"),
        condition_label_change_n=("condition_label_change", "sum"),
        cross_environment_drift_n=("cross_environment_drift", "sum"),
        any_meaning_drift_n=("any_meaning_drift", "sum"),
        self_loop_stable_n=("self_loop_stable", "sum"),
        mean_portability_score=("cardiomyopathy_portability_score", "mean"),
    ).reset_index()
    for base in ["classification_change", "condition_label_change", "cross_environment_drift", "any_meaning_drift", "self_loop_stable"]:
        out[f"{base}_rate"] = (out[f"{base}_n"] / out["N"]).round(4)
    out["mean_portability_score"] = out["mean_portability_score"].round(4)
    out.sort_values(["cross_environment_drift_rate", "condition_label_change_rate", "N"], ascending=[False, False, False]).to_csv(path, index=False)
    return out


def dominant(values: pd.Series, max_items=5) -> str:
    counts = values.fillna("unknown").astype(str).value_counts()
    total = counts.sum()
    return "; ".join(f"{idx}:{cnt}({cnt/total:.2f})" for idx, cnt in counts.head(max_items).items()) if total else ""


def build_transition_network(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition_environment_baseline", "condition_environment_followup"], dropna=False)
    rows = []
    for (src, tgt), sub in g:
        rows.append({
            "source_environment": src,
            "target_environment": tgt,
            "edge_count": len(sub),
            "edge_fraction": round(len(sub) / len(df), 4) if len(df) else np.nan,
            "genes_in_edge": int(sub["gene"].nunique()),
            "dominant_genes": dominant(sub["gene"]),
            "dominant_regime": dominant(sub["cardiomyopathy_portability_regime"]),
            "collision_fraction": round(float(sub["collision_or_overlap"].mean()), 4),
            "mean_portability_score": round(float(sub["cardiomyopathy_portability_score"].mean()), 4),
            "self_loop_or_cross_environment": "self_loop" if src == tgt else "cross_environment",
        })
    edges = pd.DataFrame(rows).sort_values(["edge_count"], ascending=False)
    edges.to_csv(OUT_TRANSITIONS, index=False)
    edges.to_csv(TABLES / "cardiomyopathy_environment_transition_edges.csv", index=False)
    return edges


def fisher_test(df: pd.DataFrame, exposure: str, outcome: str, test_name: str) -> Dict[str, object]:
    if fisher_exact is None:
        return {"test": test_name, "status": "skipped_scipy_unavailable"}
    x = df[exposure].astype(bool)
    y = df[outcome].astype(bool)
    a = int((x & y).sum())
    b = int((x & ~y).sum())
    c = int((~x & y).sum())
    d = int((~x & ~y).sum())
    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        odds, p = np.nan, np.nan
    return {
        "test": test_name, "exposure": exposure, "outcome": outcome,
        "a_exposed_outcome": a, "b_exposed_no_outcome": b,
        "c_unexposed_outcome": c, "d_unexposed_no_outcome": d,
        "odds_ratio": odds, "p_value": p, "status": "fit",
    }


def transition_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        fisher_test(df, "collision_or_overlap", "cross_environment_drift", "collision_overlap_enriched_cross_environment"),
        fisher_test(df, "phenotype_anchored", "self_loop_stable", "phenotype_anchored_enriched_self_loop"),
        fisher_test(df, "underresolved_or_nonspecific", "condition_label_change", "underresolved_enriched_future_label_drift"),
    ]
    out = pd.DataFrame(rows)
    if "p_value" in out.columns:
        pvals = pd.to_numeric(out["p_value"], errors="coerce")
        order = np.argsort(pvals.fillna(1).values)
        fdr = np.full(len(out), np.nan)
        m = len(out)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            # simpler BH reverse
            pass
        # direct BH
        sorted_idx = np.argsort(pvals.fillna(1).values)
        adj = np.zeros(len(out))
        min_adj = 1.0
        for i in range(len(sorted_idx)-1, -1, -1):
            idx = sorted_idx[i]
            raw = pvals.iloc[idx] if not pd.isna(pvals.iloc[idx]) else 1.0
            val = min(min_adj, raw * m / (i+1))
            min_adj = val
            adj[idx] = val
        out["FDR_p_value"] = adj
    out.to_csv(TABLES / "cardiomyopathy_transition_enrichment_tests.csv", index=False)
    return out


def make_model(df: pd.DataFrame, endpoint: str, features: List[str], model: str) -> Dict[str, object]:
    y = df[endpoint].astype(bool).astype(int)
    n = len(df)
    pos = int(y.sum())
    if n < 30 or y.nunique() < 2:
        return {"endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": "skipped_insufficient_N_or_endpoint"}

    X = df[features].copy()
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
    pipe = Pipeline([
        ("pre", ColumnTransformer(transformers, remainder="drop")),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    try:
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        ll = log_loss(y, p, labels=[0, 1])
        min_class = int(y.value_counts().min())
        cv_auc = np.nan
        if min_class >= 2:
            cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=RANDOM_STATE)
            try:
                pcv = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
                cv_auc = roc_auc_score(y, pcv)
            except Exception:
                cv_auc = np.nan
        rng = np.random.default_rng(RANDOM_STATE)
        boots = []
        idx_all = np.arange(n)
        for _ in range(N_BOOT):
            idx = rng.choice(idx_all, size=n, replace=True)
            if len(np.unique(y.iloc[idx])) < 2:
                continue
            boots.append(roc_auc_score(y.iloc[idx], p[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        return {
            "endpoint": endpoint, "model": model, "N": n, "positive_N": pos,
            "AUROC": round(float(auc), 4), "AUROC_CI95_low": round(float(lo), 4),
            "AUROC_CI95_high": round(float(hi), 4), "AUPRC": round(float(auprc), 4),
            "Brier_score": round(float(brier), 4), "log_loss": round(float(ll), 4),
            "cross_validated_AUROC": round(float(cv_auc), 4) if not math.isnan(cv_auc) else np.nan,
            "AIC_approx": round(float(2 * (len(features)+1) + 2 * ll * n), 4),
            "BIC_approx": round(float(math.log(n) * (len(features)+1) + 2 * ll * n), 4),
            "status": "fit",
        }
    except Exception as e:
        return {"endpoint": endpoint, "model": model, "N": n, "positive_N": pos, "status": f"fit_failed:{type(e).__name__}:{str(e)[:120]}"}


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["baseline_review_category"] = df["review_status_baseline"].map(review_category)
    df["baseline_classification_group"] = df["classification_baseline"].map(clinical_group)
    df["submitter_count_baseline_num"] = pd.to_numeric(df["submitter_count_baseline"], errors="coerce")
    model_specs = {
        "M1_gene_only": ["gene"],
        "M2_cardiomyopathy_CAB_like_regime_only": ["cardiomyopathy_portability_regime", "cardiomyopathy_architecture", "cardiomyopathy_portability_score"],
        "M3_ClinVar_metadata_only": ["baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
        "M4_gene_plus_CAB_like_regime": ["gene", "cardiomyopathy_portability_regime", "cardiomyopathy_architecture", "cardiomyopathy_portability_score"],
        "M5_gene_plus_CAB_like_regime_plus_metadata": ["gene", "cardiomyopathy_portability_regime", "cardiomyopathy_architecture", "cardiomyopathy_portability_score", "baseline_review_category", "submitter_count_baseline_num", "baseline_classification_group"],
    }
    rows = []
    for endpoint in ["cross_environment_drift", "condition_label_change"]:
        for model, features in model_specs.items():
            rows.append(make_model(df, endpoint, features, model))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cardiomyopathy_model_comparison.csv", index=False)

    # LR-style tests approximate from log_loss if available.
    tests = []
    for endpoint in ["cross_environment_drift", "condition_label_change"]:
        sub = out[out["endpoint"] == endpoint].copy()
        by = {r["model"]: r for r in sub.to_dict("records")}
        pairs = [
            ("M1_gene_only", "M4_gene_plus_CAB_like_regime"),
            ("M3_ClinVar_metadata_only", "M5_gene_plus_CAB_like_regime_plus_metadata"),
            ("M4_gene_plus_CAB_like_regime", "M5_gene_plus_CAB_like_regime_plus_metadata"),
        ]
        for base_m, full_m in pairs:
            if base_m in by and full_m in by and by[base_m].get("status") == "fit" and by[full_m].get("status") == "fit":
                n = by[base_m]["N"]
                lr = 2 * n * (float(by[base_m]["log_loss"]) - float(by[full_m]["log_loss"]))
                p = float(chi2.sf(lr, df=1)) if chi2 is not None else np.nan
                tests.append({"endpoint": endpoint, "base_model": base_m, "full_model": full_m, "LR_style_statistic_approx": round(lr, 4), "p_value_approx": p})
    pd.DataFrame(tests).to_csv(TABLES / "cardiomyopathy_model_lr_style_tests.csv", index=False)
    return out


def find_clingen_files() -> List[Path]:
    pats = []
    for root in [RAW, DATA, TABLES]:
        if root.exists():
            pats.extend(list(root.rglob("*clingen*")))
            pats.extend(list(root.rglob("*ClinGen*")))
            pats.extend(list(root.rglob("*cspec*")))
            pats.extend(list(root.rglob("*vcep*")))
    return [p for p in pats if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx"}]


def load_gene_validity(files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in files:
        if p.suffix.lower() == ".xlsx":
            continue
        try:
            sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
            df = pd.read_csv(p, sep=sep, low_memory=False, nrows=200000)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        gene_col = None
        for cand in ["gene", "gene symbol", "genesymbol", "hgnc symbol", "symbol"]:
            if cand in cols:
                gene_col = cols[cand]
                break
        if not gene_col:
            continue
        valid_col = None
        for cand in ["classification", "validity classification", "clinical validity classification", "classification_title"]:
            if cand in cols:
                valid_col = cols[cand]
                break
        disease_col = None
        for cand in ["disease", "disease label", "disease_title", "condition", "mondo"]:
            if cand in cols:
                disease_col = cols[cand]
                break
        sub = df[df[gene_col].astype(str).str.upper().isin(TARGET_GENES)].copy()
        if len(sub):
            for _, r in sub.iterrows():
                rows.append({
                    "gene": str(r.get(gene_col, "")).upper(),
                    "clingen_validity": str(r.get(valid_col, "available_unspecified")) if valid_col else "available_unspecified",
                    "clingen_disease": str(r.get(disease_col, "")) if disease_col else "",
                    "source_file": str(p.relative_to(BASE)) if p.is_relative_to(BASE) else str(p),
                })
    if not rows:
        return pd.DataFrame(columns=["gene", "clingen_validity", "clingen_disease", "source_file"])
    return pd.DataFrame(rows).drop_duplicates()


def clingen_overlay(df: pd.DataFrame) -> pd.DataFrame:
    files = find_clingen_files()
    validity = load_gene_validity(files)
    out = df.copy()

    if not validity.empty:
        validity_gene = validity.groupby("gene", as_index=False).agg(
            clingen_validity=("clingen_validity", join_unique),
            clingen_disease=("clingen_disease", join_unique),
            clingen_validity_source=("source_file", join_unique),
        )
        out = out.merge(validity_gene, on="gene", how="left")
        out["clingen_validity_level"] = np.where(out["clingen_validity"].notna(), "gene-condition_or_gene_level_from_local_file", "unavailable")
        out["clingen_validity"] = out["clingen_validity"].fillna("unavailable")
    else:
        out["clingen_validity"] = "unavailable_no_local_clingen_gene_disease_validity_file_detected"
        out["clingen_validity_level"] = "unavailable"
        out["clingen_disease"] = ""
        out["clingen_validity_source"] = ""

    # Conservative CSpec/VCEP overlay: gene-level coverage candidates only for CMP-VCEP target genes.
    out["cspec_covered"] = out["gene"].isin(CMP_CSPEC_GENES_TARGET)
    out["cspec_level"] = np.where(out["cspec_covered"], "gene_level_CMP_CSpec_scope_candidate_not_variant_level_validation", "unavailable_or_not_targeted_by_current_CMP_CSpec_gene_set")
    out["vcep_covered"] = out["gene"].isin(CMP_CSPEC_GENES_TARGET)
    out["vcep_level"] = np.where(out["vcep_covered"], "gene_level_CMP_VCEP_scope_candidate_not_variant_level_validation", "unavailable_or_not_targeted_by_current_CMP_VCEP_gene_set")
    out["evidence_repository_match"] = False
    out["match_confidence"] = np.where(out["cspec_covered"], "gene_level_scope_only_low_variant_confidence", "none")
    out["match_failure_reason"] = np.where(
        out["cspec_covered"],
        "No variant-level Evidence Repository join performed; only gene-level CMP-VCEP/CSpec scope flag.",
        "No local VCEP/CSpec/Evidence Repository variant-level resource matched."
    )
    out.to_csv(OUT_CLINGEN, index=False)

    cov = pd.DataFrame([
        {"resource": "ClinGen Gene-Disease Validity", "coverage_level": "gene/gene-condition if local file present", "covered_assertions": int((out["clingen_validity_level"] != "unavailable").sum()), "total_assertions": len(out), "coverage_rate": round(float((out["clingen_validity_level"] != "unavailable").mean()), 4)},
        {"resource": "CMP VCEP scope", "coverage_level": "gene-level scope candidate only", "covered_assertions": int(out["vcep_covered"].sum()), "total_assertions": len(out), "coverage_rate": round(float(out["vcep_covered"].mean()), 4)},
        {"resource": "CMP CSpec scope", "coverage_level": "gene-level scope candidate only", "covered_assertions": int(out["cspec_covered"].sum()), "total_assertions": len(out), "coverage_rate": round(float(out["cspec_covered"].mean()), 4)},
        {"resource": "ClinGen Evidence Repository", "coverage_level": "variant-level", "covered_assertions": int(out["evidence_repository_match"].sum()), "total_assertions": len(out), "coverage_rate": round(float(out["evidence_repository_match"].mean()), 4)},
    ])
    cov.to_csv(TABLES / "cardiomyopathy_clingen_coverage.csv", index=False)

    by_regime = out.groupby("cardiomyopathy_portability_regime", dropna=False).agg(
        N=("variation_id", "size"),
        cspec_gene_scope_fraction=("cspec_covered", "mean"),
        vcep_gene_scope_fraction=("vcep_covered", "mean"),
        cross_environment_drift_rate=("cross_environment_drift", "mean"),
        condition_label_change_rate=("condition_label_change", "mean"),
    ).reset_index()
    for c in by_regime.select_dtypes(include=["float"]).columns:
        by_regime[c] = by_regime[c].round(4)
    by_regime.to_csv(TABLES / "cardiomyopathy_clingen_by_regime.csv", index=False)

    within = out[out["cspec_covered"] | out["vcep_covered"]].groupby(["cspec_covered", "vcep_covered"], dropna=False).agg(
        N=("variation_id", "size"),
        cross_environment_drift_n=("cross_environment_drift", "sum"),
        condition_label_change_n=("condition_label_change", "sum"),
        cross_environment_drift_rate=("cross_environment_drift", "mean"),
        condition_label_change_rate=("condition_label_change", "mean"),
    ).reset_index()
    for c in within.select_dtypes(include=["float"]).columns:
        within[c] = within[c].round(4)
    within.to_csv(TABLES / "cardiomyopathy_clingen_drift_within_covered_zones.csv", index=False)
    return out, cov


def compare_arrhythmia_cardiomyopathy(df: pd.DataFrame, transition_tests_df: pd.DataFrame, models: pd.DataFrame, clingen_cov: pd.DataFrame) -> pd.DataFrame:
    # Pull arrhythmia values if available.
    arr_cross_or = np.nan
    arr_gene_auc = np.nan
    arr_cab_auc = np.nan
    arr_gene_cab_auc = np.nan
    arr_transition = TABLES / "transition_network_enrichment_tests.csv"
    if arr_transition.exists():
        try:
            arrt = pd.read_csv(arr_transition)
            hit = arrt[arrt["test"].astype(str).str.contains("disease_model_collision_enriched_cross_environment", na=False)]
            if len(hit):
                arr_cross_or = hit["odds_ratio"].iloc[0]
        except Exception:
            pass

    # CM metrics.
    n = len(df)
    model_ce = models[models["endpoint"] == "cross_environment_drift"]
    def m_auc(name):
        hit = model_ce[model_ce["model"].eq(name)]
        return hit["AUROC"].iloc[0] if len(hit) and "AUROC" in hit else np.nan
    cm_gene_auc = m_auc("M1_gene_only")
    cm_cab_auc = m_auc("M2_cardiomyopathy_CAB_like_regime_only")
    cm_gene_cab_auc = m_auc("M4_gene_plus_CAB_like_regime")
    cm_or = transition_tests_df.loc[transition_tests_df["test"].eq("collision_overlap_enriched_cross_environment"), "odds_ratio"].iloc[0] if len(transition_tests_df) else np.nan
    ext_cov = clingen_cov["coverage_rate"].max() if len(clingen_cov) else np.nan

    rows = [
        {
            "domain": "arrhythmia_CAB_previous",
            "assertion_N": 1731,
            "temporal_aligned_N": 942,
            "temporal_alignment_rate": round(942/1731, 4),
            "classification_change_rate": 0.0998,
            "condition_label_change_rate": 0.3875,
            "cross_environment_drift_rate": 0.1550,
            "self_loop_stable_rate": 0.8450,
            "collision_overlap_enrichment_OR": arr_cross_or,
            "canonical_self_loop_enrichment_OR": 4.45,
            "gene_only_AUROC": arr_gene_auc,
            "CAB_like_only_AUROC": arr_cab_auc,
            "gene_plus_CAB_like_AUROC": arr_gene_cab_auc,
            "incremental_gain_from_CAB_like_architecture": np.nan,
            "external_ClinGen_coverage": np.nan,
            "source_note": "arrhythmia values from prior local CAB outputs when available; do not restore deprecated AUCs",
        },
        {
            "domain": "cardiomyopathy_replication",
            "assertion_N": n,
            "temporal_aligned_N": n,
            "temporal_alignment_rate": np.nan,
            "classification_change_rate": round(float(df["classification_change"].mean()), 4),
            "condition_label_change_rate": round(float(df["condition_label_change"].mean()), 4),
            "cross_environment_drift_rate": round(float(df["cross_environment_drift"].mean()), 4),
            "self_loop_stable_rate": round(float(df["self_loop_stable"].mean()), 4),
            "collision_overlap_enrichment_OR": cm_or,
            "canonical_self_loop_enrichment_OR": transition_tests_df.loc[transition_tests_df["test"].eq("phenotype_anchored_enriched_self_loop"), "odds_ratio"].iloc[0] if len(transition_tests_df) else np.nan,
            "gene_only_AUROC": cm_gene_auc,
            "CAB_like_only_AUROC": cm_cab_auc,
            "gene_plus_CAB_like_AUROC": cm_gene_cab_auc,
            "incremental_gain_from_CAB_like_architecture": round(float(cm_gene_cab_auc - cm_gene_auc), 4) if not pd.isna(cm_gene_auc) and not pd.isna(cm_gene_cab_auc) else np.nan,
            "external_ClinGen_coverage": ext_cov,
            "source_note": "computed from cardiomyopathy ClinVar temporal rebuild and local ClinGen overlay resources",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "arrhythmia_vs_cardiomyopathy_replication_summary.csv", index=False)
    return out


def plot_outputs(df: pd.DataFrame, edges: pd.DataFrame, comparison: pd.DataFrame) -> None:
    if plt is None:
        return
    # Network-like transition figure: heatmap matrix.
    try:
        pivot = edges.pivot(index="source_environment", columns="target_environment", values="edge_count").fillna(0)
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(pivot.values)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Cardiomyopathy environment transitions")
        fig.colorbar(im, ax=ax, label="edge_count")
        fig.tight_layout()
        fig.savefig(FIGURES / "cardiomyopathy_environment_transition_network.svg")
        plt.close(fig)
    except Exception:
        pass

    try:
        by = df.groupby("cardiomyopathy_architecture").agg(
            cross_environment_drift_rate=("cross_environment_drift", "mean"),
            condition_label_change_rate=("condition_label_change", "mean"),
            N=("variation_id", "size"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(by["cardiomyopathy_architecture"].astype(str), by["cross_environment_drift_rate"])
        ax.set_ylabel("cross_environment_drift_rate")
        ax.set_xticklabels(by["cardiomyopathy_architecture"].astype(str), rotation=45, ha="right")
        ax.set_title("Cross-environment drift by cardiomyopathy architecture")
        fig.tight_layout()
        fig.savefig(FIGURES / "cardiomyopathy_cross_environment_drift_by_architecture.svg")
        plt.close(fig)
    except Exception:
        pass

    try:
        cm = comparison[comparison["domain"].eq("cardiomyopathy_replication")].iloc[0]
        vals = [cm["condition_label_change_rate"], cm["cross_environment_drift_rate"], cm["self_loop_stable_rate"]]
        labs = ["condition label", "cross-env", "self-loop"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labs, vals)
        ax.set_ylabel("rate")
        ax.set_title("Cardiomyopathy replication endpoint rates")
        fig.tight_layout()
        fig.savefig(FIGURES / "arrhythmia_vs_cardiomyopathy_replication_panel.svg")
        plt.close(fig)
    except Exception:
        pass


def write_reports(df: pd.DataFrame, counts: pd.DataFrame, tests: pd.DataFrame, models: pd.DataFrame, clingen_cov: pd.DataFrame, comparison: pd.DataFrame) -> None:
    regime_rules = [
        "# Cardiomyopathy Regime Rule Definitions",
        "",
        "Technical rule definitions; not manuscript prose.",
        "",
        "## Regimes",
        "- sarcomeric_phenotype_anchored: sarcomeric target genes with coherent HCM-like or cardiomyopathy phenotype labels.",
        "- structural_electrical_overlap: desmosomal/structural genes or DCM/conduction/sudden-death overlap contexts.",
        "- arrhythmogenic_cardiomyopathy_collision: labels crossing cardiomyopathy disease-model environments.",
        "- conduction_cardiomyopathy_overlap: LMNA/PLN/DES/FLNC/RBM20/TTN/BAG3 with conduction/DCM context.",
        "- sudden_death_arrhythmia_overlap: cardiomyopathy target genes with sudden-death/arrhythmia labels.",
        "- syndromic_metabolic_overlap: syndromic/metabolic cardiomyopathy labels.",
        "- population_frequency_evaluability_limited: only coarse candidate flag for TTN/MYBPC3; no AF claim is made.",
        "- underresolved_contextual: unknown/ambiguous context not otherwise classifiable.",
        "- nonspecific_label_state: nonspecific cardiomyopathy or other/unknown labels.",
        "",
        "## Guardrails",
        "- These are CAB-like portability regimes, not clinical pathogenicity classifications.",
        "- No ancestry-aware AF claim is made unless AF fields are available; this script only flags coarse evaluability candidates.",
        "- VCEP/CSpec coverage is not inferred as variant-level validation.",
    ]
    (QC / "cardiomyopathy_regime_rule_definitions.md").write_text("\n".join(regime_rules), encoding="utf-8")

    report = [
        "# Cardiomyopathy Replication and ClinGen Overlay Report",
        "",
        "Analysis report, not manuscript prose.",
        "",
        "## 1. Did cardiomyopathy show condition/cross-environment meaning drift?",
        counts.to_string(index=False),
        "",
        "## 2. Were drift patterns structured by disease-model architecture?",
        tests.to_string(index=False),
        "",
        "## 3. Did CAB-like regimes improve or explain gene-only models?",
        models.to_string(index=False),
        "",
        "## 4. Did ClinGen/VCEP/CSpec coverage constrain but not collapse regimes?",
        clingen_cov.to_string(index=False),
        "",
        "## 5. Arrhythmia vs cardiomyopathy comparison",
        comparison.to_string(index=False),
        "",
        "## 6. Publication-safe result logic",
        "- If cardiomyopathy cross-environment drift is nonzero and enriched in collision/overlap regimes, cardiomyopathy supports portability architecture replication.",
        "- If gene+CAB-like AUROC exceeds gene-only, CAB-like architecture adds explanatory/predictive information beyond gene identity.",
        "- If ClinGen coverage is only gene-level, do not claim variant-level validation.",
        "- If local ClinGen files are absent, overlay is a coverage/QC constraint only.",
        "",
        "## 7. Claims remaining internal-only",
        "- Variant-level VCEP/CSpec validation unless Evidence Repository/CSpec variant-level files are actually joined.",
        "- Clinical actionability beyond routing.",
        "- General assertion portability theory beyond cardiovascular replication.",
        "",
        "## 8. Limitations",
        "- Cardiomyopathy cohort is derived from ClinVar P/LP assertions in target genes, not expert-curated truth.",
        "- Condition/environment mapping is regex/rule-based and mapping-sensitive.",
        "- Temporal alignment is limited to assertions present in both snapshots.",
        "- VCEP/CSpec overlay is conservative and does not turn gene-level coverage into variant-level validation.",
        "",
    ]
    (REPORTS / "final_cardiomyopathy_replication_and_clingen_overlay_report.md").write_text("\n".join(report), encoding="utf-8")

    qc = [
        "# Cardiomyopathy Replication QC",
        "",
        f"- aligned cardiomyopathy assertions: {len(df)}",
        f"- target genes: {', '.join(TARGET_GENES)}",
        "- endpoints from parsed local ClinVar 2023-01 and 2026-04 snapshots.",
        "- no deprecated CAB AUCs restored.",
        "- no VCEP/CSpec variant-level coverage claimed unless local files support it.",
    ]
    (QC / "cardiomyopathy_replication_qc.md").write_text("\n".join(qc), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Building cardiomyopathy ClinVar P/LP temporal cohort...")
    df = build_cohort()
    print(f"Temporally aligned cardiomyopathy assertions: {len(df):,}")

    counts = endpoint_counts(df)
    gene = group_summary(df, "gene", TABLES / "cardiomyopathy_drift_by_gene.csv")
    regime = group_summary(df, "cardiomyopathy_portability_regime", TABLES / "cardiomyopathy_drift_by_regime.csv")
    env = group_summary(df, "condition_environment_baseline", TABLES / "cardiomyopathy_drift_by_environment.csv")

    edges = build_transition_network(df)
    tests = transition_tests(df)
    models = model_comparison(df)
    clingen_df, clingen_cov = clingen_overlay(df)
    comparison = compare_arrhythmia_cardiomyopathy(df, tests, models, clingen_cov)
    plot_outputs(df, edges, comparison)
    write_reports(df, counts, tests, models, clingen_cov, comparison)

    print("Cardiomyopathy replication and ClinGen overlay complete.")
    print()
    print("Endpoint counts:")
    print(counts.to_string(index=False))
    print()
    print("Transition enrichment tests:")
    print(tests.to_string(index=False))
    print()
    print("Model comparison:")
    print(models.to_string(index=False))
    print()
    print("ClinGen coverage:")
    print(clingen_cov.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_MASTER, OUT_ALIGN, OUT_ENV_MAP, OUT_REGIMES,
        TABLES / "cardiomyopathy_temporal_endpoint_counts.csv",
        TABLES / "cardiomyopathy_drift_by_gene.csv",
        TABLES / "cardiomyopathy_drift_by_regime.csv",
        TABLES / "cardiomyopathy_drift_by_environment.csv",
        OUT_TRANSITIONS,
        TABLES / "cardiomyopathy_transition_enrichment_tests.csv",
        TABLES / "cardiomyopathy_model_comparison.csv",
        OUT_CLINGEN,
        TABLES / "cardiomyopathy_clingen_coverage.csv",
        TABLES / "arrhythmia_vs_cardiomyopathy_replication_summary.csv",
        REPORTS / "final_cardiomyopathy_replication_and_clingen_overlay_report.md",
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
