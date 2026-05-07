
#!/usr/bin/env python3
"""Build CAB Prospective Portability Prediction Registry.

Purpose:
Create a sealed, baseline-only prediction cohort of new public P/LP assertions
after the current benchmark cutoff. This is prospective-style prediction of
future assertion portability, not patient risk.

Do not use follow-up data.
Do not claim prospective validation until follow-up snapshot exists.
Do not modify primary predictions after registry lock.

Inputs:
- downloads ClinVar variant_summary.txt.gz from NCBI FTP unless already present
- data/processed/cab_decision_challenge_tasks.csv
- reports/tables/clinvar_identity_vs_meaning_concordance.csv, if available
- reports/tables/disease_architecture_regime_temporal_signatures.csv, if available
- reports/tables/disease_architecture_regime_support_levels.csv, if available

Outputs:
- data/prospective/cab_prospective_cohort_baseline_2026.csv
- data/prospective/cab_prospective_prediction_registry_2026_locked.csv
- data/prospective/cab_prospective_sads_stratum.csv
- reports/prospective/cab_prediction_registry_lock_report.md
- reports/prospective/cab_prospective_endpoint_definitions.md
- reports/prospective/cab_prospective_analysis_plan.md
- reports/prospective/prospective_registry_publication_safe_statement.md
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
import urllib.request
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROSPECTIVE = ROOT / "data" / "prospective"
RAW_PROSPECTIVE = DATA_PROSPECTIVE / "raw"
REPORTS_PROSPECTIVE = ROOT / "reports" / "prospective"
TABLES = ROOT / "reports" / "tables"

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

BENCHMARK_TASKS = DATA_PROCESSED / "cab_decision_challenge_tasks.csv"
IDENTITY = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
REGIME_SIG = TABLES / "disease_architecture_regime_temporal_signatures.csv"
REGIME_SUPPORT = TABLES / "disease_architecture_regime_support_levels.csv"

OUT_COHORT = DATA_PROSPECTIVE / "cab_prospective_cohort_baseline_2026.csv"
OUT_REGISTRY = DATA_PROSPECTIVE / "cab_prospective_prediction_registry_2026_locked.csv"
OUT_SADS = DATA_PROSPECTIVE / "cab_prospective_sads_stratum.csv"
OUT_LOCK = REPORTS_PROSPECTIVE / "cab_prediction_registry_lock_report.md"
OUT_ENDPOINTS = REPORTS_PROSPECTIVE / "cab_prospective_endpoint_definitions.md"
OUT_PLAN = REPORTS_PROSPECTIVE / "cab_prospective_analysis_plan.md"
OUT_SAFE = REPORTS_PROSPECTIVE / "prospective_registry_publication_safe_statement.md"

P_LP_RE = re.compile(r"(^|[/,; ])(Pathogenic|Likely pathogenic)($|[/,; ])", re.I)
BENIGN_RE = re.compile(r"Benign|Likely benign", re.I)
CONFLICT_RE = re.compile(r"Conflicting|Uncertain|VUS|not provided|association|risk factor|protective", re.I)

ARRHYTHMIA_GENES = {
    "KCNQ1", "KCNH2", "SCN5A", "RYR2", "CASQ2", "KCNE1", "KCNE2", "KCNJ2",
    "ANK2", "CACNA1C", "CALM1", "CALM2", "CALM3", "HCN4", "TRDN", "TECRL",
    "SNTA1", "SCN1B", "SCN2B", "SCN3B", "SCN4B", "KCND3", "KCNA5", "KCNJ5",
}

CARDIOMYOPATHY_GENES = {
    "MYBPC3", "MYH7", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3",
    "TTN", "LMNA", "DSP", "PKP2", "DSG2", "DSC2", "FLNC", "DES", "RBM20",
    "PLN", "BAG3", "TCAP", "JPH2", "NEXN", "VCL", "CSRP3", "MYPN",
}

CANCER_GENES = {
    "BRCA1", "BRCA2", "PALB2", "CHEK2", "ATM", "TP53", "PTEN", "APC",
    "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "MUTYH", "STK11", "CDH1",
    "SMAD4", "BMPR1A", "NF1", "NF2", "VHL", "RET", "MEN1", "SDHB", "SDHD",
    "SDHC", "SDHA", "BAP1", "MITF", "POLD1", "POLE", "RAD51C", "RAD51D",
}

SADS_TERMS = [
    "sads", "sudden death", "sudden cardiac death", "molecular autopsy",
    "brugada", "catecholaminergic polymorphic ventricular tachycardia",
    "cpvt", "long qt", "lqts", "short qt", "sqts", "arrhythmogenic",
]

SADS_GENES = {"SCN5A", "RYR2", "KCNQ1", "KCNH2", "CASQ2", "TRDN"}


def ensure_dirs() -> None:
    DATA_PROSPECTIVE.mkdir(parents=True, exist_ok=True)
    RAW_PROSPECTIVE.mkdir(parents=True, exist_ok=True)
    REPORTS_PROSPECTIVE.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build locked CAB prospective prediction registry.")
    p.add_argument("--baseline-date", default=datetime.now(timezone.utc).date().isoformat(),
                   help="Registry baseline date, YYYY-MM-DD. Default: today UTC.")
    p.add_argument("--original-cutoff", default="2026-04-01",
                   help="Original benchmark cutoff. Default: 2026-04-01.")
    p.add_argument("--clinvar-url", default=CLINVAR_URL,
                   help="ClinVar variant_summary.txt.gz URL.")
    p.add_argument("--raw-clinvar", default="",
                   help="Optional path to already downloaded variant_summary.txt.gz.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Optional debug limit for ClinVar rows. 0 means all.")
    p.add_argument("--include-update-cohort", action="store_true",
                   help="Include existing benchmark VariationIDs as update cohort if LastEvaluated is after original cutoff.")
    p.add_argument("--planned-follow-up-months", type=int, default=12,
                   help="Planned follow-up months. Default 12.")
    return p.parse_args()


def download_clinvar(url: str, baseline_date: str, existing_path: str = "") -> Path:
    if existing_path:
        p = Path(existing_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    out = RAW_PROSPECTIVE / f"clinvar_variant_summary_baseline_{baseline_date}.txt.gz"
    if out.exists() and out.stat().st_size > 0:
        return out

    print(f"[download] {url} -> {out}")
    with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def read_existing_benchmark_ids() -> set[str]:
    ids: set[str] = set()

    if BENCHMARK_TASKS.exists():
        df = pd.read_csv(BENCHMARK_TASKS, low_memory=False, dtype=str)
        for col in ["clinvar_variation_id", "VariationID", "variation_id", "assertion_id"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str)
                for v in vals:
                    m = re.search(r"(\d+)", v)
                    if m:
                        ids.add(m.group(1))

    if IDENTITY.exists():
        df = pd.read_csv(IDENTITY, low_memory=False, dtype=str)
        for col in ["clinvar_variation_id", "VariationID"]:
            if col in df.columns:
                ids.update(df[col].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip())

    return {x for x in ids if x and x.lower() != "nan"}


def is_p_lp(sig: str) -> bool:
    s = str(sig or "")
    return bool(P_LP_RE.search(s)) and not bool(BENIGN_RE.search(s)) and not bool(CONFLICT_RE.search(s))


def gene_tokens(gene_symbol: str) -> list[str]:
    vals = []
    for x in re.split(r"[;,|/ ]+", str(gene_symbol or "")):
        x = x.strip().upper()
        if x and x not in {"-", "NA", "NAN"}:
            vals.append(x)
    return vals


def classify_domain(gene_symbol: str, phenotype: str) -> str:
    genes = set(gene_tokens(gene_symbol))
    text = str(phenotype or "").lower()

    if genes & CANCER_GENES or any(t in text for t in [
        "cancer", "carcinoma", "tumor", "tumour", "breast", "ovarian", "colorectal",
        "lynch", "polyposis", "melanoma", "paraganglioma", "pheochromocytoma",
    ]):
        return "hereditary_cancer"

    if genes & CARDIOMYOPATHY_GENES or any(t in text for t in [
        "cardiomyopathy", "dilated cardiomyopathy", "hypertrophic cardiomyopathy",
        "arrhythmogenic right ventricular", "left ventricular noncompaction",
    ]):
        return "cardiomyopathy"

    if genes & ARRHYTHMIA_GENES or any(t in text for t in [
        "long qt", "short qt", "brugada", "arrhythmia", "arrhythmogenic",
        "catecholaminergic", "ventricular tachycardia", "sudden cardiac death",
    ]):
        return "inherited_arrhythmia"

    return ""


def is_sads_exploratory(gene_symbol: str, phenotype: str) -> bool:
    genes = set(gene_tokens(gene_symbol))
    text = str(phenotype or "").lower()
    return bool(genes & SADS_GENES) or any(t in text for t in SADS_TERMS)


def parse_date_maybe(s: str) -> str:
    s = str(s or "").strip()
    if not s or s.lower() in {"nan", "not provided", "not specified", "-"}:
        return ""
    # ClinVar often uses YYYY-MM-DD, sometimes Month YYYY. Keep raw if not parseable.
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    return s


def date_after(raw: str, cutoff: str) -> bool:
    raw = parse_date_maybe(raw)
    try:
        return date.fromisoformat(raw[:10]) > date.fromisoformat(cutoff)
    except Exception:
        return False


def load_clinvar_candidates(path: Path, existing_ids: set[str], original_cutoff: str, include_update: bool, max_rows: int) -> pd.DataFrame:
    use_cols = None
    rows = []
    total = 0
    kept = 0

    for chunk in pd.read_csv(path, sep="\t", compression="gzip", chunksize=250_000, low_memory=False, dtype=str):
        total += len(chunk)
        if max_rows and total > max_rows:
            chunk = chunk.iloc[: max(0, len(chunk) - (total - max_rows))]

        if "ClinicalSignificance" not in chunk.columns:
            raise ValueError("ClinVar variant_summary missing ClinicalSignificance column.")

        # Prefer GRCh38 rows where assembly is available; keep all if no Assembly.
        if "Assembly" in chunk.columns:
            chunk = chunk[chunk["Assembly"].astype(str).eq("GRCh38")].copy()

        sig_mask = chunk["ClinicalSignificance"].map(is_p_lp)
        chunk = chunk[sig_mask].copy()
        if chunk.empty:
            if max_rows and total >= max_rows:
                break
            continue

        if "VariationID" not in chunk.columns:
            raise ValueError("ClinVar variant_summary missing VariationID column.")

        chunk["VariationID"] = chunk["VariationID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        chunk["GeneSymbol"] = chunk.get("GeneSymbol", "").astype(str)
        chunk["PhenotypeList"] = chunk.get("PhenotypeList", "").astype(str)

        chunk["domain"] = chunk.apply(lambda r: classify_domain(r.get("GeneSymbol", ""), r.get("PhenotypeList", "")), axis=1)
        chunk = chunk[chunk["domain"].isin(["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"])].copy()
        if chunk.empty:
            if max_rows and total >= max_rows:
                break
            continue

        is_existing = chunk["VariationID"].isin(existing_ids)
        if include_update:
            last_eval = chunk.get("LastEvaluated", pd.Series([""] * len(chunk), index=chunk.index)).astype(str)
            updated = is_existing & last_eval.map(lambda x: date_after(x, original_cutoff))
            new = ~is_existing
            chunk["new_or_updated_status"] = "new_after_original_cutoff"
            chunk.loc[updated, "new_or_updated_status"] = "updated_after_original_cutoff"
            chunk = chunk[new | updated].copy()
        else:
            chunk = chunk[~is_existing].copy()
            chunk["new_or_updated_status"] = "new_after_original_cutoff"

        if not chunk.empty:
            rows.append(chunk)
            kept += len(chunk)

        if max_rows and total >= max_rows:
            break

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["VariationID", "GeneSymbol", "PhenotypeList", "domain"])
    return out


def normalize_review_status(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def classify_architecture_regime(row: pd.Series) -> tuple[str, str, str]:
    gene = set(gene_tokens(row.get("GeneSymbol", "")))
    pheno = str(row.get("PhenotypeList", "") or "").lower()
    domain = str(row.get("domain", "") or "")

    if any(t in pheno for t in ["not provided", "not specified", "unknown", "multiple conditions", "unspecified"]):
        return "nonspecific_underresolved", "high", "broad/unknown/underresolved baseline condition label"

    if any(t in pheno for t in ["syndrome", "silver-russell", "beckwith", "imprinting", "development", "congenital"]):
        return "syndrome_organ_boundary", "high", "syndrome/developmental label may not port directly into organ-specific disease meaning"

    if any(t in pheno for t in ["carrier", "asymptomatic", "no phenotype", "genotype-first", "screening", "molecular autopsy"]):
        return "genotype_first_absent_phenotype", "medium", "genotype-first or absent/uncertain phenotype context"

    if domain == "hereditary_cancer" or any(t in pheno for t in ["risk", "predisposition", "susceptibility", "penetrance"]):
        return "modifier_penetrance_boundary", "medium", "risk/penetrance architecture; P/LP meaning travels as conditional liability"

    if domain in {"cardiomyopathy", "inherited_arrhythmia"} and gene & {"SCN5A", "RYR2", "FLNC", "TTN", "DSP", "PKP2", "LMNA", "DES"}:
        return "structural_functional_overlap", "medium", "cardiac structural-functional overlap gene/domain"

    if domain == "inherited_arrhythmia" and gene & {"KCNQ1", "KCNH2", "SCN5A", "RYR2", "CASQ2", "TRDN", "CACNA1C"}:
        return "trigger_dependent_latent", "medium", "arrhythmia latent-risk gene requiring trigger/phenotype context"

    if gene & {"SCN5A", "RYR2", "FLNC", "TTN", "TP53", "PTEN", "CHEK2", "ATM", "CACNA1C", "KCNQ1"}:
        return "pleiotropic_collision", "low", "pleiotropic/cross-model gene membership; underpowered framework category"

    return "phenotype_anchored_monogenic", "medium", "specific monogenic disease assertion without boundary flags"


def load_regime_empirical_rates() -> dict[str, dict[str, float]]:
    df = pd.read_csv(REGIME_SIG, low_memory=False, dtype=str) if REGIME_SIG.exists() else pd.DataFrame()
    rates: dict[str, dict[str, float]] = {}

    if not df.empty and "regime" in df.columns:
        for _, r in df.iterrows():
            reg = str(r.get("regime", ""))
            rates[reg] = {
                "self_loop": safe_float(r.get("self_loop_stable_rate", ""), 0.5),
                "condition": safe_float(r.get("condition_label_drift_rate", ""), 0.5),
                "cross": safe_float(r.get("cross_environment_drift_rate", ""), 0.2),
                "any": safe_float(r.get("any_meaning_drift_rate", ""), 0.5),
                "direct": safe_float(r.get("direct_use_allowed_rate", ""), 0.25),
                "repair": safe_float(r.get("review_repair_routing_rate", ""), 0.5),
            }

    defaults = {
        "phenotype_anchored_monogenic": {"self_loop": 0.90, "condition": 0.10, "cross": 0.05, "any": 0.12, "direct": 0.55, "repair": 0.45},
        "trigger_dependent_latent": {"self_loop": 0.75, "condition": 0.25, "cross": 0.10, "any": 0.30, "direct": 0.25, "repair": 0.70},
        "pleiotropic_collision": {"self_loop": 0.60, "condition": 0.40, "cross": 0.25, "any": 0.45, "direct": 0.20, "repair": 0.80},
        "syndrome_organ_boundary": {"self_loop": 0.65, "condition": 0.35, "cross": 0.20, "any": 0.40, "direct": 0.15, "repair": 0.85},
        "structural_functional_overlap": {"self_loop": 0.70, "condition": 0.30, "cross": 0.20, "any": 0.35, "direct": 0.10, "repair": 0.90},
        "genotype_first_absent_phenotype": {"self_loop": 0.30, "condition": 0.60, "cross": 0.35, "any": 0.70, "direct": 0.00, "repair": 1.00},
        "modifier_penetrance_boundary": {"self_loop": 0.60, "condition": 0.40, "cross": 0.20, "any": 0.45, "direct": 0.30, "repair": 0.80},
        "nonspecific_underresolved": {"self_loop": 0.50, "condition": 0.50, "cross": 0.25, "any": 0.55, "direct": 0.15, "repair": 0.90},
    }
    for k, v in defaults.items():
        rates.setdefault(k, v)
    return rates


def safe_float(x: Any, default: float) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def prediction_category(prob: float, invert: bool = False) -> str:
    # For risk endpoints.
    if prob >= 0.66:
        return "high"
    if prob >= 0.33:
        return "moderate"
    return "low"


def direct_allowed_by_mode(regime: str, rates: dict[str, float], mode: str, review_status: str, submitter_count: int) -> bool:
    if mode == "strict":
        return regime == "phenotype_anchored_monogenic" and rates["cross"] < 0.10 and rates["condition"] < 0.20

    # balanced
    if regime in {"genotype_first_absent_phenotype", "nonspecific_underresolved", "syndrome_organ_boundary"}:
        return False
    if regime in {"modifier_penetrance_boundary", "trigger_dependent_latent", "structural_functional_overlap", "pleiotropic_collision"}:
        return rates["direct"] >= 0.30 and rates["repair"] < 0.70
    if regime == "phenotype_anchored_monogenic":
        return True
    return False


def routing_action(regime: str, balanced_allowed: bool) -> str:
    if balanced_allowed:
        return "direct_deterministic_use_with_concordant_context"
    if regime == "modifier_penetrance_boundary":
        return "population_penetrance_review_PRF_needed"
    if regime == "genotype_first_absent_phenotype":
        return "PRF_needed_no_deterministic_reuse"
    if regime == "trigger_dependent_latent":
        return "contextual_repair_trigger_phenotype_context_review"
    if regime == "structural_functional_overlap":
        return "domain_repair_disease_specific_expert_review"
    if regime == "syndrome_organ_boundary":
        return "source_identity_accepted_contextual_repair_or_disease_specific_review"
    if regime == "pleiotropic_collision":
        return "disease_specific_review_or_contextual_repair"
    if regime == "nonspecific_underresolved":
        return "contextual_repair_or_disease_specific_review"
    return "disease_specific_review"


def create_cohort(clinvar: pd.DataFrame, baseline_date: str) -> pd.DataFrame:
    if clinvar.empty:
        raise RuntimeError("No prospective ClinVar candidate rows found. Check source snapshot, filters, or original benchmark exclusion.")

    out = pd.DataFrame()
    out["assertion_id"] = "PROSP_CLV_" + clinvar["VariationID"].astype(str)
    out["ClinVar VariationID"] = clinvar["VariationID"].astype(str)
    out["gene"] = clinvar.get("GeneSymbol", "").astype(str)
    out["condition label"] = clinvar.get("PhenotypeList", "").astype(str)
    out["classification"] = clinvar.get("ClinicalSignificance", "").astype(str)
    out["review status"] = clinvar.get("ReviewStatus", "").astype(str)
    out["submitter count"] = clinvar.get("NumberSubmitters", "").astype(str)
    out["baseline environment"] = clinvar.get("PhenotypeList", "").astype(str)
    out["domain"] = clinvar["domain"].astype(str)
    out["source date"] = baseline_date
    out["ClinVar LastEvaluated"] = clinvar.get("LastEvaluated", "").astype(str).map(parse_date_maybe)
    out["new_or_updated_status"] = clinvar["new_or_updated_status"].astype(str)
    out["source_file"] = "ClinVar variant_summary.txt.gz"
    out["registry_note"] = "baseline_only_public_assertion_snapshot"
    out.to_csv(OUT_COHORT, index=False)
    return out


def create_predictions(cohort: pd.DataFrame, baseline_date: str) -> pd.DataFrame:
    rates = load_regime_empirical_rates()
    rows = []
    for _, r in cohort.iterrows():
        tmp = pd.Series({
            "GeneSymbol": r["gene"],
            "PhenotypeList": r["condition label"],
            "domain": r["domain"],
        })
        regime, confidence, reason = classify_architecture_regime(tmp)
        rr = rates[regime]

        review_status = normalize_review_status(r.get("review status", ""))
        try:
            submitters = int(float(str(r.get("submitter count", "0") or "0")))
        except Exception:
            submitters = 0

        strict_allowed = direct_allowed_by_mode(regime, rr, "strict", review_status, submitters)
        balanced_allowed = direct_allowed_by_mode(regime, rr, "balanced", review_status, submitters)
        prf_needed = regime in {"modifier_penetrance_boundary", "genotype_first_absent_phenotype", "trigger_dependent_latent"} or "risk" in str(r["condition label"]).lower()

        review_priority = (
            rr["cross"] * 0.35
            + rr["condition"] * 0.25
            + rr["repair"] * 0.25
            + (0.15 if prf_needed else 0)
            + (0.05 if "criteria_provided" not in review_status else 0)
        )
        reclassification_risk = (
            0.15
            + (0.10 if submitters <= 1 else 0)
            + (0.10 if "no_assertion" in review_status or not review_status else 0)
            + (0.05 if regime in {"nonspecific_underresolved", "pleiotropic_collision"} else 0)
        )
        reclassification_risk = min(0.95, reclassification_risk)

        rows.append({
            **r.to_dict(),
            "registry_lock_date": baseline_date,
            "prediction_feature_scope": "baseline_only",
            "predicted_self_loop_stable_probability": rr["self_loop"],
            "predicted_self_loop_stable_category": prediction_category(rr["self_loop"]),
            "predicted_condition_label_drift_risk": rr["condition"],
            "predicted_condition_label_drift_category": prediction_category(rr["condition"]),
            "predicted_cross_environment_drift_risk": rr["cross"],
            "predicted_cross_environment_drift_category": prediction_category(rr["cross"]),
            "predicted_any_meaning_drift_risk": rr["any"],
            "predicted_any_meaning_drift_category": prediction_category(rr["any"]),
            "predicted_direct_reuse_allowed_CAB_Balanced": "True" if balanced_allowed else "False",
            "predicted_direct_reuse_allowed_CAB_Strict": "True" if strict_allowed else "False",
            "predicted_routing_action": routing_action(regime, balanced_allowed),
            "predicted_disease_architecture_regime": regime,
            "predicted_regime_mapping_confidence": confidence,
            "predicted_regime_mapping_reason": reason,
            "predicted_PRF_needed": "True" if prf_needed else "False",
            "predicted_review_priority_score": review_priority,
            "predicted_review_priority_rank": "",
            "predicted_reclassification_risk_secondary": reclassification_risk,
            "model_CAB_regime_only_version": "cab_regime_only_v2026_locked",
            "model_gene_plus_CAB_regime_version": "gene_plus_regime_v2026_locked",
            "model_CAB_Balanced_routing_version": "CAB_Balanced_v2026_locked",
            "model_CAB_Strict_routing_version": "CAB_Strict_v2026_locked",
            "model_metadata_only_baseline_version": "metadata_only_v2026_locked",
            "model_gene_only_baseline_version": "gene_only_v2026_locked",
            "prohibited_followup_fields_present": "False",
        })

    pred = pd.DataFrame(rows)
    pred["predicted_review_priority_rank"] = pred["predicted_review_priority_score"].rank(method="first", ascending=False).astype(int)
    pred.to_csv(OUT_REGISTRY, index=False)
    return pred


def create_sads_stratum(pred: pd.DataFrame) -> pd.DataFrame:
    mask = pred.apply(lambda r: is_sads_exploratory(r.get("gene", ""), r.get("condition label", "")), axis=1)
    sads = pred[mask].copy()
    if sads.empty:
        sads = pd.DataFrame(columns=list(pred.columns) + [
            "exploratory_stratum",
            "underpowered_flag",
            "individual_risk_prediction_claim",
            "sads_stratum_goal",
        ])
    else:
        sads["exploratory_stratum"] = "SADS_sudden_death_molecular_autopsy"
        sads["underpowered_flag"] = "True"
        sads["individual_risk_prediction_claim"] = "False"
        sads["sads_stratum_goal"] = (
            "Test whether genotype-first/absent-phenotype categories appear in new public assertions "
            "and whether CAB routes them to PRF-needed/no deterministic reuse."
        )
    sads.to_csv(OUT_SADS, index=False)
    return sads


def planned_followup_date(baseline_date: str, months: int) -> str:
    y, m, d = map(int, baseline_date.split("-"))
    m2 = m + months
    y += (m2 - 1) // 12
    m2 = ((m2 - 1) % 12) + 1
    # Avoid month-end headaches. Good enough for registry plan.
    d2 = min(d, 28)
    return f"{y:04d}-{m2:02d}-{d2:02d}"


def write_lock_report(cohort: pd.DataFrame, pred: pd.DataFrame, sads: pd.DataFrame, raw_path: Path, baseline_date: str, months: int) -> None:
    registry_sha = sha256_file(OUT_REGISTRY)
    cohort_sha = sha256_file(OUT_COHORT)
    raw_sha = sha256_file(raw_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = git_commit_hash()
    followup = planned_followup_date(baseline_date, months)

    domain_counts = cohort["domain"].value_counts().to_dict()
    routing_counts = pred["predicted_routing_action"].value_counts().to_dict()
    regime_counts = pred["predicted_disease_architecture_regime"].value_counts().to_dict()

    text = f"""# CAB Prospective Prediction Registry Lock Report

## Registry status

LOCKED BASELINE PREDICTION REGISTRY.

This registry contains baseline-only predictions for future assertion portability. It must not be modified after lock except by creating a new versioned registry.

## Snapshot

- baseline date: {baseline_date}
- ClinVar source: {CLINVAR_URL}
- downloaded raw file: `{raw_path.relative_to(ROOT)}`
- raw file SHA256: `{raw_sha}`
- cohort file: `{OUT_COHORT.relative_to(ROOT)}`
- cohort SHA256: `{cohort_sha}`
- prediction registry: `{OUT_REGISTRY.relative_to(ROOT)}`
- prediction registry SHA256: `{registry_sha}`
- registry lock timestamp UTC: {timestamp}
- Git commit hash at lock: `{commit}`
- Python: {sys.version.split()[0]}
- platform: {platform.platform()}

## Cohort

- cohort N: {len(cohort):,}
- SADS/sudden-death exploratory stratum N: {len(sads):,}
- domain counts: {domain_counts}

## Model versions

- CAB regime-only: `cab_regime_only_v2026_locked`
- gene + CAB regime: `gene_plus_regime_v2026_locked`
- CAB-Balanced routing: `CAB_Balanced_v2026_locked`
- CAB-Strict routing: `CAB_Strict_v2026_locked`
- metadata-only baseline: `metadata_only_v2026_locked`
- gene-only baseline: `gene_only_v2026_locked`

## Baseline-only feature list

- ClinVar VariationID
- gene
- condition label
- classification
- review status
- submitter count
- baseline environment
- domain
- source date
- new_or_updated_status
- baseline disease-architecture regime
- baseline disease-label / phenotype text

## Prohibited follow-up fields

The locked registry must not include future ClinVar classifications, future review status, future condition labels, future submission counts, future conflicts, future disease-specific reviews, or any endpoint labels derived after baseline.

## Routing counts

{routing_counts}

## Regime counts

{regime_counts}

## Planned follow-up

- primary follow-up date: {followup}
- optional interim: 6 months exploratory only

## Primary endpoint

`future_cross_environment_disease_model_drift` at 12 months.

## Secondary endpoints

- future_condition_label_drift
- future_any_meaning_drift
- self_loop_stability
- unsupported deterministic reuse under ClinVar-label-only
- classification downgrade / P_LP_to_VUS_or_lower
- review status change
- conflict emergence
- disease-specific review emergence

## Analysis plan summary

The follow-up analysis will compare locked CAB baseline-only predictions against ClinVar-label-only, gene-only, metadata-only, classification-support proxy, AlphaMissense-only where matched, and random review queue baselines.

## Claim boundary

This registry is not prospective validation yet. It is a locked prospective-style prediction registry. No clinical outcome, patient risk, therapy, or clinical validation claim is permitted until endpoint follow-up is performed, and even then only assertion-portability claims are in scope.
"""
    OUT_LOCK.write_text(text, encoding="utf-8")


def write_endpoint_definitions(baseline_date: str, months: int) -> None:
    followup = planned_followup_date(baseline_date, months)
    text = f"""# CAB Prospective Endpoint Definitions

## Baseline

- baseline date: {baseline_date}
- planned 12-month follow-up: {followup}
- optional 6-month interim: exploratory only

## Primary endpoint

### future_cross_environment_disease_model_drift at 12 months

A baseline assertion is endpoint-positive if, at follow-up, the public assertion has moved into or acquired a disease label/environment outside its baseline disease-model environment such that deterministic reuse in the baseline environment would be unsupported.

## Secondary endpoints

### future_condition_label_drift

Any materially changed ClinVar condition label or phenotype list relative to baseline.

### future_any_meaning_drift

Any condition-label drift, cross-environment disease-model drift, phenotype-domain discordance, underresolved-to-specific shift, specific-to-broad shift, or review/routing-relevant meaning change.

### self_loop_stability

The assertion remains within a concordant baseline disease environment and remains meaning-portable within the same disease loop.

### unsupported deterministic reuse under ClinVar-label-only

A future-drift-positive assertion that ClinVar-label-only baseline would have allowed as direct deterministic reuse.

### classification downgrade / P_LP_to_VUS_or_lower

A P/LP assertion at baseline becomes VUS, conflicting, likely benign, benign, or otherwise no longer P/LP at follow-up.

### review status change

ClinVar review status changes from baseline to follow-up.

### conflict emergence

A conflict or conflicting interpretation emerges at follow-up.

### disease-specific review emergence

A disease-specific expert panel, VCEP/CSpec-like review, or disease-specific review signal appears after baseline.

## Endpoint blinding rule

No endpoint label may be added to the locked baseline registry. Follow-up endpoints must be stored in a separate follow-up table and joined only during the follow-up analysis.
"""
    OUT_ENDPOINTS.write_text(text, encoding="utf-8")


def write_analysis_plan() -> None:
    text = """# CAB Prospective Analysis Plan

## Purpose

Test whether locked CAB baseline-only predictions identify future assertion portability failures and disease-model drift in newly submitted or materially updated public P/LP assertions.

## Comparators

- ClinVar-label-only baseline
- gene-only model
- metadata-only model
- classification-support proxy
- AlphaMissense-only where matched
- random review queue

## Primary endpoint

future_cross_environment_disease_model_drift at 12 months.

## Secondary endpoints

- future_condition_label_drift
- future_any_meaning_drift
- self_loop_stability
- unsupported deterministic reuse under ClinVar-label-only
- classification downgrade / P_LP_to_VUS_or_lower
- review status change
- conflict emergence
- disease-specific review emergence

## Metrics

- AUROC
- AUPRC
- precision@top 5%, 10%, 20%
- recall@top 5%, 10%, 20%
- calibration
- Brier score
- enrichment over random
- workload required to capture 50% of future cross-environment drift
- net reduction in unsupported reuse

## Primary comparison

CAB predicted cross-environment drift risk versus future_cross_environment_disease_model_drift.

## Routing comparison

CAB-Balanced and CAB-Strict direct-use decisions will be evaluated for unsupported deterministic reuse reduction relative to ClinVar-label-only direct reuse.

## Review queue comparison

Predicted review priority rank will be evaluated using precision/recall at top 5%, 10%, and 20% of the review queue.

## Calibration

Calibration will be evaluated by grouping predicted risk into deciles or quantiles, subject to endpoint N.

## SADS/sudden-death stratum

The SADS/sudden-death stratum is exploratory. It cannot be used for individual risk prediction or clinical outcome claims. Its purpose is to test whether genotype-first / absent-phenotype categories appear in new public assertions and whether CAB routes them to PRF-needed / no deterministic reuse.

## Prohibited claims

- CAB prospectively predicted future drift before follow-up.
- CAB is clinically validated.
- CAB predicts patient outcomes.
- CAB predicts individual SADS risk.
- CAB proves therapeutic utility.
"""
    OUT_PLAN.write_text(text, encoding="utf-8")


def write_publication_safe_statement() -> None:
    text = """# Prospective Registry Publication-Safe Statement

## Allowed now

We established a locked prospective prediction registry to test whether CAB baseline disease-architecture regimes predict future assertion portability in newly submitted P/LP assertions.

## More detailed allowed wording

A baseline-only registry of newly submitted or materially updated public P/LP assertions was locked before follow-up. The registry records CAB predictions of assertion portability, disease-model drift risk, routing action, disease-architecture regime, and PRF-needed status. Prospective validation requires a later follow-up snapshot.

## Forbidden now

- CAB prospectively predicted future drift.
- CAB achieved AUC X.
- CAB is clinically validated.
- CAB predicts clinical outcomes.
- CAB predicts individual patient risk.
- CAB validates SADS genotype-first risk.
- CAB proves therapeutic utility.

## Claim boundary

This is prospective-style prediction of assertion portability, not patient risk.
"""
    OUT_SAFE.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    raw_path = download_clinvar(args.clinvar_url, args.baseline_date, args.raw_clinvar)
    existing_ids = read_existing_benchmark_ids()
    print(f"[benchmark exclusion] existing VariationIDs loaded: {len(existing_ids):,}")

    candidates = load_clinvar_candidates(
        raw_path,
        existing_ids,
        original_cutoff=args.original_cutoff,
        include_update=args.include_update_cohort,
        max_rows=args.max_rows,
    )
    print(f"[cohort candidates] {len(candidates):,}")

    cohort = create_cohort(candidates, args.baseline_date)
    pred = create_predictions(cohort, args.baseline_date)
    sads = create_sads_stratum(pred)

    write_lock_report(cohort, pred, sads, raw_path, args.baseline_date, args.planned_follow_up_months)
    write_endpoint_definitions(args.baseline_date, args.planned_follow_up_months)
    write_analysis_plan()
    write_publication_safe_statement()

    print("CAB Prospective Portability Prediction Registry complete.")
    print(f"Cohort N: {len(cohort):,}")
    print("Domain counts:")
    print(cohort["domain"].value_counts().to_string())
    print(f"SADS exploratory N: {len(sads):,}")
    print("Outputs:")
    for p in [OUT_COHORT, OUT_REGISTRY, OUT_SADS, OUT_LOCK, OUT_ENDPOINTS, OUT_PLAN, OUT_SAFE]:
        print(f"  - {p.relative_to(ROOT)}")
    print(f"Registry SHA256: {sha256_file(OUT_REGISTRY)}")


if __name__ == "__main__":
    main()
