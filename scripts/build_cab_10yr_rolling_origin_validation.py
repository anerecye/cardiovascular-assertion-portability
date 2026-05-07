#!/usr/bin/env python3
"""CAB 10-year rolling-origin historical prospective validation.

This is a historical prospective emulation / temporal backtest, not true
prospective validation. It uses baseline-only features from each ClinVar
historical snapshot and evaluates against later snapshots.

Creates inventory, rolling-origin plan, baseline cohorts, locked baseline
predictions, leakage audits, follow-up endpoints, regime signatures, model
comparisons, review-queue simulations, long-horizon analyses, publication-safe
claims, and SVG figures.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path.cwd()
RAW_DIRS = [
    ROOT / "data" / "rolling_10yr" / "raw",
    ROOT / "data" / "prospective" / "raw",
    ROOT / "data" / "raw" / "external" / "clinvar",
]
ROLLING_DIR = ROOT / "data" / "rolling_10yr"
REPORTS_ROLLING = ROOT / "reports" / "rolling_10yr"
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"

ARCHIVE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/"
TARGET_YEARS = list(range(2015, 2027))
ANNUAL_WINDOWS = [(y, y + 1) for y in range(2015, 2026)]
LONG_WINDOWS = [(2015, 2017), (2015, 2020), (2015, 2025), (2018, 2020), (2020, 2023), (2022, 2026)]

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
REGIME_ORDER = [
    "phenotype_anchored_monogenic", "trigger_dependent_latent", "pleiotropic_collision",
    "syndrome_organ_boundary", "structural_functional_overlap", "genotype_first_absent_phenotype",
    "modifier_penetrance_boundary", "nonspecific_underresolved",
]
REGIME_LABELS = {
    "phenotype_anchored_monogenic": "Phenotype-anchored monogenic",
    "trigger_dependent_latent": "Trigger-dependent latent",
    "pleiotropic_collision": "Pleiotropic collision",
    "syndrome_organ_boundary": "Syndrome-organ boundary",
    "structural_functional_overlap": "Structural-functional overlap",
    "genotype_first_absent_phenotype": "Genotype-first absent phenotype",
    "modifier_penetrance_boundary": "Modifier/penetrance boundary",
    "nonspecific_underresolved": "Nonspecific/underresolved",
}
ENDPOINT_COLS = [
    "condition_label_drift", "cross_environment_drift", "any_meaning_drift",
    "semantic_drift_without_reclassification", "classification_change", "P_LP_to_VUS_or_lower",
    "review_status_change", "submitter_count_change", "self_loop_stable",
    "unsupported_deterministic_reuse_ClinVar_label_only", "unsupported_reuse_CAB_Strict",
    "unsupported_reuse_CAB_Balanced",
]


def ensure_dirs() -> None:
    for d in RAW_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    ROLLING_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_ROLLING.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAB 10-year rolling-origin temporal backtest.")
    p.add_argument("--download-missing", action="store_true")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--preferred-month", default="01")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--min-origin-n", type=int, default=50)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_archive_links() -> dict[str, str]:
    try:
        txt = urllib.request.urlopen(ARCHIVE_URL, timeout=90).read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] remote archive listing failed: {e}", file=sys.stderr)
        return {}
    links = re.findall(r'href="([^"]+)"', txt)
    return {Path(x).name: ARCHIVE_URL + x for x in links if x.endswith(".txt.gz")}


def local_file_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for d in RAW_DIRS:
        if d.exists():
            for p in d.glob("*.txt.gz"):
                idx[p.name] = p
    return idx


def best_snapshot_for_year(year: int, local: dict[str, Path], remote: dict[str, str], preferred_month: str) -> tuple[str, str, str]:
    pat = re.compile(rf"variant_summary_({year})-(\d{{2}})\.txt\.gz$")
    candidates: list[tuple[str, str, str]] = []
    for name, p in local.items():
        m = pat.search(name)
        if m:
            candidates.append((m.group(2), str(p), ""))
    for name, url in remote.items():
        m = pat.search(name)
        if m:
            candidates.append((m.group(2), "", url))
    if not candidates:
        return str(year), "", ""
    months = sorted({x[0] for x in candidates})
    chosen = preferred_month if preferred_month in months else (months[-1] if year == 2026 else months[0])
    local_match = [x for x in candidates if x[0] == chosen and x[1]]
    if local_match:
        return f"{year}-{chosen}", local_match[0][1], ""
    remote_match = [x for x in candidates if x[0] == chosen and x[2]]
    if remote_match:
        return f"{year}-{chosen}", "", remote_match[0][2]
    return f"{year}-{chosen}", "", ""


def companion(kind: str, ym: str, local: dict[str, Path], remote: dict[str, str]) -> tuple[str, str]:
    fname = f"{kind}_{ym}.txt.gz"
    if fname in local:
        return "yes", str(local[fname])
    if fname in remote:
        return "yes_remote", remote[fname]
    return "no", ""


def build_inventory(download_missing: bool, preferred_month: str) -> pd.DataFrame:
    remote = remote_archive_links()
    local = local_file_index()
    rows = []
    for year in TARGET_YEARS:
        snap, local_path, remote_url = best_snapshot_for_year(year, local, remote, preferred_month)
        if download_missing and not local_path and remote_url:
            out = RAW_DIRS[0] / Path(remote_url).name
            if not out.exists():
                print(f"[download] {remote_url} -> {out}")
                urllib.request.urlretrieve(remote_url, out)
            local_path = str(out)
            local = local_file_index()
        source = local_path or remote_url
        ym = snap if re.match(r"\d{4}-\d{2}$", snap) else ""
        sub_status, sub_src = companion("submission_summary", ym, local, remote) if ym else ("no", "")
        digest = sha256_file(Path(local_path)) if local_path and Path(local_path).exists() else ""
        rows.append({
            "snapshot_date": snap,
            "file_available": "yes" if local_path else ("remote_only" if remote_url else "no"),
            "source_url_or_path": source,
            "file_type": "variant_summary.txt.gz",
            "md5_or_sha256": digest,
            "variant_summary_available": "yes" if local_path else ("yes_remote" if remote_url else "no"),
            "submission_summary_available": sub_status,
            "submission_summary_source": sub_src,
            "XML_available": "no",
            "XML_source": "",
            "notes": "monthly archive preferred; annual origin uses selected monthly snapshot",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "clinvar_historical_snapshot_inventory.csv", index=False)
    return out


def era_for_year(y: int) -> str:
    if y < 2018:
        return "pre_2018_early_expansion"
    if y <= 2020:
        return "2018_2020"
    if y <= 2023:
        return "2021_2023"
    return "2024_2026"


def build_plan(inv: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(r["snapshot_date"])[:4]: r for _, r in inv.iterrows()}
    def make(a: int, b: int) -> dict[str, Any]:
        ba, fb = lookup.get(str(a)), lookup.get(str(b))
        bshot = str(ba.get("snapshot_date", a)) if ba is not None else str(a)
        fshot = str(fb.get("snapshot_date", b)) if fb is not None else str(b)
        avail = ba is not None and fb is not None and str(ba.get("file_available")) == "yes" and str(fb.get("file_available")) == "yes"
        return {
            "origin_id": f"{bshot}_to_{fshot}".replace("-", ""),
            "baseline_snapshot": bshot,
            "followup_snapshot": fshot,
            "horizon_months": (b - a) * 12,
            "baseline_era": era_for_year(a),
            "followup_era": era_for_year(b),
            "available": "yes" if avail else "no",
            "notes": "annual rolling origin" if b - a == 1 else "long-horizon window",
        }
    out = pd.DataFrame([make(a,b) for a,b in ANNUAL_WINDOWS + LONG_WINDOWS])
    out.to_csv(TABLES / "cab_10yr_rolling_origin_plan.csv", index=False)
    return out


def is_p_lp(sig: str) -> bool:
    s = str(sig or "")
    return bool(P_LP_RE.search(s)) and not bool(BENIGN_RE.search(s)) and not bool(CONFLICT_RE.search(s))


def gene_tokens(gene: str) -> list[str]:
    return [x.strip().upper() for x in re.split(r"[;,|/ ]+", str(gene or "")) if x.strip() and x.strip().upper() not in {"NA", "NAN", "-"}]


def classify_domain(gene: str, phenotype: str) -> str:
    genes, text = set(gene_tokens(gene)), str(phenotype or "").lower()
    if genes & CANCER_GENES or any(t in text for t in ["cancer", "carcinoma", "tumor", "tumour", "breast", "ovarian", "colorectal", "lynch", "polyposis", "melanoma"]):
        return "hereditary_cancer"
    if genes & CARDIOMYOPATHY_GENES or any(t in text for t in ["cardiomyopathy", "dilated cardiomyopathy", "hypertrophic cardiomyopathy", "left ventricular noncompaction"]):
        return "cardiomyopathy"
    if genes & ARRHYTHMIA_GENES or any(t in text for t in ["long qt", "short qt", "brugada", "arrhythmia", "catecholaminergic", "ventricular tachycardia", "sudden cardiac death"]):
        return "inherited_arrhythmia"
    return ""


def normalize_environment(label: str, gene: str = "", domain: str = "") -> str:
    text = re.sub(r"\s+", " ", re.sub(r"[\[\]\(\),;:/_+-]+", " ", str(label or "").lower())).strip()
    if not text or text in {"not provided", "not specified", "unknown", "nan"}:
        return "underresolved"
    rules = [
        ("long qt", "long_qt_syndrome"), ("lqts", "long_qt_syndrome"), ("brugada", "brugada_syndrome"),
        ("catecholaminergic", "cpvt"), ("cpvt", "cpvt"), ("sudden", "sudden_death_sads"),
        ("hypertrophic cardiomyopathy", "hypertrophic_cardiomyopathy"), ("dilated cardiomyopathy", "dilated_cardiomyopathy"),
        ("arrhythmogenic cardiomyopathy", "arrhythmogenic_cardiomyopathy"), ("left ventricular noncompaction", "left_ventricular_noncompaction"),
        ("cardiomyopathy", "cardiomyopathy_unspecified"), ("hereditary breast", "hereditary_breast_ovarian_cancer"),
        ("ovarian cancer", "hereditary_breast_ovarian_cancer"), ("lynch", "lynch_syndrome"),
        ("polyposis", "polyposis_colorectal_cancer"), ("li fraumeni", "li_fraumeni_syndrome"),
        ("cancer", "hereditary_cancer_unspecified"), ("carcinoma", "hereditary_cancer_unspecified"), ("syndrome", "syndrome_other"),
    ]
    for key, val in rules:
        if key in text:
            return val
    return f"{domain}_underresolved" if domain else "underresolved"


def disease_architecture_regime(gene: str, label: str, domain: str) -> tuple[str, str]:
    genes, text = set(gene_tokens(gene)), str(label or "").lower()
    env = normalize_environment(label, gene, domain)
    if env == "underresolved" or any(t in text for t in ["not provided", "not specified", "multiple conditions"]):
        return "nonspecific_underresolved", "broad/unknown/underresolved baseline condition label"
    if any(t in text for t in ["carrier", "asymptomatic", "no phenotype", "genotype-first", "molecular autopsy"]):
        return "genotype_first_absent_phenotype", "genotype-first or absent/uncertain phenotype context"
    if "syndrome" in text and domain in {"inherited_arrhythmia", "hereditary_cancer"} and env not in {"long_qt_syndrome", "brugada_syndrome", "lynch_syndrome", "li_fraumeni_syndrome"}:
        return "syndrome_organ_boundary", "syndrome/organ boundary label"
    if domain == "hereditary_cancer" or any(t in text for t in ["risk", "predisposition", "susceptibility", "penetrance"]):
        return "modifier_penetrance_boundary", "risk/penetrance architecture"
    if domain in {"cardiomyopathy", "inherited_arrhythmia"} and genes & {"SCN5A", "RYR2", "FLNC", "TTN", "DSP", "PKP2", "LMNA", "DES"}:
        return "structural_functional_overlap", "cardiac structural-functional overlap gene/domain"
    if domain == "inherited_arrhythmia" and genes & {"KCNQ1", "KCNH2", "SCN5A", "RYR2", "CASQ2", "TRDN", "CACNA1C"}:
        return "trigger_dependent_latent", "arrhythmia latent-risk gene requiring trigger/phenotype context"
    if genes & {"SCN5A", "RYR2", "FLNC", "TTN", "TP53", "PTEN", "CHEK2", "ATM", "CACNA1C", "KCNQ1"}:
        return "pleiotropic_collision", "pleiotropic/cross-model gene membership"
    return "phenotype_anchored_monogenic", "specific monogenic disease assertion without boundary flags"


def routing_from_regime(regime: str) -> tuple[bool, bool, str, float]:
    table = {
        "phenotype_anchored_monogenic": (True, True, "direct_deterministic_use_with_concordant_context", 0.10),
        "trigger_dependent_latent": (False, False, "contextual_repair_trigger_phenotype_context_review", 0.65),
        "pleiotropic_collision": (False, False, "disease_specific_review_or_contextual_repair", 0.75),
        "syndrome_organ_boundary": (False, False, "source_identity_accepted_contextual_repair_or_disease_specific_review", 0.80),
        "structural_functional_overlap": (False, False, "domain_repair_disease_specific_expert_review", 0.75),
        "genotype_first_absent_phenotype": (False, False, "PRF_needed_no_deterministic_reuse", 0.90),
        "modifier_penetrance_boundary": (False, False, "population_penetrance_review_PRF_needed", 0.70),
        "nonspecific_underresolved": (False, False, "contextual_repair_or_disease_specific_review", 0.85),
    }
    return table.get(regime, table["nonspecific_underresolved"])


def load_snapshot(path: Path, snapshot_date: str, max_rows: int = 0) -> pd.DataFrame:
    rows, total = [], 0
    for chunk in pd.read_csv(path, sep="\t", compression="gzip", chunksize=200_000, low_memory=False, dtype=str):
        total += len(chunk)
        if max_rows and total > max_rows:
            chunk = chunk.iloc[: max(0, len(chunk) - (total - max_rows))]
        if "ClinicalSignificance" not in chunk.columns or "VariationID" not in chunk.columns:
            if max_rows and total >= max_rows: break
            continue
        if "Assembly" in chunk.columns:
            chunk = chunk[chunk["Assembly"].astype(str).eq("GRCh38")].copy()
        chunk = chunk[chunk["ClinicalSignificance"].map(is_p_lp)].copy()
        if chunk.empty:
            if max_rows and total >= max_rows: break
            continue
        chunk["VariationID"] = chunk["VariationID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        chunk["GeneSymbol"] = chunk.get("GeneSymbol", "").astype(str)
        chunk["PhenotypeList"] = chunk.get("PhenotypeList", "").astype(str)
        chunk["domain"] = chunk.apply(lambda r: classify_domain(r.get("GeneSymbol", ""), r.get("PhenotypeList", "")), axis=1)
        chunk = chunk[chunk["domain"].isin(["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"])].copy()
        if chunk.empty:
            if max_rows and total >= max_rows: break
            continue
        chunk["snapshot_date"] = snapshot_date
        chunk["environment"] = chunk.apply(lambda r: normalize_environment(r.get("PhenotypeList", ""), r.get("GeneSymbol", ""), r.get("domain", "")), axis=1)
        chunk["assertion_id"] = "CLV_" + chunk["VariationID"].astype(str) + "_" + chunk["GeneSymbol"].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        for k in ["ReviewStatus", "NumberSubmitters"]:
            if k not in chunk.columns: chunk[k] = ""
        rows.append(chunk[["assertion_id", "VariationID", "GeneSymbol", "PhenotypeList", "ClinicalSignificance", "ReviewStatus", "NumberSubmitters", "domain", "environment", "snapshot_date"]])
        if max_rows and total >= max_rows: break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["VariationID", "GeneSymbol", "domain", "environment", "PhenotypeList"])


def snapshot_path_for_date(inv: pd.DataFrame, snap: str) -> Path | None:
    row = inv[inv["snapshot_date"].astype(str).eq(str(snap))]
    if row.empty: return None
    src = str(row.iloc[0].get("source_url_or_path", ""))
    return Path(src) if src and Path(src).exists() else None


def metadata_score(r: pd.Series) -> float:
    review = str(r.get("ReviewStatus", "")).lower()
    try: submitters = float(str(r.get("NumberSubmitters", "") or 0))
    except Exception: submitters = 0.0
    score = 0.5 - (0.15 if "criteria_provided" in review else 0) - (0.25 if "expert_panel" in review or "practice_guideline" in review else 0) + (0.15 if submitters <= 1 else 0)
    return max(0.0, min(1.0, score))


def classification_support_score(review: str, submitters: str) -> float:
    review = str(review or "").lower()
    try: n = float(str(submitters or 0))
    except Exception: n = 0.0
    score = 0.25 + (0.25 if "criteria_provided" in review else 0) + (0.20 if "multiple_submitters" in review else 0) + (0.35 if "expert_panel" in review else 0) + (0.10 if n >= 3 else 0)
    return max(0.0, min(1.0, score))


def gene_score(gene: str) -> float:
    genes = gene_tokens(gene)
    if any(g in {"SCN5A", "RYR2", "FLNC", "TTN", "CHEK2", "ATM", "KCNQ1", "TP53", "PTEN"} for g in genes): return 0.75
    if any(g in {"BRCA1", "BRCA2", "MYBPC3", "MYH7", "PKP2", "DSP", "LMNA", "KCNH2"} for g in genes): return 0.55
    return 0.35


def create_origin_files(origin: dict[str, Any], inv: pd.DataFrame, max_rows: int):
    oid = origin["origin_id"]
    outdir, rdir = ROLLING_DIR / oid, REPORTS_ROLLING / oid
    outdir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    bpath, fpath = snapshot_path_for_date(inv, origin["baseline_snapshot"]), snapshot_path_for_date(inv, origin["followup_snapshot"])
    if bpath is None or fpath is None: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    print(f"[origin] {oid}: {bpath.name} -> {fpath.name}")
    base, follow = load_snapshot(bpath, origin["baseline_snapshot"], max_rows), load_snapshot(fpath, origin["followup_snapshot"], max_rows)
    preds = []
    for _, r in base.iterrows():
        reg, reason = disease_architecture_regime(r["GeneSymbol"], r["PhenotypeList"], r["domain"])
        strict, balanced, action, risk = routing_from_regime(reg)
        preds.append({
            "assertion_id": r["assertion_id"], "variation_id": r["VariationID"], "gene": r["GeneSymbol"],
            "condition_label_baseline": r["PhenotypeList"], "environment_baseline": r["environment"],
            "classification_baseline": r["ClinicalSignificance"], "review_status_baseline": r["ReviewStatus"],
            "submitter_count_baseline": r["NumberSubmitters"], "domain": r["domain"],
            "disease_architecture_regime_baseline": reg, "regime_mapping_reason_baseline": reason,
            "cab_strict_direct_use_allowed": strict, "cab_balanced_direct_use_allowed": balanced,
            "routing_action": action, "cab_balanced_routing_score": risk, "cab_strict_routing_score": 1.0 if not strict else 0.05,
            "metadata_only_score": metadata_score(r), "gene_only_score": gene_score(r["GeneSymbol"]), "regime_only_score": risk,
            "gene_plus_regime_score": 0.5 * gene_score(r["GeneSymbol"]) + 0.5 * risk,
            "classification_support_proxy_score": classification_support_score(r["ReviewStatus"], r["NumberSubmitters"]),
            "prediction_timestamp": pd.Timestamp.utcnow().isoformat(),
        })
    pred = pd.DataFrame(preds)
    base.rename(columns={"VariationID":"variation_id", "GeneSymbol":"gene", "PhenotypeList":"condition_label_baseline", "environment":"environment_baseline", "ClinicalSignificance":"classification_baseline", "ReviewStatus":"review_status_baseline", "NumberSubmitters":"submitter_count_baseline"}).to_csv(outdir / "baseline_assertions.csv", index=False)
    pred.to_csv(outdir / "cab_baseline_predictions.csv", index=False)
    endpoints = build_followup_endpoints(pred, follow, origin)
    endpoints.to_csv(outdir / "followup_endpoints.csv", index=False)
    endpoint_counts(endpoints).to_csv(rdir / "endpoint_counts.csv", index=False)
    leakage_audit(pred).to_csv(rdir / "leakage_audit.csv", index=False)
    return base, pred, endpoints


def build_followup_endpoints(pred: pd.DataFrame, follow: pd.DataFrame, origin: dict[str, Any]) -> pd.DataFrame:
    if follow.empty:
        out = pred.copy()
        for c in ENDPOINT_COLS: out[c] = False
        return out
    follow["variation_id"] = follow["VariationID"].astype(str); follow["gene"] = follow["GeneSymbol"].astype(str)
    grouped = follow.groupby(["variation_id", "gene"], dropna=False).agg({
        "PhenotypeList": lambda x: "|".join(sorted(set(map(str, x)))), "environment": lambda x: "|".join(sorted(set(map(str, x)))),
        "ClinicalSignificance": lambda x: "|".join(sorted(set(map(str, x)))), "ReviewStatus": lambda x: "|".join(sorted(set(map(str, x)))),
        "NumberSubmitters": lambda x: "|".join(sorted(set(map(str, x)))), "domain": lambda x: "|".join(sorted(set(map(str, x)))),
    }).reset_index()
    m = pred.merge(grouped, on=["variation_id", "gene"], how="left", suffixes=("", "_followup"))
    m["followup_present"] = m["PhenotypeList"].notna()
    env_b, env_f = m["environment_baseline"].fillna("").astype(str), m["environment"].fillna("").astype(str)
    dom_b, dom_f = m["domain"].fillna("").astype(str), m["domain_followup"].fillna("").astype(str)
    m["condition_label_drift"] = m["followup_present"] & (m["condition_label_baseline"].fillna("").astype(str) != m["PhenotypeList"].fillna("").astype(str))
    m["cross_environment_drift"] = m["followup_present"] & ((~env_f.eq("")) & (~env_b.eq(env_f)) & (~env_f.str.contains(env_b, regex=False, na=False)))
    m["cross_environment_drift"] = m["cross_environment_drift"] | (m["followup_present"] & (~dom_b.eq(dom_f)) & (~dom_f.eq("")))
    m["classification_change"] = m["followup_present"] & (m["classification_baseline"].fillna("").astype(str) != m["ClinicalSignificance"].fillna("").astype(str))
    m["P_LP_to_VUS_or_lower"] = m["classification_change"] & (~m["ClinicalSignificance"].fillna("").astype(str).map(is_p_lp))
    m["review_status_change"] = m["followup_present"] & (m["review_status_baseline"].fillna("").astype(str) != m["ReviewStatus"].fillna("").astype(str))
    m["submitter_count_change"] = m["followup_present"] & (m["submitter_count_baseline"].fillna("").astype(str) != m["NumberSubmitters"].fillna("").astype(str))
    m["semantic_drift_without_reclassification"] = (m["condition_label_drift"] | m["cross_environment_drift"]) & (~m["classification_change"])
    m["any_meaning_drift"] = m["condition_label_drift"] | m["cross_environment_drift"] | m["semantic_drift_without_reclassification"]
    m["self_loop_stable"] = m["followup_present"] & (~m["any_meaning_drift"]) & (~m["classification_change"])
    m["unsupported_deterministic_reuse_ClinVar_label_only"] = m["cross_environment_drift"] | m["condition_label_drift"]
    m["unsupported_reuse_CAB_Strict"] = m["unsupported_deterministic_reuse_ClinVar_label_only"] & m["cab_strict_direct_use_allowed"].astype(bool)
    m["unsupported_reuse_CAB_Balanced"] = m["unsupported_deterministic_reuse_ClinVar_label_only"] & m["cab_balanced_direct_use_allowed"].astype(bool)
    m["origin_id"], m["baseline_snapshot"], m["followup_snapshot"], m["horizon_months"] = origin["origin_id"], origin["baseline_snapshot"], origin["followup_snapshot"], origin["horizon_months"]
    return m


def endpoint_counts(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"endpoint": c, "N": len(df), "positive_N": int(df[c].astype(bool).sum()), "rate": float(df[c].astype(bool).mean()) if len(df) else ""} for c in ENDPOINT_COLS if c in df.columns])


def leakage_audit(pred: pd.DataFrame) -> pd.DataFrame:
    pats = ["followup", "future_", "_followup", "endpoint", "outcome", "condition_label_drift", "cross_environment_drift", "any_meaning_drift", "classification_change", "self_loop_stable"]
    return pd.DataFrame([{"column": c, "uses_forbidden_future_field": "yes" if any(p in c for p in pats) else "no", "matched_patterns": "|".join([p for p in pats if p in c])} for c in sorted(pred.columns)])


def bool_rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].astype(bool).mean()) if len(df) and col in df.columns else float("nan")


def auc_rank(y: pd.Series, score: pd.Series) -> float:
    y, s = y.astype(bool), pd.to_numeric(score, errors="coerce")
    valid = s.notna(); y, s = y[valid], s[valid]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    ranks = s.rank(method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc_step(y: pd.Series, score: pd.Series) -> float:
    y, s = y.astype(bool), pd.to_numeric(score, errors="coerce")
    valid = s.notna(); y, s = y[valid], s[valid]
    if len(y) == 0 or y.sum() == 0: return float("nan")
    order = s.sort_values(ascending=False).index; y_sorted = y.loc[order].astype(int).to_numpy()
    tp = fp = 0; prev = area = 0.0; total = y.sum()
    for v in y_sorted:
        tp += int(v == 1); fp += int(v == 0)
        rec, prec = tp / total, tp / (tp + fp)
        area += prec * max(0.0, rec - prev); prev = rec
    return float(area)


def precision_recall_at_k(y: pd.Series, score: pd.Series, frac: float) -> tuple[float, float]:
    y, s = y.astype(bool), pd.to_numeric(score, errors="coerce"); valid = s.notna(); y, s = y[valid], s[valid]
    if len(y) == 0: return float("nan"), float("nan")
    k = max(1, int(math.ceil(len(y) * frac))); top = s.sort_values(ascending=False).index[:k]
    cap, pos = int(y.loc[top].sum()), int(y.sum())
    return cap / k, cap / pos if pos else float("nan")


def brier(y: pd.Series, score: pd.Series) -> float:
    y, s = y.astype(bool).astype(float), pd.to_numeric(score, errors="coerce"); valid = s.notna()
    return float(((s[valid] - y[valid]) ** 2).mean()) if valid.sum() else float("nan")


def workload_to_capture(y: pd.Series, score: pd.Series, target: float) -> float:
    y, s = y.astype(bool), pd.to_numeric(score, errors="coerce"); valid = s.notna(); y, s = y[valid], s[valid]
    total = int(y.sum())
    if len(y) == 0 or total == 0: return float("nan")
    order = s.sort_values(ascending=False).index; cum = y.loc[order].astype(int).cumsum(); idx = cum >= total * target
    return float((idx.to_numpy().argmax() + 1) / len(y)) if idx.any() else float("nan")


def summarize(all_endpoints: list[pd.DataFrame]):
    if not all_endpoints: return
    df = pd.concat(all_endpoints, ignore_index=True)
    sig_rows = []
    for (oid, reg), sub in df.groupby(["origin_id", "disease_architecture_regime_baseline"], dropna=False):
        sig_rows.append({"origin_id": oid, "baseline_snapshot": sub["baseline_snapshot"].iloc[0], "followup_snapshot": sub["followup_snapshot"].iloc[0], "horizon_months": sub["horizon_months"].iloc[0], "regime": reg, "N": len(sub), "condition_label_drift_rate": bool_rate(sub, "condition_label_drift"), "cross_environment_drift_rate": bool_rate(sub, "cross_environment_drift"), "any_meaning_drift_rate": bool_rate(sub, "any_meaning_drift"), "classification_change_rate": bool_rate(sub, "classification_change"), "self_loop_stable_rate": bool_rate(sub, "self_loop_stable"), "CAB_Strict_unsupported_reuse_rate": bool_rate(sub, "unsupported_reuse_CAB_Strict"), "CAB_Balanced_unsupported_reuse_rate": bool_rate(sub, "unsupported_reuse_CAB_Balanced"), "direct_use_allowed_rate": bool_rate(sub, "cab_balanced_direct_use_allowed"), "review_repair_rate": 1.0 - bool_rate(sub, "cab_balanced_direct_use_allowed")})
    sig = pd.DataFrame(sig_rows); sig.to_csv(TABLES / "cab_10yr_regime_temporal_signatures.csv", index=False)

    models = {"metadata-only":"metadata_only_score", "gene-only":"gene_only_score", "disease-architecture-regime-only":"regime_only_score", "gene+regime":"gene_plus_regime_score", "CAB-Balanced routing score":"cab_balanced_routing_score", "CAB-Strict routing score":"cab_strict_routing_score", "random baseline":None, "classification-support proxy":"classification_support_proxy_score"}
    endpoints = ["cross_environment_drift", "condition_label_drift", "any_meaning_drift", "unsupported_deterministic_reuse_ClinVar_label_only", "self_loop_stable"]
    rows = []
    for oid, sub in df.groupby("origin_id", dropna=False):
        for endpoint in endpoints:
            y = sub[endpoint].astype(bool)
            for model, col in models.items():
                score = pd.Series([0.5]*len(sub), index=sub.index) if col is None else sub[col]
                p5, r5 = precision_recall_at_k(y, score, 0.05); p10, r10 = precision_recall_at_k(y, score, 0.10)
                base = y.mean() if len(y) else float("nan")
                rows.append({"origin_id": oid, "baseline_snapshot": sub["baseline_snapshot"].iloc[0], "followup_snapshot": sub["followup_snapshot"].iloc[0], "horizon_months": sub["horizon_months"].iloc[0], "endpoint": endpoint, "model": model, "N": len(sub), "positive_N": int(y.sum()), "AUROC": auc_rank(y, score), "AUPRC": auprc_step(y, score), "precision_at_top_5pct": p5, "precision_at_top_10pct": p10, "recall_at_top_5pct": r5, "recall_at_top_10pct": r10, "Brier_score": brier(y, score), "enrichment_over_random_at_10pct": p10/base if base else float("nan"), "workload_required_to_capture_50pct_future_drift": workload_to_capture(y, score, 0.50)})
    by_origin = pd.DataFrame(rows); by_origin.to_csv(TABLES / "cab_10yr_model_comparison_by_origin.csv", index=False)
    summary = by_origin.groupby(["endpoint", "model"], dropna=False).agg({"AUROC":"mean", "AUPRC":"mean", "precision_at_top_5pct":"mean", "precision_at_top_10pct":"mean", "recall_at_top_5pct":"mean", "recall_at_top_10pct":"mean", "Brier_score":"mean", "enrichment_over_random_at_10pct":"mean", "origin_id":"count"}).reset_index().rename(columns={"origin_id":"origins_evaluated"})
    summary.to_csv(TABLES / "cab_10yr_model_comparison_summary.csv", index=False)

    leak_rows = []
    for p in REPORTS_ROLLING.glob("*/leakage_audit.csv"):
        la = pd.read_csv(p, dtype=str); bad = la[la["uses_forbidden_future_field"].eq("yes")]
        leak_rows.append({"origin_id": p.parent.name, "columns_checked": len(la), "forbidden_predictor_columns": len(bad), "status": "pass" if bad.empty else "fail", "bad_columns": "|".join(bad["column"].tolist()[:20])})
    pd.DataFrame(leak_rows).to_csv(TABLES / "cab_10yr_leakage_audit_summary.csv", index=False)

    # Additional tables
    dyn = []
    for (oid, reg), sub in df.groupby(["origin_id", "disease_architecture_regime_baseline"], dropna=False):
        broad = sub["environment_baseline"].astype(str).str.contains("underresolved|unspecified", regex=True, na=False)
        spec = ~sub["environment"].fillna("").astype(str).str.contains("underresolved|unspecified", regex=True, na=False)
        dyn.append({"origin_id": oid, "regime": reg, "N": len(sub), "transition_underresolved_to_anchored_rate": float((broad & spec).mean()) if len(sub) else "", "transition_broad_to_specific_rate": float((broad & spec).mean()) if len(sub) else "", "transition_unstable_to_self_loop_rate": bool_rate(sub, "self_loop_stable"), "persistent_non_portability_rate": bool_rate(sub, "any_meaning_drift"), "classification_unchanged_but_meaning_changed_rate": bool_rate(sub, "semantic_drift_without_reclassification")})
    pd.DataFrame(dyn).to_csv(TABLES / "cab_10yr_meaning_stabilization_dynamics.csv", index=False)

    df["baseline_year"] = df["baseline_snapshot"].astype(str).str.slice(0,4).astype(int); df["baseline_era"] = df["baseline_year"].map(era_for_year)
    era_rows = []
    for era, sub in df.groupby("baseline_era", dropna=False):
        era_rows.append({"era": era, "N": len(sub), "domain_composition": counts_str(sub["domain"]), "regime_composition": counts_str(sub["disease_architecture_regime_baseline"]), "condition_label_drift_rate": bool_rate(sub, "condition_label_drift"), "cross_environment_drift_rate": bool_rate(sub, "cross_environment_drift"), "any_meaning_drift_rate": bool_rate(sub, "any_meaning_drift"), "classification_change_rate": bool_rate(sub, "classification_change"), "self_loop_stable_rate": bool_rate(sub, "self_loop_stable")})
    pd.DataFrame(era_rows).to_csv(TABLES / "cab_10yr_curation_era_analysis.csv", index=False)

    rq_rows = []
    policies = {"random":None, "metadata-only":"metadata_only_score", "gene-only":"gene_only_score", "regime-only":"regime_only_score", "gene+regime":"gene_plus_regime_score", "CAB-Balanced":"cab_balanced_routing_score", "CAB-Strict":"cab_strict_routing_score"}
    for oid, sub in df.groupby("origin_id", dropna=False):
        for endpoint in ["cross_environment_drift", "condition_label_drift", "any_meaning_drift"]:
            y = sub[endpoint].astype(bool); base = y.mean() if len(y) else float("nan")
            for policy, col in policies.items():
                score = pd.Series([0.5]*len(sub), index=sub.index) if col is None else sub[col]
                for frac in [0.05, 0.10, 0.20]:
                    p, r = precision_recall_at_k(y, score, frac)
                    rq_rows.append({"origin_id": oid, "endpoint": endpoint, "policy": policy, "top_fraction": frac, "N": len(sub), "endpoint_positive_N": int(y.sum()), "precision_at_K": p, "recall_at_K": r, "enrichment_over_random": p/base if base else float("nan"), "number_needed_to_review": 1/p if p and not math.isnan(p) else float("nan"), "workload_to_capture_50pct_drift": workload_to_capture(y, score, 0.50)})
    pd.DataFrame(rq_rows).to_csv(TABLES / "cab_10yr_review_queue_results.csv", index=False)

    long_rows = []
    for (h, reg), sub in df.groupby(["horizon_months", "disease_architecture_regime_baseline"], dropna=False):
        long_rows.append({"horizon_months": h, "regime": reg, "N": len(sub), "condition_label_drift_rate": bool_rate(sub, "condition_label_drift"), "cross_environment_drift_rate": bool_rate(sub, "cross_environment_drift"), "any_meaning_drift_rate": bool_rate(sub, "any_meaning_drift"), "classification_stable_meaning_changed_rate": bool_rate(sub, "semantic_drift_without_reclassification"), "self_loop_stable_rate": bool_rate(sub, "self_loop_stable")})
    pd.DataFrame(long_rows).to_csv(TABLES / "cab_long_horizon_drift_accumulation.csv", index=False)

    claims = [
        ("allowed_if_supported", "CAB disease-architecture regimes predict future assertion meaning drift across rolling historical origins.", "historical prospective emulation, not true prospective validation"),
        ("allowed_if_supported", "Pathogenic meaning evolves over time according to disease architecture.", "assertion-portability claim only"),
        ("allowed_if_supported", "Anchored regimes show persistent self-loop stability.", "only if observed across sufficient origins"),
        ("allowed_if_supported", "Underresolved/modifier/structural-overlap regimes show persistent non-portability or repair/review routing.", "do not overclaim underpowered regimes"),
        ("allowed_if_supported", "CAB review queues enrich future boundary-crossing assertions across historical eras.", "review queue enrichment, not clinical utility"),
        ("forbidden", "CAB has true prospective validation.", "forbidden"), ("forbidden", "CAB predicts clinical outcomes.", "forbidden"),
        ("forbidden", "CAB predicts patient risk.", "forbidden"), ("forbidden", "CAB is clinically deployed.", "forbidden"),
        ("forbidden", "CAB explains all ClinVar changes.", "forbidden"), ("forbidden", "CAB replaces expert curation.", "forbidden"),
    ]
    pd.DataFrame([{"claim_type": a, "claim_text": b, "support_source": "10yr rolling-origin output tables", "claim_boundary": c} for a,b,c in claims]).to_csv(TABLES / "cab_10yr_publication_safe_claims.csv", index=False)
    make_figures(sig, summary, pd.DataFrame(rq_rows), pd.DataFrame(dyn), pd.DataFrame(era_rows), pd.DataFrame(long_rows))


def counts_str(s: pd.Series, n: int = 8) -> str:
    return "|".join([f"{k}:{int(v)}" for k,v in s.fillna("").astype(str).value_counts().head(n).items()])


def esc(x: Any) -> str:
    return html.escape(str(x))


def write_svg(path: Path, title: str, lines: list[str]):
    width, height = 1200, max(220, 80 + 22*len(lines))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#fff"/>', f'<text x="30" y="40" font-family="Arial" font-size="22" font-weight="700">{esc(title)}</text>']
    y = 75
    for line in lines:
        parts.append(f'<text x="40" y="{y}" font-family="Arial" font-size="13">{esc(line)}</text>'); y += 22
    parts.append('</svg>'); path.write_text('\n'.join(parts), encoding='utf-8')


def make_figures(sig, perf, rq, dyn, era, long):
    write_svg(FIGS / "cab_10yr_regime_signature_heatmap.svg", "CAB 10-year regime signature heatmap", [f"{r.regime}: N={r.N}, cross={float(r.cross_environment_drift_rate):.3f}, self={float(r.self_loop_stable_rate):.3f}" for _, r in sig.head(40).iterrows()] if not sig.empty else ["No data"])
    write_svg(FIGS / "cab_10yr_model_performance_over_time.svg", "CAB 10-year model performance over time", [f"{r.endpoint} / {r.model}: AUROC={float(r.AUROC):.3f}, AUPRC={float(r.AUPRC):.3f}, origins={r.origins_evaluated}" for _, r in perf.head(45).iterrows()] if not perf.empty else ["No data"])
    write_svg(FIGS / "cab_10yr_review_queue_capture.svg", "CAB 10-year review queue capture", [f"{r.policy} {r.endpoint} top={r.top_fraction}: precision={float(r.precision_at_K):.3f}, recall={float(r.recall_at_K):.3f}" for _, r in rq.head(45).iterrows()] if not rq.empty else ["No data"])
    write_svg(FIGS / "cab_10yr_meaning_stabilization_curve.svg", "CAB 10-year meaning stabilization curve", [f"{r.regime}: self-loop={float(r.transition_unstable_to_self_loop_rate):.3f}, persistent non-port={float(r.persistent_non_portability_rate):.3f}" for _, r in dyn.head(45).iterrows()] if not dyn.empty else ["No data"])
    write_svg(FIGS / "cab_10yr_curation_era_panel.svg", "CAB 10-year curation-era panel", [f"{r.era}: N={r.N}, cross={float(r.cross_environment_drift_rate):.3f}, meaning={float(r.any_meaning_drift_rate):.3f}" for _, r in era.iterrows()] if not era.empty else ["No data"])
    write_svg(FIGS / "cab_long_horizon_drift_curves.svg", "CAB long-horizon drift curves", [f"{r.horizon_months}mo {r.regime}: any={float(r.any_meaning_drift_rate):.3f}, self={float(r.self_loop_stable_rate):.3f}" for _, r in long.head(60).iterrows()] if not long.empty else ["No data"])
    final_lines = ["A. 10-year rolling-origin design", "B. Classification stability vs meaning drift", "C. Regime temporal signatures across years", "D. Model performance over time", "E. Review queue enrichment", "F. Meaning stabilization trajectories", "Claim boundary: historical prospective emulation, not true prospective validation or clinical outcome prediction."]
    write_svg(FIGS / "final_cab_10yr_temporal_backtest.svg", "Final CAB 10-year temporal backtest", final_lines)


def main():
    args = parse_args(); ensure_dirs()
    inv = build_inventory(args.download_missing, args.preferred_month)
    plan = build_plan(inv)
    if args.inventory_only:
        print("Inventory and rolling-origin plan written."); return
    endpoints = []
    for _, origin in plan[plan.available.eq("yes")].iterrows():
        _, pred, ep = create_origin_files(origin.to_dict(), inv, args.max_rows)
        if len(pred) < args.min_origin_n:
            print(f"[skip] {origin.origin_id} insufficient N={len(pred)}"); continue
        endpoints.append(ep)
    summarize(endpoints)
    print("CAB 10-year rolling-origin historical prospective emulation complete.")
    print(f"Available origins evaluated: {len(endpoints)}")
    for p in [TABLES/'clinvar_historical_snapshot_inventory.csv', TABLES/'cab_10yr_rolling_origin_plan.csv', TABLES/'cab_10yr_leakage_audit_summary.csv', TABLES/'cab_10yr_regime_temporal_signatures.csv', TABLES/'cab_10yr_model_comparison_by_origin.csv', TABLES/'cab_10yr_model_comparison_summary.csv', TABLES/'cab_10yr_meaning_stabilization_dynamics.csv', TABLES/'cab_10yr_curation_era_analysis.csv', TABLES/'cab_10yr_review_queue_results.csv', TABLES/'cab_long_horizon_drift_accumulation.csv', TABLES/'cab_10yr_publication_safe_claims.csv', FIGS/'final_cab_10yr_temporal_backtest.svg']:
        print(f"  - {p.relative_to(ROOT)}")

if __name__ == '__main__':
    main()
