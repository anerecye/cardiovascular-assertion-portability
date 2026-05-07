#!/usr/bin/env python3
"""Download and analyze open external comparator data for CAB/PRF.

This is the *real download/analysis* layer, distinct from the literature-derived
external proxy stack.

Open/no-application downloads implemented:
- ClinVar bulk TSV: variant_summary.txt.gz
- ClinVar gene-condition file: gene_condition_source_id
- PhysioNet ECG-arrhythmia metadata: RECORDS, ConditionNames_SNOMED-CT.csv, LICENSE
- Optional full PhysioNet waveforms via wget/AWS command instructions, not default.
- Optional local/manual PGP / LOVD / GPCards hooks if files are placed by user.

Not implemented as direct download:
- eMERGE row-level genotype/EHR data: not open bulk no-application in the way needed.
- DiscovEHR/Geisinger row-level genotype/EHR data: not open bulk no-application.

Outputs:
- reports/tables/external_downloadable_resource_status.csv
- reports/tables/external_clinvar_cab_gene_assertion_summary.csv
- reports/tables/external_clinvar_cab_benchmark_join.csv
- reports/tables/external_physionet_rhythm_label_summary.csv
- reports/tables/external_open_dataset_analysis_claims.csv
- reports/qc/external_download_analysis_limitations.md
- reports/qc/external_download_analysis_runlog.md

CAB/PRF is research software. This does not establish clinical validation,
patient-outcome validation, or prospective deployment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path.cwd()
RAW = ROOT / "data" / "raw" / "external"
PROCESSED = ROOT / "data" / "processed" / "external"
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

CLINVAR_URLS = {
    "variant_summary": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
    "gene_condition_source_id": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id",
}
PHYSIONET_URLS = {
    "records": "https://physionet.org/files/ecg-arrhythmia/1.0.0/RECORDS",
    "condition_names": "https://physionet.org/files/ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv",
    "license": "https://physionet.org/files/ecg-arrhythmia/1.0.0/LICENSE.txt",
    "sha256": "https://physionet.org/files/ecg-arrhythmia/1.0.0/SHA256SUMS.txt",
}

CAB_GENES = sorted(set("""
SCN5A KCNH2 KCNQ1 RYR2 CASQ2 TRDN CACNA1C HCN4 ANK2 KCNE1 KCNE2 KCNJ2 LMNA
MYH7 MYBPC3 TNNT2 TNNI3 TPM1 ACTC1 MYL2 MYL3 DSP PKP2 DSG2 DSC2 JUP FLNC TTN
BRCA1 BRCA2 TP53 PTEN CHEK2 ATM PALB2 MLH1 MSH2 MSH6 PMS2 APC CDH1 STK11
""".split()))

DOMAIN_GENES = {
    "inherited_arrhythmia": set("SCN5A KCNH2 KCNQ1 RYR2 CASQ2 TRDN CACNA1C HCN4 ANK2 KCNE1 KCNE2 KCNJ2 LMNA".split()),
    "cardiomyopathy": set("MYH7 MYBPC3 TNNT2 TNNI3 TPM1 ACTC1 MYL2 MYL3 DSP PKP2 DSG2 DSC2 JUP FLNC TTN LMNA".split()),
    "hereditary_cancer": set("BRCA1 BRCA2 TP53 PTEN CHEK2 ATM PALB2 MLH1 MSH2 MSH6 PMS2 APC CDH1 STK11".split()),
}


def mkdirs() -> None:
    for p in [RAW, PROCESSED, TABLES, QC, RAW / "clinvar", RAW / "physionet", RAW / "pgp", RAW / "lovd", RAW / "gpcards"]:
        p.mkdir(parents=True, exist_ok=True)


def log(msg: str, lines: List[str]) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {msg}"
    print(text)
    lines.append(text)


def download(url: str, path: Path, force: bool, runlog: List[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        log(f"SKIP existing: {path}", runlog)
        return False
    log(f"DOWNLOAD {url} -> {path}", runlog)
    req = urllib.request.Request(url, headers={"User-Agent": "CAB-external-download-analysis/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, path.open("wb") as f:
        shutil.copyfileobj(r, f)
    return True


def read_clinvar_variant_summary(path: Path, genes: List[str]) -> pd.DataFrame:
    cols = [
        "GeneSymbol", "VariationID", "AlleleID", "Type", "Name", "ClinicalSignificance",
        "ClinSigSimple", "LastEvaluated", "RS# (dbSNP)", "PhenotypeIDS", "PhenotypeList",
        "Assembly", "Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele",
        "ReviewStatus", "NumberSubmitters",
    ]
    use = []
    header = pd.read_csv(path, sep="\t", compression="gzip", nrows=0).columns.tolist()
    for c in cols:
        if c in header:
            use.append(c)
    chunks = []
    gene_set = set(genes)
    for chunk in pd.read_csv(path, sep="\t", compression="gzip", usecols=use, dtype=str, chunksize=250_000, low_memory=False):
        if "GeneSymbol" not in chunk:
            continue
        sub = chunk[chunk["GeneSymbol"].isin(gene_set)].copy()
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=use)
    return pd.concat(chunks, ignore_index=True)


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    def domain_for_gene(g: str) -> str:
        hits = [d for d, genes in DOMAIN_GENES.items() if str(g) in genes]
        return "|".join(hits) if hits else "other"
    if "GeneSymbol" in df.columns:
        df["cab_domain_gene_set"] = df["GeneSymbol"].map(domain_for_gene)
    return df


def normalize_label(x: Any) -> str:
    s = str(x or "").lower()
    s = re.sub(r"[_/|;:,]+", " ", s)
    s = re.sub(r"[^a-z0-9+ -]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def summarize_clinvar(clinvar: pd.DataFrame) -> pd.DataFrame:
    if clinvar.empty:
        return pd.DataFrame()
    rows = []
    for gene, sub in clinvar.groupby("GeneSymbol", dropna=False):
        domains = sorted(set(str(x) for x in sub.get("cab_domain_gene_set", pd.Series([""] * len(sub))).dropna()))
        sig = sub.get("ClinicalSignificance", pd.Series([""] * len(sub))).fillna("").astype(str)
        plp = sig.str.contains("Pathogenic|Likely pathogenic", case=False, regex=True, na=False)
        conditions = sub.get("PhenotypeList", pd.Series([""] * len(sub))).fillna("").astype(str)
        rows.append({
            "GeneSymbol": gene,
            "cab_domain_gene_set": "|".join(domains),
            "clinvar_rows": len(sub),
            "unique_variation_ids": sub["VariationID"].nunique() if "VariationID" in sub else "",
            "plp_or_pathogenic_label_rows": int(plp.sum()),
            "unique_condition_labels": conditions.nunique(),
            "top_condition_labels": " || ".join(conditions.value_counts().head(5).index.tolist()),
            "review_status_top": " || ".join(sub.get("ReviewStatus", pd.Series([""] * len(sub))).fillna("").astype(str).value_counts().head(5).index.tolist()),
        })
    return pd.DataFrame(rows).sort_values(["cab_domain_gene_set", "GeneSymbol"])


def benchmark_join(clinvar: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ROOT / "benchmark" / "inherited_arrhythmia" / "baseline_assertions.csv",
        ROOT / "benchmark" / "cardiomyopathy" / "baseline_assertions.csv",
        ROOT / "benchmark" / "hereditary_cancer" / "baseline_assertions.csv",
    ]
    frames = []
    for p in candidates:
        if p.exists():
            frames.append(pd.read_csv(p, dtype=str, low_memory=False))
    if not frames or clinvar.empty:
        return pd.DataFrame()
    bench = pd.concat(frames, ignore_index=True)
    if "variation_id" not in bench.columns:
        return pd.DataFrame()
    bench["variation_id_norm"] = bench["variation_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    c = clinvar.copy()
    c["variation_id_norm"] = c["VariationID"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [x for x in ["variation_id_norm", "VariationID", "GeneSymbol", "PhenotypeList", "ClinicalSignificance", "ReviewStatus", "NumberSubmitters"] if x in c.columns]
    merged = bench.merge(c[keep].drop_duplicates("variation_id_norm"), on="variation_id_norm", how="left", suffixes=("_cab", "_clinvar"))
    merged["cab_condition_norm"] = merged.get("input_condition_label", pd.Series([""] * len(merged))).map(normalize_label)
    merged["clinvar_condition_norm"] = merged.get("PhenotypeList", pd.Series([""] * len(merged))).map(normalize_label)
    merged["external_clinvar_match"] = merged["VariationID"].notna() if "VariationID" in merged else False
    merged["condition_label_exact_norm_match"] = merged["cab_condition_norm"].eq(merged["clinvar_condition_norm"])
    out_cols = [
        "assertion_id", "domain", "variation_id", "gene", "input_condition_label",
        "cab_condition_norm", "VariationID", "GeneSymbol", "PhenotypeList",
        "clinvar_condition_norm", "ClinicalSignificance", "ReviewStatus",
        "external_clinvar_match", "condition_label_exact_norm_match",
        "direct_single_model_reuse_allowed", "cab_strict_direct_use_allowed",
    ]
    return merged[[c for c in out_cols if c in merged.columns]]


def analyze_physionet(raw_dir: Path) -> pd.DataFrame:
    cond_path = raw_dir / "ConditionNames_SNOMED-CT.csv"
    rec_path = raw_dir / "RECORDS"
    rows = []
    if cond_path.exists():
        try:
            cond = pd.read_csv(cond_path)
        except Exception:
            cond = pd.read_csv(cond_path, header=None)
        for _, r in cond.iterrows():
            vals = [str(x) for x in r.tolist()]
            rows.append({
                "source": "PhysioNet ECG-arrhythmia ConditionNames_SNOMED-CT.csv",
                "label_or_record": " | ".join(vals),
                "count_or_size": "",
                "CAB_PRF_use": "phenotype/rhythm label vocabulary comparator only; no genotype-linked CAB validation",
            })
    if rec_path.exists():
        records = [x.strip() for x in rec_path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        rows.append({
            "source": "PhysioNet ECG-arrhythmia RECORDS",
            "label_or_record": "record_count",
            "count_or_size": len(records),
            "CAB_PRF_use": "arrhythmia phenotype-side dataset; no germline variant linkage",
        })
    return pd.DataFrame(rows)


def write_resource_status() -> None:
    rows = [
        {
            "resource": "ClinVar variant_summary TSV",
            "download_status": "automatic",
            "local_path": "data/raw/external/clinvar/variant_summary.txt.gz",
            "analysis_output": "external_clinvar_cab_gene_assertion_summary.csv; external_clinvar_cab_benchmark_join.csv",
            "validation_status": "external public assertion comparator, not clinical validation",
        },
        {
            "resource": "ClinVar gene_condition_source_id",
            "download_status": "automatic",
            "local_path": "data/raw/external/clinvar/gene_condition_source_id",
            "analysis_output": "downloaded for provenance/context; optional downstream parsing",
            "validation_status": "gene-condition context comparator",
        },
        {
            "resource": "PhysioNet ECG-arrhythmia metadata",
            "download_status": "automatic metadata; full waveforms optional",
            "local_path": "data/raw/external/physionet/",
            "analysis_output": "external_physionet_rhythm_label_summary.csv",
            "validation_status": "phenotype label comparator only; no genotype-linked CAB validation",
        },
        {
            "resource": "PGP",
            "download_status": "manual/optional public profile selection",
            "local_path": "data/raw/external/pgp/",
            "analysis_output": "not run unless public participant files are provided",
            "validation_status": "open individual proof-of-concept only",
        },
        {
            "resource": "LOVD",
            "download_status": "manual/optional per-gene sampling subject to terms",
            "local_path": "data/raw/external/lovd/",
            "analysis_output": "not run unless gene-specific exports are provided",
            "validation_status": "assertion comparator feasibility only",
        },
        {
            "resource": "GPCards",
            "download_status": "manual/optional if download endpoint accessible and terms allow",
            "local_path": "data/raw/external/gpcards/",
            "analysis_output": "not run unless downloaded table is provided",
            "validation_status": "genotype-phenotype comparator feasibility only",
        },
        {
            "resource": "eMERGE row-level genotype/EHR",
            "download_status": "not open no-application for needed row-level validation",
            "local_path": "",
            "analysis_output": "literature-derived proxy only unless controlled access obtained",
            "validation_status": "not downloaded; not external CAB validation",
        },
        {
            "resource": "DiscovEHR/Geisinger row-level genotype/EHR",
            "download_status": "not open no-application for needed row-level validation",
            "local_path": "",
            "analysis_output": "literature-derived penetrance rationale only unless access obtained",
            "validation_status": "not downloaded; not external CAB validation",
        },
    ]
    write_csv(TABLES / "external_downloadable_resource_status.csv", rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_claims() -> None:
    rows = [
        {
            "claim_label": "downloaded_clinvar_external_comparator",
            "allowed_claim": "ClinVar bulk public data were downloaded and used as an external assertion-label comparator for CAB genes and benchmark rows.",
            "forbidden_claim": "ClinVar download validates CAB clinical correctness.",
            "claim_strength": "external comparator analysis",
        },
        {
            "claim_label": "downloaded_physionet_phenotype_comparator",
            "allowed_claim": "PhysioNet ECG-arrhythmia metadata were downloaded and used as a rhythm/phenotype vocabulary comparator.",
            "forbidden_claim": "PhysioNet validates genotype-linked CAB portability without germline variant data.",
            "claim_strength": "phenotype-side comparator analysis",
        },
        {
            "claim_label": "not_downloaded_emerge_discovehr",
            "allowed_claim": "eMERGE and DiscovEHR remain literature-derived proxies unless row-level controlled-access data are obtained.",
            "forbidden_claim": "CAB was externally validated on eMERGE or DiscovEHR row-level data.",
            "claim_strength": "limitation",
        },
        {
            "claim_label": "not_clinical_validation",
            "allowed_claim": "The downloaded external analysis supports external comparison and feasibility, not patient-outcome validation or prospective deployment.",
            "forbidden_claim": "CAB is clinically validated.",
            "claim_strength": "required caveat",
        },
    ]
    write_csv(TABLES / "external_open_dataset_analysis_claims.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--download-physionet-waveforms", action="store_true", help="Print/optionally run large PhysioNet waveform download instructions; default false.")
    ap.add_argument("--run-large-wget", action="store_true", help="Actually run wget for full PhysioNet waveforms; requires wget and ~5.1GB uncompressed.")
    args = ap.parse_args()

    mkdirs()
    runlog: list[str] = []

    # ClinVar downloads.
    clinvar_dir = RAW / "clinvar"
    download(CLINVAR_URLS["variant_summary"], clinvar_dir / "variant_summary.txt.gz", args.force, runlog)
    download(CLINVAR_URLS["gene_condition_source_id"], clinvar_dir / "gene_condition_source_id", args.force, runlog)

    # PhysioNet metadata downloads.
    phys_dir = RAW / "physionet"
    for key, url in PHYSIONET_URLS.items():
        download(url, phys_dir / Path(url).name, args.force, runlog)

    if args.download_physionet_waveforms:
        cmd = ["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/ecg-arrhythmia/1.0.0/"]
        log("Full PhysioNet waveform download command: " + " ".join(cmd), runlog)
        if args.run_large_wget:
            log("Running full PhysioNet waveform download. This is large.", runlog)
            subprocess.run(cmd, cwd=str(phys_dir), check=True)

    # Analyze ClinVar.
    log("Reading/filtering ClinVar for CAB genes.", runlog)
    clinvar = read_clinvar_variant_summary(clinvar_dir / "variant_summary.txt.gz", CAB_GENES)
    clinvar = add_domain(clinvar)
    clinvar_out = PROCESSED / "clinvar_cab_gene_variants.tsv"
    clinvar.to_csv(clinvar_out, sep="\t", index=False)
    summary = summarize_clinvar(clinvar)
    summary.to_csv(TABLES / "external_clinvar_cab_gene_assertion_summary.csv", index=False)

    joined = benchmark_join(clinvar)
    joined.to_csv(TABLES / "external_clinvar_cab_benchmark_join.csv", index=False)

    # Analyze PhysioNet metadata.
    phys = analyze_physionet(phys_dir)
    phys.to_csv(TABLES / "external_physionet_rhythm_label_summary.csv", index=False)

    write_resource_status()
    write_claims()

    limitations = """# External Download Analysis Limitations

This is the real download/analysis layer for open external resources.

## Downloaded/analyzed

- ClinVar bulk `variant_summary.txt.gz`
- ClinVar `gene_condition_source_id`
- PhysioNet ECG-arrhythmia metadata: `RECORDS`, `ConditionNames_SNOMED-CT.csv`, license/checksum files

## Not automatically downloaded

- eMERGE row-level genotype/EHR data: not available as no-application bulk public data for this validation task.
- DiscovEHR/Geisinger row-level genotype/EHR data: not available as no-application bulk public data for this validation task.
- PGP participant-level files: open-consent but identifiable; use manual profile selection and minimum necessary extraction.
- LOVD/GPCards: use manual or per-terms download/sampling; do not scrape aggressively.

## Claim boundary

This analysis provides external public comparator analysis and phenotype-side feasibility analysis. It does not provide patient-outcome validation, prospective clinical deployment, or clinical validation of CAB/PRF.
"""
    (QC / "external_download_analysis_limitations.md").write_text(limitations, encoding="utf-8")
    (QC / "external_download_analysis_runlog.md").write_text("\n".join(runlog) + "\n", encoding="utf-8")

    print("External open dataset download/analysis complete.")
    print(f"ClinVar CAB rows: {len(clinvar):,}")
    print(f"ClinVar benchmark joined rows: {len(joined):,}")
    print(f"PhysioNet metadata rows: {len(phys):,}")
    print("Outputs:")
    for p in [
        TABLES / "external_downloadable_resource_status.csv",
        TABLES / "external_clinvar_cab_gene_assertion_summary.csv",
        TABLES / "external_clinvar_cab_benchmark_join.csv",
        TABLES / "external_physionet_rhythm_label_summary.csv",
        TABLES / "external_open_dataset_analysis_claims.csv",
        QC / "external_download_analysis_limitations.md",
        QC / "external_download_analysis_runlog.md",
    ]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
