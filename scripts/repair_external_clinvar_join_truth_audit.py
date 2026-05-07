
#!/usr/bin/env python3
"""Diagnose and repair external ClinVar join for CAB benchmark rows.

Why this exists
---------------
A left join can produce 26,725 rows even when no true external ClinVar match
occurred. The real match must be counted by `external_clinvar_match == True` or
by non-empty external ClinVar fields such as VariationID/GeneSymbol/PhenotypeList.

This script:
1. Audits reports/tables/external_clinvar_cab_benchmark_join.csv.
2. Searches data/processed/*.csv and benchmark/*/baseline_assertions.csv for a
   numeric ClinVar VariationID mapping.
3. Rebuilds a corrected join if a numeric VariationID is available.
4. Writes explicit match QC and corrected claims.

Inputs expected:
- data/raw/external/clinvar/variant_summary.txt.gz
- reports/tables/external_clinvar_cab_benchmark_join.csv
- benchmark/*/baseline_assertions.csv
- optionally data/processed/*.csv with numeric VariationID columns

Outputs:
- reports/tables/external_clinvar_join_truth_audit.csv
- reports/tables/external_clinvar_cab_benchmark_join_repaired.csv
- reports/tables/external_clinvar_join_repair_source_candidates.csv
- reports/tables/external_open_dataset_analysis_claims_repaired.csv
- reports/qc/external_clinvar_join_truth_audit.md

CAB is not clinically validated by ClinVar. This is source-traceability /
external public assertion comparator analysis only.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

OLD_JOIN = TABLES / "external_clinvar_cab_benchmark_join.csv"
CLINVAR = ROOT / "data" / "raw" / "external" / "clinvar" / "variant_summary.txt.gz"

OUT_AUDIT = TABLES / "external_clinvar_join_truth_audit.csv"
OUT_REPAIRED = TABLES / "external_clinvar_cab_benchmark_join_repaired.csv"
OUT_CANDIDATES = TABLES / "external_clinvar_join_repair_source_candidates.csv"
OUT_CLAIMS = TABLES / "external_open_dataset_analysis_claims_repaired.csv"
OUT_MD = QC / "external_clinvar_join_truth_audit.md"

DOMAINS = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]
CAB_GENES = {
    # inherited arrhythmia
    "ANK2", "CACNA1C", "CASQ2", "HCN4", "KCNE1", "KCNE2", "KCNH2", "KCNJ2", "KCNQ1",
    "RYR2", "SCN5A", "TRDN",
    # cardiomyopathy
    "ACTC1", "DSC2", "DSG2", "DSP", "FLNC", "JUP", "MYBPC3", "MYH7", "MYL2", "MYL3",
    "PKP2", "TNNI3", "TNNT2", "TPM1", "TTN",
    # hereditary cancer
    "APC", "ATM", "BRCA1", "BRCA2", "CDH1", "CHEK2", "MLH1", "MSH2", "MSH6", "PALB2",
    "PMS2", "PTEN", "STK11", "TP53",
}


def norm_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def is_numeric_id_series(s: pd.Series) -> pd.Series:
    return norm_id_series(s).str.fullmatch(r"\d+")


def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig", **kwargs)


def audit_existing_join() -> dict:
    if not OLD_JOIN.exists():
        return {"status": "missing_old_join", "rows": 0}

    df = read_csv_safe(OLD_JOIN)
    rows = len(df)

    if "external_clinvar_match" in df.columns:
        explicit_matches = int(truthy_series(df["external_clinvar_match"]).sum())
    else:
        explicit_matches = None

    ext_field_cols = [c for c in ["VariationID", "GeneSymbol", "ClinicalSignificance", "PhenotypeList", "ReviewStatus"] if c in df.columns]
    nonempty_ext = 0
    if ext_field_cols:
        nonempty_ext = int(df[ext_field_cols].fillna("").astype(str).apply(lambda r: any(x.strip() for x in r), axis=1).sum())

    numeric_variation_id = None
    if "variation_id" in df.columns:
        numeric_variation_id = int(is_numeric_id_series(df["variation_id"]).sum())

    return {
        "status": "audited",
        "rows": rows,
        "explicit_external_clinvar_match_true": explicit_matches,
        "rows_with_any_external_clinvar_fields": nonempty_ext,
        "rows_with_numeric_variation_id_in_join_file": numeric_variation_id,
        "interpretation": (
            "left_join_only_not_true_match"
            if (explicit_matches == 0 or explicit_matches is None) and nonempty_ext == 0
            else "has_some_true_external_matches"
        ),
    }


def load_benchmark_rows() -> pd.DataFrame:
    frames = []
    for domain in DOMAINS:
        p = ROOT / "benchmark" / domain / "baseline_assertions.csv"
        if p.exists():
            d = read_csv_safe(p)
            if "domain" not in d.columns:
                d["domain"] = domain
            frames.append(d)
    if not frames:
        raise FileNotFoundError("No benchmark/*/baseline_assertions.csv files found.")
    out = pd.concat(frames, ignore_index=True)
    if "assertion_id" not in out.columns:
        raise ValueError("Benchmark baseline files must contain assertion_id.")
    out["assertion_id"] = norm_id_series(out["assertion_id"])
    return out


def source_candidate_rows(path: Path, df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    id_cols = [c for c in ["assertion_id", "variation_id", "VariationID", "AlleleID"] if c in cols]
    has_numeric_variation = False
    numeric_count = 0

    for c in ["VariationID", "variation_id"]:
        if c in cols:
            n = int(is_numeric_id_series(df[c]).sum())
            if n:
                has_numeric_variation = True
                numeric_count = max(numeric_count, n)

    return {
        "source_path": str(path.relative_to(ROOT)),
        "rows": len(df),
        "columns_found": "|".join(id_cols),
        "has_assertion_id": "assertion_id" in cols,
        "has_variation_id": "variation_id" in cols,
        "has_VariationID": "VariationID" in cols,
        "numeric_variation_id_rows": numeric_count,
        "candidate_strength": (
            "strong" if ("assertion_id" in cols and has_numeric_variation)
            else "medium" if has_numeric_variation
            else "weak"
        ),
    }


def scan_mapping_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    mappings = []

    paths = list((ROOT / "data" / "processed").glob("*.csv"))
    paths += list((ROOT / "benchmark").glob("*/baseline_assertions.csv"))

    for p in paths:
        try:
            df = read_csv_safe(p)
        except Exception:
            continue

        candidates.append(source_candidate_rows(p, df))

        cols = set(df.columns)
        numeric_col = None
        for c in ["VariationID", "variation_id"]:
            if c in cols and int(is_numeric_id_series(df[c]).sum()) > 0:
                numeric_col = c
                break
        if not numeric_col:
            continue

        tmp = pd.DataFrame()
        tmp["clinvar_variation_id"] = norm_id_series(df[numeric_col])
        tmp = tmp[is_numeric_id_series(tmp["clinvar_variation_id"])].copy()

        if "assertion_id" in cols:
            tmp["assertion_id"] = norm_id_series(df.loc[tmp.index, "assertion_id"])
        else:
            tmp["assertion_id"] = ""

        if "variation_id" in cols:
            tmp["synthetic_or_local_variation_id"] = norm_id_series(df.loc[tmp.index, "variation_id"])
        else:
            tmp["synthetic_or_local_variation_id"] = ""

        if "GeneSymbol" in cols:
            tmp["source_gene"] = df.loc[tmp.index, "GeneSymbol"].astype(str)
        elif "gene" in cols:
            tmp["source_gene"] = df.loc[tmp.index, "gene"].astype(str)
        else:
            tmp["source_gene"] = ""

        tmp["mapping_source_path"] = str(p.relative_to(ROOT))
        mappings.append(tmp)

    cand_df = pd.DataFrame(candidates)
    map_df = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()
    if not map_df.empty:
        map_df = map_df.drop_duplicates()
    return cand_df, map_df


def choose_mapping(benchmark: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    b = benchmark[["assertion_id"]].copy()
    if "variation_id" in benchmark.columns:
        b["synthetic_or_local_variation_id"] = norm_id_series(benchmark["variation_id"])
    else:
        b["synthetic_or_local_variation_id"] = ""

    # First try assertion_id.
    m1 = b.merge(
        map_df[["assertion_id", "clinvar_variation_id", "mapping_source_path"]],
        on="assertion_id",
        how="left",
    )
    m1_hits = int(m1["clinvar_variation_id"].notna().sum())

    # Then try local/synthetic variation_id.
    m2 = b.merge(
        map_df[["synthetic_or_local_variation_id", "clinvar_variation_id", "mapping_source_path"]],
        on="synthetic_or_local_variation_id",
        how="left",
    )
    m2_hits = int(m2["clinvar_variation_id"].notna().sum())

    if m1_hits >= m2_hits and m1_hits > 0:
        out = m1
        out["mapping_key_used"] = "assertion_id"
    elif m2_hits > 0:
        out = m2
        out["mapping_key_used"] = "variation_id"
    else:
        return pd.DataFrame()

    out = out.drop_duplicates(subset=["assertion_id"])
    return out[["assertion_id", "clinvar_variation_id", "mapping_source_path", "mapping_key_used"]]


def load_clinvar_subset(variation_ids: set[str]) -> pd.DataFrame:
    if not CLINVAR.exists():
        raise FileNotFoundError(CLINVAR)

    chunks = []
    usecols = [
        "VariationID",
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "PhenotypeList",
        "Type",
        "Name",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAllele",
        "AlternateAllele",
    ]
    for chunk in pd.read_csv(
        CLINVAR,
        sep="\t",
        compression="gzip",
        low_memory=False,
        chunksize=250_000,
    ):
        cols = [c for c in usecols if c in chunk.columns]
        sub = chunk[cols].copy()
        sub["VariationID"] = norm_id_series(sub["VariationID"])
        hit = sub[sub["VariationID"].isin(variation_ids)].copy()
        if not hit.empty:
            chunks.append(hit)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    cv = pd.concat(chunks, ignore_index=True)
    cv = cv.drop_duplicates(subset=["VariationID"])
    return cv


def repair_join() -> tuple[pd.DataFrame, dict]:
    benchmark = load_benchmark_rows()
    cand_df, map_df = scan_mapping_sources()
    cand_df.to_csv(OUT_CANDIDATES, index=False)

    mapping = choose_mapping(benchmark, map_df)
    if mapping.empty:
        repaired = benchmark.copy()
        repaired["external_clinvar_match"] = False
        repaired["clinvar_join_failure_reason"] = "no_numeric_variation_id_mapping_found"
        return repaired, {
            "repair_status": "failed_no_numeric_mapping",
            "benchmark_rows": len(benchmark),
            "mapped_rows": 0,
            "true_external_matches": 0,
        }

    ids = set(mapping["clinvar_variation_id"].dropna().astype(str))
    cv = load_clinvar_subset(ids)

    repaired = benchmark.merge(mapping, on="assertion_id", how="left")
    repaired = repaired.merge(cv, left_on="clinvar_variation_id", right_on="VariationID", how="left", suffixes=("", "_clinvar"))

    repaired["external_clinvar_match"] = repaired["VariationID"].notna()
    repaired["clinvar_join_failure_reason"] = ""
    repaired.loc[repaired["clinvar_variation_id"].isna(), "clinvar_join_failure_reason"] = "no_numeric_variation_id_mapping"
    repaired.loc[repaired["clinvar_variation_id"].notna() & repaired["VariationID"].isna(), "clinvar_join_failure_reason"] = "numeric_variation_id_not_found_in_downloaded_clinvar"

    stats = {
        "repair_status": "completed",
        "benchmark_rows": len(repaired),
        "mapped_rows": int(repaired["clinvar_variation_id"].notna().sum()),
        "true_external_matches": int(repaired["external_clinvar_match"].sum()),
        "true_external_match_rate": float(repaired["external_clinvar_match"].mean()) if len(repaired) else 0.0,
        "mapping_sources": "; ".join(sorted(repaired["mapping_source_path"].dropna().astype(str).unique())),
    }
    return repaired, stats


def write_claims(stats: dict) -> None:
    if stats.get("true_external_matches", 0) > 0:
        allowed = (
            f"Downloaded ClinVar public data were used to produce {stats['true_external_matches']:,} true row-level "
            f"external assertion-source matches for CAB benchmark rows."
        )
        strength = "external comparator analysis"
        caveat = "This supports source traceability/comparability, not clinical validation or patient-outcome validation."
    else:
        allowed = (
            "Downloaded ClinVar public data were analyzed for CAB genes, but the current CAB benchmark rows did not contain "
            "a usable numeric ClinVar VariationID mapping for true row-level external joins."
        )
        strength = "external comparator feasibility; join repair needed"
        caveat = "Do not claim the 26,725 benchmark rows matched ClinVar until true external_clinvar_match is positive."

    rows = [
        {
            "claim_label": "downloaded_clinvar_external_comparator",
            "claim_strength": strength,
            "allowed_claim": allowed,
            "required_caveat": caveat,
        },
        {
            "claim_label": "downloaded_physionet_phenotype_comparator",
            "claim_strength": "phenotype-side comparator analysis",
            "allowed_claim": "PhysioNet ECG-arrhythmia metadata were downloaded and used as a rhythm/phenotype label vocabulary comparator.",
            "required_caveat": "This is not genotype-linked CAB validation and not waveform-level outcome validation.",
        },
        {
            "claim_label": "not_clinical_validation",
            "claim_strength": "required caveat",
            "allowed_claim": "The downloaded external analysis supports external comparison and feasibility only.",
            "required_caveat": "No patient-outcome validation, no prospective clinical deployment, no clinical validation of CAB/PRF.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_CLAIMS, index=False)


def write_markdown(audit: dict, repair_stats: dict) -> None:
    lines = [
        "# External ClinVar Join Truth Audit",
        "",
        "## Existing join audit",
        "",
        f"- rows in existing join file: {audit.get('rows', 0):,}",
        f"- explicit `external_clinvar_match == True`: {audit.get('explicit_external_clinvar_match_true')}",
        f"- rows with any external ClinVar fields populated: {audit.get('rows_with_any_external_clinvar_fields')}",
        f"- rows with numeric `variation_id` in existing join file: {audit.get('rows_with_numeric_variation_id_in_join_file')}",
        f"- interpretation: `{audit.get('interpretation')}`",
        "",
        "A 26,725-row left join is not the same as 26,725 true ClinVar matches. True matching requires a positive match flag or populated external ClinVar fields.",
        "",
        "## Repair attempt",
        "",
        f"- repair status: `{repair_stats.get('repair_status')}`",
        f"- benchmark rows: {repair_stats.get('benchmark_rows', 0):,}",
        f"- rows with numeric mapping: {repair_stats.get('mapped_rows', 0):,}",
        f"- true external ClinVar matches: {repair_stats.get('true_external_matches', 0):,}",
        f"- true external match rate: {repair_stats.get('true_external_match_rate', 0):.4f}",
        f"- mapping sources: {repair_stats.get('mapping_sources', '')}",
        "",
        "## Claim boundary",
        "",
        "ClinVar can support public assertion-source comparability and traceability. It does not clinically validate CAB, does not validate patient outcomes, and does not validate prospective deployment.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    audit = audit_existing_join()
    repaired, repair_stats = repair_join()

    pd.DataFrame([audit | repair_stats]).to_csv(OUT_AUDIT, index=False)
    repaired.to_csv(OUT_REPAIRED, index=False)
    write_claims(repair_stats)
    write_markdown(audit, repair_stats)

    print("External ClinVar join truth audit complete.")
    print(pd.DataFrame([audit | repair_stats]).to_string(index=False))
    print()
    print("Wrote:")
    print(f"  - {OUT_AUDIT.relative_to(ROOT)}")
    print(f"  - {OUT_REPAIRED.relative_to(ROOT)}")
    print(f"  - {OUT_CANDIDATES.relative_to(ROOT)}")
    print(f"  - {OUT_CLAIMS.relative_to(ROOT)}")
    print(f"  - {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
