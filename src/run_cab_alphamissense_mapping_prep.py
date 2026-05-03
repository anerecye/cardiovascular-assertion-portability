#!/usr/bin/env python3
"""Prepare CAB variants for AlphaMissense negative-control mapping.

This script does NOT download AlphaMissense and does NOT claim protein-level
scores are available. It extracts protein-substitution candidates from parsed
ClinVar snapshot Name fields for the CAB temporally aligned assertion universe.

Inputs
------
data/processed/cab_predictive_operational_framework.csv
data/processed/clinvar_snapshot_baseline_202301.csv
data/processed/clinvar_snapshot_followup_202604.csv

Outputs
-------
reports/tables/cab_alphamissense_mapping_candidates.csv
reports/tables/cab_alphamissense_mapping_qc.csv
reports/qc/cab_alphamissense_mapping_prep_report.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"

CAB_FRAMEWORK = DATA / "cab_predictive_operational_framework.csv"
BASELINE = DATA / "clinvar_snapshot_baseline_202301.csv"
FOLLOWUP = DATA / "clinvar_snapshot_followup_202604.csv"

CANDIDATES_OUT = TABLES / "cab_alphamissense_mapping_candidates.csv"
QC_OUT = TABLES / "cab_alphamissense_mapping_qc.csv"
REPORT_OUT = QC / "cab_alphamissense_mapping_prep_report.md"

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Ter": "*", "Xaa": "X",
}

# Strict missense pattern: p.Gly1046Arg, p.Arg123Trp, etc.
MISSENSE_RE = re.compile(
    r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?"
)

# Transcript/gene prefix in ClinVar Name, e.g. NM_014630.3(ZNF592):c.3136G>A
TRANSCRIPT_RE = re.compile(r"(?P<transcript>[A-Z]{2}_[0-9]+(?:\.[0-9]+)?)\((?P<gene>[^)]+)\):")

BAD_PROTEIN_TOKENS = [
    "fs", "del", "dup", "ins", "delins", "Ter", "*", "=", "?", "ext"
]


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def first_nonempty(values) -> str:
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def joined_unique(values) -> str:
    out = []
    seen = set()
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return " | ".join(out)


def load_snapshot_subset(path: Path, suffix: str, ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    usecols = [
        "VariationID", "Name", "Type", "GeneSymbol", "ReferenceAllele",
        "AlternateAllele", "ClinicalSignificance", "PhenotypeList",
        "ReviewStatus", "NumberSubmitters"
    ]
    # Some columns are expected from the project parser, but keep this robust.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in header]

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=250_000):
        chunk["variation_id"] = chunk["VariationID"].map(norm_id)
        sub = chunk[chunk["variation_id"].isin(ids)].copy()
        if len(sub):
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame({"variation_id": []})

    df = pd.concat(chunks, ignore_index=True)

    agg_map = {}
    for c in df.columns:
        if c in {"variation_id", "VariationID"}:
            continue
        if c == "Name":
            agg_map[c] = joined_unique
        elif c == "NumberSubmitters":
            agg_map[c] = "max"
        else:
            agg_map[c] = first_nonempty

    out = df.groupby("variation_id", as_index=False).agg(agg_map)
    rename = {c: f"{c}_{suffix}" for c in out.columns if c != "variation_id"}
    return out.rename(columns=rename)


def parse_transcript(name: str) -> Tuple[str, str]:
    if not isinstance(name, str):
        return "", ""
    m = TRANSCRIPT_RE.search(name)
    if not m:
        return "", ""
    return m.group("transcript"), m.group("gene")


def extract_hgvs_p_candidates(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Extract parenthetical p. terms.
    terms = re.findall(r"\((p\.[^)]+)\)", name)
    # Also catch plain p. tokens when parentheses are absent.
    terms += re.findall(r"\b(p\.[A-Za-z0-9_=*?]+)\b", name)
    out = []
    seen = set()
    for t in terms:
        t = t.strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return "|".join(out)


def parse_strict_missense(hgvs_p: str) -> Tuple[bool, str, str, float, str, str]:
    """Return feasible, ref_aa, alt_aa, protein_pos, normalized_hgvs_p, reason."""
    if not hgvs_p:
        return False, "", "", np.nan, "", "no_protein_change_detected"

    # If multiple protein terms exist, keep only strict single AA substitutions.
    terms = [x.strip() for x in str(hgvs_p).split("|") if x.strip()]
    strict_hits = []
    nonmissense_reasons = []

    for term in terms:
        term_clean = term.replace("(", "").replace(")", "")
        lower = term_clean.lower()

        if any(tok.lower() in lower for tok in BAD_PROTEIN_TOKENS):
            nonmissense_reasons.append("non_missense_or_uncertain_protein_change")
            continue

        m = MISSENSE_RE.search(term_clean)
        if not m:
            nonmissense_reasons.append("protein_change_not_strict_missense")
            continue

        ref3, pos, alt3 = m.group(1), m.group(2), m.group(3)
        ref1 = AA3_TO_1.get(ref3, "")
        alt1 = AA3_TO_1.get(alt3, "")
        if not ref1 or not alt1 or ref1 == "*" or alt1 == "*":
            nonmissense_reasons.append("invalid_or_stop_amino_acid")
            continue
        strict_hits.append((ref1, alt1, int(pos), f"p.{ref3}{pos}{alt3}"))

    strict_unique = list(dict.fromkeys(strict_hits))
    if len(strict_unique) == 1:
        ref1, alt1, pos, norm = strict_unique[0]
        return True, ref1, alt1, float(pos), norm, "strict_single_missense_substitution"

    if len(strict_unique) > 1:
        return False, "", "", np.nan, "|".join(x[3] for x in strict_unique), "multiple_missense_candidates_ambiguous"

    if nonmissense_reasons:
        return False, "", "", np.nan, "", sorted(set(nonmissense_reasons))[0]
    return False, "", "", np.nan, "", "no_strict_missense_match"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    if not CAB_FRAMEWORK.exists():
        raise FileNotFoundError(f"Missing {CAB_FRAMEWORK}; run CPI/CAB framework first.")

    cab = pd.read_csv(CAB_FRAMEWORK, low_memory=False)
    cab["variation_id"] = cab["variation_id"].map(norm_id)
    ids = set(cab["variation_id"].dropna().astype(str))

    baseline = load_snapshot_subset(BASELINE, "2023-01", ids)
    followup = load_snapshot_subset(FOLLOWUP, "2026-04", ids)

    df = cab.merge(baseline, on="variation_id", how="left").merge(followup, on="variation_id", how="left")

    # Prefer baseline Name; fall back to follow-up.
    df["clinvar_name_for_mapping"] = df.get("Name_2023-01", pd.Series("", index=df.index)).fillna("")
    if "Name_2026-04" in df.columns:
        df["clinvar_name_for_mapping"] = np.where(
            df["clinvar_name_for_mapping"].astype(str).str.len() > 0,
            df["clinvar_name_for_mapping"],
            df["Name_2026-04"].fillna("")
        )

    transcript_gene = df["clinvar_name_for_mapping"].map(parse_transcript)
    df["transcript_candidate"] = [x[0] for x in transcript_gene]
    df["transcript_gene_candidate"] = [x[1] for x in transcript_gene]

    df["hgvs_p_candidate_raw"] = df["clinvar_name_for_mapping"].map(extract_hgvs_p_candidates)

    parsed = df["hgvs_p_candidate_raw"].map(parse_strict_missense)
    df["alphamissense_mapping_feasible"] = [x[0] for x in parsed]
    df["protein_ref_aa"] = [x[1] for x in parsed]
    df["protein_alt_aa"] = [x[2] for x in parsed]
    df["protein_position"] = [x[3] for x in parsed]
    df["hgvs_p_normalized"] = [x[4] for x in parsed]
    df["alphamissense_mapping_status"] = [x[5] for x in parsed]

    df["missense_candidate_by_type_or_function"] = (
        df.get("Type_2023-01", pd.Series("", index=df.index)).fillna("").str.lower().str.contains("single nucleotide|snv|substitution")
        | df.get("functional_class", pd.Series("", index=df.index)).fillna("").str.lower().str.contains("missense")
        | df["hgvs_p_candidate_raw"].str.contains(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", regex=True, na=False)
    )

    keep = [
        "variation_id", "gene", "clinvar_id", "variant_key",
        "functional_class", "causal_architecture_category", "primary_regime",
        "cab_portability_index", "cab_portability_band",
        "classification_change", "condition_label_change",
        "failure_topology", "failure_topology_severity",
        "clinvar_name_for_mapping",
        "transcript_candidate", "transcript_gene_candidate",
        "hgvs_p_candidate_raw", "hgvs_p_normalized",
        "protein_ref_aa", "protein_position", "protein_alt_aa",
        "missense_candidate_by_type_or_function",
        "alphamissense_mapping_feasible",
        "alphamissense_mapping_status",
        "Type_2023-01", "ClinicalSignificance_2023-01", "PhenotypeList_2023-01",
        "Type_2026-04", "ClinicalSignificance_2026-04", "PhenotypeList_2026-04",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out.to_csv(CANDIDATES_OUT, index=False)

    qc_rows = []
    n = len(out)
    qc_rows.append({"metric": "cab_temporally_aligned_rows", "value": n})
    qc_rows.append({"metric": "rows_with_clinvar_name", "value": int(out["clinvar_name_for_mapping"].astype(str).str.len().gt(0).sum())})
    qc_rows.append({"metric": "rows_with_any_hgvs_p_candidate", "value": int(out["hgvs_p_candidate_raw"].astype(str).str.len().gt(0).sum())})
    qc_rows.append({"metric": "rows_strict_alphamissense_mapping_feasible", "value": int(out["alphamissense_mapping_feasible"].sum())})
    qc_rows.append({"metric": "mapping_feasible_rate", "value": round(float(out["alphamissense_mapping_feasible"].mean()), 4) if n else np.nan})
    qc_rows.append({"metric": "missense_candidate_by_type_or_function", "value": int(out["missense_candidate_by_type_or_function"].sum())})

    status_counts = out["alphamissense_mapping_status"].value_counts(dropna=False).reset_index()
    status_counts.columns = ["status", "n"]
    for row in status_counts.itertuples(index=False):
        qc_rows.append({"metric": f"mapping_status__{row.status}", "value": int(row.n)})

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(QC_OUT, index=False)

    feasible_by_gene = (
        out.groupby("gene", dropna=False)
        .agg(
            n=("variation_id", "size"),
            feasible_n=("alphamissense_mapping_feasible", "sum"),
            feasible_rate=("alphamissense_mapping_feasible", "mean"),
        )
        .reset_index()
        .sort_values(["feasible_n", "n"], ascending=[False, False])
    )
    feasible_by_gene["feasible_rate"] = feasible_by_gene["feasible_rate"].round(4)
    feasible_by_gene.to_csv(TABLES / "cab_alphamissense_mapping_feasibility_by_gene.csv", index=False)

    report = [
        "# CAB AlphaMissense Mapping Preparation Report",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "## Inputs",
        f"- `{CAB_FRAMEWORK.relative_to(BASE)}`",
        f"- `{BASELINE.relative_to(BASE)}`",
        f"- `{FOLLOWUP.relative_to(BASE)}`",
        "",
        "## Outputs",
        f"- `{CANDIDATES_OUT.relative_to(BASE)}`",
        f"- `{QC_OUT.relative_to(BASE)}`",
        "- `reports/tables/cab_alphamissense_mapping_feasibility_by_gene.csv`",
        "",
        "## QC summary",
        qc.to_string(index=False),
        "",
        "## Feasibility by gene",
        feasible_by_gene.to_string(index=False),
        "",
        "## Interpretation guardrails",
        "- AlphaMissense can only be joined for strict single amino-acid substitutions.",
        "- LOF, frameshift, splice, indel, delins, and stop/uncertain protein consequences are not valid AlphaMissense joins.",
        "- No AlphaMissense score is inferred here.",
        "- If strict mapping coverage is low, AlphaMissense negative-control analysis remains unavailable or missense-only sensitivity.",
        "",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print("CAB AlphaMissense mapping prep complete.")
    print(qc.to_string(index=False))
    print()
    print("Feasibility by gene:")
    print(feasible_by_gene.to_string(index=False))
    print(f"Wrote: {CANDIDATES_OUT}")


if __name__ == "__main__":
    main()
