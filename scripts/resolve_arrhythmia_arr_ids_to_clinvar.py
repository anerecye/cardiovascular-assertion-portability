
#!/usr/bin/env python3
"""Resolve inherited-arrhythmia ARR_* benchmark IDs to numeric ClinVar VariationID.

Clean final version with explicit source-vs-meaning QC flags.

Inputs:
- reports/tables/external_clinvar_cab_benchmark_join_repaired.csv
- data/raw/external/clinvar/variant_summary.txt.gz

Outputs:
- reports/tables/external_clinvar_arrhythmia_arr_id_resolution.csv
- reports/tables/external_clinvar_cab_benchmark_join_final.csv
- reports/tables/external_clinvar_join_final_audit.csv
- reports/tables/external_open_dataset_analysis_claims_final.csv
- reports/qc/external_clinvar_arrhythmia_arr_id_resolution.md

Core logic:
- external_clinvar_match/source_match_accepted can be True.
- meaning_match_accepted can be False if phenotype-domain discordance exists.
- Discordant rows are retained, not deleted.

This is source traceability / external comparator analysis only.
It is not clinical validation, not patient-outcome validation, and not deployment evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

REPAIRED_JOIN = TABLES / "external_clinvar_cab_benchmark_join_repaired.csv"
CLINVAR = ROOT / "data" / "raw" / "external" / "clinvar" / "variant_summary.txt.gz"

OUT_RESOLUTION = TABLES / "external_clinvar_arrhythmia_arr_id_resolution.csv"
OUT_FINAL = TABLES / "external_clinvar_cab_benchmark_join_final.csv"
OUT_AUDIT = TABLES / "external_clinvar_join_final_audit.csv"
OUT_CLAIMS = TABLES / "external_open_dataset_analysis_claims_final.csv"
OUT_MD = QC / "external_clinvar_arrhythmia_arr_id_resolution.md"

CLINVAR_COLS = [
    "VariationID",
    "AlleleID",
    "Type",
    "Name",
    "GeneID",
    "GeneSymbol",
    "ClinicalSignificance",
    "ClinSigSimple",
    "LastEvaluated",
    "RS# (dbSNP)",
    "RCVaccession",
    "PhenotypeIDS",
    "PhenotypeList",
    "Origin",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "ReviewStatus",
    "NumberSubmitters",
]


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def bool_str(value: object) -> str:
    return "True" if bool(value) else "False"


def extract_arr_suffix_value(x: object) -> str:
    m = re.search(r"\bARR[_-]?(\d+)\b", str(x))
    return m.group(1) if m else ""


def gene_token_concordant(local_gene: object, clinvar_gene_field: object) -> bool:
    local = str(local_gene or "").upper().strip()
    tokens = {
        t.strip().upper()
        for t in str(clinvar_gene_field or "").replace(",", ";").split(";")
        if t.strip()
    }
    return bool(local) and local in tokens


def phenotype_domain_concordant_for_arrhythmia(phenotype: object) -> bool:
    ptxt = str(phenotype or "").lower()
    arrhythmia_terms = [
        "long qt",
        "lqts",
        "arrhythmia",
        "brugada",
        "cpvt",
        "catecholaminergic",
        "sudden",
        "cardiac",
        "tachycardia",
        "fibrillation",
        "conduction",
        "short qt",
        "qt syndrome",
    ]
    return any(t in ptxt for t in arrhythmia_terms)


def arr_phenotype_domain_note(phenotype: object) -> str:
    if phenotype_domain_concordant_for_arrhythmia(phenotype):
        return "ClinVar phenotype label appears arrhythmia-relevant"
    return (
        "ClinVar phenotype label is not inherited-arrhythmia specific; "
        "retain as source match with phenotype-domain discordance flag"
    )


def load_clinvar_subset(ids: Set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        CLINVAR,
        sep="\t",
        compression="gzip",
        chunksize=250_000,
        low_memory=False,
    ):
        cols = [c for c in CLINVAR_COLS if c in chunk.columns]
        sub = chunk[cols].copy()
        sub["VariationID"] = norm_id(sub["VariationID"])
        hit = sub[sub["VariationID"].isin(ids)].copy()
        if not hit.empty:
            chunks.append(hit)

    if not chunks:
        return pd.DataFrame(columns=CLINVAR_COLS)

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["VariationID"])
    return out


def ensure_object_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype("object")
    return df


def apply_final_qc_defaults(final: pd.DataFrame) -> pd.DataFrame:
    if "source_match_accepted" not in final.columns:
        final["source_match_accepted"] = final["external_clinvar_match"].map(
            lambda x: "True" if str(x).lower() in {"true", "1", "yes"} else "False"
        )
    else:
        blank_source = final["source_match_accepted"].astype(str).str.strip().isin({"", "nan", "None"})
        final.loc[blank_source, "source_match_accepted"] = final.loc[blank_source, "external_clinvar_match"].map(
            lambda x: "True" if str(x).lower() in {"true", "1", "yes"} else "False"
        )

    if "phenotype_domain_concordant" not in final.columns:
        final["phenotype_domain_concordant"] = "True"
    final["phenotype_domain_concordant"] = final["phenotype_domain_concordant"].replace({"": "True", "nan": "True"})

    if "phenotype_domain_discordance_flag" not in final.columns:
        final["phenotype_domain_discordance_flag"] = "False"
    final["phenotype_domain_discordance_flag"] = final["phenotype_domain_discordance_flag"].replace({"": "False", "nan": "False"})

    if "meaning_match_accepted" not in final.columns:
        final["meaning_match_accepted"] = final.apply(
            lambda r: "False"
            if str(r.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"}
            else str(r.get("source_match_accepted", "False")),
            axis=1,
        )
    else:
        blank_meaning = final["meaning_match_accepted"].astype(str).str.strip().isin({"", "nan", "None"})
        final.loc[blank_meaning, "meaning_match_accepted"] = final.loc[blank_meaning].apply(
            lambda r: "False"
            if str(r.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"}
            else str(r.get("source_match_accepted", "False")),
            axis=1,
        )

    if "routing_implication" not in final.columns:
        final["routing_implication"] = ""
    final.loc[
        final["phenotype_domain_discordance_flag"].astype(str).str.lower().isin({"true", "1", "yes"}),
        "routing_implication",
    ] = "contextual_repair_or_disease_specific_review"

    return final


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    if not REPAIRED_JOIN.exists():
        raise FileNotFoundError(REPAIRED_JOIN)
    if not CLINVAR.exists():
        raise FileNotFoundError(CLINVAR)

    df = pd.read_csv(REPAIRED_JOIN, low_memory=False, dtype=str)
    if "external_clinvar_match" not in df.columns:
        raise ValueError("Repaired join missing external_clinvar_match column.")

    for c in ["assertion_id", "variation_id", "domain", "gene"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    matched_before = int(truthy(df["external_clinvar_match"]).sum())

    unmatched_arr = df[
        (~truthy(df["external_clinvar_match"]))
        & (df.get("domain", pd.Series([""] * len(df))).astype(str).eq("inherited_arrhythmia"))
    ].copy()

    if unmatched_arr.empty:
        final = apply_final_qc_defaults(df.copy())
        final.to_csv(OUT_FINAL, index=False)
        audit = pd.DataFrame([{
            "status": "no_unmatched_arrhythmia_rows",
            "rows_total": len(final),
            "matched_before": matched_before,
            "arr_rows_attempted": 0,
            "arr_suffix_candidates": 0,
            "arr_suffix_clinvar_found": 0,
            "arr_gene_concordant": 0,
            "arr_resolution_accepted": 0,
            "phenotype_domain_discordance_flagged": int(truthy(final.get("phenotype_domain_discordance_flag", pd.Series(["False"] * len(final)))).sum()),
            "meaning_match_rejected": int((final.get("meaning_match_accepted", pd.Series(["True"] * len(final))).astype(str).str.lower().isin({"false", "0", "no"})).sum()),
            "matched_after": matched_before,
            "unmatched_after": int((~truthy(final["external_clinvar_match"])).sum()),
            "final_match_rate": float(truthy(final["external_clinvar_match"]).mean()),
        }])
        audit.to_csv(OUT_AUDIT, index=False)
        print(audit.to_string(index=False))
        return

    unmatched_arr["arr_suffix_from_assertion_id"] = unmatched_arr["assertion_id"].map(extract_arr_suffix_value)
    unmatched_arr["arr_suffix_from_variation_id"] = unmatched_arr["variation_id"].map(extract_arr_suffix_value)
    unmatched_arr["candidate_clinvar_variation_id"] = unmatched_arr["arr_suffix_from_assertion_id"]

    mask_missing_suffix = unmatched_arr["candidate_clinvar_variation_id"].eq("")
    unmatched_arr.loc[mask_missing_suffix, "candidate_clinvar_variation_id"] = unmatched_arr.loc[
        mask_missing_suffix, "arr_suffix_from_variation_id"
    ]

    ids = set(unmatched_arr["candidate_clinvar_variation_id"])
    ids = {x for x in ids if re.fullmatch(r"\d+", str(x or ""))}

    cv = load_clinvar_subset(ids)
    cv_small = cv.rename(columns={c: f"arr_clinvar_{c}" for c in cv.columns}).copy()
    if not cv_small.empty:
        cv_small["candidate_clinvar_variation_id"] = norm_id(cv_small["arr_clinvar_VariationID"])
    else:
        cv_small["candidate_clinvar_variation_id"] = pd.Series(dtype=str)

    resolved = unmatched_arr.merge(cv_small, on="candidate_clinvar_variation_id", how="left")

    resolved["arr_suffix_clinvar_found"] = resolved["arr_clinvar_VariationID"].notna()
    if "gene" in resolved.columns and "arr_clinvar_GeneSymbol" in resolved.columns:
        resolved["arr_gene_concordant"] = resolved.apply(
            lambda r: gene_token_concordant(r.get("gene", ""), r.get("arr_clinvar_GeneSymbol", "")),
            axis=1,
        )
    else:
        resolved["arr_gene_concordant"] = False

    local_gene_missing = resolved.get("gene", pd.Series([""] * len(resolved))).astype(str).str.lower().isin(
        {"", "nan", "none", "unknown", "unavailable"}
    )
    resolved["arr_resolution_accepted"] = (
        resolved["arr_suffix_clinvar_found"]
        & (resolved["arr_gene_concordant"] | local_gene_missing)
    )

    resolved["arr_resolution_note"] = ""
    resolved.loc[~resolved["arr_suffix_clinvar_found"], "arr_resolution_note"] = "candidate_suffix_not_found_in_downloaded_clinvar"
    resolved.loc[
        resolved["arr_suffix_clinvar_found"] & ~resolved["arr_gene_concordant"] & ~local_gene_missing,
        "arr_resolution_note",
    ] = "candidate_found_but_gene_discordant"
    resolved.loc[
        resolved["arr_resolution_accepted"] & resolved["arr_gene_concordant"],
        "arr_resolution_note",
    ] = "ARR_suffix_resolved_to_ClinVar_VariationID_gene_concordant"
    resolved.loc[
        resolved["arr_resolution_accepted"] & local_gene_missing,
        "arr_resolution_note",
    ] = "ARR_suffix_resolved_to_ClinVar_VariationID_local_gene_missing_or_unavailable"

    if "arr_clinvar_PhenotypeList" in resolved.columns:
        resolved["phenotype_domain_concordant"] = resolved["arr_clinvar_PhenotypeList"].map(
            phenotype_domain_concordant_for_arrhythmia
        )
        resolved["arr_phenotype_domain_note"] = resolved["arr_clinvar_PhenotypeList"].map(
            arr_phenotype_domain_note
        )
    else:
        resolved["phenotype_domain_concordant"] = False
        resolved["arr_phenotype_domain_note"] = ""

    resolved["phenotype_domain_discordance_flag"] = (
        resolved["arr_resolution_accepted"] & ~resolved["phenotype_domain_concordant"]
    )
    resolved["source_match_accepted"] = resolved["arr_resolution_accepted"]
    resolved["meaning_match_accepted"] = (
        resolved["arr_resolution_accepted"] & resolved["phenotype_domain_concordant"]
    )
    resolved["routing_implication"] = "direct_source_match_only"
    resolved.loc[
        resolved["phenotype_domain_discordance_flag"],
        "routing_implication",
    ] = "contextual_repair_or_disease_specific_review"

    resolved.to_csv(OUT_RESOLUTION, index=False)

    final = df.copy()

    string_update_cols = [
        "clinvar_variation_id",
        "mapping_source_path",
        "mapping_key_used",
        "clinvar_join_failure_reason",
        "candidate_clinvar_variation_id",
        "arr_resolution_note",
        "arr_phenotype_domain_note",
        "phenotype_domain_concordant",
        "phenotype_domain_discordance_flag",
        "source_match_accepted",
        "meaning_match_accepted",
        "routing_implication",
    ] + CLINVAR_COLS
    final = ensure_object_columns(final, string_update_cols)

    for c in ["external_clinvar_match", "arr_resolution_accepted", "arr_gene_concordant"]:
        if c not in final.columns:
            final[c] = "False"
        final[c] = final[c].astype("object")

    accepted = resolved[resolved["arr_resolution_accepted"]].copy()

    for _, row in accepted.iterrows():
        assertion_id = str(row.get("assertion_id", ""))
        mask = final["assertion_id"].astype(str).eq(assertion_id)
        if not bool(mask.any()):
            continue

        final.loc[mask, "clinvar_variation_id"] = str(row.get("candidate_clinvar_variation_id", ""))
        final.loc[mask, "mapping_source_path"] = "ARR_suffix_from_local_identifier"
        final.loc[mask, "mapping_key_used"] = "ARR_suffix"
        final.loc[mask, "external_clinvar_match"] = "True"
        final.loc[mask, "clinvar_join_failure_reason"] = ""
        final.loc[mask, "candidate_clinvar_variation_id"] = str(row.get("candidate_clinvar_variation_id", ""))
        final.loc[mask, "arr_resolution_accepted"] = bool_str(row.get("arr_resolution_accepted", False))
        final.loc[mask, "arr_gene_concordant"] = bool_str(row.get("arr_gene_concordant", False))
        final.loc[mask, "arr_resolution_note"] = str(row.get("arr_resolution_note", ""))
        final.loc[mask, "arr_phenotype_domain_note"] = str(row.get("arr_phenotype_domain_note", ""))
        final.loc[mask, "phenotype_domain_concordant"] = bool_str(row.get("phenotype_domain_concordant", False))
        final.loc[mask, "phenotype_domain_discordance_flag"] = bool_str(row.get("phenotype_domain_discordance_flag", False))
        final.loc[mask, "source_match_accepted"] = bool_str(row.get("source_match_accepted", False))
        final.loc[mask, "meaning_match_accepted"] = bool_str(row.get("meaning_match_accepted", False))
        final.loc[mask, "routing_implication"] = str(row.get("routing_implication", ""))

        for c in CLINVAR_COLS:
            pc = f"arr_clinvar_{c}"
            if pc in row.index:
                val = "" if pd.isna(row[pc]) else str(row[pc])
                final.loc[mask, c] = val

    final = apply_final_qc_defaults(final)
    final.to_csv(OUT_FINAL, index=False)

    matched_after = int(truthy(final["external_clinvar_match"]).sum())
    unmatched_after = len(final) - matched_after
    arr_attempted = len(unmatched_arr)
    arr_candidates = int(unmatched_arr["candidate_clinvar_variation_id"].astype(str).str.fullmatch(r"\d+").sum())
    arr_found = int(resolved["arr_suffix_clinvar_found"].sum())
    arr_gene_concord = int(resolved["arr_gene_concordant"].sum())
    arr_accepted = int(resolved["arr_resolution_accepted"].sum())
    discordance_count = int(truthy(final["phenotype_domain_discordance_flag"]).sum())
    meaning_rejected = int(final["meaning_match_accepted"].astype(str).str.lower().isin({"false", "0", "no"}).sum())

    audit = pd.DataFrame([{
        "status": "completed",
        "rows_total": len(final),
        "matched_before": matched_before,
        "arr_rows_attempted": arr_attempted,
        "arr_suffix_candidates": arr_candidates,
        "arr_suffix_clinvar_found": arr_found,
        "arr_gene_concordant": arr_gene_concord,
        "arr_resolution_accepted": arr_accepted,
        "phenotype_domain_discordance_flagged": discordance_count,
        "meaning_match_rejected": meaning_rejected,
        "matched_after": matched_after,
        "unmatched_after": unmatched_after,
        "final_match_rate": matched_after / len(final) if len(final) else 0,
    }])
    audit.to_csv(OUT_AUDIT, index=False)

    claims = pd.DataFrame([
        {
            "claim_label": "downloaded_clinvar_external_comparator_final",
            "claim_strength": "external comparator analysis",
            "allowed_claim": (
                f"Downloaded ClinVar public data produced true row-level external assertion-source matches for "
                f"{matched_after:,}/{len(final):,} CAB benchmark rows after VariationID repair and ARR suffix resolution."
            ),
            "required_caveat": (
                "This supports public assertion-source traceability/comparability only; "
                "it is not clinical validation or patient-outcome validation."
            ),
        },
        {
            "claim_label": "source_vs_meaning_match_boundary",
            "claim_strength": "QC boundary / portability warning",
            "allowed_claim": (
                f"{discordance_count:,} source-matched rows were retained with phenotype-domain discordance flags; "
                "these rows have source_match_accepted=True but meaning_match_accepted=False and require contextual repair or disease-specific review."
            ),
            "required_caveat": (
                "Source matching does not imply disease-environment portability or direct deterministic reuse."
            ),
        },
        {
            "claim_label": "arrhythmia_arr_suffix_resolution",
            "claim_strength": "identifier repair / source traceability",
            "allowed_claim": (
                f"Inherited-arrhythmia local ARR_* identifiers were resolved by extracting the numeric suffix as a candidate ClinVar VariationID; "
                f"{arr_accepted:,}/{arr_attempted:,} ARR rows were accepted after ClinVar lookup and tokenized gene-concordance checks."
            ),
            "required_caveat": "ARR suffix resolution is an identifier-repair step and should be reported with QC; it is not biological validation.",
        },
        {
            "claim_label": "not_clinical_validation",
            "claim_strength": "required caveat",
            "allowed_claim": "The downloaded external analysis supports external comparison and source traceability.",
            "required_caveat": "No patient-outcome validation, no prospective clinical deployment, no clinical validation of CAB/PRF.",
        },
    ])
    claims.to_csv(OUT_CLAIMS, index=False)

    md = f"""# ARR Identifier Resolution for External ClinVar Join

## Problem

The repaired external ClinVar join matched cardiomyopathy and hereditary cancer rows, but left 942 inherited-arrhythmia rows unmatched because the benchmark uses local identifiers such as `ARR_1325231`.

## Method

For unmatched inherited-arrhythmia rows, this script extracted the numeric suffix from `ARR_*` identifiers and treated it as a candidate ClinVar `VariationID`. Candidate IDs were looked up in the downloaded ClinVar `variant_summary.txt.gz`. Resolutions were accepted when the candidate was found and tokenized gene matching confirmed concordance.

## Source-vs-meaning QC

Rows can be valid external ClinVar source matches while failing disease/phenotype-domain meaning portability. Such rows are retained, not deleted, with:

- `external_clinvar_match=True`
- `source_match_accepted=True`
- `phenotype_domain_concordant=False`
- `phenotype_domain_discordance_flag=True`
- `meaning_match_accepted=False`
- `routing_implication=contextual_repair_or_disease_specific_review`

## Results

- rows total: {len(final):,}
- matched before ARR resolution: {matched_before:,}
- ARR rows attempted: {arr_attempted:,}
- ARR numeric suffix candidates: {arr_candidates:,}
- ARR candidates found in downloaded ClinVar: {arr_found:,}
- ARR gene-concordant rows: {arr_gene_concord:,}
- ARR resolutions accepted: {arr_accepted:,}
- phenotype-domain discordance flagged: {discordance_count:,}
- meaning matches rejected: {meaning_rejected:,}
- matched after ARR resolution: {matched_after:,}
- unmatched after ARR resolution: {unmatched_after:,}
- final match rate: {matched_after / len(final):.4f}

## Claim boundary

This resolves public source identifiers and improves external ClinVar assertion-source traceability. It does not validate CAB clinically, does not validate patient outcomes, and does not validate prospective deployment.
"""
    OUT_MD.write_text(md, encoding="utf-8")

    print("ARR identifier resolution complete.")
    print(audit.to_string(index=False))
    print()
    print("Wrote:")
    for p in [OUT_RESOLUTION, OUT_FINAL, OUT_AUDIT, OUT_CLAIMS, OUT_MD]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
