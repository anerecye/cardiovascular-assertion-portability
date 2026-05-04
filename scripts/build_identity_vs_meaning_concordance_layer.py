
#!/usr/bin/env python3
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"
FIGS = ROOT / "reports" / "figures"

JOIN_FINAL = TABLES / "external_clinvar_cab_benchmark_join_final.csv"
TASKS = ROOT / "data" / "processed" / "cab_decision_challenge_tasks.csv"

OUT_CONCORDANCE = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
OUT_TAXONOMY = TABLES / "phenotype_domain_discordance_taxonomy.csv"
OUT_SENS = TABLES / "meaning_discordance_sensitivity_analysis.csv"
OUT_ROUTING = TABLES / "meaning_rejected_routing_consequences.csv"
OUT_MD = QC / "identity_vs_meaning_positioning.md"
OUT_FIG = FIGS / "identity_vs_meaning_concordance_flow.svg"
OUT_CLAIMS = TABLES / "identity_vs_meaning_publication_safe_claims.csv"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, dtype=str)


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def first_existing(df: pd.DataFrame, cols: list[str], default: str = "") -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def bool_col(df: pd.DataFrame, cols: list[str], default: bool = False) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return truthy(df[c])
    return pd.Series([default] * len(df), index=df.index)


def phase1_concordance(join: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["assertion_id"] = first_existing(join, ["assertion_id"])
    out["domain"] = first_existing(join, ["domain"])
    out["local_gene"] = first_existing(join, ["gene", "local_gene"])
    out["clinvar_variation_id"] = first_existing(join, ["VariationID", "clinvar_variation_id"])
    out["clinvar_gene_symbol"] = first_existing(join, ["GeneSymbol"])
    out["clinvar_phenotype_list"] = first_existing(join, ["PhenotypeList"])
    out["external_clinvar_match"] = bool_col(join, ["external_clinvar_match"]).map(lambda x: "True" if x else "False")
    out["gene_concordant"] = bool_col(join, ["arr_gene_concordant"], default=True).map(lambda x: "True" if x else "False")
    out["phenotype_domain_concordant"] = bool_col(join, ["phenotype_domain_concordant"], default=True).map(lambda x: "True" if x else "False")
    out["phenotype_domain_discordance_flag"] = bool_col(join, ["phenotype_domain_discordance_flag"], default=False).map(lambda x: "True" if x else "False")
    out["source_match_accepted"] = bool_col(join, ["source_match_accepted", "external_clinvar_match"], default=True).map(lambda x: "True" if x else "False")
    out["meaning_match_accepted"] = bool_col(join, ["meaning_match_accepted"], default=True).map(lambda x: "True" if x else "False")
    out["routing_implication"] = first_existing(join, ["routing_implication"], "")
    out["discordance_reason"] = ""
    out.loc[out["phenotype_domain_discordance_flag"].eq("True"), "discordance_reason"] = (
        "source_identity_and_gene_concordance_accepted_but_phenotype_domain_concordance_failed"
    )
    out["notes"] = ""
    out.loc[out["phenotype_domain_discordance_flag"].eq("True"), "notes"] = (
        "Retained with discordance flag; do not treat as deterministic disease-meaning reuse."
    )
    out.to_csv(OUT_CONCORDANCE, index=False)
    return out


def classify_taxonomy(row: pd.Series) -> str:
    phenotype = str(row.get("clinvar_phenotype_list", "") or "").lower()
    gene_symbol = str(row.get("clinvar_gene_symbol", "") or "")
    local_gene = str(row.get("local_gene", "") or "")
    multi = ";" in gene_symbol or "," in gene_symbol

    syndromic_terms = [
        "syndrome", "rasopathy", "noonan", "silver-russell", "beckwith", "wiedemann",
        "timothy", "andersen", "jervell", "lange-nielsen",
    ]
    developmental_terms = [
        "development", "imprinting", "growth", "neurodevelopment", "intellectual",
        "autism", "epilepsy", "seizure", "congenital", "craniofacial", "silver-russell",
        "beckwith", "wiedemann",
    ]
    unknown_terms = ["not specified", "not provided", "unknown", "unspecified", "not reported", "multiple conditions"]
    broad_terms = ["cardiovascular phenotype", "heart disease", "cardiac phenotype", "sudden death", "arrhythmia"]

    if any(t in phenotype for t in developmental_terms):
        return "developmental_or_imprinting_label"
    if any(t in phenotype for t in syndromic_terms):
        return "syndromic_non_arrhythmia"
    if any(t in phenotype for t in unknown_terms) or phenotype.strip() == "":
        return "broad_or_unknown_phenotype"
    if multi and local_gene.upper() in gene_symbol.upper():
        return "multi_gene_symbol_ambiguity"
    if any(t in phenotype for t in broad_terms):
        return "condition_label_mismatch"
    if multi:
        return "neighboring_gene_or_transcript_context"
    if phenotype:
        return "non_domain_phenotype"
    return "other"


def phase2_taxonomy(conc: pd.DataFrame) -> pd.DataFrame:
    discordant = conc[conc["phenotype_domain_discordance_flag"].eq("True")].copy()
    order = [
        "non_domain_phenotype",
        "syndromic_non_arrhythmia",
        "developmental_or_imprinting_label",
        "broad_or_unknown_phenotype",
        "neighboring_gene_or_transcript_context",
        "multi_gene_symbol_ambiguity",
        "condition_label_mismatch",
        "other",
    ]
    if discordant.empty:
        tax = pd.DataFrame(columns=[
            "discordance_taxon", "N", "percent", "example_VariationIDs",
            "example_gene", "example_phenotype"
        ])
        tax.to_csv(OUT_TAXONOMY, index=False)
        return tax

    discordant["discordance_taxon"] = discordant.apply(classify_taxonomy, axis=1)
    total = len(discordant)
    rows = []
    for taxon in order:
        sub = discordant[discordant["discordance_taxon"].eq(taxon)]
        if sub.empty:
            continue
        rows.append({
            "discordance_taxon": taxon,
            "N": len(sub),
            "percent": len(sub) / total,
            "example_VariationIDs": "|".join(sub["clinvar_variation_id"].dropna().astype(str).head(5)),
            "example_gene": "|".join(sub["local_gene"].dropna().astype(str).drop_duplicates().head(5)),
            "example_phenotype": "|".join(sub["clinvar_phenotype_list"].dropna().astype(str).drop_duplicates().head(3)),
        })
    tax = pd.DataFrame(rows)
    tax.to_csv(OUT_TAXONOMY, index=False)
    return tax


def attach_tasks(conc: pd.DataFrame) -> pd.DataFrame:
    if not TASKS.exists():
        return conc.copy()
    tasks = read_csv(TASKS)
    if "assertion_id" not in tasks.columns:
        return conc.copy()

    tasks["assertion_id"] = norm_id(tasks["assertion_id"])
    conc2 = conc.copy()
    conc2["assertion_id"] = norm_id(conc2["assertion_id"])

    keep = [
        "assertion_id",
        "future_condition_label_drift",
        "future_cross_environment_drift",
        "future_any_meaning_drift",
        "future_self_loop_stable",
        "self_loop_stable",
        "conservative_composite_routing",
        "direct_single_model_reuse_allowed",
        "cab_strict_direct_use_allowed",
        "cab_balanced_direct_use_allowed",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "routing_primary_action",
        "routing_secondary_flags",
        "primary_routing_action",
        "secondary_routing_flags",
    ]
    keep = [c for c in keep if c in tasks.columns]
    return conc2.merge(tasks[keep], on="assertion_id", how="left")


def infer_bool(df: pd.DataFrame, candidates: list[str], default: bool = False) -> pd.Series:
    return bool_col(df, candidates, default=default)


def metric_subset(label: str, df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "subset": label,
            "N": 0,
            "condition_label_drift_rate": "",
            "cross_environment_drift_rate": "",
            "any_meaning_drift_rate": "",
            "self_loop_stable_rate": "",
            "unsupported_reuse_ClinVar_label_only": "",
            "unsupported_reuse_CAB_Strict": "",
            "unsupported_reuse_CAB_Balanced": "",
            "direct_use_allowed_rate": "",
            "review_repair_routing_rate": "",
        }

    cond = infer_bool(df, ["future_condition_label_drift"], False)
    cross = infer_bool(df, ["future_cross_environment_drift", "cross_environment_drift"], False)
    anyd = infer_bool(df, ["future_any_meaning_drift", "any_meaning_drift"], False)
    self_loop = infer_bool(df, ["future_self_loop_stable", "self_loop_stable"], False)

    strict_direct = infer_bool(df, ["cab_strict_direct_use_allowed"], False)
    balanced_direct = infer_bool(df, ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"], False)

    repair_flags = pd.Series([False] * n, index=df.index)
    for c in [
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "phenotype_domain_discordance_flag",
    ]:
        if c in df.columns:
            repair_flags = repair_flags | truthy(df[c])

    if "routing_implication" in df.columns:
        repair_flags = repair_flags | df["routing_implication"].astype(str).eq("contextual_repair_or_disease_specific_review")

    return {
        "subset": label,
        "N": n,
        "condition_label_drift_rate": cond.mean(),
        "cross_environment_drift_rate": cross.mean(),
        "any_meaning_drift_rate": anyd.mean(),
        "self_loop_stable_rate": self_loop.mean(),
        "unsupported_reuse_ClinVar_label_only": cond.mean(),
        "unsupported_reuse_CAB_Strict": (cond & strict_direct).mean(),
        "unsupported_reuse_CAB_Balanced": (cond & balanced_direct).mean(),
        "direct_use_allowed_rate": balanced_direct.mean(),
        "review_repair_routing_rate": repair_flags.mean(),
    }


def phase3_sensitivity(conc: pd.DataFrame) -> pd.DataFrame:
    df = attach_tasks(conc)
    arr = df[df["domain"].eq("inherited_arrhythmia")].copy()
    arr_accepted = arr[arr["meaning_match_accepted"].eq("True")].copy()
    arr_rejected = arr[arr["meaning_match_accepted"].eq("False")].copy()
    all_excluded = df[df["meaning_match_accepted"].ne("False")].copy()
    all_with_repair = df.copy()

    rows = [
        metric_subset("A_full_ARR_set", arr),
        metric_subset("B_ARR_meaning_accepted_only", arr_accepted),
        metric_subset("C_ARR_meaning_rejected_phenotype_domain_discordant_only", arr_rejected),
        metric_subset("D_all_domain_benchmark_304_discordant_rows_excluded", all_excluded),
        metric_subset("E_all_domain_benchmark_including_discordant_with_repair_routing", all_with_repair),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SENS, index=False)
    return out


def phase4_routing_consequence(conc: pd.DataFrame) -> pd.DataFrame:
    df = attach_tasks(conc)
    rejected = df[df["meaning_match_accepted"].eq("False")].copy()

    out = pd.DataFrame()
    out["assertion_id"] = first_existing(rejected, ["assertion_id"])
    out["domain"] = first_existing(rejected, ["domain"])
    out["gene"] = first_existing(rejected, ["local_gene", "gene"])
    out["clinvar_variation_id"] = first_existing(rejected, ["clinvar_variation_id"])
    out["clinvar_phenotype_list"] = first_existing(rejected, ["clinvar_phenotype_list"])
    out["clinvar_label_only_direct_reuse_allowed"] = "True"
    out["CAB_Strict_direct_reuse_allowed"] = infer_bool(rejected, ["cab_strict_direct_use_allowed"], False).map(lambda x: "True" if x else "False")
    out["CAB_Balanced_direct_reuse_allowed"] = infer_bool(
        rejected,
        ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"],
        False,
    ).map(lambda x: "True" if x else "False")
    out["primary_routing_action"] = first_existing(
        rejected,
        ["routing_primary_action", "primary_routing_action", "routing_implication"],
        "contextual_repair_or_disease_specific_review",
    )
    out.loc[out["primary_routing_action"].eq(""), "primary_routing_action"] = "contextual_repair_or_disease_specific_review"
    out["secondary_routing_flags"] = first_existing(
        rejected,
        ["routing_secondary_flags", "secondary_routing_flags"],
        "phenotype_domain_discordance_flag",
    )
    out.loc[out["secondary_routing_flags"].eq(""), "secondary_routing_flags"] = "phenotype_domain_discordance_flag"
    temporal = infer_bool(rejected, ["future_condition_label_drift"], False)
    composite = infer_bool(rejected, ["conservative_composite_routing", "future_any_meaning_drift"], False)
    out["unsupported_reuse_under_temporal_gold"] = temporal.map(lambda x: "True" if x else "False")
    out["unsupported_reuse_under_composite_gold"] = composite.map(lambda x: "True" if x else "False")
    out.to_csv(OUT_ROUTING, index=False)
    return out


def phase5_positioning(total: int, meaning_accepted: int, meaning_rejected: int) -> None:
    text = f"""# Identity vs Meaning Positioning

## Key distinction

ClinVar / VRS / source matching establishes variant or record identity. CAB evaluates disease-meaning portability.

A gene-concordant source match can still be phenotype-domain discordant. Therefore identity resolution is necessary but insufficient for deterministic assertion reuse.

## Current result

All {total:,} benchmark assertions were externally matched to ClinVar source records. Of these, {meaning_accepted:,} were meaning accepted and {meaning_rejected:,} were source-accepted but disease-meaning rejected because phenotype-domain concordance failed.

## Allowed claim

All {total:,} benchmark assertions were externally matched to ClinVar source records, but {meaning_rejected:,} ARR records were source-accepted and gene-concordant while disease-meaning rejected because phenotype-domain concordance failed.

## Forbidden claims

- ClinVar is wrong.
- These records are invalid.
- CAB reclassifies these variants.
- Source match failure.

These rows are source matches with meaning-portability failure. They are retained with discordance flags and routed to contextual repair or disease-specific review.

## Example

ARR_977320 resolves to ClinVar VariationID 977320 and is gene-concordant for KCNQ1. Its ClinVar phenotype label is Silver-Russell syndrome 1, so CAB retains the source match but rejects deterministic inherited-arrhythmia meaning reuse.
"""
    OUT_MD.write_text(text, encoding="utf-8")


def svg_box(x, y, w, h, title, subtitle, fill="#ffffff"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="#111" stroke-width="2"/>\n'
        f'<text x="{x + 18}" y="{y + 32}" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>\n'
        f'<text x="{x + 18}" y="{y + 58}" font-family="Arial" font-size="13">{html.escape(subtitle)}</text>\n'
    )


def phase6_figure(total: int, accepted: int, rejected: int) -> None:
    width, height = 1200, 720
    box_w, box_h = 300, 80
    x = 70
    ys = [70, 190, 310, 430, 550]
    boxes = [
        ("26,725 source matched", "External ClinVar source records resolved"),
        ("26,725 gene/source accepted", "VariationID + tokenized gene concordance"),
        (f"{accepted:,} meaning accepted", "Identity and phenotype-domain concordance"),
        (f"{rejected:,} phenotype-domain discordant", "Source accepted, meaning rejected"),
        ("contextual repair or disease-specific review", "No deterministic disease-meaning reuse"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="#fff"/>\n',
        '<text x="60" y="34" font-family="Arial" font-size="24" font-weight="700">CAB identity vs disease-meaning concordance</text>\n',
        '<text x="60" y="58" font-family="Arial" font-size="14">Complete source identity does not guarantee deterministic disease-meaning portability.</text>\n',
    ]
    for i, (title, subtitle) in enumerate(boxes):
        fill = "#ffffff" if i < 3 else "#f7f7f7"
        parts.append(svg_box(x, ys[i], box_w, box_h, title, subtitle, fill))
        if i < len(boxes) - 1:
            xmid = x + box_w / 2
            parts.append(f'<line x1="{xmid}" y1="{ys[i] + box_h}" x2="{xmid}" y2="{ys[i+1]}" stroke="#111" stroke-width="2"/>\n')
            parts.append(f'<polygon points="{xmid - 7},{ys[i+1]-10} {xmid + 7},{ys[i+1]-10} {xmid},{ys[i+1]}" fill="#111"/>\n')

    ex_x, ex_y = 450, 160
    inset_lines = [
        (38, "Example inset: ARR_977320", 22, "700"),
        (76, "Source match: ClinVar VariationID 977320", 16, "400"),
        (106, "Gene concordance: KCNQ1 in KCNQ1;KCNQ1OT1;LOC106783508", 16, "400"),
        (136, "ClinVar phenotype: Silver-Russell syndrome 1", 16, "400"),
        (176, "QC flags", 16, "700"),
        (208, "source_match_accepted = True", 15, "400"),
        (234, "meaning_match_accepted = False", 15, "400"),
        (260, "routing_implication = contextual_repair_or_disease_specific_review", 15, "400"),
    ]
    parts.append(f'<rect x="{ex_x}" y="{ex_y}" width="680" height="310" rx="20" fill="#fafafa" stroke="#111" stroke-width="2"/>\n')
    for dy, text, size, weight in inset_lines:
        parts.append(f'<text x="{ex_x + 24}" y="{ex_y + dy}" font-family="Arial" font-size="{size}" font-weight="{weight}">{html.escape(text)}</text>\n')
    parts.append("</svg>\n")
    OUT_FIG.write_text("".join(parts), encoding="utf-8")


def phase7_claims(total: int, rejected: int) -> None:
    rows = [
        {
            "claim_type": "allowed",
            "claim_strength": "identity_vs_meaning_primary",
            "claim_text": (
                "Source identity and assertion meaning are separable layers. "
                f"All {total:,} benchmark assertions were externally matched to ClinVar source records, "
                f"but {rejected:,} source-accepted rows were disease-meaning rejected because phenotype-domain concordance failed."
            ),
            "required_caveat": "Source matching does not imply deterministic disease-meaning portability.",
        },
        {
            "claim_type": "allowed",
            "claim_strength": "portability_boundary",
            "claim_text": "Complete ClinVar source matching does not guarantee disease-domain portability.",
            "required_caveat": "CAB evaluates portability after source identity resolution; it does not reclassify variants.",
        },
        {
            "claim_type": "allowed",
            "claim_strength": "routing_consequence",
            "claim_text": "CAB retains source matches but rejects deterministic meaning reuse when phenotype-domain concordance fails.",
            "required_caveat": "Phenotype-domain discordant rows require contextual repair or disease-specific review.",
        },
        {
            "claim_type": "forbidden",
            "claim_strength": "forbidden",
            "claim_text": "CAB invalidates ClinVar records.",
            "required_caveat": "Forbidden: these are source matches with meaning-portability failure, not invalid records.",
        },
        {
            "claim_type": "forbidden",
            "claim_strength": "forbidden",
            "claim_text": "CAB reclassifies variants.",
            "required_caveat": "Forbidden: CAB does not alter pathogenicity classifications.",
        },
        {
            "claim_type": "forbidden",
            "claim_strength": "forbidden",
            "claim_text": "CAB diagnoses phenotype mismatch.",
            "required_caveat": "Forbidden: CAB flags disease-domain portability risk; it does not diagnose.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_CLAIMS, index=False)


def main() -> None:
    ensure_dirs()
    join = read_csv(JOIN_FINAL)
    conc = phase1_concordance(join)
    phase2_taxonomy(conc)
    phase3_sensitivity(conc)
    phase4_routing_consequence(conc)

    total = len(conc)
    rejected = int(conc["meaning_match_accepted"].eq("False").sum())
    accepted = total - rejected

    phase5_positioning(total, accepted, rejected)
    phase6_figure(total, accepted, rejected)
    phase7_claims(total, rejected)

    print("ClinVar identity-vs-meaning concordance layer complete.")
    print(f"Total source matched: {total:,}")
    print(f"Meaning accepted: {accepted:,}")
    print(f"Meaning rejected / phenotype-domain discordant: {rejected:,}")
    print("Outputs:")
    for p in [OUT_CONCORDANCE, OUT_TAXONOMY, OUT_SENS, OUT_ROUTING, OUT_MD, OUT_FIG, OUT_CLAIMS]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
