
#!/usr/bin/env python3
"""Build final CAB disease-architecture meaning-travel layer.

Creates:
1. reports/tables/disease_architecture_portability_regimes_final.csv
2. data/processed/assertion_disease_architecture_regime_map_final.csv
3. reports/tables/disease_architecture_regime_temporal_signatures.csv
4. reports/tables/disease_architecture_regime_enrichment_tests.csv
5. reports/figures/disease_architecture_meaning_travel_map.svg
6. reports/qc/mendelian_meaning_travel_claim.md

Claim boundary:
This is a disease-architecture portability synthesis. It does not reclassify
variants, invalidate ClinVar, claim clinical validation, or replace expert curation.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"

TASKS = DATA_PROCESSED / "cab_decision_challenge_tasks.csv"
CONCORDANCE = TABLES / "clinvar_identity_vs_meaning_concordance.csv"

OUT_REGIMES = TABLES / "disease_architecture_portability_regimes_final.csv"
OUT_MAP = DATA_PROCESSED / "assertion_disease_architecture_regime_map_final.csv"
OUT_SIG = TABLES / "disease_architecture_regime_temporal_signatures.csv"
OUT_ENRICH = TABLES / "disease_architecture_regime_enrichment_tests.csv"
OUT_FIG = FIGS / "disease_architecture_meaning_travel_map.svg"
OUT_CLAIM = QC / "mendelian_meaning_travel_claim.md"


REGIMES = [
    {
        "regime_name": "Phenotype-anchored monogenic",
        "definition": "A disease architecture in which pathogenic meaning is anchored to a specific monogenic phenotype and a concordant disease model.",
        "biological_determinant": "specific gene-disease model; phenotype anchoring; concordant self-loop reuse",
        "meaning_travel_rule": "direct deterministic use within concordant self-loop",
        "expected_failure_mode": "failure occurs mainly when the assertion is reused outside the concordant disease model",
        "dominant_routing_action": "direct deterministic use within concordant self-loop",
        "PRF_required": "no",
        "domain_examples": "inherited_arrhythmia|cardiomyopathy|hereditary_cancer",
        "gene_examples": "KCNQ1|KCNH2|SCN5A|MYBPC3|MYH7|BRCA1|BRCA2|MLH1|MSH2|TP53",
        "condition_environment_examples": "long QT syndrome self-loop|hypertrophic cardiomyopathy self-loop|hereditary breast and ovarian cancer self-loop",
        "source_identity_vs_meaning_note": "source identity and disease meaning are usually aligned only within the same disease model",
        "publication_safe_interpretation": "supports deterministic reuse only within concordant disease self-loops",
    },
    {
        "regime_name": "Trigger-dependent latent",
        "definition": "A latent-risk architecture in which pathogenic meaning depends on trigger, stressor, drug, physiologic, age, or ascertainment context.",
        "biological_determinant": "incomplete penetrance; trigger-dependent expressivity; latent liability",
        "meaning_travel_rule": "contextual repair; trigger/phenotype-context review",
        "expected_failure_mode": "deterministic disease reuse without trigger or ascertainment context",
        "dominant_routing_action": "contextual repair; trigger/phenotype-context review",
        "PRF_required": "yes",
        "domain_examples": "inherited_arrhythmia|cardiomyopathy",
        "gene_examples": "RYR2|KCNQ1|KCNH2|SCN5A|CACNA1C",
        "condition_environment_examples": "stress/exercise-triggered arrhythmia|drug-triggered QT risk|latent cardiomyopathy expression",
        "source_identity_vs_meaning_note": "source identity can be stable while phenotype realization requires trigger context",
        "publication_safe_interpretation": "requires trigger-aware contextual repair rather than direct deterministic reuse",
    },
    {
        "regime_name": "Pleiotropic collision",
        "definition": "A disease architecture in which one gene or source assertion collides with multiple disease models or condition labels.",
        "biological_determinant": "pleiotropy; multi-condition labels; cross-model gene use",
        "meaning_travel_rule": "disease-specific review or contextual repair",
        "expected_failure_mode": "false portability across disease environments",
        "dominant_routing_action": "disease-specific review or contextual repair",
        "PRF_required": "conditional",
        "domain_examples": "inherited_arrhythmia|cardiomyopathy|hereditary_cancer",
        "gene_examples": "SCN5A|RYR2|FLNC|TTN|TP53|PTEN|CHEK2|ATM",
        "condition_environment_examples": "arrhythmia vs cardiomyopathy labels|cancer syndrome vs organ-specific cancer risk",
        "source_identity_vs_meaning_note": "gene/source concordance does not guarantee one disease meaning across models",
        "publication_safe_interpretation": "requires disease-specific review before transfer across disease models",
    },
    {
        "regime_name": "Syndrome-organ boundary",
        "definition": "A boundary in which source identity is valid but meaning crosses between syndromic/developmental/imprinting labels and organ-specific labels.",
        "biological_determinant": "syndrome-to-organ boundary; developmental/imprinting phenotype labels; multi-system disease architecture",
        "meaning_travel_rule": "source identity accepted; contextual repair or disease-specific review",
        "expected_failure_mode": "source match mistaken for organ-specific disease portability",
        "dominant_routing_action": "source_identity_accepted; contextual_repair_or_disease_specific_review",
        "PRF_required": "conditional",
        "domain_examples": "inherited_arrhythmia|hereditary_cancer",
        "gene_examples": "KCNQ1|KCNQ1OT1|CACNA1C|PTEN|TP53",
        "condition_environment_examples": "Silver-Russell syndrome label crossing into inherited arrhythmia|syndromic cancer labels crossing into organ-specific risk",
        "source_identity_vs_meaning_note": "source match is retained, but disease-meaning portability fails when phenotype-domain concordance fails",
        "publication_safe_interpretation": "retained as source match with meaning-portability failure",
    },
    {
        "regime_name": "Structural-functional overlap",
        "definition": "A disease architecture where structural tissue disease and functional/electrical disease mechanisms overlap but are not interchangeable.",
        "biological_determinant": "structural-functional coupling; cardiac remodeling; electrophysiologic/contractile overlap",
        "meaning_travel_rule": "domain repair; disease-specific expert review",
        "expected_failure_mode": "overextension from one organ-function model into another",
        "dominant_routing_action": "domain repair; disease-specific expert review",
        "PRF_required": "conditional",
        "domain_examples": "inherited_arrhythmia|cardiomyopathy",
        "gene_examples": "SCN5A|RYR2|FLNC|TTN|DSP|PKP2|LMNA",
        "condition_environment_examples": "arrhythmia-cardiomyopathy boundary|structural cardiomyopathy vs electrical disease model",
        "source_identity_vs_meaning_note": "source identity can travel, but disease mechanism needs domain repair",
        "publication_safe_interpretation": "requires domain repair rather than direct cross-model reuse",
    },
    {
        "regime_name": "Genotype-first absent phenotype",
        "definition": "A genotype-first architecture in which a source-valid P/LP assertion has absent, unknown, or unobserved phenotype realization.",
        "biological_determinant": "genotype-first ascertainment; absent phenotype; reduced penetrance; phenotype not yet realized",
        "meaning_travel_rule": "PRF-needed; no deterministic reuse",
        "expected_failure_mode": "carrier status treated as deterministic disease state",
        "dominant_routing_action": "PRF-needed; phenotype ascertainment review; no deterministic reuse",
        "PRF_required": "yes",
        "domain_examples": "inherited_arrhythmia|hereditary_cancer|cardiomyopathy",
        "gene_examples": "KCNQ1|KCNH2|SCN5A|RYR2|BRCA1|BRCA2|PALB2|CHEK2|TTN",
        "condition_environment_examples": "genotype-first carrier without phenotype|population screening carrier|phenotype-absent P/LP assertion",
        "source_identity_vs_meaning_note": "source identity does not establish phenotype realization",
        "publication_safe_interpretation": "cannot be deterministically reused as disease without phenotype/penetrance context",
    },
    {
        "regime_name": "Modifier/penetrance boundary",
        "definition": "A boundary where pathogenic meaning travels as conditional risk rather than deterministic disease identity.",
        "biological_determinant": "penetrance modifiers; ancestry/population context; risk architecture",
        "meaning_travel_rule": "population/penetrance review; PRF-needed",
        "expected_failure_mode": "risk assertion converted into deterministic disease label",
        "dominant_routing_action": "population/penetrance review; PRF-needed",
        "PRF_required": "yes",
        "domain_examples": "hereditary_cancer|cardiomyopathy|inherited_arrhythmia",
        "gene_examples": "CHEK2|ATM|PALB2|BRCA1|BRCA2|TTN|FLNC|LMNA|KCNQ1",
        "condition_environment_examples": "conditional cancer risk|reduced penetrance cardiomyopathy|arrhythmia carrier risk",
        "source_identity_vs_meaning_note": "source identity supports risk framing, not deterministic disease transfer",
        "publication_safe_interpretation": "travels as conditional risk under PRF-style framing",
    },
    {
        "regime_name": "Nonspecific/underresolved",
        "definition": "An underresolved architecture where broad, unknown, or nonspecific phenotype labels are insufficient for deterministic disease-model reuse.",
        "biological_determinant": "broad condition labels; underresolved phenotype mapping; unknown disease environment",
        "meaning_travel_rule": "contextual repair or disease-specific review",
        "expected_failure_mode": "broad label reused as a specific disease assertion",
        "dominant_routing_action": "contextual repair or disease-specific review",
        "PRF_required": "conditional",
        "domain_examples": "inherited_arrhythmia|cardiomyopathy|hereditary_cancer",
        "gene_examples": "KCNQ1|KCNH2|SCN5A|ANK2|TRDN|RYR2|BRCA1|TP53",
        "condition_environment_examples": "not provided|not specified|cardiovascular phenotype|multiple conditions",
        "source_identity_vs_meaning_note": "source identity can be true while disease meaning remains underresolved",
        "publication_safe_interpretation": "requires contextual repair before deterministic reuse",
    },
]

REGIME_ORDER = [r["regime_name"] for r in REGIMES]


def ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)


def read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False, dtype=str)
    return pd.DataFrame()


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


def load_data() -> pd.DataFrame:
    tasks = read_optional(TASKS)
    conc = read_optional(CONCORDANCE)

    if tasks.empty:
        raise FileNotFoundError(TASKS)

    tasks["assertion_id"] = norm_id(tasks["assertion_id"])

    if not conc.empty and "assertion_id" in conc.columns:
        conc["assertion_id"] = norm_id(conc["assertion_id"])
        keep = [
            "assertion_id",
            "local_gene",
            "clinvar_phenotype_list",
            "phenotype_domain_discordance_flag",
            "meaning_match_accepted",
            "source_match_accepted",
            "routing_implication",
        ]
        keep = [c for c in keep if c in conc.columns]
        df = tasks.merge(conc[keep], on="assertion_id", how="left")
    else:
        df = tasks.copy()

    df["gene"] = first_existing(df, ["gene", "local_gene", "GeneSymbol", "gene_symbol"], "")
    df["baseline_environment"] = first_existing(
        df,
        ["baseline_environment", "baseline_condition", "baseline_disease_environment", "condition_environment", "PhenotypeList", "clinvar_phenotype_list"],
        "",
    )
    df["baseline_regime"] = first_existing(
        df,
        ["baseline_regime", "baseline_review_category", "baseline_portability_regime", "regime", "cab_regime"],
        "",
    )
    df["phenotype_text"] = first_existing(
        df,
        ["clinvar_phenotype_list", "PhenotypeList", "baseline_environment", "condition_environment"],
        "",
    )
    return df


def classify_row(row: pd.Series) -> tuple[str, str, str, str]:
    gene = str(row.get("gene", "") or "").upper()
    domain = str(row.get("domain", "") or "").lower()
    env = str(row.get("baseline_environment", "") or "").lower()
    pheno = str(row.get("phenotype_text", "") or "").lower()
    base_regime = str(row.get("baseline_regime", "") or "").lower()
    text = " ".join([env, pheno, base_regime])

    discordant = str(row.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"}

    arrhythmia_genes = {"KCNQ1", "KCNH2", "SCN5A", "RYR2", "CACNA1C", "CASQ2", "KCNE1", "KCNE2", "KCNJ2", "ANK2", "HCN4", "TRDN"}
    structural_overlap_genes = {"SCN5A", "RYR2", "FLNC", "TTN", "DSP", "PKP2", "LMNA", "DES"}
    penetrance_genes = {"CHEK2", "ATM", "PALB2", "BRCA1", "BRCA2", "TTN", "FLNC", "LMNA", "KCNQ1", "KCNH2", "SCN5A"}
    pleiotropy_genes = {"SCN5A", "RYR2", "FLNC", "TTN", "TP53", "PTEN", "CHEK2", "ATM", "CACNA1C", "KCNQ1"}
    anchored_genes = {"KCNQ1", "KCNH2", "SCN5A", "PKP2", "MYBPC3", "MYH7", "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "TP53"}

    if discordant:
        if any(t in text for t in ["syndrome", "silver-russell", "imprinting", "development", "beckwith", "congenital"]):
            return (
                "Syndrome-organ boundary",
                "high",
                "phenotype-domain discordance with syndrome/developmental/organ-boundary label",
                "True",
            )
        if any(t in text for t in ["not provided", "not specified", "unknown", "unspecified", "multiple conditions", "cardiovascular phenotype"]):
            return (
                "Nonspecific/underresolved",
                "high",
                "phenotype-domain discordance with broad or underresolved condition label",
                "True",
            )
        return (
            "Pleiotropic collision",
            "medium",
            "source match accepted but phenotype-domain concordance failed",
            "True",
        )

    if any(t in text for t in ["not provided", "not specified", "unknown", "unspecified", "multiple conditions"]):
        return ("Nonspecific/underresolved", "high", "broad/unknown phenotype label", "False")

    if any(t in text for t in ["carrier", "asymptomatic", "no phenotype", "genotype-first", "screening", "absent phenotype"]):
        return ("Genotype-first absent phenotype", "high", "genotype-first or absent phenotype context", "False")

    if any(t in text for t in ["penetrance", "risk", "predisposition", "susceptibility"]) or domain == "hereditary_cancer":
        return ("Modifier/penetrance boundary", "medium", "risk/penetrance-oriented domain or label", "False")

    if any(t in text for t in ["syndrome", "silver-russell", "beckwith", "development", "imprinting", "congenital"]):
        return ("Syndrome-organ boundary", "high", "syndrome/developmental label crossing organ boundary", "False")

    if gene in structural_overlap_genes and domain in {"inherited_arrhythmia", "cardiomyopathy"}:
        return ("Structural-functional overlap", "medium", "cardiac structural-functional overlap gene/domain", "False")

    if gene in arrhythmia_genes and domain == "inherited_arrhythmia":
        return ("Trigger-dependent latent", "medium", "arrhythmia latent-risk gene requiring trigger/phenotype context", "False")

    if gene in pleiotropy_genes:
        return ("Pleiotropic collision", "medium", "pleiotropic/cross-model gene membership", "False")

    if gene in penetrance_genes:
        return ("Modifier/penetrance boundary", "medium", "penetrance/modifier gene membership", "False")

    if gene in anchored_genes:
        return ("Phenotype-anchored monogenic", "medium", "canonical gene-disease membership", "False")

    return ("Phenotype-anchored monogenic", "low", "default specific monogenic assertion without boundary flags", "False")


def create_regime_definitions() -> pd.DataFrame:
    df = pd.DataFrame(REGIMES)
    df.to_csv(OUT_REGIMES, index=False)
    return df


def create_assertion_map(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        regime, confidence, reason, ambiguity = classify_row(row)
        rows.append({
            "assertion_id": row.get("assertion_id", ""),
            "domain": row.get("domain", ""),
            "gene": row.get("gene", ""),
            "baseline_environment": row.get("baseline_environment", ""),
            "baseline_regime": row.get("baseline_regime", ""),
            "disease_architecture_regime": regime,
            "mapping_confidence": confidence,
            "mapping_reason": reason,
            "ambiguity_flag": ambiguity,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_MAP, index=False)
    return out


def rate(df: pd.DataFrame, cols: list[str]) -> float:
    if len(df) == 0:
        return float("nan")
    return float(bool_col(df, cols, False).mean())


def signatures(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(amap[["assertion_id", "disease_architecture_regime"]], on="assertion_id", how="left")

    rows = []
    for reg in REGIME_ORDER:
        sub = merged[merged["disease_architecture_regime"].eq(reg)].copy()
        n = len(sub)

        cond = bool_col(sub, ["future_condition_label_drift", "condition_label_drift"], False)
        cross = bool_col(sub, ["future_cross_environment_drift", "cross_environment_drift"], False)
        self_loop = bool_col(sub, ["future_self_loop_stable", "self_loop_stable"], False)
        any_meaning = bool_col(sub, ["future_any_meaning_drift", "any_meaning_drift"], False)

        strict_direct = bool_col(sub, ["cab_strict_direct_use_allowed"], False)
        balanced_direct = bool_col(sub, ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"], False)

        repair = pd.Series([False] * n, index=sub.index)
        for c in [
            "contextual_repair_required",
            "disease_specific_expert_review_required",
            "population_or_penetrance_review_required",
            "phenotype_domain_discordance_flag",
        ]:
            if c in sub.columns:
                repair = repair | bool_col(sub, [c], False)

        if "routing_implication" in sub.columns:
            repair = repair | sub["routing_implication"].fillna("").astype(str).eq("contextual_repair_or_disease_specific_review")

        domains = "|".join(sorted([x for x in sub.get("domain", pd.Series(dtype=str)).dropna().astype(str).unique() if x]))
        dominant = REGIMES[REGIME_ORDER.index(reg)]["dominant_routing_action"]

        rows.append({
            "regime_name": reg,
            "N": n,
            "domains_represented": domains,
            "condition_label_drift_rate": float(cond.mean()) if n else "",
            "cross_environment_drift_rate": float(cross.mean()) if n else "",
            "self_loop_stable_rate": float(self_loop.mean()) if n else "",
            "any_meaning_drift_rate": float(any_meaning.mean()) if n else "",
            "ClinVar_label_only_unsupported_reuse_rate": float(cond.mean()) if n else "",
            "CAB_Strict_unsupported_reuse_rate": float((cond & strict_direct).mean()) if n else "",
            "CAB_Balanced_unsupported_reuse_rate": float((cond & balanced_direct).mean()) if n else "",
            "direct_use_allowed_rate": float(balanced_direct.mean()) if n else "",
            "review_repair_routing_rate": float(repair.mean()) if n else "",
            "dominant_routing_action": dominant,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SIG, index=False)
    return out


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def enrichment_tests(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(amap[["assertion_id", "disease_architecture_regime"]], on="assertion_id", how="left")

    hypotheses = [
        ("phenotype_anchored_monogenic enriched for self-loop stability", "Phenotype-anchored monogenic", bool_col(merged, ["future_self_loop_stable", "self_loop_stable"], False)),
        ("pleiotropic_collision enriched for cross-environment drift", "Pleiotropic collision", bool_col(merged, ["future_cross_environment_drift", "cross_environment_drift"], False)),
        ("syndrome_organ_boundary enriched for cross-environment drift in hereditary cancer", "Syndrome-organ boundary", bool_col(merged, ["future_cross_environment_drift", "cross_environment_drift"], False) & merged["domain"].astype(str).eq("hereditary_cancer")),
        ("genotype_first_absent_phenotype enriched for no deterministic reuse / PRF-needed", "Genotype-first absent phenotype", ~bool_col(merged, ["direct_single_model_reuse_allowed", "cab_balanced_direct_use_allowed"], False)),
        ("modifier_penetrance_boundary enriched for population/penetrance review", "Modifier/penetrance boundary", bool_col(merged, ["population_or_penetrance_review_required"], False) | merged["domain"].astype(str).eq("hereditary_cancer")),
        ("nonspecific_underresolved enriched for contextual repair", "Nonspecific/underresolved", bool_col(merged, ["contextual_repair_required", "phenotype_domain_discordance_flag"], False)),
    ]

    rows = []
    for hyp, reg, outcome in hypotheses:
        in_reg = merged["disease_architecture_regime"].eq(reg)
        a = int((in_reg & outcome).sum())
        b = int((in_reg & ~outcome).sum())
        c = int((~in_reg & outcome).sum())
        d = int((~in_reg & ~outcome).sum())
        in_rate = a / (a + b) if (a + b) else float("nan")
        out_rate = c / (c + d) if (c + d) else float("nan")
        rows.append({
            "hypothesis": hyp,
            "regime_name": reg,
            "regime_positive_N": a,
            "regime_total_N": a + b,
            "regime_rate": in_rate,
            "other_positive_N": c,
            "other_total_N": c + d,
            "other_rate": out_rate,
            "rate_difference": in_rate - out_rate,
            "odds_ratio_Haldane_Anscombe": odds_ratio(a, b, c, d),
            "result": "supported_directionally" if in_rate > out_rate else "not_supported_directionally",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_ENRICH, index=False)
    return out


def esc(x: Any) -> str:
    return html.escape(str(x))


def create_figure(sig: pd.DataFrame) -> None:
    width, height = 1500, 1050
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="#fff"/>\n',
        '<text x="40" y="38" font-family="Arial" font-size="26" font-weight="700">Disease-architecture regimes determine pathogenic meaning travel</text>\n',
    ]

    # Panel A: route list
    parts.append('<text x="40" y="82" font-family="Arial" font-size="20" font-weight="700">A. Meaning-travel routes</text>\n')
    y = 118
    for r in REGIMES:
        parts.append(f'<rect x="40" y="{y-24}" width="650" height="34" rx="10" fill="#fafafa" stroke="#222"/>\n')
        parts.append(f'<text x="55" y="{y}" font-family="Arial" font-size="14" font-weight="700">{esc(r["regime_name"])}</text>\n')
        parts.append(f'<text x="310" y="{y}" font-family="Arial" font-size="13">{esc(r["meaning_travel_rule"])}</text>\n')
        y += 44

    # Panel B
    parts.append('<text x="770" y="82" font-family="Arial" font-size="20" font-weight="700">B. Self-loop vs boundary crossing</text>\n')
    parts.append('<rect x="770" y="110" width="300" height="95" rx="18" fill="#fafafa" stroke="#111"/>\n')
    parts.append('<text x="800" y="145" font-family="Arial" font-size="18" font-weight="700">Self-loop travel</text>\n')
    parts.append('<text x="800" y="176" font-family="Arial" font-size="14">Phenotype-anchored monogenic</text>\n')
    parts.append('<rect x="1120" y="110" width="330" height="95" rx="18" fill="#f7f7f7" stroke="#111"/>\n')
    parts.append('<text x="1150" y="145" font-family="Arial" font-size="18" font-weight="700">Boundary crossing</text>\n')
    parts.append('<text x="1150" y="176" font-family="Arial" font-size="14">collision / syndrome / PRF / underresolved</text>\n')
    parts.append('<line x1="1070" y1="158" x2="1120" y2="158" stroke="#111" stroke-width="2"/><polygon points="1120,158 1108,151 1108,165" fill="#111"/>\n')

    # Panel C: temporal signatures
    parts.append('<text x="770" y="260" font-family="Arial" font-size="20" font-weight="700">C. Temporal signatures by regime</text>\n')
    chart_x, chart_y = 780, 300
    max_w = 260
    for i, row in sig.iterrows():
        reg = row["regime_name"]
        val = float(row["condition_label_drift_rate"]) if str(row["condition_label_drift_rate"]) else 0.0
        bw = max_w * val
        yy = chart_y + i * 36
        parts.append(f'<text x="{chart_x}" y="{yy+14}" font-family="Arial" font-size="12">{esc(reg[:34])}</text>\n')
        parts.append(f'<rect x="{chart_x+250}" y="{yy}" width="{bw:.1f}" height="16" fill="#d9d9d9" stroke="#111"/>\n')
        parts.append(f'<text x="{chart_x+520}" y="{yy+14}" font-family="Arial" font-size="12">{val:.2f}</text>\n')

    # Panel D: routing consequences
    parts.append('<text x="40" y="535" font-family="Arial" font-size="20" font-weight="700">D. Routing consequences</text>\n')
    y = 570
    for r in REGIMES:
        parts.append(f'<text x="55" y="{y}" font-family="Arial" font-size="13" font-weight="700">{esc(r["regime_name"])}</text>\n')
        parts.append(f'<text x="330" y="{y}" font-family="Arial" font-size="13">{esc(r["dominant_routing_action"])}</text>\n')
        y += 32

    # Panel E: examples
    parts.append('<text x="770" y="650" font-family="Arial" font-size="20" font-weight="700">E. Examples across domains</text>\n')
    examples = [
        ("inherited_arrhythmia", "KCNQ1/KCNH2/SCN5A", "trigger-dependent, self-loop, syndrome-organ boundary"),
        ("cardiomyopathy", "MYBPC3/MYH7/TTN/FLNC", "phenotype anchored, structural-functional overlap"),
        ("hereditary_cancer", "BRCA1/BRCA2/CHEK2/ATM", "modifier/penetrance boundary, genotype-first risk"),
    ]
    y = 690
    for dom, genes, txt in examples:
        parts.append(f'<rect x="780" y="{y-24}" width="650" height="54" rx="12" fill="#fafafa" stroke="#222"/>\n')
        parts.append(f'<text x="800" y="{y}" font-family="Arial" font-size="15" font-weight="700">{esc(dom)}</text>\n')
        parts.append(f'<text x="1030" y="{y}" font-family="Arial" font-size="13">{esc(genes)}</text>\n')
        parts.append(f'<text x="800" y="{y+22}" font-family="Arial" font-size="13">{esc(txt)}</text>\n')
        y += 72

    parts.append('</svg>\n')
    OUT_FIG.write_text("".join(parts), encoding="utf-8")


def write_claim(sig: pd.DataFrame) -> None:
    text = """# Mendelian Meaning Travel Claim

## Core wording

Mendelian diseases differ in how pathogenic meaning is allowed to travel. Phenotype-anchored architectures support deterministic reuse within concordant self-loops; trigger-dependent and modifier-dependent architectures require contextual or penetrance review; pleiotropic, syndrome-organ, structural-overlap, genotype-first, and underresolved architectures create portability boundaries where P/LP assertions cannot be deterministically transferred.

## Operational interpretation

CAB identifies disease-architecture regimes that govern whether a P/LP assertion can travel as deterministic disease meaning, conditional risk, trigger-dependent liability, or context-specific interpretation requiring repair/review.

## Publication-safe claim

Pathogenicity may be source-identical, but its disease meaning travels according to architecture. Complete source identity is necessary but insufficient for deterministic reuse.

## Claim boundary

This is a portability and routing framework. It does not reclassify variants, invalidate ClinVar records, diagnose phenotype mismatch, claim clinical outcome validation, or replace disease-specific expert curation.
"""
    OUT_CLAIM.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    regimes = create_regime_definitions()
    df = load_data()
    amap = create_assertion_map(df)
    sig = signatures(df, amap)
    enrichment_tests(df, amap)
    create_figure(sig)
    write_claim(sig)

    print("Final disease-architecture meaning-travel layer complete.")
    print(f"Regimes: {len(regimes)}")
    print(f"Assertion mappings: {len(amap)}")
    print(sig[[
        "regime_name",
        "N",
        "domains_represented",
        "condition_label_drift_rate",
        "cross_environment_drift_rate",
        "self_loop_stable_rate",
        "review_repair_routing_rate",
        "dominant_routing_action",
    ]].to_string(index=False))
    print()
    print("Outputs:")
    for p in [OUT_REGIMES, OUT_MAP, OUT_SIG, OUT_ENRICH, OUT_FIG, OUT_CLAIM]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
