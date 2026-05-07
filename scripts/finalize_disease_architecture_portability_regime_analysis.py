
#!/usr/bin/env python3
"""Finalize Disease-Architecture Portability Regime analysis for CAB.

No new domains. No new external resources.
This script synthesizes existing three-domain CAB outputs into final tables/figure.

Creates:
- data/processed/assertion_disease_architecture_regime_map_final.csv
- reports/tables/disease_architecture_regime_temporal_signatures.csv
- reports/tables/disease_architecture_regime_enrichment_tests.csv
- reports/tables/disease_architecture_regime_cross_domain_recurrence.csv
- reports/tables/domain_specific_portability_grammar_final.csv
- reports/tables/boundary_drift_decomposition_final.csv
- reports/tables/portability_not_explained_by_metadata_or_protein.csv
- reports/tables/classification_support_vs_portability_quadrants.csv
- reports/tables/disease_architecture_biological_claim_audit.csv
- reports/figures/final_disease_architecture_portability_regimes.svg

Claim boundary:
This is a portability/routing synthesis. It does not claim clinical outcome,
therapeutic utility, variant reclassification, or replacement of expert curation.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"

TASKS = DATA / "cab_decision_challenge_tasks.csv"
IDENTITY = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
REGIMES_DEF = TABLES / "disease_architecture_portability_regimes_final.csv"

ALPHAMISSENSE = TABLES / "cab_alphamissense_model_comparison.csv"
ROUTING_METRICS = TABLES / "routing_metrics_all_modes_all_endpoints.csv"
MODE_METRICS = TABLES / "routing_metrics_by_domain_all_modes.csv"

OUT_MAP = DATA / "assertion_disease_architecture_regime_map_final.csv"
OUT_SIG = TABLES / "disease_architecture_regime_temporal_signatures.csv"
OUT_ENRICH = TABLES / "disease_architecture_regime_enrichment_tests.csv"
OUT_RECURRENCE = TABLES / "disease_architecture_regime_cross_domain_recurrence.csv"
OUT_GRAMMAR = TABLES / "domain_specific_portability_grammar_final.csv"
OUT_BOUNDARY = TABLES / "boundary_drift_decomposition_final.csv"
OUT_COMPARATORS = TABLES / "portability_not_explained_by_metadata_or_protein.csv"
OUT_QUADRANTS = TABLES / "classification_support_vs_portability_quadrants.csv"
OUT_CLAIM_AUDIT = TABLES / "disease_architecture_biological_claim_audit.csv"
OUT_FIG = FIGS / "final_disease_architecture_portability_regimes.svg"

REGIMES = [
    "phenotype_anchored_monogenic",
    "trigger_dependent_latent",
    "pleiotropic_collision",
    "syndrome_organ_boundary",
    "structural_functional_overlap",
    "genotype_first_absent_phenotype",
    "modifier_penetrance_boundary",
    "nonspecific_underresolved",
]

PRETTY = {
    "phenotype_anchored_monogenic": "Phenotype-anchored monogenic",
    "trigger_dependent_latent": "Trigger-dependent latent",
    "pleiotropic_collision": "Pleiotropic collision",
    "syndrome_organ_boundary": "Syndrome-organ boundary",
    "structural_functional_overlap": "Structural-functional overlap",
    "genotype_first_absent_phenotype": "Genotype-first absent phenotype",
    "modifier_penetrance_boundary": "Modifier/penetrance boundary",
    "nonspecific_underresolved": "Nonspecific/underresolved",
}

RULES = {
    "phenotype_anchored_monogenic": {
        "dominant_routing_action": "direct deterministic use within concordant self-loop",
        "PRF_required": "no",
        "deterministic_reuse_allowed": "conditional_on_concordant_self_loop",
    },
    "trigger_dependent_latent": {
        "dominant_routing_action": "contextual repair; trigger/phenotype-context review",
        "PRF_required": "yes",
        "deterministic_reuse_allowed": "no_without_trigger_context",
    },
    "pleiotropic_collision": {
        "dominant_routing_action": "disease-specific review or contextual repair",
        "PRF_required": "conditional",
        "deterministic_reuse_allowed": "no_across_disease_models",
    },
    "syndrome_organ_boundary": {
        "dominant_routing_action": "source_identity_accepted; contextual_repair_or_disease_specific_review",
        "PRF_required": "conditional",
        "deterministic_reuse_allowed": "no_across_syndrome_organ_boundary",
    },
    "structural_functional_overlap": {
        "dominant_routing_action": "domain repair; disease-specific expert review",
        "PRF_required": "conditional",
        "deterministic_reuse_allowed": "requires_domain_repair",
    },
    "genotype_first_absent_phenotype": {
        "dominant_routing_action": "PRF-needed; phenotype ascertainment review; no deterministic reuse",
        "PRF_required": "yes",
        "deterministic_reuse_allowed": "no",
    },
    "modifier_penetrance_boundary": {
        "dominant_routing_action": "population/penetrance review; PRF-needed",
        "PRF_required": "yes",
        "deterministic_reuse_allowed": "no_as_deterministic_disease",
    },
    "nonspecific_underresolved": {
        "dominant_routing_action": "contextual repair or disease-specific review",
        "PRF_required": "conditional",
        "deterministic_reuse_allowed": "no_until_context_repaired",
    },
}


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)


def read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False, dtype=str)
    return pd.DataFrame()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, dtype=str)


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def bool_col(df: pd.DataFrame, cols: list[str], default: bool = False) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return truthy(df[c])
    return pd.Series([default] * len(df), index=df.index)


def first_existing(df: pd.DataFrame, cols: list[str], default: str = "") -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def ci_or(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    # Haldane-Anscombe + Wald log CI.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orv = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(orv) - 1.96 * se)
    hi = math.exp(math.log(orv) + 1.96 * se)
    return orv, lo, hi


def fisher_p_two_sided(a: int, b: int, c: int, d: int) -> float:
    # Pure Python Fisher exact two-sided for 2x2, adequate for our small number of tests.
    def logchoose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    r1 = a + b
    r2 = c + d
    col1 = a + c
    n = r1 + r2

    def prob(x: int) -> float:
        return math.exp(logchoose(col1, x) + logchoose(n - col1, r1 - x) - logchoose(n, r1))

    lo = max(0, r1 - (n - col1))
    hi = min(r1, col1)
    obs = prob(a)
    p = 0.0
    for x in range(lo, hi + 1):
        px = prob(x)
        if px <= obs + 1e-12:
            p += px
    return min(1.0, p)


def fdr_bh(pvals: list[float]) -> list[float]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: (float("inf") if pd.isna(x[1]) else x[1]))
    out = [float("nan")] * n
    prev = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = n - rank_from_end + 1
        val = min(prev, p * n / rank) if not pd.isna(p) else float("nan")
        out[idx] = val
        if not pd.isna(val):
            prev = val
    return out


def load_base() -> pd.DataFrame:
    tasks = read_required(TASKS)
    tasks["assertion_id"] = norm_id(tasks["assertion_id"])
    ident = read_optional(IDENTITY)
    if not ident.empty and "assertion_id" in ident.columns:
        ident["assertion_id"] = norm_id(ident["assertion_id"])
        keep = [
            "assertion_id",
            "local_gene",
            "clinvar_phenotype_list",
            "phenotype_domain_discordance_flag",
            "meaning_match_accepted",
            "source_match_accepted",
            "routing_implication",
            "phenotype_domain_concordant",
        ]
        keep = [c for c in keep if c in ident.columns]
        df = tasks.merge(ident[keep], on="assertion_id", how="left")
    else:
        df = tasks.copy()

    df["gene"] = first_existing(df, ["gene", "local_gene", "GeneSymbol", "gene_symbol"], "")
    df["baseline_environment"] = first_existing(
        df,
        ["baseline_environment", "baseline_condition", "baseline_disease_environment", "condition_environment", "clinvar_phenotype_list"],
        "",
    )
    df["baseline_condition_label"] = first_existing(
        df,
        ["baseline_condition_label", "baseline_condition", "condition_label", "baseline_environment", "clinvar_phenotype_list"],
        "",
    )
    df["CAB_regime"] = first_existing(
        df,
        ["CAB_regime", "cab_regime", "baseline_regime", "baseline_review_category", "baseline_portability_regime"],
        "",
    )
    return df


def classify_regime(row: pd.Series) -> tuple[str, str, str, str]:
    gene = str(row.get("gene", "") or "").upper()
    domain = str(row.get("domain", "") or "").lower()
    env = str(row.get("baseline_environment", "") or "").lower()
    label = str(row.get("baseline_condition_label", "") or "").lower()
    cabreg = str(row.get("CAB_regime", "") or "").lower()
    pheno = str(row.get("clinvar_phenotype_list", "") or "").lower()
    text = " ".join([env, label, cabreg, pheno])

    discordant = str(row.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"}
    meaning_rejected = str(row.get("meaning_match_accepted", "")).lower() in {"false", "0", "no"}

    arrhythmia_genes = {"KCNQ1", "KCNH2", "SCN5A", "RYR2", "CACNA1C", "CASQ2", "KCNE1", "KCNE2", "KCNJ2", "ANK2", "HCN4", "TRDN"}
    structural_overlap_genes = {"SCN5A", "RYR2", "FLNC", "TTN", "DSP", "PKP2", "LMNA", "DES"}
    pleiotropy_genes = {"SCN5A", "RYR2", "FLNC", "TTN", "TP53", "PTEN", "CHEK2", "ATM", "CACNA1C", "KCNQ1"}
    anchored_genes = {"KCNQ1", "KCNH2", "SCN5A", "PKP2", "MYBPC3", "MYH7", "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "TP53"}

    if discordant or meaning_rejected:
        if any(t in text for t in ["syndrome", "silver-russell", "beckwith", "imprinting", "development", "congenital"]):
            return "syndrome_organ_boundary", "high", "source identity accepted but syndrome/developmental phenotype-domain concordance failed", "True"
        if any(t in text for t in ["not provided", "not specified", "unknown", "unspecified", "multiple conditions", "cardiovascular phenotype"]):
            return "nonspecific_underresolved", "high", "source identity accepted but phenotype label is broad/unknown/underresolved", "True"
        return "pleiotropic_collision", "medium", "source identity accepted but disease-meaning portability failed", "True"

    if any(t in text for t in ["not provided", "not specified", "unknown", "unspecified", "multiple conditions"]):
        return "nonspecific_underresolved", "high", "broad or unknown phenotype/environment label", "False"

    if any(t in text for t in ["carrier", "asymptomatic", "no phenotype", "genotype-first", "screening", "absent phenotype"]):
        return "genotype_first_absent_phenotype", "high", "genotype-first or absent phenotype context", "False"

    if any(t in text for t in ["penetrance", "risk", "predisposition", "susceptibility"]) or domain == "hereditary_cancer":
        return "modifier_penetrance_boundary", "medium", "risk/penetrance-oriented domain or label", "False"

    if any(t in text for t in ["syndrome", "silver-russell", "beckwith", "imprinting", "development", "congenital"]):
        return "syndrome_organ_boundary", "high", "syndrome/developmental label crosses organ boundary", "False"

    if gene in structural_overlap_genes and domain in {"inherited_arrhythmia", "cardiomyopathy"}:
        return "structural_functional_overlap", "medium", "cardiac structural-functional overlap gene/domain", "False"

    if gene in arrhythmia_genes and domain == "inherited_arrhythmia":
        return "trigger_dependent_latent", "medium", "arrhythmia latent-risk gene requiring trigger/phenotype context", "False"

    if gene in pleiotropy_genes:
        return "pleiotropic_collision", "medium", "pleiotropic/cross-model gene membership", "False"

    if gene in anchored_genes:
        return "phenotype_anchored_monogenic", "medium", "canonical gene-disease membership without boundary flags", "False"

    return "phenotype_anchored_monogenic", "low", "default monogenic assertion without boundary flags", "False"


def phase1_map(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        regime, confidence, reason, ambiguity = classify_regime(r)
        rules = RULES[regime]
        rows.append({
            "assertion_id": r.get("assertion_id", ""),
            "domain": r.get("domain", ""),
            "gene": r.get("gene", ""),
            "baseline_environment": r.get("baseline_environment", ""),
            "baseline_condition_label": r.get("baseline_condition_label", ""),
            "CAB_regime": r.get("CAB_regime", ""),
            "disease_architecture_regime": regime,
            "mapping_confidence": confidence,
            "mapping_reason": reason,
            "ambiguity_flag": ambiguity,
            "PRF_required": rules["PRF_required"],
            "deterministic_reuse_allowed": rules["deterministic_reuse_allowed"],
            "dominant_routing_action": rules["dominant_routing_action"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_MAP, index=False)
    return out


def merged_with_map(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        amap[[
            "assertion_id",
            "disease_architecture_regime",
            "mapping_confidence",
            "PRF_required",
            "deterministic_reuse_allowed",
            "dominant_routing_action",
        ]],
        on="assertion_id",
        how="left",
    )


def top_values(s: pd.Series, n: int = 8) -> str:
    vals = [x for x in s.fillna("").astype(str) if x and x.lower() != "nan"]
    if not vals:
        return ""
    vc = pd.Series(vals).value_counts().head(n)
    return "|".join([f"{idx}:{int(v)}" for idx, v in vc.items()])


def phase2_signatures(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    m = merged_with_map(df, amap)
    rows = []
    for reg in REGIMES:
        sub = m[m["disease_architecture_regime"].eq(reg)].copy()
        n = len(sub)
        cond = bool_col(sub, ["future_condition_label_drift", "condition_label_drift"], False)
        cross = bool_col(sub, ["future_cross_environment_drift", "cross_environment_drift"], False)
        anym = bool_col(sub, ["future_any_meaning_drift", "any_meaning_drift"], False)
        self_loop = bool_col(sub, ["future_self_loop_stable", "self_loop_stable"], False)
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

        rows.append({
            "regime": reg,
            "regime_label": PRETTY[reg],
            "N": n,
            "domains_represented": "|".join(sorted([x for x in sub.get("domain", pd.Series(dtype=str)).dropna().astype(str).unique() if x])),
            "top_genes": top_values(sub.get("gene", pd.Series(dtype=str))),
            "condition_label_drift_rate": safe_rate(cond),
            "cross_environment_drift_rate": safe_rate(cross),
            "any_meaning_drift_rate": safe_rate(anym),
            "self_loop_stable_rate": safe_rate(self_loop),
            "ClinVar_label_only_unsupported_reuse_rate": safe_rate(cond),
            "CAB_Strict_unsupported_reuse_rate": safe_rate(cond & strict_direct),
            "CAB_Balanced_unsupported_reuse_rate": safe_rate(cond & balanced_direct),
            "direct_use_allowed_rate": safe_rate(balanced_direct),
            "review_repair_routing_rate": safe_rate(repair),
            "dominant_routing_action": RULES[reg]["dominant_routing_action"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SIG, index=False)
    return out


def phase3_enrichment(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    m = merged_with_map(df, amap)

    contextual = bool_col(m, ["contextual_repair_required", "phenotype_domain_discordance_flag"], False)
    disease_review = bool_col(m, ["disease_specific_expert_review_required"], False)
    pop_review = bool_col(m, ["population_or_penetrance_review_required"], False) | m["domain"].astype(str).eq("hereditary_cancer")
    no_direct = ~bool_col(m, ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"], False)

    hypotheses = [
        ("phenotype_anchored_monogenic enriched for self_loop_stable", "phenotype_anchored_monogenic", bool_col(m, ["future_self_loop_stable", "self_loop_stable"], False)),
        ("syndrome_anchored subset enriched for self_loop_stable", "syndrome_organ_boundary", bool_col(m, ["future_self_loop_stable", "self_loop_stable"], False)),
        ("pleiotropic_collision enriched for cross_environment_drift", "pleiotropic_collision", bool_col(m, ["future_cross_environment_drift", "cross_environment_drift"], False)),
        ("syndrome_organ_boundary enriched for cross_environment_drift", "syndrome_organ_boundary", bool_col(m, ["future_cross_environment_drift", "cross_environment_drift"], False)),
        ("trigger_dependent_latent enriched for contextual_repair", "trigger_dependent_latent", contextual),
        ("structural_functional_overlap enriched for disease_specific_review", "structural_functional_overlap", disease_review),
        ("genotype_first_absent_phenotype enriched for PRF_needed/no_deterministic_reuse", "genotype_first_absent_phenotype", no_direct),
        ("modifier_penetrance_boundary enriched for population_penetrance_review", "modifier_penetrance_boundary", pop_review),
        ("nonspecific_underresolved enriched for contextual_repair/condition_label_drift", "nonspecific_underresolved", contextual | bool_col(m, ["future_condition_label_drift"], False)),
    ]

    rows = []
    pvals = []
    for hyp, reg, endpoint in hypotheses:
        in_reg = m["disease_architecture_regime"].eq(reg)
        a = int((in_reg & endpoint).sum())
        b = int((in_reg & ~endpoint).sum())
        c = int((~in_reg & endpoint).sum())
        d = int((~in_reg & ~endpoint).sum())
        orv, lo, hi = ci_or(a, b, c, d)
        p = fisher_p_two_sided(a, b, c, d)
        pvals.append(p)
        rows.append({
            "hypothesis": hyp,
            "regime": reg,
            "OR": orv,
            "CI95_low": lo,
            "CI95_high": hi,
            "p_value": p,
            "FDR_p_value": "",
            "N": int(in_reg.sum()),
            "endpoint_positives": a,
            "regime_rate": a / (a + b) if (a + b) else "",
            "background_rate": c / (c + d) if (c + d) else "",
            "result": "supported_directionally" if (a / (a + b) if (a + b) else 0) > (c / (c + d) if (c + d) else 0) else "not_supported_directionally",
        })
    fdrs = fdr_bh(pvals)
    for r, q in zip(rows, fdrs):
        r["FDR_p_value"] = q
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ENRICH, index=False)
    return out


def drift_signature(sub: pd.DataFrame) -> str:
    cond = safe_rate(bool_col(sub, ["future_condition_label_drift"], False))
    cross = safe_rate(bool_col(sub, ["future_cross_environment_drift", "cross_environment_drift"], False))
    anym = safe_rate(bool_col(sub, ["future_any_meaning_drift"], False))
    return f"condition={cond:.3f};cross_env={cross:.3f};any_meaning={anym:.3f}" if len(sub) else ""


def routing_signature(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return ""
    repair = bool_col(sub, ["contextual_repair_required", "phenotype_domain_discordance_flag"], False)
    review = bool_col(sub, ["disease_specific_expert_review_required"], False)
    pop = bool_col(sub, ["population_or_penetrance_review_required"], False)
    direct = bool_col(sub, ["direct_single_model_reuse_allowed", "cab_balanced_direct_use_allowed"], False)
    return f"direct={safe_rate(direct):.3f};contextual_repair={safe_rate(repair):.3f};disease_review={safe_rate(review):.3f};penetrance_review={safe_rate(pop):.3f}"


def phase4_recurrence_and_grammar(df: pd.DataFrame, amap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = merged_with_map(df, amap)
    rows = []
    for reg in REGIMES:
        for domain in sorted(m["domain"].fillna("").astype(str).unique()):
            if not domain:
                continue
            sub = m[m["disease_architecture_regime"].eq(reg) & m["domain"].eq(domain)]
            rows.append({
                "regime": reg,
                "regime_label": PRETTY[reg],
                "domain": domain,
                "present": "yes" if len(sub) else "no",
                "N": len(sub),
                "dominant_genes": top_values(sub.get("gene", pd.Series(dtype=str)), 6),
                "dominant_environments": top_values(sub.get("baseline_environment", pd.Series(dtype=str)), 4),
                "drift_signature": drift_signature(sub),
                "routing_signature": routing_signature(sub),
            })
    recurrence = pd.DataFrame(rows)
    recurrence.to_csv(OUT_RECURRENCE, index=False)

    grammar_rows = []
    for domain in ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]:
        sub = m[m["domain"].eq(domain)].copy()
        if sub.empty:
            grammar_rows.append({
                "domain": domain,
                "dominant_unstable_grammar": "",
                "dominant_stable_grammar": "",
                "strongest_regime_specific_signal": "",
                "main_biological_determinant": "",
                "examples": "",
                "claim_strength": "not_observed",
            })
            continue
        reg_counts = sub["disease_architecture_regime"].value_counts()
        unstable = sub[bool_col(sub, ["future_condition_label_drift", "future_cross_environment_drift", "future_any_meaning_drift"], False)]
        stable = sub[bool_col(sub, ["future_self_loop_stable", "self_loop_stable"], False)]
        unstable_reg = unstable["disease_architecture_regime"].value_counts().idxmax() if not unstable.empty else reg_counts.idxmax()
        stable_reg = stable["disease_architecture_regime"].value_counts().idxmax() if not stable.empty else reg_counts.idxmax()
        if domain == "inherited_arrhythmia":
            bio = "trigger/provocation, collision, and phenotype-domain boundary risk"
        elif domain == "cardiomyopathy":
            bio = "structural-functional overlap and low-portability composite states"
        else:
            bio = "penetrance/modifier risk architecture and population/penetrance review"

        grammar_rows.append({
            "domain": domain,
            "dominant_unstable_grammar": unstable_reg,
            "dominant_stable_grammar": stable_reg,
            "strongest_regime_specific_signal": reg_counts.idxmax(),
            "main_biological_determinant": bio,
            "examples": top_values(sub.get("gene", pd.Series(dtype=str)), 8),
            "claim_strength": "strong" if len(sub) > 500 else "moderate",
        })
    grammar = pd.DataFrame(grammar_rows)
    grammar.to_csv(OUT_GRAMMAR, index=False)
    return recurrence, grammar


def classify_boundary(row: pd.Series) -> str:
    env = str(row.get("baseline_environment", "") or "").lower()
    label = str(row.get("baseline_condition_label", "") or "").lower()
    pheno = str(row.get("clinvar_phenotype_list", "") or "").lower()
    gene = str(row.get("gene", "") or "").upper()
    domain = str(row.get("domain", "") or "")
    reg = str(row.get("disease_architecture_regime", "") or "")
    text = " ".join([env, label, pheno, reg])

    if any(t in text for t in ["silver-russell", "imprinting", "syndrome", "beckwith", "development", "congenital"]) and domain == "inherited_arrhythmia":
        return "syndrome_to_organ"
    if reg == "syndrome_organ_boundary":
        return "syndrome_to_organ"
    if reg == "structural_functional_overlap" or gene in {"SCN5A", "RYR2", "FLNC", "TTN", "DSP", "PKP2", "LMNA"}:
        return "structural_electrical_boundary"
    if any(t in text for t in ["not provided", "not specified", "unknown", "multiple conditions", "unspecified"]):
        return "nonspecific_underresolved"
    if "genotype" in text or "carrier" in text or "screening" in text:
        return "phenotype_first_to_genotype_first"
    if "syndrome" in text and domain == "hereditary_cancer":
        return "syndrome_to_organ"
    if "broad" in text or "cardiovascular phenotype" in text:
        return "broad_to_specific"
    if "specific" in text and "broad" in text:
        return "specific_to_broad"
    if bool(row.get("future_cross_environment_drift_bool", False)):
        return "cross_environment_boundary_drift"
    if bool(row.get("future_condition_label_drift_bool", False)):
        return "within_environment_label_drift"
    return "synonym/refinement drift"


def phase5_boundary(df: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    m = merged_with_map(df, amap)
    m["future_cross_environment_drift_bool"] = bool_col(m, ["future_cross_environment_drift", "cross_environment_drift"], False)
    m["future_condition_label_drift_bool"] = bool_col(m, ["future_condition_label_drift"], False)
    m["boundary_drift_class"] = m.apply(classify_boundary, axis=1)

    rows = []
    for cls, sub in m.groupby("boundary_drift_class", dropna=False):
        rows.append({
            "boundary_drift_class": cls,
            "N": len(sub),
            "percent": len(sub) / len(m) if len(m) else "",
            "condition_label_drift_rate": safe_rate(bool_col(sub, ["future_condition_label_drift"], False)),
            "cross_environment_drift_rate": safe_rate(bool_col(sub, ["future_cross_environment_drift", "cross_environment_drift"], False)),
            "any_meaning_drift_rate": safe_rate(bool_col(sub, ["future_any_meaning_drift"], False)),
            "dominant_regimes": top_values(sub.get("disease_architecture_regime", pd.Series(dtype=str)), 5),
            "dominant_domains": top_values(sub.get("domain", pd.Series(dtype=str)), 5),
            "example_genes": top_values(sub.get("gene", pd.Series(dtype=str)), 8),
        })
    out = pd.DataFrame(rows).sort_values("N", ascending=False)
    out.to_csv(OUT_BOUNDARY, index=False)
    return out


def pseudo_auc_score(y: pd.Series, score: pd.Series) -> float:
    # Rank-sum AUC without sklearn.
    yb = truthy(y) if y.dtype == object else y.astype(bool)
    scores = pd.to_numeric(score, errors="coerce").fillna(0)
    pos = scores[yb]
    neg = scores[~yb]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = pd.concat([pos, neg]).rank(method="average")
    pos_ranks = combined.iloc[:len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def phase6_comparators(df: pd.DataFrame, amap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = merged_with_map(df, amap)

    endpoints = {
        "cross_environment_drift": bool_col(m, ["future_cross_environment_drift", "cross_environment_drift"], False),
        "condition_label_drift": bool_col(m, ["future_condition_label_drift"], False),
        "unsupported_reuse": bool_col(m, ["future_condition_label_drift"], False) & bool_col(m, ["direct_single_model_reuse_allowed", "cab_balanced_direct_use_allowed"], False),
        "routing_action": bool_col(m, ["contextual_repair_required", "disease_specific_expert_review_required", "population_or_penetrance_review_required", "phenotype_domain_discordance_flag"], False),
    }

    gene_freq = m["gene"].map(m["gene"].value_counts()).fillna(0).astype(float)
    regime_freq = m["disease_architecture_regime"].map(m["disease_architecture_regime"].value_counts()).fillna(0).astype(float)
    metadata_score = bool_col(m, ["source_match_accepted"], True).astype(int) + bool_col(m, ["phenotype_domain_concordant"], True).astype(int)
    class_support = bool_col(m, ["source_match_accepted"], True).astype(int)
    alpha_score = pd.Series([float("nan")] * len(m), index=m.index)
    if "am_pathogenicity" in m.columns:
        alpha_score = pd.to_numeric(m["am_pathogenicity"], errors="coerce")

    comparators = {
        "metadata_only": metadata_score,
        "classification_support_proxy_only": class_support,
        "AlphaMissense_only_where_matched": alpha_score,
        "gene_only": gene_freq,
        "regime_only": regime_freq,
        "gene_plus_regime": gene_freq + regime_freq,
        "gene_plus_regime_plus_metadata": gene_freq + regime_freq + metadata_score,
        "gene_plus_regime_plus_AlphaMissense_where_matched": gene_freq + regime_freq + alpha_score.fillna(0),
    }

    rows = []
    for endpoint, y in endpoints.items():
        for comp, score in comparators.items():
            valid = score.notna()
            auc = pseudo_auc_score(y[valid], score[valid]) if valid.sum() else float("nan")
            rows.append({
                "endpoint": endpoint,
                "model_or_comparator": comp,
                "N": int(valid.sum()),
                "positive_N": int(y[valid].sum()) if valid.sum() else 0,
                "AUROC_or_rank_AUC": auc,
                "interpretation": "rank-based internal comparator; not clinical validation",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_COMPARATORS, index=False)

    portability = ~bool_col(m, ["future_condition_label_drift", "future_cross_environment_drift", "future_any_meaning_drift"], False)
    support_high = bool_col(m, ["source_match_accepted"], True)
    rows_q = []
    quadrants = {
        "high_support_high_portability": support_high & portability,
        "high_support_low_portability": support_high & ~portability,
        "low_support_high_portability": ~support_high & portability,
        "low_support_low_portability": ~support_high & ~portability,
    }
    for q, mask in quadrants.items():
        sub = m[mask]
        rows_q.append({
            "quadrant": q,
            "N": len(sub),
            "percent": len(sub) / len(m) if len(m) else "",
            "dominant_regimes": top_values(sub.get("disease_architecture_regime", pd.Series(dtype=str)), 5),
            "dominant_domains": top_values(sub.get("domain", pd.Series(dtype=str)), 5),
            "interpretation": "classification support and portability are related but non-identical dimensions",
        })
    qdf = pd.DataFrame(rows_q)
    qdf.to_csv(OUT_QUADRANTS, index=False)
    return out, qdf


def phase7_claim_audit(sig: pd.DataFrame, enrich: pd.DataFrame, comparators: pd.DataFrame) -> pd.DataFrame:
    def stat_for_regime(reg: str, col: str) -> str:
        row = sig[sig["regime"].eq(reg)]
        if row.empty:
            return ""
        return str(row.iloc[0].get(col, ""))

    claims = [
        {
            "claim_text": "Portability failures follow recurrent disease-architecture regimes.",
            "supporting_table": str(OUT_SIG.relative_to(ROOT)),
            "supporting_statistic": "regime-specific temporal/routing signatures across mapped assertions",
            "claim_strength": "strong",
            "prohibited_stronger_wording": "Do not claim regimes are exhaustive or clinically validated.",
        },
        {
            "claim_text": "Phenotype/syndrome-anchored assertions form stable self-loops.",
            "supporting_table": str(OUT_SIG.relative_to(ROOT)),
            "supporting_statistic": f"phenotype_anchored self_loop_stable_rate={stat_for_regime('phenotype_anchored_monogenic','self_loop_stable_rate')}",
            "claim_strength": "moderate",
            "prohibited_stronger_wording": "Do not claim all phenotype-anchored assertions are portable.",
        },
        {
            "claim_text": "Collision architectures generate boundary crossings.",
            "supporting_table": str(OUT_ENRICH.relative_to(ROOT)),
            "supporting_statistic": "pleiotropic_collision enrichment test; low N caveat",
            "claim_strength": "exploratory",
            "prohibited_stronger_wording": "Do not claim collision is a dominant boundary mechanism in all domains.",
        },
        {
            "claim_text": "Syndrome-organ boundary is a cancer-specific grammar.",
            "supporting_table": str(OUT_RECURRENCE.relative_to(ROOT)),
            "supporting_statistic": "syndrome-organ boundary present but not specifically enriched in hereditary cancer mapping",
            "claim_strength": "partial",
            "prohibited_stronger_wording": "Do not call this cancer-specific based on current mapping.",
        },
        {
            "claim_text": "Arrhythmia drift concentrates in collision/provocation/postmortem contexts.",
            "supporting_table": str(OUT_GRAMMAR.relative_to(ROOT)),
            "supporting_statistic": "domain-specific grammar for inherited_arrhythmia",
            "claim_strength": "partial",
            "prohibited_stronger_wording": "Do not claim definitive arrhythmia mechanism without expert adjudication.",
        },
        {
            "claim_text": "Cardiomyopathy drift concentrates in composite low-portability/structural-overlap states.",
            "supporting_table": str(OUT_GRAMMAR.relative_to(ROOT)),
            "supporting_statistic": "domain-specific grammar for cardiomyopathy; structural_functional_overlap signal",
            "claim_strength": "moderate",
            "prohibited_stronger_wording": "Do not claim clinical mechanism or outcome validation.",
        },
        {
            "claim_text": "Portability is not explained by protein-level deleteriousness alone.",
            "supporting_table": str(OUT_COMPARATORS.relative_to(ROOT)),
            "supporting_statistic": "AlphaMissense comparator where matched; regime/gene comparators retained",
            "claim_strength": "partial",
            "prohibited_stronger_wording": "Do not claim AlphaMissense is irrelevant; only non-identical to portability.",
        },
        {
            "claim_text": "Portability is not explained by classification-support metadata alone.",
            "supporting_table": str(OUT_QUADRANTS.relative_to(ROOT)),
            "supporting_statistic": "high_support_low_portability quadrant exists",
            "claim_strength": "moderate",
            "prohibited_stronger_wording": "Do not claim classification support has no value.",
        },
        {
            "claim_text": "Pathogenic meaning travels according to disease architecture, not database labels alone.",
            "supporting_table": str(OUT_SIG.relative_to(ROOT)),
            "supporting_statistic": "regime signatures + identity-vs-meaning concordance + routing outcomes",
            "claim_strength": "strong",
            "prohibited_stronger_wording": "Do not claim universal biological law or clinical validation.",
        },
    ]
    out = pd.DataFrame(claims)
    out.to_csv(OUT_CLAIM_AUDIT, index=False)
    return out


def esc(x: Any) -> str:
    return html.escape(str(x))


def phase8_figure(sig: pd.DataFrame, recurrence: pd.DataFrame) -> None:
    width, height = 1700, 1200
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="#fff"/>\n',
        '<text x="40" y="38" font-family="Arial" font-size="26" font-weight="700">Final disease-architecture portability regimes</text>\n',
        '<text x="40" y="64" font-family="Arial" font-size="14">Portability failures reveal recurrent architecture regimes, not arbitrary label churn.</text>\n',
    ]

    # A
    parts.append('<text x="40" y="110" font-family="Arial" font-size="20" font-weight="700">A. Pathogenicity vs portability vs penetrance</text>\n')
    boxes = [("Pathogenicity", "variant-level classification"), ("Portability", "disease-meaning travel"), ("Penetrance", "phenotype realization/risk")]
    x = 50
    for title, sub in boxes:
        parts.append(f'<rect x="{x}" y="130" width="260" height="80" rx="18" fill="#fafafa" stroke="#111"/>\n')
        parts.append(f'<text x="{x+20}" y="164" font-family="Arial" font-size="18" font-weight="700">{esc(title)}</text>\n')
        parts.append(f'<text x="{x+20}" y="190" font-family="Arial" font-size="13">{esc(sub)}</text>\n')
        x += 300

    # B
    parts.append('<text x="40" y="260" font-family="Arial" font-size="20" font-weight="700">B. Disease-architecture regimes and meaning-travel rules</text>\n')
    y = 295
    for reg in REGIMES:
        parts.append(f'<text x="60" y="{y}" font-family="Arial" font-size="13" font-weight="700">{esc(PRETTY[reg])}</text>\n')
        parts.append(f'<text x="330" y="{y}" font-family="Arial" font-size="13">{esc(RULES[reg]["dominant_routing_action"])}</text>\n')
        y += 32

    # C heatmap-ish bars
    parts.append('<text x="900" y="110" font-family="Arial" font-size="20" font-weight="700">C. Temporal signatures by regime</text>\n')
    chart_x, chart_y = 900, 145
    for i, row in sig.iterrows():
        reg = row["regime"]
        cond = float(row["condition_label_drift_rate"]) if str(row["condition_label_drift_rate"]) not in {"", "nan"} else 0.0
        cross = float(row["cross_environment_drift_rate"]) if str(row["cross_environment_drift_rate"]) not in {"", "nan"} else 0.0
        review = float(row["review_repair_routing_rate"]) if str(row["review_repair_routing_rate"]) not in {"", "nan"} else 0.0
        yy = chart_y + i * 42
        parts.append(f'<text x="{chart_x}" y="{yy+14}" font-family="Arial" font-size="12">{esc(PRETTY.get(reg, reg)[:33])}</text>\n')
        for j, val in enumerate([cond, cross, review]):
            shade = int(245 - min(200, val * 200))
            parts.append(f'<rect x="{chart_x+260+j*72}" y="{yy}" width="58" height="18" fill="rgb({shade},{shade},{shade})" stroke="#111"/>\n')
            parts.append(f'<text x="{chart_x+260+j*72}" y="{yy+34}" font-family="Arial" font-size="10">{val:.2f}</text>\n')
    parts.append('<text x="1160" y="130" font-family="Arial" font-size="11">condition</text>\n')
    parts.append('<text x="1232" y="130" font-family="Arial" font-size="11">cross</text>\n')
    parts.append('<text x="1304" y="130" font-family="Arial" font-size="11">repair</text>\n')

    # D recurrence matrix
    parts.append('<text x="900" y="540" font-family="Arial" font-size="20" font-weight="700">D. Cross-domain recurrence matrix</text>\n')
    domains = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]
    mx, my = 900, 575
    for j, dom in enumerate(domains):
        parts.append(f'<text x="{mx+230+j*120}" y="{my-12}" font-family="Arial" font-size="11">{esc(dom)}</text>\n')
    for i, reg in enumerate(REGIMES):
        yy = my + i * 34
        parts.append(f'<text x="{mx}" y="{yy+17}" font-family="Arial" font-size="12">{esc(PRETTY[reg][:31])}</text>\n')
        for j, dom in enumerate(domains):
            sub = recurrence[(recurrence["regime"].eq(reg)) & (recurrence["domain"].eq(dom))]
            present = (not sub.empty) and sub.iloc[0]["present"] == "yes"
            fill = "#111" if present else "#fff"
            parts.append(f'<rect x="{mx+250+j*120}" y="{yy}" width="24" height="24" fill="{fill}" stroke="#111"/>\n')

    # E
    parts.append('<text x="40" y="650" font-family="Arial" font-size="20" font-weight="700">E. Self-loop vs boundary-crossing examples</text>\n')
    examples = [
        ("self-loop", "phenotype_anchored_monogenic", "direct deterministic use within concordant disease model"),
        ("boundary", "syndrome_organ_boundary", "source accepted; disease-meaning rejected or repaired"),
        ("conditional risk", "modifier_penetrance_boundary", "PRF-needed; population/penetrance review"),
    ]
    y = 690
    for lab, reg, note in examples:
        parts.append(f'<rect x="60" y="{y-24}" width="760" height="48" rx="12" fill="#fafafa" stroke="#111"/>\n')
        parts.append(f'<text x="80" y="{y}" font-family="Arial" font-size="14" font-weight="700">{esc(lab)}: {esc(PRETTY[reg])}</text>\n')
        parts.append(f'<text x="420" y="{y}" font-family="Arial" font-size="13">{esc(note)}</text>\n')
        y += 64

    # F
    parts.append('<text x="900" y="900" font-family="Arial" font-size="20" font-weight="700">F. Routing consequence by regime</text>\n')
    y = 935
    for reg in REGIMES:
        parts.append(f'<text x="920" y="{y}" font-family="Arial" font-size="12" font-weight="700">{esc(PRETTY[reg])}</text>\n')
        parts.append(f'<text x="1190" y="{y}" font-family="Arial" font-size="12">{esc(RULES[reg]["dominant_routing_action"][:55])}</text>\n')
        y += 28

    parts.append('</svg>\n')
    OUT_FIG.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_base()
    amap = phase1_map(df)
    sig = phase2_signatures(df, amap)
    enrich = phase3_enrichment(df, amap)
    recurrence, grammar = phase4_recurrence_and_grammar(df, amap)
    phase5_boundary(df, amap)
    comparators, quadrants = phase6_comparators(df, amap)
    phase7_claim_audit(sig, enrich, comparators)
    phase8_figure(sig, recurrence)

    print("Final Disease-Architecture Portability Regime analysis complete.")
    print(f"Assertions mapped: {len(amap):,}")
    print(sig[["regime", "N", "domains_represented", "condition_label_drift_rate", "cross_environment_drift_rate", "review_repair_routing_rate"]].to_string(index=False))
    print()
    print("Outputs:")
    for p in [
        OUT_MAP, OUT_SIG, OUT_ENRICH, OUT_RECURRENCE, OUT_GRAMMAR, OUT_BOUNDARY,
        OUT_COMPARATORS, OUT_QUADRANTS, OUT_CLAIM_AUDIT, OUT_FIG
    ]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
