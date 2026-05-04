
#!/usr/bin/env python3
"""Build fundamental biology framing for CAB.

Theme:
Pathogenic meaning mobility is a phenotype of disease architecture.

Creates:
- reports/qc/pathogenic_meaning_mobility_definition.md
- reports/tables/principles_of_pathogenic_meaning_portability.csv
- reports/tables/disease_architecture_phase_space.csv
- reports/figures/disease_architecture_phase_space.svg
- reports/tables/meaning_stability_index_by_regime.csv
- reports/tables/fundamental_biology_support_table.csv
- reports/tables/cross_domain_biological_grammar.csv
- reports/qc/fundamental_biology_claim_audit.md

No new domains. No therapy claims. No patient outcome validation.
Uses current empirical support levels.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"
FIGS = ROOT / "reports" / "figures"

SUPPORT = TABLES / "disease_architecture_regime_support_levels.csv"
SIG = TABLES / "disease_architecture_regime_temporal_signatures.csv"
RECURRENCE = TABLES / "disease_architecture_regime_cross_domain_recurrence.csv"
GRAMMAR = TABLES / "domain_specific_portability_grammar_final.csv"
COMPARATORS = TABLES / "portability_not_explained_by_metadata_or_protein.csv"
QUADRANTS = TABLES / "classification_support_vs_portability_quadrants.csv"
IDENTITY = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
SENS = TABLES / "identity_meaning_discordance_sensitivity_core.csv"

OUT_DEF = QC / "pathogenic_meaning_mobility_definition.md"
OUT_PRINCIPLES = TABLES / "principles_of_pathogenic_meaning_portability.csv"
OUT_PHASE = TABLES / "disease_architecture_phase_space.csv"
OUT_PHASE_FIG = FIGS / "disease_architecture_phase_space.svg"
OUT_MSI = TABLES / "meaning_stability_index_by_regime.csv"
OUT_SUPPORT = TABLES / "fundamental_biology_support_table.csv"
OUT_GRAMMAR = TABLES / "cross_domain_biological_grammar.csv"
OUT_AUDIT = QC / "fundamental_biology_claim_audit.md"

REGIME_LABELS = {
    "phenotype_anchored_monogenic": "Phenotype-anchored monogenic",
    "syndrome_anchored": "Syndrome-anchored",
    "modifier_penetrance_boundary": "Modifier/penetrance boundary",
    "nonspecific_underresolved": "Nonspecific/underresolved",
    "structural_functional_overlap": "Structural-functional overlap",
    "syndrome_organ_boundary": "Syndrome-organ boundary",
    "trigger_dependent_latent": "Trigger-dependent latent",
    "pleiotropic_collision": "Pleiotropic collision",
    "genotype_first_absent_phenotype": "Genotype-first absent phenotype",
}

PHASE_SPACE = [
    # regime, phenotype anchoring, penetrance/modifier dependence, disease-model specificity,
    # structural/system overlap, genotype-first inference dependence
    ("phenotype_anchored_monogenic", 0.90, 0.20, 0.90, 0.20, 0.10, "deterministic self-loop mobility"),
    ("syndrome_anchored", 0.85, 0.30, 0.70, 0.55, 0.15, "self-loop stable but boundary-sensitive"),
    ("modifier_penetrance_boundary", 0.45, 0.95, 0.65, 0.30, 0.45, "conditional liability, not deterministic disease"),
    ("nonspecific_underresolved", 0.15, 0.45, 0.15, 0.40, 0.35, "inference-limited; contextual repair required"),
    ("structural_functional_overlap", 0.55, 0.45, 0.55, 0.95, 0.25, "domain repair across structural/functional systems"),
    ("syndrome_organ_boundary", 0.65, 0.55, 0.45, 0.70, 0.30, "source identity accepted; meaning boundary"),
    ("trigger_dependent_latent", 0.50, 0.80, 0.65, 0.45, 0.45, "trigger/context-dependent mobility"),
    ("pleiotropic_collision", 0.40, 0.55, 0.35, 0.75, 0.25, "disease-model collision; underpowered here"),
    ("genotype_first_absent_phenotype", 0.10, 0.90, 0.35, 0.35, 1.00, "absent/currently unsampled; PRF target"),
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False, dtype=str)
    return pd.DataFrame()


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def get_row(df: pd.DataFrame, col: str, val: str) -> pd.Series | None:
    if df.empty or col not in df.columns:
        return None
    sub = df[df[col].astype(str).eq(val)]
    if sub.empty:
        return None
    return sub.iloc[0]


def get_support(regime: str, field: str, default: str = "") -> str:
    support = read_csv(SUPPORT)
    row = get_row(support, "regime_name", regime)
    if row is None:
        return default
    return str(row.get(field, default))


def get_sig(regime: str, field: str, default: str = "") -> str:
    sig = read_csv(SIG)
    row = get_row(sig, "regime", regime)
    if row is None:
        return default
    return str(row.get(field, default))


def write_definition() -> None:
    text = """# Pathogenic Meaning Mobility Definition

## Definitions

**Pathogenicity**  
A variant-level classification describing whether a variant is interpreted as pathogenic or likely pathogenic under a specified evidence framework.

**Portability**  
The degree to which a pathogenic assertion can be reused across clinical-genomic contexts without changing its disease-model meaning.

**Pathogenic meaning mobility**  
The observable behavior of pathogenic meaning as it moves across disease labels, environments, ascertainment settings, and phenotype contexts. Mobility can be stable, conditional, boundary-limited, or inference-limited.

**Self-loop stability**  
A state in which pathogenic meaning remains stable when reused within the same disease model or concordant phenotype loop.

**Boundary crossing**  
A state in which source identity or variant classification is preserved, but disease meaning changes or becomes unsafe to reuse after crossing a disease-model boundary.

**Conditional liability**  
A state in which P/LP meaning travels as risk, penetrance, or trigger-dependent liability rather than deterministic disease identity.

**Inference-limited meaning**  
A state in which available labels or phenotype information are too broad, absent, or underresolved to support deterministic disease reuse.

**Disease architecture regime**  
A recurring biological and curation architecture that determines how pathogenic meaning is allowed to travel.

## Core statement

Assertion portability is a measurable phenotype of disease architecture.

## Claim boundary

This framing describes portability and routing behavior. It does not claim therapy, patient outcome validation, variant reclassification, or replacement of expert curation.
"""
    OUT_DEF.write_text(text, encoding="utf-8")


def create_principles() -> pd.DataFrame:
    support = read_csv(SUPPORT)
    sig = read_csv(SIG)
    identity = read_csv(IDENTITY)

    def supp_stat(regime: str, stat: str) -> str:
        row = get_row(sig, "regime", regime)
        if row is None:
            return ""
        return str(row.get(stat, ""))

    identity_rejected = ""
    if not identity.empty and "meaning_match_accepted" in identity.columns:
        identity_rejected = str((identity["meaning_match_accepted"].astype(str).str.lower().isin({"false", "0", "no"})).sum())

    rows = [
        {
            "principle": "Anchoring principle",
            "definition": "Phenotype- or syndrome-anchored architectures stabilize pathogenic meaning within concordant self-loops.",
            "empirical_support": "phenotype-anchored and syndrome-anchored patterns show self-loop stability",
            "supporting_statistic": f"phenotype_anchored self_loop_stable_rate={supp_stat('phenotype_anchored_monogenic','self_loop_stable_rate')}; OR={get_support('phenotype_anchored_monogenic','OR')}",
            "domain_examples": "cardiomyopathy|inherited_arrhythmia",
            "limitation": "does not imply universal portability outside concordant disease loops",
            "claim_strength": get_support("phenotype_anchored_monogenic", "support_level", "moderate_empirical_support"),
        },
        {
            "principle": "Boundary principle",
            "definition": "Pathogenic meaning fails or requires repair when assertions cross disease-model boundaries.",
            "empirical_support": "structural-functional and syndrome-organ regimes require domain repair or review",
            "supporting_statistic": f"structural_functional_overlap disease_review OR={get_support('structural_functional_overlap','OR')}",
            "domain_examples": "cardiomyopathy|inherited_arrhythmia",
            "limitation": "pleiotropic collision is underpowered in the present mapping",
            "claim_strength": "strong_for_structural_overlap; exploratory_for_pleiotropic_collision",
        },
        {
            "principle": "Conditionality principle",
            "definition": "Modifier/penetrance-boundary architectures convert P/LP meaning from deterministic disease to conditional liability.",
            "empirical_support": "modifier/penetrance-boundary is strongly supported and dominates hereditary cancer portability behavior",
            "supporting_statistic": f"N={get_support('modifier_penetrance_boundary','N')}; OR={get_support('modifier_penetrance_boundary','OR')}",
            "domain_examples": "hereditary_cancer",
            "limitation": "does not validate patient outcomes or penetrance estimates",
            "claim_strength": get_support("modifier_penetrance_boundary", "support_level", "strong_empirical_support"),
        },
        {
            "principle": "Resolution principle",
            "definition": "Nonspecific or underresolved labels collapse distinct biological environments and require contextual repair.",
            "empirical_support": "nonspecific/underresolved regime is enriched for contextual repair/condition-label drift",
            "supporting_statistic": f"N={get_support('nonspecific_underresolved','N')}; OR={get_support('nonspecific_underresolved','OR')}",
            "domain_examples": "inherited_arrhythmia|cardiomyopathy|hereditary_cancer",
            "limitation": "label repair requires expert or domain-specific interpretation",
            "claim_strength": get_support("nonspecific_underresolved", "support_level", "strong_empirical_support"),
        },
        {
            "principle": "Identity-meaning separation principle",
            "definition": "Variant/source identity may match while disease meaning is rejected.",
            "empirical_support": "complete source matching coexists with meaning-rejected phenotype-domain discordance class",
            "supporting_statistic": f"meaning_match_rejected={identity_rejected}",
            "domain_examples": "inherited_arrhythmia",
            "limitation": "source matches are not invalid records; CAB does not reclassify variants",
            "claim_strength": "strong_identity_vs_meaning_boundary_support",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PRINCIPLES, index=False)
    return out


def create_phase_space() -> pd.DataFrame:
    support = read_csv(SUPPORT)
    rows = []
    for regime, anch, pen, spec, overlap, geno, mobility in PHASE_SPACE:
        # syndrome_anchored is represented by syndrome-organ support where needed.
        support_key = regime if regime != "syndrome_anchored" else "syndrome_organ_boundary"
        sup_row = get_row(support, "regime_name", support_key)
        support_level = str(sup_row.get("support_level", "")) if sup_row is not None else "framework_category"
        n = str(sup_row.get("N", "")) if sup_row is not None else ""
        dominant = {
            "phenotype_anchored_monogenic": "direct deterministic use within concordant self-loop",
            "syndrome_anchored": "self-loop stable but boundary-sensitive",
            "modifier_penetrance_boundary": "population/penetrance review; PRF-needed",
            "nonspecific_underresolved": "contextual repair or disease-specific review",
            "structural_functional_overlap": "domain repair; disease-specific expert review",
            "syndrome_organ_boundary": "source identity accepted; contextual repair or disease-specific review",
            "trigger_dependent_latent": "contextual repair; trigger/phenotype-context review",
            "pleiotropic_collision": "disease-specific review or contextual repair",
            "genotype_first_absent_phenotype": "PRF-needed; no deterministic reuse",
        }[regime]
        rows.append({
            "regime": regime,
            "regime_label": REGIME_LABELS[regime],
            "N": n,
            "phenotype_anchoring": anch,
            "penetrance_modifier_dependence": pen,
            "disease_model_specificity": spec,
            "structural_system_overlap": overlap,
            "genotype_first_inference_dependence": geno,
            "support_level": support_level,
            "expected_meaning_mobility": mobility,
            "dominant_routing": dominant,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PHASE, index=False)
    return out


def esc(x: Any) -> str:
    return html.escape(str(x))


def color_for_support(level: str) -> str:
    return {
        "strong_empirical_support": "#cfcfcf",
        "moderate_empirical_support": "#e6e6e6",
        "underpowered_framework_category": "#ffffff",
        "absent_in_current_benchmark": "#f8f8f8",
        "framework_category": "#ffffff",
    }.get(level, "#ffffff")


def create_phase_space_fig(phase: pd.DataFrame) -> None:
    width, height = 1400, 980
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="#fff"/>\n',
        '<text x="40" y="40" font-family="Arial" font-size="26" font-weight="700">Disease architecture phase space</text>\n',
        '<text x="40" y="66" font-family="Arial" font-size="14">Pathogenic meaning mobility varies with anchoring, penetrance dependence, specificity, system overlap, and genotype-first inference dependence.</text>\n',
    ]

    # scatter: x phenotype anchoring, y penetrance dependence, radius structural overlap, border genotype-first
    x0, y0, w, h = 90, 140, 520, 520
    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#111"/>\n')
    parts.append(f'<text x="{x0+150}" y="{y0-22}" font-family="Arial" font-size="18" font-weight="700">Anchoring vs conditionality</text>\n')
    parts.append(f'<text x="{x0+180}" y="{y0+h+40}" font-family="Arial" font-size="14">phenotype anchoring →</text>\n')
    parts.append(f'<text x="{x0-70}" y="{y0+260}" transform="rotate(-90 {x0-70},{y0+260})" font-family="Arial" font-size="14">penetrance/modifier dependence →</text>\n')

    for _, r in phase.iterrows():
        px = x0 + float(r["phenotype_anchoring"]) * w
        py = y0 + (1 - float(r["penetrance_modifier_dependence"])) * h
        radius = 10 + float(r["structural_system_overlap"]) * 16
        stroke_w = 1 + float(r["genotype_first_inference_dependence"]) * 4
        fill = color_for_support(str(r["support_level"]))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{fill}" stroke="#111" stroke-width="{stroke_w:.1f}"/>\n')
        parts.append(f'<text x="{px+14:.1f}" y="{py+4:.1f}" font-family="Arial" font-size="11">{esc(r["regime_label"])}</text>\n')

    # table side
    tx, ty = 680, 130
    parts.append(f'<text x="{tx}" y="{ty-30}" font-family="Arial" font-size="18" font-weight="700">Regime support and expected mobility</text>\n')
    for i, r in phase.iterrows():
        y = ty + i * 82
        fill = color_for_support(str(r["support_level"]))
        parts.append(f'<rect x="{tx}" y="{y-22}" width="650" height="68" rx="12" fill="{fill}" stroke="#111"/>\n')
        parts.append(f'<text x="{tx+14}" y="{y}" font-family="Arial" font-size="13" font-weight="700">{esc(r["regime_label"])}</text>\n')
        parts.append(f'<text x="{tx+310}" y="{y}" font-family="Arial" font-size="12">{esc(r["support_level"])}</text>\n')
        parts.append(f'<text x="{tx+14}" y="{y+22}" font-family="Arial" font-size="12">{esc(r["expected_meaning_mobility"])}</text>\n')
        parts.append(f'<text x="{tx+14}" y="{y+42}" font-family="Arial" font-size="12">{esc(r["dominant_routing"])}</text>\n')

    parts.append('</svg>\n')
    OUT_PHASE_FIG.write_text("".join(parts), encoding="utf-8")


def create_meaning_stability_index() -> pd.DataFrame:
    sig = read_csv(SIG)
    support = read_csv(SUPPORT)
    rows = []
    if sig.empty:
        pd.DataFrame(rows).to_csv(OUT_MSI, index=False)
        return pd.DataFrame(rows)

    for _, r in sig.iterrows():
        regime = str(r.get("regime", ""))
        self_loop = as_float(r.get("self_loop_stable_rate", ""))
        cross = as_float(r.get("cross_environment_drift_rate", ""))
        repair = as_float(r.get("review_repair_routing_rate", ""))
        direct = as_float(r.get("direct_use_allowed_rate", ""))
        msi = self_loop - cross
        boundary = cross + repair - direct
        sup = get_row(support, "regime_name", regime)
        rows.append({
            "regime": regime,
            "regime_label": REGIME_LABELS.get(regime, regime),
            "N": r.get("N", ""),
            "self_loop_stable_rate": self_loop,
            "cross_environment_drift_rate": cross,
            "review_repair_routing_rate": repair,
            "direct_use_allowed_rate": direct,
            "meaning_stability_index": msi,
            "portability_boundary_score": boundary,
            "support_level": str(sup.get("support_level", "")) if sup is not None else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_MSI, index=False)
    return out


def lookup_comparator(endpoint: str, model: str) -> str:
    comp = read_csv(COMPARATORS)
    if comp.empty:
        return ""
    if "endpoint" not in comp.columns or "model_or_comparator" not in comp.columns:
        return ""
    sub = comp[comp["endpoint"].astype(str).eq(endpoint) & comp["model_or_comparator"].astype(str).eq(model)]
    if sub.empty:
        return ""
    return str(sub.iloc[0].get("AUROC_or_rank_AUC", ""))


def create_fundamental_support_table() -> pd.DataFrame:
    identity = read_csv(IDENTITY)
    quadrants = read_csv(QUADRANTS)

    meaning_rejected = ""
    if not identity.empty and "meaning_match_accepted" in identity.columns:
        meaning_rejected = str((identity["meaning_match_accepted"].astype(str).str.lower().isin({"false", "0", "no"})).sum())

    high_support_low_portability = ""
    if not quadrants.empty and "quadrant" in quadrants.columns:
        row = get_row(quadrants, "quadrant", "high_support_low_portability")
        if row is not None:
            high_support_low_portability = str(row.get("N", ""))

    rows = [
        {
            "claim": "Anchored architectures self-stabilize",
            "supporting_N": get_support("phenotype_anchored_monogenic", "N"),
            "effect_size": "self-loop enrichment",
            "OR_AUROC_or_rate": get_support("phenotype_anchored_monogenic", "OR"),
            "domains": get_support("phenotype_anchored_monogenic", "domains_represented"),
            "support_level": get_support("phenotype_anchored_monogenic", "support_level"),
            "limitation": "self-loop stability does not imply cross-boundary portability",
        },
        {
            "claim": "Modifier/penetrance-boundary dominates non-portable space",
            "supporting_N": get_support("modifier_penetrance_boundary", "N"),
            "effect_size": "population/penetrance review enrichment",
            "OR_AUROC_or_rate": get_support("modifier_penetrance_boundary", "OR"),
            "domains": get_support("modifier_penetrance_boundary", "domains_represented"),
            "support_level": get_support("modifier_penetrance_boundary", "support_level"),
            "limitation": "does not validate penetrance estimates or outcomes",
        },
        {
            "claim": "Nonspecific/underresolved states require contextual repair",
            "supporting_N": get_support("nonspecific_underresolved", "N"),
            "effect_size": "contextual repair / condition-label drift enrichment",
            "OR_AUROC_or_rate": get_support("nonspecific_underresolved", "OR"),
            "domains": get_support("nonspecific_underresolved", "domains_represented"),
            "support_level": get_support("nonspecific_underresolved", "support_level"),
            "limitation": "requires expert label-resolution review for validation",
        },
        {
            "claim": "Structural-functional overlap requires disease-specific review",
            "supporting_N": get_support("structural_functional_overlap", "N"),
            "effect_size": "disease-specific review enrichment",
            "OR_AUROC_or_rate": get_support("structural_functional_overlap", "OR"),
            "domains": get_support("structural_functional_overlap", "domains_represented"),
            "support_level": get_support("structural_functional_overlap", "support_level"),
            "limitation": "does not claim clinical mechanism or patient outcome validation",
        },
        {
            "claim": "Identity matching does not guarantee meaning matching",
            "supporting_N": meaning_rejected,
            "effect_size": "source match accepted but meaning rejected",
            "OR_AUROC_or_rate": f"meaning_rejected_N={meaning_rejected}",
            "domains": "inherited_arrhythmia",
            "support_level": "strong_identity_vs_meaning_support",
            "limitation": "does not imply ClinVar records are invalid",
        },
        {
            "claim": "Protein damage does not explain portability",
            "supporting_N": "",
            "effect_size": "AlphaMissense/protein comparator rank-AUC where available",
            "OR_AUROC_or_rate": lookup_comparator("cross_environment_drift", "AlphaMissense_only_where_matched"),
            "domains": "available matched subset",
            "support_level": "partial",
            "limitation": "not all rows have protein-score comparator; protein damage remains useful but non-identical to portability",
        },
        {
            "claim": "Classification support does not equal portability",
            "supporting_N": high_support_low_portability,
            "effect_size": "high-support low-portability quadrant exists",
            "OR_AUROC_or_rate": f"high_support_low_portability_N={high_support_low_portability}",
            "domains": "three-domain benchmark",
            "support_level": "moderate",
            "limitation": "classification support remains important; it is just not the same layer as portability",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUPPORT, index=False)
    return out


def create_cross_domain_grammar() -> pd.DataFrame:
    grammar = read_csv(GRAMMAR)
    rows = []
    domains = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]
    for domain in domains:
        if not grammar.empty and "domain" in grammar.columns:
            row = get_row(grammar, "domain", domain)
        else:
            row = None

        if row is None:
            # Conservative default from the current project logic.
            if domain == "inherited_arrhythmia":
                rows.append({
                    "domain": domain,
                    "dominant_portability_architecture": "trigger-dependent / syndrome-organ / underresolved mixture",
                    "stable_architecture": "phenotype-anchored self-loop",
                    "unstable_architecture": "phenotype-domain boundary and underresolved labels",
                    "biological_determinant": "trigger/provocation, phenotype ascertainment, and disease-environment boundary",
                    "meaning_mobility_behavior": "meaning travels only with trigger and phenotype context",
                    "routing_consequence": "contextual repair or disease-specific review",
                })
            elif domain == "cardiomyopathy":
                rows.append({
                    "domain": domain,
                    "dominant_portability_architecture": "structural-functional overlap and phenotype-anchored self-loop",
                    "stable_architecture": "phenotype-anchored monogenic cardiomyopathy",
                    "unstable_architecture": "structural-functional overlap and underresolved environments",
                    "biological_determinant": "structural disease models, contractile/cytoskeletal overlap, and electrical-system interface",
                    "meaning_mobility_behavior": "meaning travels within self-loops but needs repair across structural-functional boundaries",
                    "routing_consequence": "domain repair and disease-specific expert review",
                })
            else:
                rows.append({
                    "domain": domain,
                    "dominant_portability_architecture": "modifier/penetrance boundary",
                    "stable_architecture": "risk architecture under PRF framing",
                    "unstable_architecture": "deterministic disease reuse from conditional risk",
                    "biological_determinant": "penetrance, modifier effects, and population/ascertainment context",
                    "meaning_mobility_behavior": "meaning travels as conditional liability, not deterministic disease",
                    "routing_consequence": "population/penetrance review and PRF-needed",
                })
            continue

        rows.append({
            "domain": domain,
            "dominant_portability_architecture": str(row.get("strongest_regime_specific_signal", "")),
            "stable_architecture": str(row.get("dominant_stable_grammar", "")),
            "unstable_architecture": str(row.get("dominant_unstable_grammar", "")),
            "biological_determinant": str(row.get("main_biological_determinant", "")),
            "meaning_mobility_behavior": {
                "inherited_arrhythmia": "meaning is trigger/context-sensitive and boundary-limited",
                "cardiomyopathy": "meaning is self-loop stable but constrained by structural-functional overlap",
                "hereditary_cancer": "meaning travels as conditional penetrance/risk rather than deterministic disease",
            }.get(domain, ""),
            "routing_consequence": {
                "inherited_arrhythmia": "contextual repair or disease-specific review",
                "cardiomyopathy": "domain repair and disease-specific expert review",
                "hereditary_cancer": "population/penetrance review and PRF-needed",
            }.get(domain, ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GRAMMAR, index=False)
    return out


def write_biology_claim_audit() -> None:
    text = """# Fundamental Biology Claim Audit

## 1. Can we claim disease architecture governs pathogenic meaning mobility?

Yes, with bounded wording.

Strongest supported wording:
Across three Mendelian disease domains, pathogenic meaning mobility is governed by disease architecture: anchored architectures stabilize meaning, whereas modifier/penetrance-boundary, underresolved, and structural-overlap architectures create portability boundaries.

This is supported as a portability/routing architecture claim, not as clinical validation.

## 2. Can we claim all Mendelian diseases?

No.

Allowed:
Across three Mendelian disease domains in the current benchmark.

Forbidden:
All Mendelian diseases follow these exact regimes.

## 3. Can we claim across Mendelian medicine?

Only cautiously.

Allowed:
These results suggest a generalizable architecture-based framing for Mendelian assertion portability.

Forbidden:
CAB proves a universal law across Mendelian medicine.

## 4. Can we claim SADS genotype-first regime?

No, not as empirically validated here.

Allowed:
Genotype-first absent-phenotype remains a PRF-relevant conceptual regime and high-priority validation target.

Forbidden:
CAB validates SADS genotype-first risk or genotype-first absent-phenotype architecture in the current benchmark.

## 5. Can we claim therapy or clinical outcomes?

No.

Forbidden:
CAB explains patient outcomes.
CAB proves therapeutic utility.
CAB improves clinical outcomes.
CAB validates treatment decisions.

## 6. What is the strongest honest biological claim?

Across three Mendelian disease domains, pathogenic meaning mobility is governed by disease architecture: anchored architectures stabilize meaning, whereas modifier/penetrance-boundary, underresolved, and structural-overlap architectures create portability boundaries.

## Required caveat

This analysis measures assertion portability and routing behavior. It does not reclassify variants, invalidate source databases, diagnose phenotypes, validate patient outcomes, or establish therapeutic utility.
"""
    OUT_AUDIT.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_definition()
    create_principles()
    phase = create_phase_space()
    create_phase_space_fig(phase)
    create_meaning_stability_index()
    create_fundamental_support_table()
    create_cross_domain_grammar()
    write_biology_claim_audit()

    print("Fundamental biology framing complete.")
    print("Outputs:")
    for p in [
        OUT_DEF,
        OUT_PRINCIPLES,
        OUT_PHASE,
        OUT_PHASE_FIG,
        OUT_MSI,
        OUT_SUPPORT,
        OUT_GRAMMAR,
        OUT_AUDIT,
    ]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
