
#!/usr/bin/env python3
"""Update disease-architecture regime claims by empirical support level.

Creates:
- reports/tables/disease_architecture_regime_support_levels.csv
- reports/qc/abstract_regime_language_safe.md

Updates:
- reports/tables/disease_architecture_biological_claim_audit.csv
- reports/figures/final_disease_architecture_portability_regimes.svg

Goal:
Separate empirically supported regimes from underpowered conceptual CAB/PRF categories.
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

SIG = TABLES / "disease_architecture_regime_temporal_signatures.csv"
ENRICH = TABLES / "disease_architecture_regime_enrichment_tests.csv"
CLAIM_AUDIT = TABLES / "disease_architecture_biological_claim_audit.csv"
OUT_SUPPORT = TABLES / "disease_architecture_regime_support_levels.csv"
OUT_ABSTRACT = QC / "abstract_regime_language_safe.md"
OUT_FIG = FIGS / "final_disease_architecture_portability_regimes.svg"

SUPPORT_ORDER = [
    "modifier_penetrance_boundary",
    "nonspecific_underresolved",
    "structural_functional_overlap",
    "phenotype_anchored_monogenic",
    "syndrome_organ_boundary",
    "trigger_dependent_latent",
    "pleiotropic_collision",
    "genotype_first_absent_phenotype",
]

LABELS = {
    "modifier_penetrance_boundary": "Modifier/penetrance boundary",
    "nonspecific_underresolved": "Nonspecific/underresolved",
    "structural_functional_overlap": "Structural-functional overlap",
    "phenotype_anchored_monogenic": "Phenotype-anchored monogenic",
    "syndrome_organ_boundary": "Syndrome-organ boundary",
    "trigger_dependent_latent": "Trigger-dependent latent",
    "pleiotropic_collision": "Pleiotropic collision",
    "genotype_first_absent_phenotype": "Genotype-first absent phenotype",
}

KEY_ENRICHMENT_LOOKUP = {
    "modifier_penetrance_boundary": "modifier_penetrance_boundary enriched for population_penetrance_review",
    "nonspecific_underresolved": "nonspecific_underresolved enriched for contextual_repair/condition_label_drift",
    "structural_functional_overlap": "structural_functional_overlap enriched for disease_specific_review",
    "phenotype_anchored_monogenic": "phenotype_anchored_monogenic enriched for self_loop_stable",
    "syndrome_organ_boundary": "syndrome_anchored subset enriched for self_loop_stable",
    "trigger_dependent_latent": "trigger_dependent_latent enriched for contextual_repair",
    "pleiotropic_collision": "pleiotropic_collision enriched for cross_environment_drift",
    "genotype_first_absent_phenotype": "genotype_first_absent_phenotype enriched for PRF_needed/no_deterministic_reuse",
}


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False, dtype=str)
    return pd.DataFrame()


def to_float(x: Any) -> float:
    try:
        if x is None or str(x).strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def get_sig(sig: pd.DataFrame, regime: str, col: str, default: str = "") -> str:
    if sig.empty:
        return default
    row = sig[sig["regime"].eq(regime)]
    if row.empty:
        return default
    return str(row.iloc[0].get(col, default))


def get_enrichment(enrich: pd.DataFrame, regime: str) -> dict[str, Any]:
    if enrich.empty:
        return {"key_enrichment": "", "OR": "", "FDR": "", "CI": ""}
    hyp = KEY_ENRICHMENT_LOOKUP.get(regime, "")
    rows = enrich[enrich["hypothesis"].eq(hyp)]
    if rows.empty:
        rows = enrich[enrich["regime"].eq(regime)]
    if rows.empty:
        return {"key_enrichment": hyp, "OR": "", "FDR": "", "CI": ""}
    r = rows.iloc[0]
    ci = f"{r.get('CI95_low', '')}–{r.get('CI95_high', '')}"
    return {
        "key_enrichment": str(r.get("hypothesis", hyp)),
        "OR": str(r.get("OR", "")),
        "FDR": str(r.get("FDR_p_value", "")),
        "CI": ci,
    }


def assign_support(regime: str, n: int, fdr: float, orv: float) -> str:
    if regime == "genotype_first_absent_phenotype" or n == 0:
        return "absent_in_current_benchmark"
    if regime in {"trigger_dependent_latent", "pleiotropic_collision"}:
        return "underpowered_framework_category"
    if regime in {"modifier_penetrance_boundary", "nonspecific_underresolved", "structural_functional_overlap"}:
        return "strong_empirical_support"
    if regime in {"phenotype_anchored_monogenic", "syndrome_organ_boundary"}:
        if n >= 100 and (not pd.isna(fdr)) and fdr < 0.05 and (not pd.isna(orv)) and orv > 1:
            return "strong_empirical_support"
        return "moderate_empirical_support"
    return "underpowered_framework_category"


def allowed_claim(regime: str, support: str) -> str:
    if regime == "modifier_penetrance_boundary":
        return "Modifier/penetrance-boundary architecture is strongly supported as a portability regime requiring population/penetrance review and PRF-style conditional-risk framing."
    if regime == "nonspecific_underresolved":
        return "Nonspecific/underresolved architecture is strongly supported as a portability regime requiring contextual repair or disease-specific review."
    if regime == "structural_functional_overlap":
        return "Structural-functional overlap is strongly supported as a cardiac portability regime enriched for disease-specific review."
    if regime == "phenotype_anchored_monogenic":
        return "Phenotype-anchored monogenic assertions are empirically supported as a self-loop-stable portability regime."
    if regime == "syndrome_organ_boundary":
        return "Syndrome-organ boundary is supported as an identity-versus-meaning boundary class; its domain-specific enrichment should be stated cautiously."
    if regime == "trigger_dependent_latent":
        return "Trigger-dependent latent architecture is a biologically motivated CAB/PRF framework category and high-priority validation target, but is underpowered here."
    if regime == "pleiotropic_collision":
        return "Pleiotropic-collision architecture is a biologically motivated framework category but is underpowered here and should not be claimed as robustly driving boundary crossing."
    if regime == "genotype_first_absent_phenotype":
        return "Genotype-first absent-phenotype architecture is retained as a PRF-relevant conceptual regime, not empirically supported in the current mapping."
    return "Use cautious support-tiered wording."


def prohibited_claim(regime: str, support: str) -> str:
    if support == "strong_empirical_support":
        return "Do not claim clinical validation, therapeutic utility, or universality beyond the current benchmark."
    if support == "moderate_empirical_support":
        return "Do not claim this regime is fully validated across all domains or all disease models."
    if support == "underpowered_framework_category":
        return "Do not claim robust enrichment, validation, or boundary-driving behavior from current data."
    if support == "absent_in_current_benchmark":
        return "Do not claim empirical support in the current benchmark."
    return "Do not overclaim beyond support tier."


def future_target(regime: str, support: str) -> str:
    if regime == "trigger_dependent_latent":
        return "Add trigger/provocation-rich arrhythmia and latent-risk datasets or expert adjudication."
    if regime == "pleiotropic_collision":
        return "Increase cross-domain pleiotropic gene sampling and expert-reviewed boundary labels."
    if regime == "genotype_first_absent_phenotype":
        return "Use genotype-first cohorts with explicit absent/unknown phenotype fields and penetrance follow-up."
    if regime == "syndrome_organ_boundary":
        return "Validate syndrome-to-organ and organ-to-syndrome boundaries across curated disease-specific labels."
    if regime == "phenotype_anchored_monogenic":
        return "Replicate self-loop stability in additional disease-specific curation settings."
    if regime == "structural_functional_overlap":
        return "Validate structural/electrical overlap with expert cardiac domain review."
    if regime == "modifier_penetrance_boundary":
        return "Validate PRF framing against penetrance-aware curated cohorts."
    if regime == "nonspecific_underresolved":
        return "Validate contextual repair decisions against expert label-resolution review."
    return "External expert adjudication."


def create_support_table() -> pd.DataFrame:
    sig = read_csv(SIG)
    enrich = read_csv(ENRICH)
    rows = []
    for regime in SUPPORT_ORDER:
        n = int(float(get_sig(sig, regime, "N", "0") or 0))
        domains = get_sig(sig, regime, "domains_represented", "")
        e = get_enrichment(enrich, regime)
        orv = to_float(e["OR"])
        fdr = to_float(e["FDR"])
        support = assign_support(regime, n, fdr, orv)

        rows.append({
            "regime_name": regime,
            "regime_label": LABELS[regime],
            "N": n,
            "domains_represented": domains,
            "key_enrichment": e["key_enrichment"],
            "OR": e["OR"],
            "CI95": e["CI"],
            "FDR_p_value": e["FDR"],
            "support_level": support,
            "allowed_claim": allowed_claim(regime, support),
            "prohibited_claim": prohibited_claim(regime, support),
            "future_validation_target": future_target(regime, support),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUPPORT, index=False)
    return out


def update_claim_audit(support: pd.DataFrame) -> pd.DataFrame:
    audit = read_csv(CLAIM_AUDIT)
    if audit.empty:
        audit = pd.DataFrame(columns=[
            "claim_text",
            "supporting_table",
            "supporting_statistic",
            "claim_strength",
            "prohibited_stronger_wording",
        ])

    replacement_claim = (
        "Trigger-dependent, pleiotropic-collision, and genotype-first absent-phenotype regimes "
        "are biologically motivated CAB/PRF categories but were underpowered or absent in the current benchmark."
    )

    overstrong_patterns = [
        "pleiotropic, trigger-dependent, genotype-first regimes generated predictable boundary crossings",
        "Pleiotropic, trigger-dependent, genotype-first regimes generated predictable boundary crossings",
        "trigger-dependent, pleiotropic, genotype-first regimes generated predictable boundary crossings",
    ]

    if "claim_text" in audit.columns:
        mask = pd.Series([False] * len(audit))
        for pat in overstrong_patterns:
            mask = mask | audit["claim_text"].astype(str).str.contains(pat, regex=False, case=False, na=False)
        audit.loc[mask, "claim_text"] = replacement_claim
        audit.loc[mask, "claim_strength"] = "exploratory"
        audit.loc[mask, "supporting_table"] = str(OUT_SUPPORT.relative_to(ROOT))
        audit.loc[mask, "supporting_statistic"] = "trigger_dependent_latent N=45; pleiotropic_collision N=11; genotype_first_absent_phenotype N=0"
        audit.loc[mask, "prohibited_stronger_wording"] = "Do not claim these regimes were empirically validated or robustly enriched in the current benchmark."

    # Patch existing related rows to cautious support.
    if "claim_text" in audit.columns:
        collision_mask = audit["claim_text"].astype(str).str.contains("Collision architectures generate boundary crossings", regex=False, na=False)
        audit.loc[collision_mask, "claim_strength"] = "exploratory"
        audit.loc[collision_mask, "supporting_table"] = str(OUT_SUPPORT.relative_to(ROOT))
        audit.loc[collision_mask, "supporting_statistic"] = "pleiotropic_collision N=11; underpowered framework category"
        audit.loc[collision_mask, "prohibited_stronger_wording"] = "Do not claim robust boundary-crossing enrichment for pleiotropic collision unless N/CI/FDR support exists."

    # Add explicit support-tier audit row if absent.
    if not audit["claim_text"].astype(str).str.contains("underpowered or absent", case=False, na=False).any():
        audit = pd.concat([audit, pd.DataFrame([{
            "claim_text": replacement_claim,
            "supporting_table": str(OUT_SUPPORT.relative_to(ROOT)),
            "supporting_statistic": "trigger_dependent_latent N=45; pleiotropic_collision N=11; genotype_first_absent_phenotype N=0",
            "claim_strength": "exploratory",
            "prohibited_stronger_wording": "Do not state that all proposed regimes were validated.",
        }])], ignore_index=True)

    audit.to_csv(CLAIM_AUDIT, index=False)
    return audit


def write_abstract_safe() -> None:
    text = """# Abstract-Safe Disease-Architecture Regime Language

## Allowed wording

Empirically supported portability regimes were dominated by modifier/penetrance-boundary, nonspecific/underresolved, and structural-functional-overlap architectures. Phenotype-anchored and syndrome-anchored assertions showed self-loop stability. Trigger-dependent and genotype-first regimes remain high-priority validation targets.

## More detailed allowed wording

Disease-architecture regime analysis separated empirically supported portability regimes from underpowered framework categories. Modifier/penetrance-boundary, nonspecific/underresolved, structural-functional-overlap, phenotype-anchored, and syndrome-anchored patterns were supported in the current three-domain benchmark. Trigger-dependent latent, pleiotropic-collision, and genotype-first absent-phenotype regimes should be described as biologically motivated CAB/PRF categories requiring additional validation rather than as validated findings.

## Forbidden wording

- All proposed regimes were validated.
- Genotype-first absent-phenotype architecture was empirically supported.
- Pleiotropic collision robustly drove boundary crossing, unless N/CI/FDR support exists.
- Trigger-dependent latent architecture was validated in the current benchmark.
- CAB demonstrates clinical outcome or therapeutic utility from these regimes.

## Required caveat

This is a portability/routing architecture analysis, not clinical validation, therapeutic validation, or variant reclassification.
"""
    OUT_ABSTRACT.write_text(text, encoding="utf-8")


def esc(x: Any) -> str:
    return html.escape(str(x))


def support_color(level: str) -> str:
    return {
        "strong_empirical_support": "#d9d9d9",
        "moderate_empirical_support": "#eeeeee",
        "underpowered_framework_category": "#ffffff",
        "absent_in_current_benchmark": "#f8f8f8",
    }.get(level, "#ffffff")


def update_figure_with_support(support: pd.DataFrame) -> None:
    # Append a support legend and regime tier panel to existing SVG if possible.
    if OUT_FIG.exists():
        old = OUT_FIG.read_text(encoding="utf-8")
        if "Support-level overlay" in old:
            return
        insert_at = old.rfind("</svg>")
        if insert_at == -1:
            old = ""
    else:
        old = ""

    panel = []
    panel.append('\n<!-- Support-level overlay -->\n')
    panel.append('<text x="40" y="1040" font-family="Arial" font-size="20" font-weight="700">Support-level overlay</text>\n')
    panel.append('<text x="40" y="1064" font-family="Arial" font-size="13">Regimes are tiered by empirical support in the current benchmark, not treated as equally validated.</text>\n')

    x0, y0 = 40, 1090
    for i, row in support.iterrows():
        level = str(row["support_level"])
        fill = support_color(level)
        x = x0 + (i % 2) * 780
        y = y0 + (i // 2) * 38
        panel.append(f'<rect x="{x}" y="{y-20}" width="740" height="30" rx="8" fill="{fill}" stroke="#111"/>\n')
        panel.append(f'<text x="{x+12}" y="{y}" font-family="Arial" font-size="12" font-weight="700">{esc(row["regime_label"])}</text>\n')
        panel.append(f'<text x="{x+275}" y="{y}" font-family="Arial" font-size="12">{esc(level)}</text>\n')
        panel.append(f'<text x="{x+535}" y="{y}" font-family="Arial" font-size="12">N={esc(row["N"])}</text>\n')

    # If the old figure has height 1200, bump it to 1280 so overlay fits.
    if old:
        old2 = old.replace('height="1200"', 'height="1280"', 1).replace('viewBox="0 0 1700 1200"', 'viewBox="0 0 1700 1280"', 1)
        new = old2[:insert_at] + "".join(panel) + old2[insert_at:]
    else:
        new = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="1280" viewBox="0 0 1700 1280">\n'
            '<rect width="100%" height="100%" fill="#fff"/>\n'
            '<text x="40" y="40" font-family="Arial" font-size="26" font-weight="700">Disease-architecture support levels</text>\n'
            + "".join(panel) + '</svg>\n'
        )
    OUT_FIG.write_text(new, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    support = create_support_table()
    update_claim_audit(support)
    write_abstract_safe()
    update_figure_with_support(support)

    print("Disease-architecture support-level update complete.")
    print(support[["regime_name", "N", "key_enrichment", "OR", "FDR_p_value", "support_level"]].to_string(index=False))
    print()
    print("Outputs:")
    for p in [OUT_SUPPORT, CLAIM_AUDIT, OUT_ABSTRACT, OUT_FIG]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
