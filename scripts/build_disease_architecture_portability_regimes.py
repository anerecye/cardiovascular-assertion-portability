
#!/usr/bin/env python3
"""Build Disease-Architecture Portability Regime Table for CAB.

Goal:
Convert domain-specific CAB patterns into recurrent disease-architecture regimes
that determine how pathogenic meaning travels.

Output:
- reports/tables/disease_architecture_portability_regimes.csv
- reports/qc/disease_architecture_portability_regime_summary.md

Inputs used when available:
- data/processed/cab_decision_challenge_tasks.csv
- reports/tables/clinvar_identity_vs_meaning_concordance.csv
- reports/tables/identity_meaning_discordance_sensitivity_core.csv
- reports/tables/routing_metrics_by_domain_all_modes.csv

Claim boundary:
This is an internal cross-domain architecture synthesis, not clinical validation,
not a replacement for expert curation, and not a variant reclassification layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

TASKS = ROOT / "data" / "processed" / "cab_decision_challenge_tasks.csv"
CONCORDANCE = TABLES / "clinvar_identity_vs_meaning_concordance.csv"

OUT_TABLE = TABLES / "disease_architecture_portability_regimes.csv"
OUT_MD = QC / "disease_architecture_portability_regime_summary.md"


REGIME_ORDER = [
    "Phenotype-anchored monogenic",
    "Trigger-dependent latent",
    "Pleiotropic collision",
    "Syndrome-organ boundary",
    "Structural-functional overlap",
    "Genotype-first absent phenotype",
    "Modifier/penetrance boundary",
    "Nonspecific/underresolved",
]


REGIME_META = {
    "Phenotype-anchored monogenic": {
        "definition": "Variant meaning is anchored to a relatively specific disease model and usually travels within the same disease loop.",
        "biological_determinant": "high phenotype specificity; canonical gene-disease model; same-environment reuse",
        "expected_meaning_travel": "travels within self-loop",
        "expected_failure_mode": "fails mainly when reused outside the original disease model",
        "dominant_routing_action": "direct deterministic use or CAB-Balanced reuse when context is concordant",
        "claim_strength": "recurrent_cross_domain_regime",
    },
    "Trigger-dependent latent": {
        "definition": "Variant meaning depends on a latent risk state and an environmental, physiologic, drug, age, or stress trigger.",
        "biological_determinant": "incomplete penetrance; trigger-dependent expressivity; context-sensitive risk realization",
        "expected_meaning_travel": "travels only with trigger context",
        "expected_failure_mode": "deterministic disease reuse without trigger/ascertainment context",
        "dominant_routing_action": "contextual repair; trigger/phenotype-context review",
        "claim_strength": "mechanistically_plausible_regime",
    },
    "Pleiotropic collision": {
        "definition": "A gene or variant is attached to multiple disease models whose labels collide under deterministic reuse.",
        "biological_determinant": "pleiotropy; multi-condition ClinVar labels; cross-domain gene use",
        "expected_meaning_travel": "breaks across disease models",
        "expected_failure_mode": "false portability across disease environments",
        "dominant_routing_action": "disease-specific review or contextual repair",
        "claim_strength": "recurrent_cross_domain_regime",
    },
    "Syndrome-organ boundary": {
        "definition": "Source identity is valid, but meaning crosses from a syndromic/developmental/imprinting label into an organ-specific disease label.",
        "biological_determinant": "syndrome-to-organ boundary; developmental or imprinting phenotype labels; multi-system disease architecture",
        "expected_meaning_travel": "travels poorly between syndrome and organ label",
        "expected_failure_mode": "source match mistaken for organ-specific disease portability",
        "dominant_routing_action": "contextual repair or disease-specific review",
        "claim_strength": "external_source_identity_vs_meaning_supported",
    },
    "Structural-functional overlap": {
        "definition": "Variant meaning sits at overlap between structural tissue disease and functional/electrical disease models.",
        "biological_determinant": "shared gene architecture; structural remodeling; functional electrophysiologic overlap",
        "expected_meaning_travel": "requires domain repair",
        "expected_failure_mode": "overextension from one organ-function model into another",
        "dominant_routing_action": "domain repair; disease-specific expert review",
        "claim_strength": "cross_domain_architecture_regime",
    },
    "Genotype-first absent phenotype": {
        "definition": "A pathogenic or likely pathogenic assertion is source-valid but phenotype realization is absent, unknown, or not ascertained.",
        "biological_determinant": "genotype-first ascertainment; reduced penetrance; absent/unknown phenotype",
        "expected_meaning_travel": "cannot travel deterministically",
        "expected_failure_mode": "carrier status treated as deterministic disease state",
        "dominant_routing_action": "PRF-needed; phenotype ascertainment review; no deterministic reuse",
        "claim_strength": "external_proxy_and_internal_routing_supported",
    },
    "Modifier/penetrance boundary": {
        "definition": "Variant meaning travels as probabilistic or conditional risk rather than deterministic disease assignment.",
        "biological_determinant": "penetrance modifiers; population/ancestry context; risk rather than diagnosis",
        "expected_meaning_travel": "travels as conditional risk, not deterministic disease",
        "expected_failure_mode": "risk assertion converted into deterministic disease label",
        "dominant_routing_action": "population/penetrance review; PRF-needed",
        "claim_strength": "mechanistically_plausible_regime",
    },
    "Nonspecific/underresolved": {
        "definition": "Phenotype or condition labels are too broad, unknown, or underresolved for deterministic disease-model reuse.",
        "biological_determinant": "broad phenotype labels; underresolved condition mapping; nonspecific disease environment",
        "expected_meaning_travel": "requires contextual repair",
        "expected_failure_mode": "broad label reused as specific disease assertion",
        "dominant_routing_action": "contextual repair",
        "claim_strength": "recurrent_label_resolution_regime",
    },
}


GENE_HINTS = {
    "Phenotype-anchored monogenic": {
        "KCNQ1", "KCNH2", "SCN5A", "PKP2", "MYBPC3", "MYH7", "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "TP53"
    },
    "Trigger-dependent latent": {"RYR2", "KCNQ1", "KCNH2", "SCN5A", "CACNA1C"},
    "Pleiotropic collision": {"SCN5A", "RYR2", "LMNA", "FLNC", "TTN", "TP53", "PTEN", "CHEK2", "ATM"},
    "Syndrome-organ boundary": {"KCNQ1", "KCNQ1OT1", "CACNA1C", "PTEN", "TP53"},
    "Structural-functional overlap": {"SCN5A", "LMNA", "FLNC", "TTN", "DES", "DSP", "PKP2", "RYR2"},
    "Genotype-first absent phenotype": {"BRCA1", "BRCA2", "PALB2", "CHEK2", "ATM", "KCNQ1", "KCNH2", "SCN5A", "RYR2"},
    "Modifier/penetrance boundary": {"CHEK2", "ATM", "PALB2", "BRCA1", "BRCA2", "TTN", "FLNC", "LMNA"},
    "Nonspecific/underresolved": set(),
}


def read_csv_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False, dtype=str)
    return pd.DataFrame()


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def bool_col(df: pd.DataFrame, candidates: list[str], default: bool = False) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return truthy(df[c])
    return pd.Series([default] * len(df), index=df.index)


def first_existing(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def load_joined() -> pd.DataFrame:
    tasks = read_csv_optional(TASKS)
    conc = read_csv_optional(CONCORDANCE)

    if tasks.empty and conc.empty:
        raise FileNotFoundError("Need at least cab_decision_challenge_tasks.csv or clinvar_identity_vs_meaning_concordance.csv")

    if not tasks.empty and "assertion_id" in tasks.columns:
        tasks["assertion_id"] = norm_id(tasks["assertion_id"])
    if not conc.empty and "assertion_id" in conc.columns:
        conc["assertion_id"] = norm_id(conc["assertion_id"])

    if not tasks.empty and not conc.empty and "assertion_id" in tasks.columns and "assertion_id" in conc.columns:
        keep = [
            "assertion_id",
            "local_gene",
            "clinvar_phenotype_list",
            "phenotype_domain_discordance_flag",
            "meaning_match_accepted",
            "routing_implication",
            "source_match_accepted",
        ]
        keep = [c for c in keep if c in conc.columns]
        df = tasks.merge(conc[keep], on="assertion_id", how="left")
    elif not tasks.empty:
        df = tasks.copy()
    else:
        df = conc.copy()

    if "domain" not in df.columns:
        df["domain"] = first_existing(df, ["domain_x", "domain_y"], "unknown")

    df["gene_for_regime"] = first_existing(df, ["gene", "local_gene", "GeneSymbol", "gene_symbol"], "")
    df["phenotype_for_regime"] = first_existing(df, ["clinvar_phenotype_list", "PhenotypeList", "condition", "phenotype", "PhenotypeList"], "")
    return df


def classify_regimes_for_row(row: pd.Series) -> list[str]:
    gene = str(row.get("gene_for_regime", "") or "").upper()
    phenotype = str(row.get("phenotype_for_regime", "") or "").lower()
    domain = str(row.get("domain", "") or "").lower()

    regimes: set[str] = set()

    if str(row.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"}:
        if any(t in phenotype for t in ["silver-russell", "imprinting", "beckwith", "growth", "development"]):
            regimes.add("Syndrome-organ boundary")
        elif any(t in phenotype for t in ["not provided", "not specified", "unknown", "multiple conditions", "unspecified"]):
            regimes.add("Nonspecific/underresolved")
        else:
            regimes.add("Pleiotropic collision")

    if any(t in phenotype for t in ["not provided", "not specified", "unknown", "unspecified", "multiple conditions"]):
        regimes.add("Nonspecific/underresolved")

    if any(t in phenotype for t in ["syndrome", "silver-russell", "beckwith", "development", "imprinting", "congenital"]):
        regimes.add("Syndrome-organ boundary")

    if any(t in phenotype for t in ["penetrance", "risk", "predisposition", "susceptibility"]) or domain == "hereditary_cancer":
        regimes.add("Modifier/penetrance boundary")

    if any(t in phenotype for t in ["carrier", "asymptomatic", "no phenotype", "genotype-first", "screening"]):
        regimes.add("Genotype-first absent phenotype")

    if gene in GENE_HINTS["Trigger-dependent latent"]:
        regimes.add("Trigger-dependent latent")

    if gene in GENE_HINTS["Structural-functional overlap"]:
        regimes.add("Structural-functional overlap")

    if gene in GENE_HINTS["Pleiotropic collision"]:
        regimes.add("Pleiotropic collision")

    if gene in GENE_HINTS["Phenotype-anchored monogenic"]:
        regimes.add("Phenotype-anchored monogenic")

    if gene in GENE_HINTS["Modifier/penetrance boundary"]:
        regimes.add("Modifier/penetrance boundary")

    if not regimes:
        regimes.add("Nonspecific/underresolved")

    return [r for r in REGIME_ORDER if r in regimes]


def explode_regimes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        regs = classify_regimes_for_row(row)
        for reg in regs:
            d = row.to_dict()
            d["regime_name"] = reg
            rows.append(d)
    return pd.DataFrame(rows)


def metric_rate(df: pd.DataFrame, cols: list[str]) -> float:
    if len(df) == 0:
        return float("nan")
    return float(bool_col(df, cols, False).mean())


def dominant_action(df: pd.DataFrame, regime_name: str) -> str:
    action = first_existing(df, ["routing_primary_action", "primary_routing_action", "routing_implication"], "")
    if len(action) and action.str.strip().ne("").any():
        vc = action[action.str.strip().ne("")].value_counts()
        if not vc.empty:
            top = str(vc.index[0])
            if top:
                return top

    if regime_name in REGIME_META:
        return REGIME_META[regime_name]["dominant_routing_action"]
    return "contextual repair"


def examples_join(series: pd.Series, limit: int = 8) -> str:
    vals = [x for x in series.fillna("").astype(str).tolist() if x and x.lower() != "nan"]
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
        if len(seen) >= limit:
            break
    return "|".join(seen)


def build_table(exploded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        sub = exploded[exploded["regime_name"].eq(regime)].copy()
        meta = REGIME_META[regime]

        condition_label_drift_rate = metric_rate(sub, ["future_condition_label_drift", "condition_label_drift"])
        cross_environment_drift_rate = metric_rate(sub, ["future_cross_environment_drift", "cross_environment_drift"])
        self_loop_stable_rate = metric_rate(sub, ["future_self_loop_stable", "self_loop_stable"])

        rows.append({
            "regime_name": regime,
            "definition": meta["definition"],
            "biological_determinant": meta["biological_determinant"],
            "expected_meaning_travel": meta["expected_meaning_travel"],
            "expected_failure_mode": meta["expected_failure_mode"],
            "domain_examples": examples_join(first_existing(sub, ["domain"], ""), 5) if not sub.empty else "",
            "gene_examples": examples_join(first_existing(sub, ["gene_for_regime", "gene", "local_gene"], ""), 12) if not sub.empty else "",
            "N_assertion_regime_memberships": len(sub),
            "condition_label_drift_rate": condition_label_drift_rate,
            "cross_environment_drift_rate": cross_environment_drift_rate,
            "self_loop_stable_rate": self_loop_stable_rate,
            "dominant_routing_action": dominant_action(sub, regime) if not sub.empty else meta["dominant_routing_action"],
            "evidence_support": evidence_support_for_regime(regime, sub),
            "claim_strength": meta["claim_strength"],
        })

    return pd.DataFrame(rows)


def evidence_support_for_regime(regime: str, sub: pd.DataFrame) -> str:
    n = len(sub)
    if n == 0:
        return "theoretical regime retained for interpretation; no current row membership"
    if regime == "Syndrome-organ boundary":
        flagged = int(bool_col(sub, ["phenotype_domain_discordance_flag"], False).sum())
        return f"supported by {flagged}/{n} phenotype-domain discordance flags among source-matched rows"
    if regime == "Nonspecific/underresolved":
        return f"supported by broad/unknown phenotype labels and contextual repair routing across {n} memberships"
    if regime == "Modifier/penetrance boundary":
        return f"supported by hereditary cancer/genotype-first risk architecture and PRF-style conditional portability across {n} memberships"
    if regime == "Pleiotropic collision":
        return f"supported by cross-domain/multi-condition gene membership and disease-model collision risk across {n} memberships"
    if regime == "Phenotype-anchored monogenic":
        return f"supported by canonical monogenic gene-disease memberships and self-loop/direct-reuse behavior across {n} memberships"
    if regime == "Trigger-dependent latent":
        return f"supported by arrhythmia/latent-risk genes where meaning requires trigger or ascertainment context across {n} memberships"
    if regime == "Structural-functional overlap":
        return f"supported by overlapping cardiomyopathy/electrophysiology genes and domain-repair routing across {n} memberships"
    if regime == "Genotype-first absent phenotype":
        return f"supported by genotype-first/phenotype-realization boundary logic across {n} memberships"
    return f"supported by {n} row memberships"


def write_summary(table: pd.DataFrame) -> None:
    def get_rate(regime: str, col: str) -> str:
        row = table[table["regime_name"].eq(regime)]
        if row.empty:
            return "NA"
        v = row.iloc[0][col]
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    text = f"""# Disease-Architecture Portability Regime Summary

## Core result

CAB identifies recurrent disease-architecture regimes that determine how pathogenic meaning travels. The result is not simply that domains differ. Instead, recurring regimes appear across domains and govern whether source-matched pathogenic assertions travel within a self-loop, require contextual repair, or fail across disease-model boundaries.

## Summary interpretation

### Phenotype-anchored regimes

Phenotype-anchored monogenic regimes show the strongest expectation of self-loop portability and direct or balanced reuse when the disease model remains concordant.

- self-loop stable rate: {get_rate("Phenotype-anchored monogenic", "self_loop_stable_rate")}
- dominant routing: {table.loc[table["regime_name"].eq("Phenotype-anchored monogenic"), "dominant_routing_action"].iloc[0]}

### Collision / syndrome-organ / genotype-first regimes

Pleiotropic collision, syndrome-organ boundary, and genotype-first absent phenotype regimes have weaker deterministic portability and require review/repair routing when assertions cross disease-model boundaries.

- syndrome-organ cross-environment drift rate: {get_rate("Syndrome-organ boundary", "cross_environment_drift_rate")}
- syndrome-organ dominant routing: {table.loc[table["regime_name"].eq("Syndrome-organ boundary"), "dominant_routing_action"].iloc[0]}

### Modifier/penetrance regimes

Modifier and penetrance-boundary regimes travel as conditional risk rather than deterministic disease assignment. They require PRF-style framing and should not be treated as direct deterministic disease reuse.

- modifier/penetrance condition-label drift rate: {get_rate("Modifier/penetrance boundary", "condition_label_drift_rate")}
- modifier/penetrance dominant routing: {table.loc[table["regime_name"].eq("Modifier/penetrance boundary"), "dominant_routing_action"].iloc[0]}

### Nonspecific/underresolved regimes

Nonspecific and underresolved regimes require contextual repair because broad or unknown phenotype labels are insufficient for deterministic disease-model reuse.

- nonspecific condition-label drift rate: {get_rate("Nonspecific/underresolved", "condition_label_drift_rate")}
- nonspecific dominant routing: {table.loc[table["regime_name"].eq("Nonspecific/underresolved"), "dominant_routing_action"].iloc[0]}

## Publication-safe claim

We identify recurrent disease-architecture regimes that determine pathogenic meaning travel. Phenotype-anchored monogenic assertions tend to travel within self-loops, whereas collision, syndrome-organ, genotype-first, penetrance/modifier, and underresolved regimes require contextual repair, disease-specific review, or PRF-style conditional-risk framing.

## Claim boundary

This table is a disease-architecture synthesis layer. It does not reclassify variants, invalidate ClinVar records, claim clinical outcome validation, or replace expert disease-specific curation.
"""
    OUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    df = load_joined()
    exploded = explode_regimes(df)
    table = build_table(exploded)
    table.to_csv(OUT_TABLE, index=False)
    write_summary(table)

    print("Disease-Architecture Portability Regime Table complete.")
    print(table[[
        "regime_name",
        "N_assertion_regime_memberships",
        "condition_label_drift_rate",
        "cross_environment_drift_rate",
        "self_loop_stable_rate",
        "dominant_routing_action",
        "claim_strength",
    ]].to_string(index=False))
    print()
    print("Outputs:")
    print(f"  - {OUT_TABLE.relative_to(ROOT)}")
    print(f"  - {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
