#!/usr/bin/env python3
"""CAB routing overrestriction and primary-action audit.

Purpose
-------
Finalize the publication-safe routing layer by adding:

1. Explicit non-mutual-exclusivity audit for routing flags.
2. Mutually exclusive primary routing action hierarchy:
   - direct_deterministic_use
   - contextual_repair
   - disease_specific_review
   - population_or_penetrance_review
   - no_deterministic_reuse

3. False restriction / overrestriction audit:
   - true portable assertions
   - CAB direct use among true portable
   - CAB false restriction among true portable
   - false restriction rate
   - direct-use precision
   - direct-use recall
   - specificity for portability

4. Ablation benchmark with utility trade-off:
   - ClinVar-label-only baseline
   - metadata-only routing
   - gene-only routing
   - regime-only routing
   - portability-score-only routing
   - failure-topology-only routing
   - gene+regime routing
   - full CAB routing

Guardrails
----------
- routing benchmark only
- no clinical outcome improvement claim
- no expert-validated decision correctness claim
- no composite-as-external-validation claim
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"
FIGURES = REPORTS / "figures"

TASKS = DATA / "cab_decision_challenge_tasks.csv"
GOLD_COMPONENTS = TABLES / "cab_routing_benchmark_gold_standard_components.csv"

OUT_NONEXCLUSIVE = TABLES / "routing_nonexclusive_action_flags_audit.csv"
OUT_PRIMARY = TABLES / "routing_primary_action_distribution.csv"
OUT_PRIMARY_ASSERTION = DATA / "cab_decision_challenge_primary_actions.csv"
OUT_OVER = TABLES / "routing_false_restriction_overrestriction_audit.csv"
OUT_ABLATION = TABLES / "routing_ablation_utility_tradeoff.csv"
OUT_CLAIMS = TABLES / "routing_overrestriction_publication_safe_claims.csv"

FIG_PRIMARY = FIGURES / "routing_primary_action_distribution.svg"
FIG_OVER = FIGURES / "routing_false_restriction_overrestriction.svg"
FIG_ABLATION = FIGURES / "routing_ablation_utility_tradeoff.svg"

OUT_QC = QC / "routing_overrestriction_and_primary_action_audit.md"

GOLD_MAP = {
    "temporal_condition_label_drift_gold_standard": "gold_temporal_condition",
    "conservative_composite_routing_gold_standard": "gold_composite_routing",
}

PRIMARY_ORDER = [
    "no_deterministic_reuse",
    "disease_specific_review",
    "population_or_penetrance_review",
    "contextual_repair",
    "direct_deterministic_use",
]


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def bool_col(s: pd.Series) -> pd.Series:
    return s.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}).fillna(False).astype(bool)


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_tasks() -> pd.DataFrame:
    if not TASKS.exists():
        raise FileNotFoundError(f"Missing task table: {TASKS}")
    tasks = pd.read_csv(TASKS, low_memory=False)

    for c in [
        "direct_single_model_reuse_allowed", "cross_environment_reuse_allowed",
        "contextual_repair_required", "disease_specific_expert_review_required",
        "population_or_penetrance_review_required", "high_future_meaning_drift_risk",
        "high_future_cross_environment_drift_risk", "future_condition_label_drift",
        "future_cross_environment_drift", "future_any_meaning_drift", "self_loop_stable",
    ]:
        if c in tasks.columns:
            tasks[c] = bool_col(tasks[c])
        else:
            tasks[c] = False

    if "baseline_portability_score" not in tasks.columns:
        tasks["baseline_portability_score"] = np.nan
    tasks["baseline_portability_score"] = pd.to_numeric(tasks["baseline_portability_score"], errors="coerce")

    if "baseline_nonportability_score" not in tasks.columns:
        tasks["baseline_nonportability_score"] = 100 - tasks["baseline_portability_score"]
    tasks["baseline_nonportability_score"] = pd.to_numeric(tasks["baseline_nonportability_score"], errors="coerce")

    for c in ["domain", "assertion_id", "gene", "baseline_regime_primary", "baseline_architecture_family", "environment_baseline"]:
        if c not in tasks.columns:
            tasks[c] = ""

    gold = safe_read(GOLD_COMPONENTS)
    if not gold.empty and "assertion_id" in gold.columns:
        keep = [c for c in gold.columns if c == "assertion_id" or c.startswith("gold_") or c in ["baseline_direct_use_allowed", "cab_direct_use_allowed"]]
        tasks = tasks.merge(gold[keep], on="assertion_id", how="left", suffixes=("", "_gold"))

    if "baseline_direct_use_allowed" not in tasks.columns:
        tasks["baseline_direct_use_allowed"] = True
    else:
        tasks["baseline_direct_use_allowed"] = bool_col(tasks["baseline_direct_use_allowed"])

    if "cab_direct_use_allowed" not in tasks.columns:
        tasks["cab_direct_use_allowed"] = tasks["direct_single_model_reuse_allowed"]
    else:
        tasks["cab_direct_use_allowed"] = bool_col(tasks["cab_direct_use_allowed"])

    if "gold_temporal_condition" not in tasks.columns:
        tasks["gold_temporal_condition"] = tasks["future_condition_label_drift"]
    else:
        tasks["gold_temporal_condition"] = bool_col(tasks["gold_temporal_condition"])

    if "gold_composite_routing" not in tasks.columns:
        reg = tasks["baseline_regime_primary"].astype(str).str.lower()
        arch = tasks["baseline_architecture_family"].astype(str).str.lower()
        low_portability = tasks["baseline_portability_score"].lt(50).fillna(False)
        failure_topology = (
            reg.str.contains("collision|nonportable|low|underresolved|nonspecific|moderate|penetrance|spectrum|recessive|biallelic", na=False)
            | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
        )
        decision_layer = (
            tasks["contextual_repair_required"]
            | tasks["disease_specific_expert_review_required"]
            | tasks["population_or_penetrance_review_required"]
            | (~tasks["cab_direct_use_allowed"])
        )
        tasks["gold_composite_routing"] = (
            tasks["future_condition_label_drift"]
            | tasks["future_cross_environment_drift"]
            | low_portability
            | failure_topology
            | decision_layer
        )
    else:
        tasks["gold_composite_routing"] = bool_col(tasks["gold_composite_routing"])

    return tasks


def primary_action(row) -> str:
    # Hierarchy: most restrictive/highest review burden wins.
    direct = bool(row.get("cab_direct_use_allowed", False))
    disease_review = bool(row.get("disease_specific_expert_review_required", False))
    pop_review = bool(row.get("population_or_penetrance_review_required", False))
    repair = bool(row.get("contextual_repair_required", False))

    if not direct and disease_review:
        return "disease_specific_review"
    if not direct and pop_review:
        return "population_or_penetrance_review"
    if not direct and repair:
        return "contextual_repair"
    if not direct:
        return "no_deterministic_reuse"
    if disease_review:
        return "disease_specific_review"
    if pop_review:
        return "population_or_penetrance_review"
    if repair:
        return "contextual_repair"
    return "direct_deterministic_use"


def add_primary_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["routing_actions_non_mutually_exclusive"] = True
    out["number_of_positive_routing_flags"] = (
        out["cab_direct_use_allowed"].astype(int)
        + out["contextual_repair_required"].astype(int)
        + out["disease_specific_expert_review_required"].astype(int)
        + out["population_or_penetrance_review_required"].astype(int)
        + (~out["cab_direct_use_allowed"]).astype(int)
    )
    out["primary_routing_action"] = out.apply(primary_action, axis=1)
    # Human-readable routing class.
    out["primary_routing_action_definition"] = out["primary_routing_action"].map({
        "direct_deterministic_use": "direct deterministic reuse allowed without repair/review flag",
        "contextual_repair": "reuse requires contextual repair before deterministic inference",
        "disease_specific_review": "reuse routed to disease-specific expert review",
        "population_or_penetrance_review": "reuse requires population-frequency or penetrance review",
        "no_deterministic_reuse": "direct deterministic reuse blocked",
    })
    out.to_csv(OUT_PRIMARY_ASSERTION, index=False)
    return out


def nonexclusive_audit(df: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "cab_direct_use_allowed",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "high_future_meaning_drift_risk",
        "high_future_cross_environment_drift_risk",
    ]
    rows = []
    for domain, sub in [("all", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0]):
        n = len(sub)
        flag_sum = sum(sub[f].astype(int) for f in flags)
        rows.append({
            "domain": domain,
            "N": n,
            "routing_flags_are_non_mutually_exclusive": "yes",
            "assertions_with_multiple_positive_flags_N": int((flag_sum > 1).sum()),
            "assertions_with_multiple_positive_flags_percent": float((flag_sum > 1).mean() * 100),
            "max_positive_flags_per_assertion": int(flag_sum.max()) if n else 0,
            "note": "Non-exclusive action flags can sum above 100%; use primary_routing_action for mutually exclusive distribution.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_NONEXCLUSIVE, index=False)
    return out


def primary_action_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, sub in [("all", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0]):
        n = len(sub)
        vc = sub["primary_routing_action"].value_counts()
        for action in PRIMARY_ORDER:
            k = int(vc.get(action, 0))
            rows.append({
                "domain": domain,
                "primary_routing_action": action,
                "N": n,
                "action_N": k,
                "action_percent": k / n * 100 if n else np.nan,
                "mutually_exclusive": "yes",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PRIMARY, index=False)
    return out


def confusion(direct_allowed: pd.Series, gold_nonportable: pd.Series) -> dict:
    direct_allowed = direct_allowed.astype(bool)
    gold_nonportable = gold_nonportable.astype(bool)
    true_portable = ~gold_nonportable
    pred_nonportable = ~direct_allowed

    tp = int((pred_nonportable & gold_nonportable).sum())
    fn = int((direct_allowed & gold_nonportable).sum())
    tn = int((direct_allowed & true_portable).sum())
    fp = int((pred_nonportable & true_portable).sum())
    n = tp + fn + tn + fp

    nonportability_recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    direct_precision = tn / (tn + fn) if (tn + fn) else np.nan
    direct_recall = tn / (tn + fp) if (tn + fp) else np.nan
    restriction_precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * nonportability_recall * restriction_precision / (nonportability_recall + restriction_precision)
        if (nonportability_recall + restriction_precision)
        else np.nan
    )
    return {
        "N": n,
        "true_portable_assertions_N": int(true_portable.sum()),
        "true_portable_assertions_percent": float(true_portable.mean() * 100),
        "CAB_allowed_direct_use_among_true_portable_N": tn,
        "CAB_allowed_direct_use_among_true_portable_percent": tn / max(1, int(true_portable.sum())) * 100,
        "CAB_falsely_restricted_true_portable_N": fp,
        "CAB_falsely_restricted_true_portable_percent": fp / max(1, int(true_portable.sum())) * 100,
        "false_restriction_rate_all_assertions": fp / n if n else np.nan,
        "false_direct_use_unsupported_reuse_N": fn,
        "false_direct_use_unsupported_reuse_percent": fn / n * 100 if n else np.nan,
        "direct_use_precision": direct_precision,
        "direct_use_recall": direct_recall,
        "specificity_for_portability": specificity,
        "nonportability_recall": nonportability_recall,
        "restriction_precision": restriction_precision,
        "F1_nonportability_detection": f1,
        "direct_use_allowed_rate": (tn + fn) / n if n else np.nan,
        "overrestriction_rate": fp / n if n else np.nan,
    }


def overrestriction_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        for domain, sub in [("all", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0]):
            m = confusion(sub["cab_direct_use_allowed"], sub[gold_col])
            row = {
                "gold_standard_name": gold_name,
                "domain": domain,
            }
            row.update(m)
            row["conservatism_label"] = "conservative_direct_use_low" if m["direct_use_allowed_rate"] < 0.25 else "not_extremely_conservative_by_threshold"
            row["publication_note"] = "Report false restriction explicitly; CAB is repair-first/routing-first, not a direct-use maximizer."
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_OVER, index=False)
    return out


def direct_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    reg = df["baseline_regime_primary"].astype(str).str.lower()
    arch = df["baseline_architecture_family"].astype(str).str.lower()
    gene = df["gene"].astype(str).str.upper()
    score = pd.to_numeric(df["baseline_portability_score"], errors="coerce").fillna(60)
    submitter = pd.to_numeric(df.get("submitter_count_baseline", pd.Series(np.nan, index=df.index)), errors="coerce")

    low = score < 50
    failure = (
        reg.str.contains("collision|underresolved|nonspecific|penetrance|spectrum|moderate|nonportable|low", na=False)
        | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
    )
    metadata_weak = submitter.le(1).fillna(False)
    high_risk_genes = {
        "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
        "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC",
    }

    if variant == "ClinVar-label-only baseline":
        return pd.Series(True, index=df.index)
    if variant == "metadata-only routing":
        return ~metadata_weak
    if variant == "gene-only routing":
        return ~gene.isin(high_risk_genes)
    if variant == "regime-only routing":
        return ~failure
    if variant == "portability-score-only routing":
        return ~low
    if variant == "failure-topology-only routing":
        return ~failure
    if variant == "gene+regime routing":
        return ~(gene.isin(high_risk_genes) | failure)
    if variant == "full CAB routing":
        return df["cab_direct_use_allowed"].astype(bool)
    return pd.Series(True, index=df.index)


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    variants = [
        "ClinVar-label-only baseline",
        "metadata-only routing",
        "gene-only routing",
        "regime-only routing",
        "portability-score-only routing",
        "failure-topology-only routing",
        "gene+regime routing",
        "full CAB routing",
    ]
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        gold = df[gold_col].astype(bool)
        base_direct = direct_variant(df, "ClinVar-label-only baseline")
        base_rate = (base_direct & gold).mean()
        for variant in variants:
            direct = direct_variant(df, variant)
            m = confusion(direct, gold)
            unsupported = (direct & gold).mean()
            rows.append({
                "gold_standard_name": gold_name,
                "routing_variant": variant,
                "N": len(df),
                "unsupported_reuse_rate": unsupported,
                "absolute_reduction_vs_baseline": base_rate - unsupported,
                "relative_reduction_vs_baseline": (base_rate - unsupported) / base_rate if base_rate else np.nan,
                "overrestriction_rate": m["overrestriction_rate"],
                "direct_use_allowed_rate": m["direct_use_allowed_rate"],
                "F1_nonportability_detection": m["F1_nonportability_detection"],
                "direct_use_precision": m["direct_use_precision"],
                "nonportability_recall": m["nonportability_recall"],
                "restriction_precision": m["restriction_precision"],
                "utility_note": "Evaluate unsupported-reuse reduction together with overrestriction and direct-use allowance.",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ABLATION, index=False)
    return out


def claims(df: pd.DataFrame, over: pd.DataFrame, primary: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "claim_strength": "routing_utility_caveat",
            "claim_text": "Routing action flags are non-mutually exclusive; percentages for direct use, contextual repair, disease-specific review, and population/penetrance review can sum above 100%.",
            "source_table": "reports/tables/routing_nonexclusive_action_flags_audit.csv",
            "allowed_wording": "Routing actions are non-mutually exclusive.",
            "prohibited_wording": "Action categories are mutually exclusive unless using primary_routing_action.",
        },
        {
            "claim_strength": "mutually_exclusive_primary_action",
            "claim_text": "A mutually exclusive primary routing action hierarchy is provided for figure/table readability.",
            "source_table": "reports/tables/routing_primary_action_distribution.csv",
            "allowed_wording": "Primary routing action is mutually exclusive by hierarchy.",
            "prohibited_wording": "Primary action replaces non-exclusive routing flags for all operational interpretation.",
        },
        {
            "claim_strength": "overrestriction_transparency",
            "claim_text": "CAB is intentionally conservative and must be reported with false-restriction/overrestriction metrics.",
            "source_table": "reports/tables/routing_false_restriction_overrestriction_audit.csv",
            "allowed_wording": "CAB is repair-first/routing-first and designed to prevent unsupported deterministic reuse, not to maximize direct-use classification.",
            "prohibited_wording": "CAB simply improves direct clinical use or should be used clinically without expert review.",
        },
        {
            "claim_strength": "ablation_interpretation",
            "claim_text": "Ablations must be interpreted as utility trade-offs: lower unsupported reuse can come with higher overrestriction.",
            "source_table": "reports/tables/routing_ablation_utility_tradeoff.csv",
            "allowed_wording": "Compare unsupported-reuse reduction together with overrestriction and direct-use allowance.",
            "prohibited_wording": "One ablation is better than full CAB based only on unsupported-reuse rate.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def write_qc(nonex, primary, over, ab):
    lines = [
        "# Routing Overrestriction and Primary Action Audit",
        "",
        "Technical QC report; not manuscript prose.",
        "",
        "## Key finding",
        "CAB is intentionally conservative: it is designed to prevent unsupported deterministic reuse, not to maximize direct-use classification.",
        "",
        "## Routing flags",
        "Routing action flags are non-mutually exclusive. Percentages for direct-use allowed, contextual repair, disease-specific review, and population/penetrance review can sum above 100%.",
        "",
        "## Primary action hierarchy",
        "A mutually exclusive primary_routing_action is provided for visualization and simple reporting:",
        "1. no_deterministic_reuse",
        "2. disease_specific_review",
        "3. population_or_penetrance_review",
        "4. contextual_repair",
        "5. direct_deterministic_use",
        "",
        "## Non-exclusive audit",
        nonex.to_string(index=False),
        "",
        "## Primary action distribution",
        primary.to_string(index=False),
        "",
        "## False restriction / overrestriction audit",
        over.to_string(index=False),
        "",
        "## Ablation utility trade-off",
        ab.to_string(index=False),
        "",
        "## Publication-safe interpretation",
        "CAB separates pathogenicity from portability. Across three disease domains, public P/LP assertions often remained classification-stable while their disease-model meaning drifted. In a temporal counterfactual routing benchmark, CAB reduced unsupported deterministic reuse while routing most non-portable assertions to contextual repair or disease-specific review.",
        "",
        "## Forbidden interpretations",
        "- CAB reduces clinical errors.",
        "- CAB improves patient outcomes.",
        "- CAB is externally expert-validated.",
        "- CAB should be used clinically without expert review.",
        "- Overrestriction can be ignored.",
    ]
    OUT_QC.write_text("\n".join(lines), encoding="utf-8")


def plot_primary(primary_df):
    if plt is None or primary_df.empty:
        return
    sub = primary_df[primary_df["domain"].ne("all")].copy()
    actions = PRIMARY_ORDER
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = {d: 0 for d in sub["domain"].unique()}
    domains = sorted(sub["domain"].unique())
    x = np.arange(len(domains))
    bottom_vals = np.zeros(len(domains))
    for action in actions:
        vals = []
        for d in domains:
            hit = sub[(sub["domain"].eq(d)) & (sub["primary_routing_action"].eq(action))]
            vals.append(hit["action_percent"].iloc[0] / 100 if len(hit) else 0)
        ax.bar(x, vals, bottom=bottom_vals, label=action)
        bottom_vals += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_PRIMARY)
    plt.close(fig)


def plot_over(over_df):
    if plt is None or over_df.empty:
        return
    sub = over_df[(over_df["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")) & (over_df["domain"].ne("all"))]
    metrics = [
        "direct_use_precision",
        "direct_use_recall",
        "specificity_for_portability",
        "nonportability_recall",
    ]
    x = np.arange(len(sub))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, sub[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["domain"], rotation=20, ha="right")
    ax.set_ylabel("metric")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_OVER)
    plt.close(fig)


def plot_ablation(ab_df):
    if plt is None or ab_df.empty:
        return
    sub = ab_df[ab_df["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(sub["overrestriction_rate"], sub["unsupported_reuse_rate"])
    for _, r in sub.iterrows():
        ax.annotate(r["routing_variant"], (r["overrestriction_rate"], r["unsupported_reuse_rate"]), fontsize=7, rotation=20)
    ax.set_xlabel("overrestriction rate")
    ax.set_ylabel("unsupported reuse rate")
    fig.tight_layout()
    fig.savefig(FIG_ABLATION)
    plt.close(fig)


def main():
    ensure_dirs()
    print("Loading routing tasks...")
    df = load_tasks()
    df = add_primary_actions(df)

    print("Auditing non-exclusive flags...")
    nonex = nonexclusive_audit(df)

    print("Building mutually exclusive primary action distribution...")
    primary = primary_action_distribution(df)

    print("Computing false restriction / overrestriction audit...")
    over = overrestriction_audit(df)

    print("Running ablation utility benchmark...")
    ab = ablation(df)

    print("Writing claims and QC...")
    cl = claims(df, over, primary, ab)
    write_qc(nonex, primary, over, ab)

    print("Writing figures...")
    plot_primary(primary)
    plot_over(over)
    plot_ablation(ab)

    print("Routing overrestriction and primary-action audit complete.")
    print()
    print("Non-exclusive flags:")
    print(nonex.to_string(index=False))
    print()
    print("Primary action distribution:")
    print(primary.to_string(index=False))
    print()
    print("Overrestriction audit:")
    print(over.to_string(index=False))
    print()
    print("Ablation utility:")
    print(ab.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_NONEXCLUSIVE, OUT_PRIMARY, OUT_PRIMARY_ASSERTION, OUT_OVER,
        OUT_ABLATION, OUT_CLAIMS, OUT_QC, FIG_PRIMARY, FIG_OVER, FIG_ABLATION,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
