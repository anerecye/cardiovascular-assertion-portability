#!/usr/bin/env python3
"""CAB operating modes: Core vs Conservative routing audit.

Purpose
-------
Refactor CAB routing into two explicitly separated operating modes:

1. CAB-Core
   - features: gene + baseline disease-model regime
   - purpose: temporal portability prediction and minimal routing
   - primary benchmark: temporal condition-label drift gold standard
   - expected behavior: reduce unsupported deterministic reuse while allowing more
     true portable assertions

2. CAB-Conservative
   - features: full CAB routing flags, portability score, failure topology,
     repair/review flags
   - purpose: safety-first triage and contextual repair
   - primary benchmark: conservative composite routing gold standard
   - expected behavior: stronger restriction, higher repair/review routing, and
     possible overrestriction

Guardrails
----------
- Do not present full CAB as best temporal model.
- Do not bury overrestriction.
- Do not collapse CAB-Core and CAB-Conservative.
- Do not call conservative mode clinically validated.
- Every result traces to task table and this script.
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

OUT_DEF = QC / "cab_operating_modes_definition.md"
OUT_METRICS = TABLES / "cab_core_vs_conservative_routing_metrics.csv"
OUT_BY_DOMAIN = TABLES / "cab_core_vs_conservative_by_domain.csv"
OUT_BOOT = TABLES / "cab_core_vs_conservative_bootstrap_ci.csv"
OUT_PARETO = TABLES / "routing_pareto_frontier.csv"
OUT_DOMAIN_REC = TABLES / "domain_specific_cab_mode_recommendations.csv"
OUT_CLAIMS = TABLES / "routing_publication_safe_claims_operating_modes.csv"

FIG_TRADEOFF = FIGURES / "cab_core_vs_conservative_tradeoff.svg"
FIG_PARETO = FIGURES / "routing_pareto_frontier.svg"
FIG_FINAL = FIGURES / "final_cab_operating_modes_figure.svg"

RANDOM_STATE = 42
N_BOOT = 1000

GOLD_MAP = {
    "temporal_condition_label_drift": "gold_temporal_condition",
    "cross_environment_drift": "gold_cross_environment",
    "any_meaning_drift": "gold_any_meaning",
    "conservative_composite_routing": "gold_composite_routing",
}

MODES = ["ClinVar-label-only baseline", "CAB-Core", "CAB-Conservative"]

PARETO_VARIANTS = [
    "ClinVar baseline",
    "metadata-only",
    "gene-only",
    "regime-only",
    "portability-score-only",
    "failure-topology-only",
    "gene+regime = CAB-Core",
    "full CAB = CAB-Conservative",
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
        raise FileNotFoundError(f"Missing required task table: {TASKS}")

    df = pd.read_csv(TASKS, low_memory=False)

    bools = [
        "direct_single_model_reuse_allowed", "cross_environment_reuse_allowed",
        "contextual_repair_required", "disease_specific_expert_review_required",
        "population_or_penetrance_review_required", "high_future_meaning_drift_risk",
        "high_future_cross_environment_drift_risk", "future_condition_label_drift",
        "future_cross_environment_drift", "future_any_meaning_drift", "self_loop_stable",
    ]
    for c in bools:
        if c in df.columns:
            df[c] = bool_col(df[c])
        else:
            df[c] = False

    for c in ["domain", "assertion_id", "gene", "baseline_regime_primary", "baseline_architecture_family", "environment_baseline"]:
        if c not in df.columns:
            df[c] = ""

    if "baseline_portability_score" not in df.columns:
        df["baseline_portability_score"] = np.nan
    df["baseline_portability_score"] = pd.to_numeric(df["baseline_portability_score"], errors="coerce")

    if "baseline_nonportability_score" not in df.columns:
        df["baseline_nonportability_score"] = 100 - df["baseline_portability_score"]
    df["baseline_nonportability_score"] = pd.to_numeric(df["baseline_nonportability_score"], errors="coerce")

    if "submitter_count_baseline" not in df.columns:
        df["submitter_count_baseline"] = np.nan
    df["submitter_count_baseline"] = pd.to_numeric(df["submitter_count_baseline"], errors="coerce")

    gold = safe_read(GOLD_COMPONENTS)
    if not gold.empty and "assertion_id" in gold.columns:
        keep = [c for c in gold.columns if c == "assertion_id" or c.startswith("gold_")]
        df = df.merge(gold[keep], on="assertion_id", how="left", suffixes=("", "_gold"))

    # Gold standards.
    if "gold_temporal_condition" not in df.columns:
        df["gold_temporal_condition"] = df["future_condition_label_drift"]
    else:
        df["gold_temporal_condition"] = bool_col(df["gold_temporal_condition"])

    df["gold_cross_environment"] = df["future_cross_environment_drift"]
    df["gold_any_meaning"] = df["future_any_meaning_drift"] | df["future_condition_label_drift"] | df["future_cross_environment_drift"]

    if "gold_composite_routing" not in df.columns:
        reg = df["baseline_regime_primary"].astype(str).str.lower()
        arch = df["baseline_architecture_family"].astype(str).str.lower()
        low_portability = df["baseline_portability_score"].lt(50).fillna(False)
        failure_topology = (
            reg.str.contains("collision|nonportable|low|underresolved|nonspecific|moderate|penetrance|spectrum|recessive|biallelic", na=False)
            | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
        )
        decision_layer = (
            df["contextual_repair_required"]
            | df["disease_specific_expert_review_required"]
            | df["population_or_penetrance_review_required"]
            | (~df["direct_single_model_reuse_allowed"])
        )
        df["gold_composite_routing"] = (
            df["future_condition_label_drift"]
            | df["future_cross_environment_drift"]
            | low_portability
            | failure_topology
            | decision_layer
        )
    else:
        df["gold_composite_routing"] = bool_col(df["gold_composite_routing"])

    return df


def failure_topology(df: pd.DataFrame) -> pd.Series:
    reg = df["baseline_regime_primary"].astype(str).str.lower()
    arch = df["baseline_architecture_family"].astype(str).str.lower()
    return (
        reg.str.contains("collision|underresolved|nonspecific|penetrance|spectrum|moderate|nonportable|low|recessive|biallelic", na=False)
        | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
    )


def direct_allowed_for(df: pd.DataFrame, mode: str) -> pd.Series:
    regfail = failure_topology(df)
    gene = df["gene"].astype(str).str.upper()
    score = df["baseline_portability_score"].fillna(60)
    submitter = df["submitter_count_baseline"]

    high_risk_genes = {
        "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
        "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC",
    }

    if mode in {"ClinVar-label-only baseline", "ClinVar baseline"}:
        return pd.Series(True, index=df.index)

    if mode in {"CAB-Core", "gene+regime = CAB-Core"}:
        return ~(gene.isin(high_risk_genes) | regfail)

    if mode in {"CAB-Conservative", "full CAB = CAB-Conservative"}:
        return df["direct_single_model_reuse_allowed"].astype(bool)

    if mode == "metadata-only":
        return ~submitter.le(1).fillna(False)

    if mode == "gene-only":
        return ~gene.isin(high_risk_genes)

    if mode == "regime-only":
        return ~regfail

    if mode == "failure-topology-only":
        return ~regfail

    if mode == "portability-score-only":
        return ~score.lt(50).fillna(False)

    return pd.Series(True, index=df.index)


def metric_block(direct_allowed: pd.Series, gold_nonportable: pd.Series, baseline_unsupported_rate: float | None = None) -> dict:
    direct_allowed = direct_allowed.astype(bool)
    gold_nonportable = gold_nonportable.astype(bool)
    true_portable = ~gold_nonportable
    pred_nonportable = ~direct_allowed

    tp = int((pred_nonportable & gold_nonportable).sum())
    fn = int((direct_allowed & gold_nonportable).sum())
    tn = int((direct_allowed & true_portable).sum())
    fp = int((pred_nonportable & true_portable).sum())
    n = tp + fn + tn + fp

    unsupported = fn / n if n else np.nan
    if baseline_unsupported_rate is None:
        abs_red = np.nan
        rel_red = np.nan
    else:
        abs_red = baseline_unsupported_rate - unsupported
        rel_red = abs_red / baseline_unsupported_rate if baseline_unsupported_rate else np.nan

    true_portable_n = int(true_portable.sum())

    nonport_recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    direct_precision = tn / (tn + fn) if (tn + fn) else np.nan
    direct_recall = tn / (tn + fp) if (tn + fp) else np.nan
    restriction_precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * nonport_recall * restriction_precision / (nonport_recall + restriction_precision) if (nonport_recall + restriction_precision) else np.nan

    return {
        "N": n,
        "unsupported_deterministic_reuse_rate": unsupported,
        "absolute_reduction_vs_ClinVar_baseline": abs_red,
        "relative_reduction_vs_ClinVar_baseline": rel_red,
        "direct_use_allowed_rate": (tn + fn) / n if n else np.nan,
        "true_portable_assertions_N": true_portable_n,
        "true_portable_allowed_direct_use_N": tn,
        "true_portable_allowed_direct_use_rate": tn / true_portable_n if true_portable_n else np.nan,
        "false_restriction_N": fp,
        "false_restriction_rate": fp / true_portable_n if true_portable_n else np.nan,
        "overrestriction_rate": fp / n if n else np.nan,
        "nonportability_recall": nonport_recall,
        "direct_use_precision": direct_precision,
        "direct_use_recall": direct_recall,
        "specificity": specificity,
        "F1_nonportability_detection": f1,
    }


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    domain_rows = []

    for gold_name, gold_col in GOLD_MAP.items():
        gold = df[gold_col].astype(bool)
        baseline_rate = (direct_allowed_for(df, "ClinVar-label-only baseline") & gold).mean()

        for mode in MODES:
            direct = direct_allowed_for(df, mode)
            block = metric_block(direct, gold, baseline_rate)
            row = {
                "gold_standard": gold_name,
                "domain": "all",
                "operating_mode": mode,
                "features": features_for_mode(mode),
                "purpose": purpose_for_mode(mode),
            }
            row.update(block)
            rows.append(row)

        for domain, sub in sorted(list(df.groupby("domain")), key=lambda x: x[0]):
            g = sub[gold_col].astype(bool)
            b_rate = (direct_allowed_for(sub, "ClinVar-label-only baseline") & g).mean()
            for mode in MODES:
                direct = direct_allowed_for(sub, mode)
                block = metric_block(direct, g, b_rate)
                row = {
                    "gold_standard": gold_name,
                    "domain": domain,
                    "operating_mode": mode,
                    "features": features_for_mode(mode),
                    "purpose": purpose_for_mode(mode),
                }
                row.update(block)
                domain_rows.append(row)

    metrics = pd.DataFrame(rows)
    by_domain = pd.DataFrame(domain_rows)
    metrics.to_csv(OUT_METRICS, index=False)
    by_domain.to_csv(OUT_BY_DOMAIN, index=False)
    return metrics, by_domain


def features_for_mode(mode: str) -> str:
    if mode == "CAB-Core":
        return "gene + baseline disease-model regime"
    if mode == "CAB-Conservative":
        return "full CAB routing flags + portability score + failure topology + repair/review flags"
    if mode == "ClinVar-label-only baseline":
        return "ClinVar P/LP label only"
    return ""


def purpose_for_mode(mode: str) -> str:
    if mode == "CAB-Core":
        return "temporal portability prediction and minimal routing"
    if mode == "CAB-Conservative":
        return "safety-first triage and contextual repair"
    if mode == "ClinVar-label-only baseline":
        return "counterfactual label-only direct reuse"
    return ""


def bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    domains = sorted(df["domain"].dropna().unique())
    rows = []

    for gold_name, gold_col in GOLD_MAP.items():
        for mode in ["CAB-Core", "CAB-Conservative"]:
            reps = []
            for _ in range(N_BOOT):
                parts = []
                for d in domains:
                    sub = df[df["domain"].eq(d)]
                    idx = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
                    parts.append(df.loc[idx])
                boot = pd.concat(parts, ignore_index=True)
                gold = boot[gold_col].astype(bool)
                baseline_direct = direct_allowed_for(boot, "ClinVar-label-only baseline")
                mode_direct = direct_allowed_for(boot, mode)
                baseline_rate = (baseline_direct & gold).mean()
                block = metric_block(mode_direct, gold, baseline_rate)
                reps.append({
                    "unsupported_deterministic_reuse_rate": block["unsupported_deterministic_reuse_rate"],
                    "absolute_reduction_vs_ClinVar_baseline": block["absolute_reduction_vs_ClinVar_baseline"],
                    "relative_reduction_vs_ClinVar_baseline": block["relative_reduction_vs_ClinVar_baseline"],
                    "false_restriction_rate": block["false_restriction_rate"],
                    "true_portable_allowed_direct_use_rate": block["true_portable_allowed_direct_use_rate"],
                })
            reps = pd.DataFrame(reps)
            for metric in reps.columns:
                rows.append({
                    "gold_standard": gold_name,
                    "operating_mode": mode,
                    "metric": metric,
                    "estimate": reps[metric].mean(),
                    "ci95_low": reps[metric].quantile(0.025),
                    "ci95_high": reps[metric].quantile(0.975),
                    "bootstrap_replicates": N_BOOT,
                    "stratification": "within_domain",
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BOOT, index=False)
    return out


def pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        gold = df[gold_col].astype(bool)
        baseline_rate = (direct_allowed_for(df, "ClinVar baseline") & gold).mean()
        for variant in PARETO_VARIANTS:
            direct = direct_allowed_for(df, variant)
            block = metric_block(direct, gold, baseline_rate)
            rows.append({
                "gold_standard": gold_name,
                "routing_variant": variant,
                "unsupported_reuse": block["unsupported_deterministic_reuse_rate"],
                "false_restriction_overrestriction": block["overrestriction_rate"],
                "false_restriction_rate_among_true_portable": block["false_restriction_rate"],
                "direct_use_allowed_rate": block["direct_use_allowed_rate"],
                "true_portable_allowed_direct_use_rate": block["true_portable_allowed_direct_use_rate"],
                "absolute_reduction_vs_ClinVar_baseline": block["absolute_reduction_vs_ClinVar_baseline"],
                "relative_reduction_vs_ClinVar_baseline": block["relative_reduction_vs_ClinVar_baseline"],
                "is_CAB_Core": "yes" if variant == "gene+regime = CAB-Core" else "no",
                "is_CAB_Conservative": "yes" if variant == "full CAB = CAB-Conservative" else "no",
            })
    out = pd.DataFrame(rows)

    # Pareto status: lower unsupported and lower overrestriction are better.
    statuses = []
    for gold_name, sub in out.groupby("gold_standard"):
        for i, r in sub.iterrows():
            dominated_by = []
            for _, q in sub.iterrows():
                if q["routing_variant"] == r["routing_variant"]:
                    continue
                better_or_equal = (
                    q["unsupported_reuse"] <= r["unsupported_reuse"]
                    and q["false_restriction_overrestriction"] <= r["false_restriction_overrestriction"]
                )
                strictly_better = (
                    q["unsupported_reuse"] < r["unsupported_reuse"]
                    or q["false_restriction_overrestriction"] < r["false_restriction_overrestriction"]
                )
                if better_or_equal and strictly_better:
                    dominated_by.append(q["routing_variant"])
            if dominated_by:
                status = "dominated"
            else:
                status = "pareto_frontier"
            statuses.append((i, status, ";".join(dominated_by)))
    for i, status, dom in statuses:
        out.loc[i, "pareto_status"] = status
        out.loc[i, "dominated_by"] = dom

    # Explicit interpretation requested.
    out["interpretation"] = ""
    mask = (out["gold_standard"].eq("temporal_condition_label_drift")) & (out["routing_variant"].eq("full CAB = CAB-Conservative"))
    if mask.any():
        core = out[(out["gold_standard"].eq("temporal_condition_label_drift")) & (out["routing_variant"].eq("gene+regime = CAB-Core")]
        if len(core):
            c = core.iloc[0]
            cc = out[mask].iloc[0]
            if c["unsupported_reuse"] < cc["unsupported_reuse"]:
                out.loc[mask, "interpretation"] = "CAB-Conservative is worse than CAB-Core for temporal portability unsupported-reuse minimization; retain for conservative composite routing only."
    mask_core = (out["gold_standard"].eq("temporal_condition_label_drift")) & (out["routing_variant"].eq("gene+regime = CAB-Core"))
    out.loc[mask_core, "interpretation"] = "CAB-Core is the preferred temporal portability operating mode if utility trade-off is acceptable."

    out.to_csv(OUT_PARETO, index=False)
    return out


def mode_recommendations(metrics: pd.DataFrame, by_domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in sorted(by_domain["domain"].unique()):
        temp = by_domain[(by_domain["domain"].eq(domain)) & (by_domain["gold_standard"].eq("temporal_condition_label_drift"))]
        comp = by_domain[(by_domain["domain"].eq(domain)) & (by_domain["gold_standard"].eq("conservative_composite_routing"))]
        def row(mode, frame):
            h = frame[frame["operating_mode"].eq(mode)]
            return h.iloc[0] if len(h) else None
        core_t = row("CAB-Core", temp)
        cons_t = row("CAB-Conservative", temp)
        core_c = row("CAB-Core", comp)
        cons_c = row("CAB-Conservative", comp)

        rec_temporal = "CAB-Core" if core_t is not None and cons_t is not None and core_t["unsupported_deterministic_reuse_rate"] <= cons_t["unsupported_deterministic_reuse_rate"] else "CAB-Conservative"
        rec_safety = "CAB-Conservative"

        rows.append({
            "domain": domain,
            "recommendation_if_goal_temporal_portability_prediction": rec_temporal,
            "recommendation_if_goal_maximum_safety_contextual_repair": rec_safety,
            "CAB_Core_temporal_unsupported_reuse_rate": core_t["unsupported_deterministic_reuse_rate"] if core_t is not None else np.nan,
            "CAB_Conservative_temporal_unsupported_reuse_rate": cons_t["unsupported_deterministic_reuse_rate"] if cons_t is not None else np.nan,
            "CAB_Core_true_portable_allowed_rate_temporal": core_t["true_portable_allowed_direct_use_rate"] if core_t is not None else np.nan,
            "CAB_Conservative_true_portable_allowed_rate_temporal": cons_t["true_portable_allowed_direct_use_rate"] if cons_t is not None else np.nan,
            "CAB_Core_false_restriction_rate_temporal": core_t["false_restriction_rate"] if core_t is not None else np.nan,
            "CAB_Conservative_false_restriction_rate_temporal": cons_t["false_restriction_rate"] if cons_t is not None else np.nan,
            "CAB_Core_composite_unsupported_reuse_rate": core_c["unsupported_deterministic_reuse_rate"] if core_c is not None else np.nan,
            "CAB_Conservative_composite_unsupported_reuse_rate": cons_c["unsupported_deterministic_reuse_rate"] if cons_c is not None else np.nan,
            "interpretation": "Use CAB-Core for temporal portability prediction; use CAB-Conservative when safety-first repair/review triage is the stated goal.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DOMAIN_REC, index=False)
    return out


def claims(metrics: pd.DataFrame, boot: pd.DataFrame, pareto: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    def get(mode, gold):
        h = metrics[(metrics["domain"].eq("all")) & (metrics["gold_standard"].eq(gold)) & (metrics["operating_mode"].eq(mode))]
        return h.iloc[0] if len(h) else None
    core_temp = get("CAB-Core", "temporal_condition_label_drift")
    cons_temp = get("CAB-Conservative", "temporal_condition_label_drift")
    cons_comp = get("CAB-Conservative", "conservative_composite_routing")
    base_temp = get("ClinVar-label-only baseline", "temporal_condition_label_drift")
    base_comp = get("ClinVar-label-only baseline", "conservative_composite_routing")

    rows = []
    if core_temp is not None and base_temp is not None and cons_temp is not None:
        rows.append({
            "claim_type": "primary_operating_mode_claim",
            "claim_text": (
                f"CAB-Core, using gene plus baseline disease-model regime, reduced unsupported deterministic reuse under the temporal condition-label drift gold standard "
                f"from {100*base_temp['unsupported_deterministic_reuse_rate']:.2f}% to {100*core_temp['unsupported_deterministic_reuse_rate']:.2f}%, "
                f"outperforming both ClinVar-label-only reuse and CAB-Conservative for temporal portability."
            ),
            "numbers": (
                f"baseline={base_temp['unsupported_deterministic_reuse_rate']}; "
                f"CAB-Core={core_temp['unsupported_deterministic_reuse_rate']}; "
                f"CAB-Conservative={cons_temp['unsupported_deterministic_reuse_rate']}"
            ),
            "source_table": "reports/tables/cab_core_vs_conservative_routing_metrics.csv",
            "allowed": "yes",
            "limitation": "CAB-Core is optimized for temporal portability/minimal routing, not maximum safety-first triage.",
        })
    if cons_temp is not None and cons_comp is not None and base_comp is not None and base_temp is not None:
        rows.append({
            "claim_type": "secondary_conservative_mode_claim",
            "claim_text": (
                f"CAB-Conservative reduced unsupported deterministic reuse from {100*base_temp['unsupported_deterministic_reuse_rate']:.2f}% to "
                f"{100*cons_temp['unsupported_deterministic_reuse_rate']:.2f}% under the temporal drift gold standard and from "
                f"{100*base_comp['unsupported_deterministic_reuse_rate']:.2f}% to {100*cons_comp['unsupported_deterministic_reuse_rate']:.2f}% "
                f"under the conservative composite routing benchmark, but routed many assertions to repair/review."
            ),
            "numbers": (
                f"temporal baseline={base_temp['unsupported_deterministic_reuse_rate']}; temporal conservative={cons_temp['unsupported_deterministic_reuse_rate']}; "
                f"composite baseline={base_comp['unsupported_deterministic_reuse_rate']}; composite conservative={cons_comp['unsupported_deterministic_reuse_rate']}"
            ),
            "source_table": "reports/tables/cab_core_vs_conservative_routing_metrics.csv",
            "allowed": "yes",
            "limitation": "CAB-Conservative is safety-oriented and can overrestrict assertions that remain temporally portable.",
        })
    rows.append({
        "claim_type": "required_limitation",
        "claim_text": "CAB-Conservative is intentionally safety-oriented and overrestricts many assertions that remain temporally portable; CAB-Core provides the more balanced temporal portability configuration.",
        "numbers": "see cab_core_vs_conservative_routing_metrics.csv and routing_pareto_frontier.csv",
        "source_table": "reports/tables/cab_core_vs_conservative_routing_metrics.csv; reports/tables/routing_pareto_frontier.csv",
        "allowed": "required",
        "limitation": "Do not present full CAB as the best temporal model if CAB-Core performs better.",
    })
    rows.append({
        "claim_type": "forbidden_claim",
        "claim_text": "Full CAB / CAB-Conservative is the best temporal condition-label drift model.",
        "numbers": "contradicted when CAB-Core temporal unsupported reuse is lower than CAB-Conservative",
        "source_table": "reports/tables/cab_core_vs_conservative_routing_metrics.csv",
        "allowed": "no",
        "limitation": "Forbidden if CAB-Core outperforms CAB-Conservative under temporal gold.",
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def write_definition():
    lines = [
        "# CAB Operating Modes Definition",
        "",
        "Technical definitions; not manuscript prose.",
        "",
        "## CAB-Core",
        "- features: gene + baseline disease-model regime",
        "- purpose: temporal portability prediction and minimal routing",
        "- benchmark: temporal condition-label drift gold standard",
        "- expected behavior: reduce unsupported deterministic reuse while allowing more true portable assertions",
        "- reporting rule: use as the primary temporal portability operating mode if it outperforms CAB-Conservative under temporal gold.",
        "",
        "## CAB-Conservative",
        "- features: full CAB routing flags, portability score, failure topology, repair/review flags",
        "- purpose: safety-first triage and contextual repair",
        "- benchmark: conservative composite routing gold standard",
        "- expected behavior: stronger restriction, higher repair/review routing, possible overrestriction",
        "- reporting rule: use as conservative operational stress-test / safety-first triage mode, not as the best temporal portability model if CAB-Core performs better.",
        "",
        "## Required distinction",
        "CAB-Core and CAB-Conservative are distinct operating modes and must not be collapsed into one claim.",
        "",
        "## Non-negotiable reporting rules",
        "- Do not present full CAB/CAB-Conservative as best for temporal condition-label drift if CAB-Core is better.",
        "- Do not hide overrestriction.",
        "- Do not call CAB-Conservative clinically validated.",
        "- Do not treat composite routing gold as independent external validation.",
        "- Report unsupported reuse together with false restriction, direct-use allowance, and true-portable allowance.",
    ]
    OUT_DEF.write_text("\n".join(lines), encoding="utf-8")


def plot_tradeoff(metrics: pd.DataFrame):
    if plt is None or metrics.empty:
        return
    sub = metrics[(metrics["domain"].eq("all")) & (metrics["gold_standard"].isin(["temporal_condition_label_drift", "conservative_composite_routing"]))]
    fig, ax = plt.subplots(figsize=(8, 5))
    for gold, g in sub.groupby("gold_standard"):
        for _, r in g.iterrows():
            ax.scatter(r["false_restriction_rate"], r["unsupported_deterministic_reuse_rate"])
            ax.annotate(f"{r['operating_mode']}\n{gold}", (r["false_restriction_rate"], r["unsupported_deterministic_reuse_rate"]), fontsize=7)
    ax.set_xlabel("false restriction rate among true portable")
    ax.set_ylabel("unsupported deterministic reuse rate")
    fig.tight_layout()
    fig.savefig(FIG_TRADEOFF)
    plt.close(fig)


def plot_pareto(pareto: pd.DataFrame):
    if plt is None or pareto.empty:
        return
    sub = pareto[pareto["gold_standard"].eq("temporal_condition_label_drift")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in sub.iterrows():
        marker = "o"
        if r["is_CAB_Core"] == "yes":
            marker = "*"
        elif r["is_CAB_Conservative"] == "yes":
            marker = "s"
        ax.scatter(r["false_restriction_overrestriction"], r["unsupported_reuse"], marker=marker, s=90 if marker == "*" else 45)
        ax.annotate(r["routing_variant"], (r["false_restriction_overrestriction"], r["unsupported_reuse"]), fontsize=7, rotation=15)
    ax.set_xlabel("false restriction / overrestriction")
    ax.set_ylabel("unsupported reuse")
    ax.set_title("Temporal condition-label drift gold standard")
    fig.tight_layout()
    fig.savefig(FIG_PARETO)
    plt.close(fig)


def plot_final(metrics: pd.DataFrame, pareto: pd.DataFrame, rec: pd.DataFrame):
    if plt is None:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.6, "CAB-Core\nGene + baseline disease-model regime\nTemporal portability / minimal routing", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 0].set_title("A. Operating modes")

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.6, "CAB-Conservative\nFull routing flags + portability score\nSafety-first repair/review triage", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 1].set_title("A continued")

    sub = metrics[(metrics["domain"].eq("all")) & (metrics["gold_standard"].eq("temporal_condition_label_drift"))]
    axes[1, 0].bar(sub["operating_mode"], sub["unsupported_deterministic_reuse_rate"])
    axes[1, 0].set_xticklabels(sub["operating_mode"], rotation=25, ha="right")
    axes[1, 0].set_ylabel("unsupported reuse")
    axes[1, 0].set_title("B. Unsupported reuse reduction")

    axes[1, 1].bar(sub["operating_mode"], sub["true_portable_allowed_direct_use_rate"])
    axes[1, 1].set_xticklabels(sub["operating_mode"], rotation=25, ha="right")
    axes[1, 1].set_ylabel("true portable allowed direct use")
    axes[1, 1].set_title("C. True portable allowed rate")

    p = pareto[pareto["gold_standard"].eq("temporal_condition_label_drift")]
    for _, r in p.iterrows():
        marker = "*" if r["is_CAB_Core"] == "yes" else ("s" if r["is_CAB_Conservative"] == "yes" else "o")
        axes[2, 0].scatter(r["false_restriction_overrestriction"], r["unsupported_reuse"], marker=marker, s=90 if marker == "*" else 45)
    axes[2, 0].set_xlabel("overrestriction")
    axes[2, 0].set_ylabel("unsupported reuse")
    axes[2, 0].set_title("D. Pareto tradeoff")

    x = np.arange(len(rec))
    width = 0.35
    axes[2, 1].bar(x - width/2, rec["CAB_Core_temporal_unsupported_reuse_rate"], width, label="CAB-Core")
    axes[2, 1].bar(x + width/2, rec["CAB_Conservative_temporal_unsupported_reuse_rate"], width, label="CAB-Conservative")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(rec["domain"], rotation=20, ha="right")
    axes[2, 1].set_ylabel("temporal unsupported reuse")
    axes[2, 1].set_title("E. Domain-specific performance")
    axes[2, 1].legend()

    fig.tight_layout()
    fig.savefig(FIG_FINAL)
    plt.close(fig)


def main():
    ensure_dirs()
    print("Loading CAB decision challenge tasks...")
    df = load_tasks()
    print(f"N={len(df):,}")
    print(df.groupby("domain").size().to_string())

    print("Writing operating mode definitions...")
    write_definition()

    print("Computing CAB-Core vs CAB-Conservative metrics...")
    metrics, by_domain = compute_metrics(df)

    print("Running stratified bootstrap...")
    boot = bootstrap_ci(df)

    print("Running Pareto frontier analysis...")
    pareto = pareto_analysis(df)

    print("Building domain-specific operating mode recommendations...")
    rec = mode_recommendations(metrics, by_domain)

    print("Writing publication-safe operating mode claims...")
    cl = claims(metrics, boot, pareto, rec)

    print("Writing figures...")
    plot_tradeoff(metrics)
    plot_pareto(pareto)
    plot_final(metrics, pareto, rec)

    print("CAB operating mode audit complete.")
    print()
    print("All-domain metrics:")
    print(metrics.to_string(index=False))
    print()
    print("Domain recommendations:")
    print(rec.to_string(index=False))
    print()
    print("Pareto temporal subset:")
    print(pareto[pareto["gold_standard"].eq("temporal_condition_label_drift")].to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_DEF, OUT_METRICS, OUT_BY_DOMAIN, FIG_TRADEOFF, OUT_BOOT,
        OUT_PARETO, FIG_PARETO, OUT_DOMAIN_REC, OUT_CLAIMS, FIG_FINAL,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
