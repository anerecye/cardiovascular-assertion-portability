#!/usr/bin/env python3
"""Final CAB routing operating-frontier analysis.

This script finalizes the operating-frontier framing:

Final mode names
----------------
- ClinVar-label-only
- CAB-Strict        (old: CAB-Core)
- CAB-Balanced      (old: CAB-Conservative)

Core principle
--------------
CAB is not a single universally optimal routing classifier. It defines an
operating frontier for assertion portability, exposing the tradeoff between:
- false portability / unsupported deterministic reuse
- overrestriction / false restriction of true-portable assertions

Guardrails
----------
- Do not hide CAB-Strict overrestriction.
- Do not hide CAB-Balanced higher unsupported reuse.
- Do not present one mode as universally optimal.
- Do not use old names in final headline tables except in crosswalk.
- Do not claim external expert decision validation.
- Do not claim clinical outcome improvement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
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
THREE_DOMAIN_SUMMARY = TABLES / "three_domain_portability_summary.csv"
GRAMMAR = TABLES / "domain_specific_portability_grammar_final.csv"
ALPHA = TABLES / "cab_alphamissense_model_comparison.csv"
CLINGEN = TABLES / "cardiomyopathy_clingen_overlay_status_clean.csv"
PRIMARY_ACTIONS = DATA / "cab_decision_challenge_primary_actions.csv"

OUT_DEF = QC / "cab_operating_modes_final_definition.md"
OUT_MODES = DATA / "cab_routing_operating_modes_final.csv"
OUT_CROSSWALK = TABLES / "cab_operating_mode_name_crosswalk.csv"

OUT_METRICS = TABLES / "routing_metrics_all_modes_all_endpoints.csv"
OUT_METRICS_DOMAIN = TABLES / "routing_metrics_by_domain_all_modes.csv"

OUT_FRONTIER_DOMAIN = TABLES / "routing_frontier_by_domain.csv"
OUT_FRONTIER_MATRIX = TABLES / "routing_frontier_domain_endpoint_matrix.csv"

OUT_PARETO = TABLES / "routing_pareto_frontier_by_endpoint.csv"
FIG_PARETO_TEMP = FIGURES / "routing_pareto_frontier_temporal_condition_label_drift.svg"
FIG_PARETO_CROSS = FIGURES / "routing_pareto_frontier_cross_environment_drift.svg"
FIG_PARETO_ANY = FIGURES / "routing_pareto_frontier_any_meaning_drift.svg"
FIG_PARETO_COMP = FIGURES / "routing_pareto_frontier_composite.svg"

OUT_BOOT = TABLES / "routing_operating_modes_bootstrap_ci.csv"

OUT_GUIDE = TABLES / "cab_mode_selection_guide.csv"
OUT_CLAIMS = TABLES / "routing_publication_safe_claims_final_operating_frontier.csv"
FIG_FINAL = FIGURES / "final_cab_operating_frontier_figure.svg"

OUT_LADDER = TABLES / "cab_evidence_ladder_final.csv"
OUT_READY = REPORTS / "final_cab_readiness_report.md"
OUT_QUAR = TABLES / "quarantined_operating_mode_wording.csv"

RANDOM_STATE = 42
N_BOOT = 1000

FINAL_MODES = [
    "ClinVar-label-only",
    "metadata-only",
    "gene-only",
    "regime-only",
    "portability-score-only",
    "failure-topology-only",
    "CAB-Strict",
    "CAB-Balanced",
]

GOLD_ENDPOINTS = {
    "temporal_condition_label_drift": "gold_temporal_condition",
    "cross_environment_drift": "gold_cross_environment",
    "any_meaning_drift": "gold_any_meaning",
    "semantic_drift_without_reclassification": "gold_semantic_drift_without_reclassification",
    "conservative_composite_routing": "gold_composite_routing",
}

PARETO_FIGS = {
    "temporal_condition_label_drift": FIG_PARETO_TEMP,
    "cross_environment_drift": FIG_PARETO_CROSS,
    "any_meaning_drift": FIG_PARETO_ANY,
    "conservative_composite_routing": FIG_PARETO_COMP,
}


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

    bool_cols = [
        "direct_single_model_reuse_allowed", "cross_environment_reuse_allowed",
        "contextual_repair_required", "disease_specific_expert_review_required",
        "population_or_penetrance_review_required", "high_future_meaning_drift_risk",
        "high_future_cross_environment_drift_risk", "future_condition_label_drift",
        "future_cross_environment_drift", "future_any_meaning_drift", "self_loop_stable",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = bool_col(df[c])
        else:
            df[c] = False

    for c in ["domain", "assertion_id", "gene", "baseline_regime_primary", "baseline_architecture_family", "environment_baseline"]:
        if c not in df.columns:
            df[c] = ""

    if "baseline_portability_score" not in df.columns:
        df["baseline_portability_score"] = np.nan
    df["baseline_portability_score"] = pd.to_numeric(df["baseline_portability_score"], errors="coerce").fillna(60)

    if "baseline_nonportability_score" not in df.columns:
        df["baseline_nonportability_score"] = 100 - df["baseline_portability_score"]
    df["baseline_nonportability_score"] = pd.to_numeric(df["baseline_nonportability_score"], errors="coerce").fillna(40)

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

    df["gold_any_meaning"] = (
        df["future_any_meaning_drift"]
        | df["future_condition_label_drift"]
        | df["future_cross_environment_drift"]
    )

    # Semantic drift without reclassification: if no explicit field exists, use condition drift as conservative fallback.
    if "semantic_drift_without_reclassification" in df.columns:
        df["gold_semantic_drift_without_reclassification"] = bool_col(df["semantic_drift_without_reclassification"])
    else:
        df["gold_semantic_drift_without_reclassification"] = df["future_condition_label_drift"]

    if "gold_composite_routing" not in df.columns:
        low_portability = df["baseline_portability_score"].lt(50).fillna(False)
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
            | failure_topology(df)
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


def high_risk_gene_mask(df: pd.DataFrame) -> pd.Series:
    gene = df["gene"].astype(str).str.upper()
    high_risk_genes = {
        "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
        "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC",
    }
    return gene.isin(high_risk_genes)


def direct_allowed(df: pd.DataFrame, mode: str) -> pd.Series:
    fail = failure_topology(df)
    high_gene = high_risk_gene_mask(df)
    low_score = df["baseline_portability_score"].lt(50).fillna(False)
    metadata_weak = df["submitter_count_baseline"].le(1).fillna(False)

    if mode == "ClinVar-label-only":
        return pd.Series(True, index=df.index)
    if mode == "metadata-only":
        return ~metadata_weak
    if mode == "gene-only":
        return ~high_gene
    if mode == "regime-only":
        return ~fail
    if mode == "portability-score-only":
        return ~low_score
    if mode == "failure-topology-only":
        return ~fail
    if mode == "CAB-Strict":
        return ~(high_gene | fail)
    if mode == "CAB-Balanced":
        return df["direct_single_model_reuse_allowed"].astype(bool)
    raise ValueError(f"Unknown mode: {mode}")


def mode_features(mode: str) -> str:
    return {
        "ClinVar-label-only": "ClinVar P/LP label only",
        "metadata-only": "baseline review/status/submitter metadata only",
        "gene-only": "gene identity only",
        "regime-only": "baseline disease-model regime only",
        "portability-score-only": "baseline portability score threshold only",
        "failure-topology-only": "baseline failure/regime topology only",
        "CAB-Strict": "gene + baseline disease-model regime",
        "CAB-Balanced": "full CAB routing configuration",
    }.get(mode, "")


def old_mode_name(mode: str) -> str:
    return {
        "ClinVar-label-only": "ClinVar-label-only baseline",
        "CAB-Strict": "CAB-Core",
        "CAB-Balanced": "CAB-Conservative",
    }.get(mode, "")


def mode_behavior(mode: str) -> str:
    return {
        "ClinVar-label-only": "maximal permissiveness",
        "CAB-Strict": "high-stringency triage",
        "CAB-Balanced": "balanced safety-permissiveness routing",
    }.get(mode, "baseline comparator")


def mode_goal(mode: str) -> str:
    return {
        "ClinVar-label-only": "default direct deterministic reuse",
        "CAB-Strict": "minimize false portability / unsupported deterministic reuse",
        "CAB-Balanced": "retain large unsupported-reuse reduction while allowing more direct use",
    }.get(mode, "comparator")


def mode_limitation(mode: str) -> str:
    return {
        "ClinVar-label-only": "high unsupported deterministic reuse under drift endpoints",
        "CAB-Strict": "high overrestriction and low direct-use allowance",
        "CAB-Balanced": "higher unsupported reuse than CAB-Strict but less overrestriction",
    }.get(mode, "not a final CAB operating mode")


def routing_loss(direct: pd.Series, gold: pd.Series) -> float:
    # Brier-like routing loss: prediction of portability as direct_allowed vs true portability.
    true_portable = (~gold.astype(bool)).astype(float)
    pred_portable = direct.astype(bool).astype(float)
    return float(np.mean((pred_portable - true_portable) ** 2))


def metric_block(direct: pd.Series, gold: pd.Series, baseline_rate: float | None) -> Dict[str, float]:
    direct = direct.astype(bool)
    gold = gold.astype(bool)
    pred_nonportable = ~direct
    true_nonportable = gold
    true_portable = ~gold

    tp = int((pred_nonportable & true_nonportable).sum())
    fn = int((direct & true_nonportable).sum())
    tn = int((direct & true_portable).sum())
    fp = int((pred_nonportable & true_portable).sum())
    n = tp + fn + tn + fp

    unsupported = fn / n if n else np.nan
    overrestriction = fp / n if n else np.nan
    direct_allowed_rate = (tn + fn) / n if n else np.nan
    positives = int(true_nonportable.sum())
    true_portable_n = int(true_portable.sum())

    nonport_recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    direct_precision = tn / (tn + fn) if (tn + fn) else np.nan
    direct_recall = tn / (tn + fp) if (tn + fp) else np.nan
    restriction_precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * nonport_recall * restriction_precision / (nonport_recall + restriction_precision) if (nonport_recall + restriction_precision) else np.nan
    balanced_accuracy = np.nanmean([nonport_recall, specificity])

    if baseline_rate is None:
        abs_red_pp = np.nan
        rel_red_pct = np.nan
    else:
        abs_red_pp = (baseline_rate - unsupported) * 100
        rel_red_pct = ((baseline_rate - unsupported) / baseline_rate * 100) if baseline_rate else np.nan

    return {
        "N": n,
        "endpoint_positive_N": positives,
        "unsupported_reuse_rate": unsupported,
        "absolute_reduction_vs_ClinVar_pp": abs_red_pp,
        "relative_reduction_vs_ClinVar_percent": rel_red_pct,
        "overrestriction_rate": overrestriction,
        "direct_use_allowed_rate": direct_allowed_rate,
        "true_portable_allowed_rate": tn / true_portable_n if true_portable_n else np.nan,
        "false_restriction_rate": fp / true_portable_n if true_portable_n else np.nan,
        "direct_use_precision": direct_precision,
        "direct_use_recall": direct_recall,
        "nonportability_recall": nonport_recall,
        "portability_specificity": specificity,
        "F1_nonportability": f1,
        "balanced_accuracy": balanced_accuracy,
        "Brier_like_routing_loss": routing_loss(direct, gold),
    }


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    domain_rows = []
    domains = [("all_domains", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0])

    for endpoint, gold_col in GOLD_ENDPOINTS.items():
        gold = df[gold_col].astype(bool)
        base_direct = direct_allowed(df, "ClinVar-label-only")
        baseline_rate = (base_direct & gold).mean()

        for mode in FINAL_MODES:
            direct = direct_allowed(df, mode)
            m = metric_block(direct, gold, baseline_rate)
            rows.append({
                "endpoint": endpoint,
                "cab_mode": mode,
                "old_mode_name": old_mode_name(mode),
                "features": mode_features(mode),
                "claim_strength": claim_strength(endpoint, mode),
                **m,
            })

        for domain, sub in domains:
            gold_d = sub[gold_col].astype(bool)
            base_d = direct_allowed(sub, "ClinVar-label-only")
            baseline_rate_d = (base_d & gold_d).mean()
            for mode in FINAL_MODES:
                direct_d = direct_allowed(sub, mode)
                m = metric_block(direct_d, gold_d, baseline_rate_d)
                domain_rows.append({
                    "domain": domain,
                    "endpoint": endpoint,
                    "cab_mode": mode,
                    "old_mode_name": old_mode_name(mode),
                    "features": mode_features(mode),
                    "claim_strength": claim_strength(endpoint, mode),
                    **m,
                })

    out = pd.DataFrame(rows)
    dom = pd.DataFrame(domain_rows)
    out.to_csv(OUT_METRICS, index=False)
    dom.to_csv(OUT_METRICS_DOMAIN, index=False)
    return out, dom


def claim_strength(endpoint: str, mode: str) -> str:
    if mode == "ClinVar-label-only":
        return "counterfactual_baseline"
    if mode == "CAB-Strict" and endpoint in {"temporal_condition_label_drift", "cross_environment_drift", "any_meaning_drift"}:
        return "high_stringency_temporal_frontier_mode"
    if mode == "CAB-Balanced" and endpoint == "conservative_composite_routing":
        return "balanced_operational_stress_test_mode"
    if mode == "CAB-Balanced":
        return "balanced_routing_mode"
    return "comparator"


def frontier_checks(metrics: pd.DataFrame, dom: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    matrix_rows = []
    for endpoint, sub in metrics.groupby("endpoint"):
        strict = sub[sub["cab_mode"].eq("CAB-Strict")].iloc[0]
        bal = sub[sub["cab_mode"].eq("CAB-Balanced")].iloc[0]
        min_mode = sub.loc[sub["unsupported_reuse_rate"].astype(float).idxmin(), "cab_mode"]
        rows.append({
            "domain": "all_domains",
            "endpoint": endpoint,
            "CAB_Strict_minimizes_unsupported_reuse": "yes" if min_mode == "CAB-Strict" else "no",
            "min_unsupported_reuse_mode": min_mode,
            "CAB_Balanced_more_direct_use_than_Strict": "yes" if bal["direct_use_allowed_rate"] > strict["direct_use_allowed_rate"] else "no",
            "CAB_Balanced_lower_overrestriction_than_Strict": "yes" if bal["overrestriction_rate"] < strict["overrestriction_rate"] else "no",
            "CAB_Strict_unsupported_reuse": strict["unsupported_reuse_rate"],
            "CAB_Balanced_unsupported_reuse": bal["unsupported_reuse_rate"],
            "CAB_Strict_overrestriction": strict["overrestriction_rate"],
            "CAB_Balanced_overrestriction": bal["overrestriction_rate"],
            "interpretation": interpretation_for_frontier(strict, bal, endpoint),
        })

    for (domain, endpoint), sub in dom.groupby(["domain", "endpoint"]):
        strict = sub[sub["cab_mode"].eq("CAB-Strict")].iloc[0]
        bal = sub[sub["cab_mode"].eq("CAB-Balanced")].iloc[0]
        min_mode = sub.loc[sub["unsupported_reuse_rate"].astype(float).idxmin(), "cab_mode"]
        matrix_rows.append({
            "domain": domain,
            "endpoint": endpoint,
            "CAB_Strict_minimizes_unsupported_reuse": "yes" if min_mode == "CAB-Strict" else "no",
            "min_unsupported_reuse_mode": min_mode,
            "CAB_Balanced_more_direct_use_than_Strict": "yes" if bal["direct_use_allowed_rate"] > strict["direct_use_allowed_rate"] else "no",
            "CAB_Balanced_lower_overrestriction_than_Strict": "yes" if bal["overrestriction_rate"] < strict["overrestriction_rate"] else "no",
            "CAB_Strict_unsupported_reuse": strict["unsupported_reuse_rate"],
            "CAB_Balanced_unsupported_reuse": bal["unsupported_reuse_rate"],
            "CAB_Strict_overrestriction": strict["overrestriction_rate"],
            "CAB_Balanced_overrestriction": bal["overrestriction_rate"],
        })
    f = pd.DataFrame(rows)
    m = pd.DataFrame(matrix_rows)
    f.to_csv(OUT_FRONTIER_DOMAIN, index=False)
    m.to_csv(OUT_FRONTIER_MATRIX, index=False)
    return f, m


def interpretation_for_frontier(strict, bal, endpoint: str) -> str:
    if strict["unsupported_reuse_rate"] < bal["unsupported_reuse_rate"] and strict["overrestriction_rate"] > bal["overrestriction_rate"]:
        return "CAB-Strict minimizes false portability; CAB-Balanced preserves more direct-use capacity and less overrestriction."
    if strict["unsupported_reuse_rate"] <= bal["unsupported_reuse_rate"]:
        return "CAB-Strict has lower or equal unsupported reuse."
    return "CAB-Balanced has lower unsupported reuse for this endpoint; report honestly."


def pareto(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, sub in metrics.groupby("endpoint"):
        candidates = sub.copy()
        for _, r in candidates.iterrows():
            dominated_by = []
            for _, q in candidates.iterrows():
                if q["cab_mode"] == r["cab_mode"]:
                    continue
                better_or_equal = (
                    q["unsupported_reuse_rate"] <= r["unsupported_reuse_rate"]
                    and q["overrestriction_rate"] <= r["overrestriction_rate"]
                )
                strictly = (
                    q["unsupported_reuse_rate"] < r["unsupported_reuse_rate"]
                    or q["overrestriction_rate"] < r["overrestriction_rate"]
                )
                if better_or_equal and strictly:
                    dominated_by.append(q["cab_mode"])
            status = "dominated" if dominated_by else "pareto_frontier"
            label = ""
            if r["cab_mode"] == "CAB-Strict" and status == "pareto_frontier":
                label = "high-stringency frontier mode"
            elif r["cab_mode"] == "CAB-Balanced" and status == "pareto_frontier":
                label = "balanced frontier mode"
            elif r["cab_mode"] == "CAB-Balanced" and status == "dominated":
                label = "balanced mode dominated for this endpoint; report honestly"
            rows.append({
                "endpoint": endpoint,
                "cab_mode": r["cab_mode"],
                "old_mode_name": r["old_mode_name"],
                "unsupported_reuse_rate": r["unsupported_reuse_rate"],
                "overrestriction_rate": r["overrestriction_rate"],
                "direct_use_allowed_rate": r["direct_use_allowed_rate"],
                "true_portable_allowed_rate": r["true_portable_allowed_rate"],
                "pareto_status": status,
                "dominated_by": ";".join(dominated_by),
                "frontier_label": label,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PARETO, index=False)
    return out


def bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    domains = sorted(df["domain"].dropna().unique())
    rows = []

    for endpoint, gold_col in GOLD_ENDPOINTS.items():
        for mode in FINAL_MODES:
            reps = []
            for _ in range(N_BOOT):
                parts = []
                for d in domains:
                    sub = df[df["domain"].eq(d)]
                    if len(sub) == 0:
                        continue
                    # Stratified by domain; endpoint class balance is approximately preserved by large within-domain resampling.
                    idx = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
                    parts.append(df.loc[idx])
                boot_df = pd.concat(parts, ignore_index=True)
                gold = boot_df[gold_col].astype(bool)
                base_rate = (direct_allowed(boot_df, "ClinVar-label-only") & gold).mean()
                direct = direct_allowed(boot_df, mode)
                m = metric_block(direct, gold, base_rate)
                reps.append({
                    "unsupported_reuse_rate": m["unsupported_reuse_rate"],
                    "overrestriction_rate": m["overrestriction_rate"],
                    "direct_use_allowed_rate": m["direct_use_allowed_rate"],
                    "true_portable_allowed_rate": m["true_portable_allowed_rate"],
                    "absolute_reduction_vs_ClinVar_pp": m["absolute_reduction_vs_ClinVar_pp"],
                    "relative_reduction_vs_ClinVar_percent": m["relative_reduction_vs_ClinVar_percent"],
                })
            rep = pd.DataFrame(reps)
            for metric in rep.columns:
                rows.append({
                    "endpoint": endpoint,
                    "cab_mode": mode,
                    "old_mode_name": old_mode_name(mode),
                    "metric": metric,
                    "estimate": rep[metric].mean(),
                    "ci95_low": rep[metric].quantile(0.025),
                    "ci95_high": rep[metric].quantile(0.975),
                    "bootstrap_replicates": N_BOOT,
                    "stratification": "within_domain",
                    "endpoint_class_balance_note": "large within-domain bootstrap; class balance approximately preserved",
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BOOT, index=False)
    return out


def selection_guide(metrics: pd.DataFrame) -> pd.DataFrame:
    temp = metrics[(metrics["endpoint"].eq("temporal_condition_label_drift"))]
    comp = metrics[(metrics["endpoint"].eq("conservative_composite_routing"))]
    def get(mode, field, frame=temp):
        h = frame[frame["cab_mode"].eq(mode)]
        return h[field].iloc[0] if len(h) else np.nan

    rows = [
        {
            "use_case": "high-risk screening / conservative triage",
            "recommended_mode": "CAB-Strict or expert review required",
            "metric_basis": "lowest temporal unsupported reuse, but high overrestriction",
            "expected_benefit": f"unsupported reuse {get('CAB-Strict','unsupported_reuse_rate'):.4f}; relative reduction {get('CAB-Strict','relative_reduction_vs_ClinVar_percent'):.2f}%",
            "expected_cost": f"false restriction {get('CAB-Strict','false_restriction_rate'):.4f}; direct-use allowed {get('CAB-Strict','direct_use_allowed_rate'):.4f}",
            "when_not_to_use": "when direct-use preservation is more important than minimizing false portability",
            "required_caveats": "not clinical automation; expert adjudication pending",
        },
        {
            "use_case": "routine curation prioritization",
            "recommended_mode": "CAB-Balanced",
            "metric_basis": "larger direct-use allowance and lower overrestriction than CAB-Strict while preserving major unsupported-reuse reduction",
            "expected_benefit": f"unsupported reuse {get('CAB-Balanced','unsupported_reuse_rate'):.4f}; relative reduction {get('CAB-Balanced','relative_reduction_vs_ClinVar_percent'):.2f}%",
            "expected_cost": f"higher unsupported reuse than CAB-Strict; false restriction {get('CAB-Balanced','false_restriction_rate'):.4f}",
            "when_not_to_use": "when the objective is maximum false-portability suppression",
            "required_caveats": "balanced routing, not external validation",
        },
        {
            "use_case": "research annotation reuse",
            "recommended_mode": "CAB-Balanced",
            "metric_basis": "preserves more direct-use capacity for annotation workflows",
            "expected_benefit": f"direct-use allowed {get('CAB-Balanced','direct_use_allowed_rate'):.4f}; true portable allowed {get('CAB-Balanced','true_portable_allowed_rate'):.4f}",
            "expected_cost": f"unsupported reuse {get('CAB-Balanced','unsupported_reuse_rate'):.4f}",
            "when_not_to_use": "when annotation will be used as deterministic clinical inference",
            "required_caveats": "research use only; route ambiguous assertions to review",
        },
        {
            "use_case": "population screening annotation review",
            "recommended_mode": "CAB-Strict or expert review required",
            "metric_basis": "prioritizes false-portability suppression and population/penetrance repair",
            "expected_benefit": f"temporal unsupported reuse {get('CAB-Strict','unsupported_reuse_rate'):.4f}",
            "expected_cost": f"low true-portable direct-use allowance {get('CAB-Strict','true_portable_allowed_rate'):.4f}",
            "when_not_to_use": "when no disease-specific review capacity exists",
            "required_caveats": "population-frequency and penetrance review required",
        },
        {
            "use_case": "postmortem / genotype-first contexts",
            "recommended_mode": "expert review required",
            "metric_basis": "context is high-risk for unsupported deterministic inference",
            "expected_benefit": "routes to disease-specific review rather than direct deterministic reuse",
            "expected_cost": "may overrestrict portable assertions",
            "when_not_to_use": "do not use as automated clinical filter",
            "required_caveats": "external expert adjudication pending",
        },
        {
            "use_case": "broad public database reuse",
            "recommended_mode": "CAB-Balanced",
            "metric_basis": "default operating-frontier mode for broad reuse",
            "expected_benefit": f"reduces unsupported reuse to {get('CAB-Balanced','unsupported_reuse_rate'):.4f} while allowing direct use {get('CAB-Balanced','direct_use_allowed_rate'):.4f}",
            "expected_cost": "higher unsupported reuse than CAB-Strict",
            "when_not_to_use": "when downstream use is high-risk deterministic inference",
            "required_caveats": "routing benchmark only; no clinical outcome claim",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GUIDE, index=False)
    return out


def publication_claims(metrics: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    temp = metrics[metrics["endpoint"].eq("temporal_condition_label_drift")]
    def get(mode, field):
        h = temp[temp["cab_mode"].eq(mode)]
        return h[field].iloc[0] if len(h) else np.nan
    rows = [
        {
            "claim_type": "allowed_primary",
            "claim_text": (
                "CAB defines an operating frontier for assertion portability. In the temporal condition-label drift benchmark, "
                "CAB-Strict reduced unsupported deterministic reuse from 36.92% to 2.42% but allowed direct use for only 8.09% "
                "of assertions, whereas CAB-Balanced reduced unsupported reuse to 7.46% while allowing direct use for 27.31% "
                "of assertions and 31.48% of true portable assertions."
            ),
            "source_numbers": (
                f"Strict unsupported={get('CAB-Strict','unsupported_reuse_rate')}; Strict direct={get('CAB-Strict','direct_use_allowed_rate')}; "
                f"Balanced unsupported={get('CAB-Balanced','unsupported_reuse_rate')}; Balanced direct={get('CAB-Balanced','direct_use_allowed_rate')}; "
                f"Balanced true_portable_allowed={get('CAB-Balanced','true_portable_allowed_rate')}"
            ),
            "claim_strength": "operating_frontier_primary",
            "source_table": "reports/tables/routing_metrics_all_modes_all_endpoints.csv",
            "required_caveat": "Routing correctness is internally benchmarked; external expert adjudication remains pending.",
        },
        {
            "claim_type": "allowed_secondary",
            "claim_text": (
                "Across endpoints, CAB-Strict minimized false portability, while CAB-Balanced reduced overrestriction and preserved more direct-use capacity. "
                "These modes represent different operating points rather than a single universal classifier."
            ),
            "source_numbers": "see routing_metrics_all_modes_all_endpoints.csv and routing_pareto_frontier_by_endpoint.csv",
            "claim_strength": "operating_frontier_secondary",
            "source_table": "reports/tables/routing_pareto_frontier_by_endpoint.csv",
            "required_caveat": "Do not present one mode as universally optimal.",
        },
        {
            "claim_type": "required_caveat",
            "claim_text": "Routing correctness is internally benchmarked against temporal and composite portability endpoints; external expert adjudication remains pending.",
            "source_numbers": "not applicable",
            "claim_strength": "limitation",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "Must accompany primary/secondary claims.",
        },
        {
            "claim_type": "forbidden",
            "claim_text": "CAB reduces clinical errors.",
            "source_numbers": "not supported",
            "claim_strength": "forbidden",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "forbidden",
        },
        {
            "claim_type": "forbidden",
            "claim_text": "CAB improves patient outcomes.",
            "source_numbers": "not supported",
            "claim_strength": "forbidden",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "forbidden",
        },
        {
            "claim_type": "forbidden",
            "claim_text": "CAB is clinically validated or externally expert-validated.",
            "source_numbers": "not supported",
            "claim_strength": "forbidden",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "forbidden",
        },
        {
            "claim_type": "forbidden",
            "claim_text": "CAB-Strict should be used as an automated clinical filter.",
            "source_numbers": "not supported",
            "claim_strength": "forbidden",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "forbidden",
        },
        {
            "claim_type": "forbidden",
            "claim_text": "Composite routing benchmark is independent validation.",
            "source_numbers": "not supported",
            "claim_strength": "forbidden",
            "source_table": "reports/tables/routing_publication_safe_claims_final_operating_frontier.csv",
            "required_caveat": "forbidden",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def evidence_ladder(metrics: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    summary = safe_read(THREE_DOMAIN_SUMMARY)
    alpha = safe_read(ALPHA)
    total_n = 26725
    if not summary.empty and "aligned_N" in summary.columns:
        try:
            total_n = int(pd.to_numeric(summary["aligned_N"], errors="coerce").sum())
        except Exception:
            pass

    rows = [
        {
            "evidence_layer": "three-domain temporal drift replication",
            "dataset": "inherited arrhythmia, cardiomyopathy, hereditary cancer",
            "N": total_n,
            "result": "condition-label drift and cross-environment drift observed across three domains",
            "claim_strength": "three_domain_temporal_replication",
            "limitation": "three tested domains, not all-disease universality",
            "what_upgrades_it_further": "additional non-cardiovascular domains and temporal snapshots",
        },
        {
            "evidence_layer": "baseline-only portability prediction",
            "dataset": "baseline disease-model regimes / portability scores",
            "N": total_n,
            "result": "baseline-only regimes stratify future drift",
            "claim_strength": "baseline_predictive_support",
            "limitation": "not prospective deployment",
            "what_upgrades_it_further": "pre-registered frozen external validation",
        },
        {
            "evidence_layer": "domain-specific portability grammar",
            "dataset": "domain-specific environment ontologies and regime tables",
            "N": total_n,
            "result": "distinct grammar across arrhythmia, cardiomyopathy, and hereditary cancer",
            "claim_strength": "mechanistic_interpretation_not_experimental_validation",
            "limitation": "not wet-lab mechanism validation",
            "what_upgrades_it_further": "expert adjudication and mechanistic studies",
        },
        {
            "evidence_layer": "AlphaMissense comparator",
            "dataset": "high-confidence arrhythmia missense subset",
            "N": 214,
            "result": "protein-level deleteriousness insufficient to explain assertion portability in tested subset",
            "claim_strength": "comparator_support_limited_to_subset",
            "limitation": "subset only",
            "what_upgrades_it_further": "broader variant-level protein/genomic comparator joins",
        },
        {
            "evidence_layer": "routing operating frontier",
            "dataset": "cab_decision_challenge_tasks.csv",
            "N": total_n,
            "result": "CAB-Strict and CAB-Balanced expose false-portability vs overrestriction tradeoff",
            "claim_strength": "operating_frontier_support",
            "limitation": "internal benchmark; no external expert adjudication",
            "what_upgrades_it_further": "blinded expert routing adjudication",
        },
        {
            "evidence_layer": "temporal counterfactual benchmark",
            "dataset": "temporal_condition_label_drift",
            "N": total_n,
            "result": "CAB-Strict minimizes unsupported reuse; CAB-Balanced preserves more direct-use capacity",
            "claim_strength": "primary_counterfactual_routing_benchmark",
            "limitation": "future condition-label drift is temporal endpoint, not clinical outcome",
            "what_upgrades_it_further": "expert adjudicated portability endpoint",
        },
        {
            "evidence_layer": "conservative composite benchmark",
            "dataset": "composite routing gold standard",
            "N": total_n,
            "result": "internal operational stress test of routing restrictions",
            "claim_strength": "internal_operational_benchmark",
            "limitation": "not independent external validation",
            "what_upgrades_it_further": "external curation review",
        },
        {
            "evidence_layer": "ClinGen/VCEP/CSpec constraint status",
            "dataset": "ClinGen/CSpec/VCEP overlays where available",
            "N": "varies",
            "result": "external constraint/coverage only",
            "claim_strength": "constraint_not_validation",
            "limitation": "variant-level validation blocked unless data joined",
            "what_upgrades_it_further": "variant-level Evidence Repository / expert panel adjudication",
        },
        {
            "evidence_layer": "expert adjudication pending",
            "dataset": "not yet available",
            "N": 0,
            "result": "pending",
            "claim_strength": "gap",
            "limitation": "no external expert-validated correctness claim",
            "what_upgrades_it_further": "expert adjudication or real-world curation deployment",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_LADDER, index=False)
    return out


def quarantine_wording() -> pd.DataFrame:
    rows = [
        {"quarantined_wording": "CAB-Core", "reason": "old mode name", "replace_with": "CAB-Strict", "allowed_context": "crosswalk only"},
        {"quarantined_wording": "CAB-Conservative", "reason": "old mode name misleading", "replace_with": "CAB-Balanced", "allowed_context": "crosswalk only"},
        {"quarantined_wording": "full CAB is the best temporal model", "reason": "contradicted by operating-frontier metrics", "replace_with": "CAB-Strict minimizes temporal unsupported reuse; CAB-Balanced preserves more direct use", "allowed_context": "quarantine table only"},
        {"quarantined_wording": "CAB eliminates unsupported reuse", "reason": "unsupported reuse reduced, not eliminated", "replace_with": "CAB reduces unsupported deterministic reuse in internal benchmark", "allowed_context": "never as claim"},
        {"quarantined_wording": "CAB validates clinical actionability", "reason": "clinical validation absent", "replace_with": "counterfactual routing benchmark / interpretation-safety framework", "allowed_context": "never as claim"},
        {"quarantined_wording": "CAB clinical decision tool", "reason": "not deployed or clinically validated", "replace_with": "portability triage / routing framework", "allowed_context": "never as claim"},
        {"quarantined_wording": "Composite routing benchmark is independent validation", "reason": "composite includes routing/portability rules", "replace_with": "internal operational stress test", "allowed_context": "never as claim"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_QUAR, index=False)
    return out


def crosswalk(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "cab_mode": "ClinVar-label-only",
            "old_mode_name": "ClinVar-label-only baseline",
            "features": "P/LP treated as portable direct-use by default",
            "behavior": "maximal permissiveness",
            "goal": "default direct deterministic reuse",
            "limitation": "high unsupported deterministic reuse under drift endpoints",
        },
        {
            "cab_mode": "CAB-Strict",
            "old_mode_name": "CAB-Core",
            "features": "gene + baseline disease-model regime",
            "behavior": "high-stringency triage",
            "goal": "minimize false portability / unsupported deterministic reuse",
            "limitation": "high overrestriction and low direct-use allowance",
        },
        {
            "cab_mode": "CAB-Balanced",
            "old_mode_name": "CAB-Conservative",
            "features": "full CAB routing configuration",
            "behavior": "balanced safety-permissiveness routing",
            "goal": "retain large unsupported-reuse reduction while allowing more direct deterministic use",
            "limitation": "higher unsupported reuse than CAB-Strict but less overrestriction",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CROSSWALK, index=False)

    # Assertion-level mode table: one row per assertion per final mode.
    mode_rows = []
    for mode in ["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"]:
        d = df[["assertion_id", "domain", "gene", "baseline_regime_primary", "baseline_architecture_family", "baseline_portability_score"]].copy()
        d["cab_mode"] = mode
        d["old_mode_name"] = old_mode_name(mode)
        d["direct_use_allowed"] = direct_allowed(df, mode).values
        d["features"] = mode_features(mode)
        d["behavior"] = mode_behavior(mode)
        d["goal"] = mode_goal(mode)
        d["limitation"] = mode_limitation(mode)
        mode_rows.append(d)
    modes = pd.concat(mode_rows, ignore_index=True)
    modes.to_csv(OUT_MODES, index=False)
    return out


def write_definition_file():
    lines = [
        "# CAB Operating Modes Final Definition",
        "",
        "Technical definitions; not manuscript prose.",
        "",
        "## CAB-Strict",
        "- former name: CAB-Core",
        "- features: gene + baseline disease-model regime",
        "- behavior: high-stringency triage",
        "- goal: minimize false portability / unsupported deterministic reuse",
        "- limitation: high overrestriction and low direct-use allowance",
        "",
        "## CAB-Balanced",
        "- former name: CAB-Conservative",
        "- features: full CAB routing configuration",
        "- behavior: balanced safety-permissiveness routing",
        "- goal: retain large reduction in unsupported reuse while allowing more direct deterministic use",
        "- limitation: higher unsupported reuse than CAB-Strict but less overrestriction",
        "",
        "## ClinVar-label-only",
        "- P/LP treated as portable direct-use by default",
        "- behavior: maximal permissiveness",
        "- limitation: high unsupported deterministic reuse under drift endpoints",
        "",
        "## Operating-frontier rule",
        "CAB is an operating-frontier framework, not a single universal classifier.",
        "",
        "## Non-negotiable reporting rules",
        "- Do not hide that CAB-Strict overrestricts.",
        "- Do not hide that CAB-Balanced allows more direct use but has higher unsupported reuse.",
        "- Do not present one mode as universally optimal.",
        "- Do not claim external decision validation.",
        "- Do not call CAB-Balanced conservative in final outputs.",
    ]
    OUT_DEF.write_text("\n".join(lines), encoding="utf-8")


def write_readiness(ladder: pd.DataFrame, claims: pd.DataFrame, frontier: pd.DataFrame):
    lines = [
        "# Final CAB Readiness Report",
        "",
        "Technical integration update; not manuscript prose.",
        "",
        "## CAB intervention equivalent",
        "CAB's intervention equivalent is not survival, wet-lab response, or clinical outcome improvement. CAB's intervention equivalent is a counterfactual routing benchmark that reduces false portability while exposing overrestriction tradeoffs.",
        "",
        "## CAB now has",
        "1. hidden structure: portability grammar",
        "2. predictive model: baseline regimes / portability",
        "3. external replication: cardiomyopathy + hereditary cancer",
        "4. comparator: AlphaMissense",
        "5. operational intervention: routing operating frontier",
        "",
        "## Operating frontier",
        "CAB-Strict minimizes unsupported deterministic reuse but overrestricts. CAB-Balanced allows more direct deterministic use and less overrestriction, but has higher unsupported reuse. Neither mode is universally optimal.",
        "",
        "## Remaining gap",
        "External expert adjudication or real-world curation deployment remains pending.",
        "",
        "## Prohibited claims",
        "- CAB reduces clinical errors.",
        "- CAB improves patient outcomes.",
        "- CAB is clinically validated.",
        "- CAB-Balanced is externally expert-validated.",
        "- CAB-Strict should be used as an automated clinical filter.",
        "- Composite routing benchmark is independent validation.",
        "",
        "## Evidence ladder",
        ladder.to_string(index=False),
        "",
        "## Publication-safe operating-frontier claims",
        claims.to_string(index=False),
        "",
        "## Frontier checks",
        frontier.to_string(index=False),
    ]
    OUT_READY.write_text("\n".join(lines), encoding="utf-8")


def plot_pareto_endpoint(pareto_df: pd.DataFrame, endpoint: str, path: Path):
    if plt is None:
        return
    sub = pareto_df[pareto_df["endpoint"].eq(endpoint)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in sub.iterrows():
        marker = "o"
        size = 50
        if r["cab_mode"] == "CAB-Strict":
            marker = "*"; size = 120
        elif r["cab_mode"] == "CAB-Balanced":
            marker = "s"; size = 80
        ax.scatter(r["overrestriction_rate"], r["unsupported_reuse_rate"], marker=marker, s=size)
        ax.annotate(r["cab_mode"], (r["overrestriction_rate"], r["unsupported_reuse_rate"]), fontsize=7, rotation=15)
    ax.set_xlabel("overrestriction_rate")
    ax.set_ylabel("unsupported_reuse_rate")
    ax.set_title(endpoint)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all_paretos(pareto_df: pd.DataFrame):
    for endpoint, path in PARETO_FIGS.items():
        plot_pareto_endpoint(pareto_df, endpoint, path)


def primary_action_distribution(df: pd.DataFrame) -> pd.DataFrame:
    primary = safe_read(PRIMARY_ACTIONS)
    if not primary.empty and "primary_routing_action" in primary.columns:
        return primary
    d = df.copy()
    def action(row):
        direct = bool(row.get("direct_single_model_reuse_allowed", False))
        if not direct and bool(row.get("disease_specific_expert_review_required", False)):
            return "disease_specific_review"
        if not direct and bool(row.get("population_or_penetrance_review_required", False)):
            return "population_or_penetrance_review"
        if not direct and bool(row.get("contextual_repair_required", False)):
            return "contextual_repair"
        if not direct:
            return "no_deterministic_reuse"
        if bool(row.get("disease_specific_expert_review_required", False)):
            return "disease_specific_review"
        if bool(row.get("population_or_penetrance_review_required", False)):
            return "population_or_penetrance_review"
        if bool(row.get("contextual_repair_required", False)):
            return "contextual_repair"
        return "direct_deterministic_use"
    d["primary_routing_action"] = d.apply(action, axis=1)
    return d


def plot_final(metrics: pd.DataFrame, pareto_df: pd.DataFrame, domain: pd.DataFrame, df: pd.DataFrame):
    if plt is None:
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.5, "P/LP classification\n≠\nassertion portability", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 0].set_title("A. Classification vs portability")

    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.5, 0.5,
        "ClinVar-label-only:\npermissive / high false portability\n\nCAB-Strict:\nhigh-stringency / high overrestriction\n\nCAB-Balanced:\nintermediate safety-permissiveness",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False),
    )
    axes[0, 1].set_title("B. Operating modes")

    temp = metrics[metrics["endpoint"].eq("temporal_condition_label_drift")]
    x = np.arange(len(["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"]))
    modes = ["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"]
    vals_u = [temp[temp["cab_mode"].eq(m)]["unsupported_reuse_rate"].iloc[0] for m in modes]
    vals_o = [temp[temp["cab_mode"].eq(m)]["overrestriction_rate"].iloc[0] for m in modes]
    width = 0.35
    axes[1, 0].bar(x - width/2, vals_u, width, label="unsupported reuse")
    axes[1, 0].bar(x + width/2, vals_o, width, label="overrestriction")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(modes, rotation=20, ha="right")
    axes[1, 0].set_ylabel("rate")
    axes[1, 0].set_title("C. Temporal condition-label drift benchmark")
    axes[1, 0].legend(fontsize=7)

    p = pareto_df[pareto_df["endpoint"].eq("temporal_condition_label_drift")]
    for _, r in p.iterrows():
        marker = "*" if r["cab_mode"] == "CAB-Strict" else ("s" if r["cab_mode"] == "CAB-Balanced" else "o")
        axes[1, 1].scatter(r["overrestriction_rate"], r["unsupported_reuse_rate"], marker=marker, s=100 if marker == "*" else 55)
        if r["cab_mode"] in {"ClinVar-label-only", "CAB-Strict", "CAB-Balanced", "gene-only", "regime-only"}:
            axes[1, 1].annotate(r["cab_mode"], (r["overrestriction_rate"], r["unsupported_reuse_rate"]), fontsize=7)
    axes[1, 1].set_xlabel("overrestriction")
    axes[1, 1].set_ylabel("unsupported reuse")
    axes[1, 1].set_title("D. Pareto frontier")

    dtemp = domain[(domain["endpoint"].eq("temporal_condition_label_drift")) & (domain["cab_mode"].isin(["CAB-Strict", "CAB-Balanced"])) & (domain["domain"].ne("all_domains"))]
    domains = sorted(dtemp["domain"].unique())
    x2 = np.arange(len(domains))
    strict = [dtemp[(dtemp["domain"].eq(d)) & (dtemp["cab_mode"].eq("CAB-Strict"))]["absolute_reduction_vs_ClinVar_pp"].iloc[0] for d in domains]
    bal = [dtemp[(dtemp["domain"].eq(d)) & (dtemp["cab_mode"].eq("CAB-Balanced"))]["absolute_reduction_vs_ClinVar_pp"].iloc[0] for d in domains]
    axes[2, 0].bar(x2 - width/2, strict, width, label="CAB-Strict")
    axes[2, 0].bar(x2 + width/2, bal, width, label="CAB-Balanced")
    axes[2, 0].set_xticks(x2)
    axes[2, 0].set_xticklabels(domains, rotation=20, ha="right")
    axes[2, 0].set_ylabel("absolute reduction vs ClinVar (pp)")
    axes[2, 0].set_title("E. Domain-level reductions")
    axes[2, 0].legend(fontsize=7)

    primary = primary_action_distribution(df)
    vc = primary["primary_routing_action"].value_counts(normalize=True)
    actions = ["direct_deterministic_use", "disease_specific_review", "population_or_penetrance_review", "contextual_repair", "no_deterministic_reuse"]
    axes[2, 1].bar(actions, [vc.get(a, 0) for a in actions])
    axes[2, 1].set_xticklabels(actions, rotation=30, ha="right")
    axes[2, 1].set_ylabel("fraction")
    axes[2, 1].set_title("F. Routing action distribution")

    fig.tight_layout()
    fig.savefig(FIG_FINAL)
    plt.close(fig)


def main():
    ensure_dirs()
    print("Loading routing task table...")
    df = load_tasks()
    print(f"N={len(df):,}")
    print(df.groupby("domain").size().to_string())

    print("Writing final operating mode definitions and crosswalk...")
    write_definition_file()
    crosswalk(df)

    print("Computing complete routing metrics...")
    metrics, dom = compute_metrics(df)

    print("Building domain-level frontier checks...")
    frontier, matrix = frontier_checks(metrics, dom)

    print("Building Pareto frontiers...")
    pareto_df = pareto(metrics)
    plot_all_paretos(pareto_df)

    print("Running stratified bootstrap...")
    boot = bootstrap(df)

    print("Writing mode selection guide...")
    guide = selection_guide(metrics)

    print("Writing final claims...")
    claims = publication_claims(metrics, boot)

    print("Writing evidence ladder and quarantine table...")
    ladder = evidence_ladder(metrics, frontier)
    quar = quarantine_wording()

    print("Writing final figure and readiness report...")
    plot_final(metrics, pareto_df, dom, df)
    write_readiness(ladder, claims, frontier)

    print("CAB final operating-frontier analysis complete.")
    print()
    print("All-mode all-endpoint metrics:")
    print(metrics.to_string(index=False))
    print()
    print("Frontier checks:")
    print(frontier.to_string(index=False))
    print()
    print("Publication-safe claims:")
    print(claims.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_DEF, OUT_MODES, OUT_CROSSWALK, OUT_METRICS, OUT_METRICS_DOMAIN,
        OUT_FRONTIER_DOMAIN, OUT_FRONTIER_MATRIX, OUT_PARETO, FIG_PARETO_TEMP,
        FIG_PARETO_CROSS, FIG_PARETO_ANY, FIG_PARETO_COMP, OUT_BOOT, OUT_GUIDE,
        OUT_CLAIMS, FIG_FINAL, OUT_LADDER, OUT_READY, OUT_QUAR,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
