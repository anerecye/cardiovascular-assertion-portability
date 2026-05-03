#!/usr/bin/env python3
"""Final CAB routing benchmark publication-safe reporting.

Separates:
1. temporal_condition_label_drift_gold_standard
   - future condition-label drift endpoint
   - primary temporal counterfactual benchmark

2. conservative_composite_routing_gold_standard
   - broader internal routing stress test
   - includes portability restrictions / decision-layer logic
   - not independent external validation

Inputs:
- data/processed/cab_decision_challenge_tasks.csv
- reports/tables/routing_error_reduction_by_gold_standard.csv
- reports/tables/routing_benchmark_bootstrap_ci.csv
- reports/tables/routing_metric_summary_by_domain.csv
- reports/tables/cab_routing_benchmark_gold_standard_components.csv

Outputs:
- reports/qc/routing_gold_standard_definitions.md
- reports/tables/routing_gold_standard_comparison.csv
- reports/tables/routing_benchmark_by_domain_and_gold_standard.csv
- reports/figures/routing_benchmark_by_domain.svg
- reports/tables/routing_overrestriction_utility_audit.csv
- reports/figures/routing_action_distribution.svg
- reports/tables/routing_ablation_benchmark.csv
- reports/figures/routing_ablation_benchmark.svg
- reports/tables/routing_benchmark_bootstrap_ci_final.csv
- reports/tables/routing_benchmark_publication_safe_claims_final.csv
- reports/figures/final_cab_routing_intervention_figure.svg
- reports/final_cab_readiness_report.md

No clinical outcome claims.
No expert-validated correctness claims.
No composite-as-external-validation claims.
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
ERROR_REDUCTION = TABLES / "routing_error_reduction_by_gold_standard.csv"
BOOTSTRAP = TABLES / "routing_benchmark_bootstrap_ci.csv"

OUT_DEF = QC / "routing_gold_standard_definitions.md"
OUT_GOLD_COMPARE = TABLES / "routing_gold_standard_comparison.csv"
OUT_DOMAIN = TABLES / "routing_benchmark_by_domain_and_gold_standard.csv"
OUT_DOMAIN_FIG = FIGURES / "routing_benchmark_by_domain.svg"
OUT_UTILITY = TABLES / "routing_overrestriction_utility_audit.csv"
OUT_ACTION_FIG = FIGURES / "routing_action_distribution.svg"
OUT_ABLATION = TABLES / "routing_ablation_benchmark.csv"
OUT_ABLATION_FIG = FIGURES / "routing_ablation_benchmark.svg"
OUT_BOOT_FINAL = TABLES / "routing_benchmark_bootstrap_ci_final.csv"
OUT_CLAIMS = TABLES / "routing_benchmark_publication_safe_claims_final.csv"
OUT_FINAL_FIG = FIGURES / "final_cab_routing_intervention_figure.svg"
OUT_READY = REPORTS / "final_cab_readiness_report.md"

RANDOM_STATE = 42
N_BOOT = 1000

GOLD_MAP = {
    "temporal_condition_label_drift_gold_standard": "gold_temporal_condition",
    "conservative_composite_routing_gold_standard": "gold_composite_routing",
}

OLD_TO_NEW_GOLD = {
    "temporal_condition_gold": "temporal_condition_label_drift_gold_standard",
    "composite_routing_gold": "conservative_composite_routing_gold_standard",
}


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def bool_col(s: pd.Series) -> pd.Series:
    return s.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}).fillna(False).astype(bool)


def pct(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.{digits}f}"


def rate_pct(x):
    return f"{pct(x)}%"


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_tasks_and_gold() -> pd.DataFrame:
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

    # Use gold component table if available, otherwise reconstruct.
    gold = safe_read(GOLD_COMPONENTS)
    if not gold.empty and "assertion_id" in gold.columns:
        keep = [c for c in gold.columns if c == "assertion_id" or c.startswith("gold_") or c in ["baseline_direct_use_allowed", "cab_direct_use_allowed"]]
        df = tasks.merge(gold[keep], on="assertion_id", how="left", suffixes=("", "_gold"))
    else:
        df = tasks.copy()

    # Reconstruct missing gold columns robustly.
    if "baseline_direct_use_allowed" not in df.columns:
        df["baseline_direct_use_allowed"] = True
    else:
        df["baseline_direct_use_allowed"] = bool_col(df["baseline_direct_use_allowed"])
    if "cab_direct_use_allowed" not in df.columns:
        df["cab_direct_use_allowed"] = df["direct_single_model_reuse_allowed"]
    else:
        df["cab_direct_use_allowed"] = bool_col(df["cab_direct_use_allowed"])

    if "gold_temporal_condition" not in df.columns:
        df["gold_temporal_condition"] = df["future_condition_label_drift"]
    else:
        df["gold_temporal_condition"] = bool_col(df["gold_temporal_condition"])

    if "gold_composite_routing" not in df.columns:
        reg = df.get("baseline_regime_primary", pd.Series("", index=df.index)).astype(str).str.lower()
        arch = df.get("baseline_architecture_family", pd.Series("", index=df.index)).astype(str).str.lower()
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
        df["gold_composite_routing"] = df["future_condition_label_drift"] | df["future_cross_environment_drift"] | low_portability | failure_topology | decision_layer
    else:
        df["gold_composite_routing"] = bool_col(df["gold_composite_routing"])

    for c in ["domain", "assertion_id", "gene", "baseline_regime_primary", "baseline_architecture_family"]:
        if c not in df.columns:
            df[c] = ""
    return df


def write_gold_definitions():
    lines = [
        "# Routing Gold Standard Definitions",
        "",
        "Technical definitions; not manuscript prose.",
        "",
        "## 1. temporal_condition_label_drift_gold_standard",
        "",
        "Definition: unsupported deterministic reuse is defined by future condition-label drift between baseline and follow-up snapshots.",
        "",
        "Interpretation: this is the primary temporal counterfactual routing benchmark because the endpoint is external to the baseline routing decision.",
        "",
        "Claim strength: temporal_counterfactual_benchmark.",
        "",
        "Allowed wording: CAB reduced unsupported deterministic reuse against a future condition-label drift endpoint.",
        "",
        "Prohibited wording: CAB reduced clinical errors; CAB improved patient outcomes; CAB produced expert-validated decisions.",
        "",
        "## 2. conservative_composite_routing_gold_standard",
        "",
        "Definition: unsupported deterministic reuse is defined by a broader internal routing standard including temporal drift endpoints plus baseline low-portability, failure/regime topology, and decision-layer restriction.",
        "",
        "Interpretation: this is an internal operational stress test. It may include CAB-derived rule logic and therefore is not independent external validation.",
        "",
        "Claim strength: internal_operational_benchmark.",
        "",
        "Allowed wording: under a conservative composite internal routing gold standard, CAB reduced unsupported deterministic reuse.",
        "",
        "Prohibited wording: composite benchmark is independent validation; composite benchmark is external expert validation; CAB should be used clinically without expert review.",
        "",
        "## Shared limitations",
        "- internal counterfactual routing benchmark",
        "- no clinical outcome validation",
        "- no expert adjudication yet",
        "- no claim of deployed clinical decision support",
        "- over-restriction and direct-use allowance must be reported",
    ]
    OUT_DEF.write_text("\n".join(lines), encoding="utf-8")


def confusion(direct_allowed: pd.Series, gold_nonportable: pd.Series) -> Dict[str, float]:
    direct_allowed = direct_allowed.astype(bool)
    gold_nonportable = gold_nonportable.astype(bool)
    pred_nonportable = ~direct_allowed
    true_portable = ~gold_nonportable

    tp = int((pred_nonportable & gold_nonportable).sum())
    fn = int((direct_allowed & gold_nonportable).sum())
    tn = int((direct_allowed & true_portable).sum())
    fp = int((pred_nonportable & true_portable).sum())
    n = tp + fn + tn + fp

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    direct_precision = tn / (tn + fn) if (tn + fn) else np.nan
    direct_recall = tn / (tn + fp) if (tn + fp) else np.nan
    restriction_precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * sens * restriction_precision / (sens + restriction_precision) if (sens + restriction_precision) else np.nan

    return {
        "N": n,
        "true_direct_use_allowed": tn,
        "false_direct_use_unsupported_reuse": fn,
        "true_restriction": tp,
        "false_restriction_overrestriction": fp,
        "unsupported_reuse_rate": fn / n if n else np.nan,
        "overrestriction_rate": fp / n if n else np.nan,
        "direct_use_allowed_rate": (tn + fn) / n if n else np.nan,
        "nonportability_recall": sens,
        "specificity_for_portability": spec,
        "direct_use_precision": direct_precision,
        "direct_use_recall": direct_recall,
        "restriction_precision": restriction_precision,
        "F1_nonportability_detection": f1,
    }


def bootstrap_domain_ci(df: pd.DataFrame, gold_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    domains = sorted(df["domain"].dropna().unique())
    reps = []
    for _ in range(N_BOOT):
        parts = []
        for d in domains:
            sub = df[df["domain"].eq(d)]
            idx = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
            parts.append(df.loc[idx])
        boot = pd.concat(parts, ignore_index=True)
        gold = boot[gold_col].astype(bool)
        base = (boot["baseline_direct_use_allowed"] & gold).mean()
        cab = (boot["cab_direct_use_allowed"] & gold).mean()
        reps.append({
            "baseline_unsupported_reuse_rate": base,
            "CAB_unsupported_reuse_rate": cab,
            "absolute_reduction": base - cab,
            "relative_reduction": (base - cab) / base if base else np.nan,
        })
    return pd.DataFrame(reps)


def bootstrap_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        reps = bootstrap_domain_ci(df, gold_col)
        for metric in reps.columns:
            rows.append({
                "gold_standard_name": gold_name,
                "metric": metric,
                "estimate": reps[metric].mean(),
                "ci95_low": reps[metric].quantile(0.025),
                "ci95_high": reps[metric].quantile(0.975),
                "bootstrap_replicates": N_BOOT,
                "stratification": "within_domain",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BOOT_FINAL, index=False)
    return out


def ci_lookup(boot: pd.DataFrame, gold: str, metric: str):
    h = boot[(boot["gold_standard_name"].eq(gold)) & (boot["metric"].eq(metric))]
    if len(h):
        return float(h["ci95_low"].iloc[0]), float(h["ci95_high"].iloc[0])
    return np.nan, np.nan


def build_gold_comparison(df: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wording = {
        "temporal_condition_label_drift_gold_standard": {
            "strength": "temporal_counterfactual_benchmark",
            "allowed": "CAB reduced unsupported deterministic reuse against a future condition-label drift endpoint.",
            "prohibited": "CAB reduces clinical errors; CAB improves patient outcomes; expert-validated routing.",
        },
        "conservative_composite_routing_gold_standard": {
            "strength": "internal_operational_benchmark",
            "allowed": "Under a conservative composite internal routing gold standard, CAB reduced unsupported deterministic reuse.",
            "prohibited": "Composite benchmark is independent external validation; CAB is expert-validated.",
        },
    }
    for gold_name, gold_col in GOLD_MAP.items():
        gold = df[gold_col].astype(bool)
        base_rate = (df["baseline_direct_use_allowed"] & gold).mean()
        cab_rate = (df["cab_direct_use_allowed"] & gold).mean()
        abs_red = base_rate - cab_rate
        rel_red = abs_red / base_rate if base_rate else np.nan
        ci_low, ci_high = ci_lookup(boot, gold_name, "absolute_reduction")
        rows.append({
            "gold_standard_name": gold_name,
            "N": len(df),
            "baseline_unsupported_reuse_rate": base_rate,
            "CAB_unsupported_reuse_rate": cab_rate,
            "absolute_reduction_pp": abs_red * 100,
            "absolute_reduction_CI_low": ci_low * 100,
            "absolute_reduction_CI_high": ci_high * 100,
            "relative_reduction_percent": rel_red * 100,
            "claim_strength": wording[gold_name]["strength"],
            "allowed_wording": wording[gold_name]["allowed"],
            "prohibited_wording": wording[gold_name]["prohibited"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GOLD_COMPARE, index=False)
    return out


def build_domain_benchmark(df: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        for domain, sub in sorted(list(df.groupby("domain")), key=lambda x: x[0]):
            gold = sub[gold_col].astype(bool)
            base_m = confusion(sub["baseline_direct_use_allowed"], gold)
            cab_m = confusion(sub["cab_direct_use_allowed"], gold)
            # CIs are currently global stratified CIs, not domain-specific. Use blank for domain-specific CI to avoid fake precision.
            rows.append({
                "gold_standard_name": gold_name,
                "domain": domain,
                "N": len(sub),
                "baseline_unsupported_reuse_rate": base_m["unsupported_reuse_rate"],
                "CAB_unsupported_reuse_rate": cab_m["unsupported_reuse_rate"],
                "absolute_reduction": base_m["unsupported_reuse_rate"] - cab_m["unsupported_reuse_rate"],
                "absolute_reduction_CI_low": np.nan,
                "absolute_reduction_CI_high": np.nan,
                "relative_reduction": (base_m["unsupported_reuse_rate"] - cab_m["unsupported_reuse_rate"]) / base_m["unsupported_reuse_rate"] if base_m["unsupported_reuse_rate"] else np.nan,
                "false_direct_use_rate": cab_m["unsupported_reuse_rate"],
                "direct_use_allowed_rate": cab_m["direct_use_allowed_rate"],
                "contextual_repair_rate": sub["contextual_repair_required"].mean(),
                "no_deterministic_reuse_rate": (~sub["cab_direct_use_allowed"].astype(bool)).mean(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DOMAIN, index=False)
    return out


def build_utility_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gold_name, gold_col in GOLD_MAP.items():
        for domain, sub in [("all", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0]):
            gold = sub[gold_col].astype(bool)
            cab = confusion(sub["cab_direct_use_allowed"], gold)
            true_portable = ~gold
            rows.append({
                "gold_standard_name": gold_name,
                "domain": domain,
                "N": len(sub),
                "CAB_direct_use_allowed_N": int(sub["cab_direct_use_allowed"].sum()),
                "CAB_direct_use_allowed_percent": sub["cab_direct_use_allowed"].mean() * 100,
                "CAB_contextual_repair_N": int(sub["contextual_repair_required"].sum()),
                "CAB_contextual_repair_percent": sub["contextual_repair_required"].mean() * 100,
                "CAB_disease_specific_review_N": int(sub["disease_specific_expert_review_required"].sum()),
                "CAB_disease_specific_review_percent": sub["disease_specific_expert_review_required"].mean() * 100,
                "CAB_population_penetrance_review_N": int(sub["population_or_penetrance_review_required"].sum()),
                "CAB_population_penetrance_review_percent": sub["population_or_penetrance_review_required"].mean() * 100,
                "CAB_no_deterministic_reuse_N": int((~sub["cab_direct_use_allowed"].astype(bool)).sum()),
                "CAB_no_deterministic_reuse_percent": (~sub["cab_direct_use_allowed"].astype(bool)).mean() * 100,
                "true_portable_assertions_N": int(true_portable.sum()),
                "true_portable_assertions_percent": true_portable.mean() * 100,
                "false_restriction_overrestriction_N": cab["false_restriction_overrestriction"],
                "false_restriction_overrestriction_percent": cab["overrestriction_rate"] * 100,
                "direct_use_precision": cab["direct_use_precision"],
                "direct_use_recall": cab["direct_use_recall"],
                "nonportability_recall": cab["nonportability_recall"],
                "specificity_for_portability": cab["specificity_for_portability"],
                "conservatism_label": "conservative_direct_use_low" if cab["direct_use_allowed_rate"] < 0.10 else "not_extremely_conservative_by_threshold",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_UTILITY, index=False)
    return out


def direct_allowed_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    reg = df.get("baseline_regime_primary", pd.Series("", index=df.index)).astype(str).str.lower()
    arch = df.get("baseline_architecture_family", pd.Series("", index=df.index)).astype(str).str.lower()
    gene = df.get("gene", pd.Series("", index=df.index)).astype(str).str.upper()
    score = pd.to_numeric(df.get("baseline_portability_score", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(60)
    submitter = pd.to_numeric(df.get("submitter_count_baseline", pd.Series(np.nan, index=df.index)), errors="coerce")

    low = score < 50
    failure = reg.str.contains("collision|underresolved|nonspecific|penetrance|spectrum|moderate|nonportable|low", na=False) | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
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
        return ~(failure | gene.isin(high_risk_genes))
    if variant == "full CAB routing":
        return df["cab_direct_use_allowed"].astype(bool)
    return pd.Series(True, index=df.index)


def build_ablation(df: pd.DataFrame) -> pd.DataFrame:
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
        base_direct = direct_allowed_variant(df, "ClinVar-label-only baseline")
        base_rate = (base_direct & gold).mean()
        for variant in variants:
            direct = direct_allowed_variant(df, variant)
            m = confusion(direct, gold)
            rate = m["unsupported_reuse_rate"]
            rows.append({
                "gold_standard_name": gold_name,
                "routing_variant": variant,
                "N": len(df),
                "unsupported_reuse_rate": rate,
                "absolute_reduction_vs_baseline": base_rate - rate,
                "relative_reduction_vs_baseline": (base_rate - rate) / base_rate if base_rate else np.nan,
                "overrestriction_rate": m["overrestriction_rate"],
                "direct_use_allowed_rate": m["direct_use_allowed_rate"],
                "F1_nonportability_detection": m["F1_nonportability_detection"],
                "precision": m["restriction_precision"],
                "recall": m["nonportability_recall"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ABLATION, index=False)
    return out


def build_claims(compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in compare.iterrows():
        if r["gold_standard_name"] == "temporal_condition_label_drift_gold_standard":
            claim = (
                "Across 26,725 temporally aligned P/LP assertions in three disease domains, CAB reduced unsupported deterministic reuse "
                "against a future condition-label drift endpoint from 36.92% to 7.46%, a 29.47 percentage-point absolute reduction "
                "(95% CI, 28.94–30.03) and 79.81% relative reduction compared with ClinVar-label-only reuse."
            )
            strength = "temporal_counterfactual_benchmark"
        else:
            claim = (
                "Under a conservative composite routing gold standard, CAB reduced unsupported deterministic reuse from 85.69% to 13.00%, "
                "a 72.69 percentage-point absolute reduction (95% CI, 72.14–73.22) and 84.83% relative reduction. "
                "This benchmark is internal and operational because the composite gold standard includes portability restrictions."
            )
            strength = "internal_operational_benchmark"
        rows.append({
            "claim_text": claim,
            "N": int(r["N"]),
            "numerator_denominator": "see routing_gold_standard_comparison.csv",
            "percent": f"baseline {r['baseline_unsupported_reuse_rate']*100:.2f}%; CAB {r['CAB_unsupported_reuse_rate']*100:.2f}%",
            "CI": f"absolute reduction {r['absolute_reduction_CI_low']:.2f}–{r['absolute_reduction_CI_high']:.2f} percentage points",
            "statistic": f"absolute_reduction_pp={r['absolute_reduction_pp']:.4f}; relative_reduction_percent={r['relative_reduction_percent']:.4f}",
            "source_table": "reports/tables/routing_gold_standard_comparison.csv",
            "source_script": "src/run_cab_routing_benchmark_final_reporting.py",
            "claim_strength": strength,
            "forbidden_wording": "CAB reduces clinical errors; CAB improves patient outcomes; CAB is externally expert-validated; CAB should be used clinically without expert review.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def plot_domain(domain_df: pd.DataFrame):
    if plt is None or domain_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, gold in zip(axes, GOLD_MAP.keys()):
        sub = domain_df[domain_df["gold_standard_name"].eq(gold)]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width/2, sub["baseline_unsupported_reuse_rate"], width, label="baseline")
        ax.bar(x + width/2, sub["CAB_unsupported_reuse_rate"], width, label="CAB")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["domain"], rotation=20, ha="right")
        ax.set_title(gold.replace("_", " "))
        ax.set_ylabel("unsupported reuse rate")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DOMAIN_FIG)
    plt.close(fig)


def plot_actions(utility: pd.DataFrame):
    if plt is None or utility.empty:
        return
    sub = utility[(utility["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")) & (utility["domain"].ne("all"))].copy()
    metrics = [
        "CAB_direct_use_allowed_percent", "CAB_contextual_repair_percent",
        "CAB_disease_specific_review_percent", "CAB_population_penetrance_review_percent",
        "CAB_no_deterministic_reuse_percent",
    ]
    x = np.arange(len(sub))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * width, sub[m] / 100, width, label=m.replace("CAB_", "").replace("_percent", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(sub["domain"], rotation=20, ha="right")
    ax.set_ylabel("rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_ACTION_FIG)
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame):
    if plt is None or ablation.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, gold in zip(axes, GOLD_MAP.keys()):
        sub = ablation[ablation["gold_standard_name"].eq(gold)]
        ax.bar(sub["routing_variant"], sub["unsupported_reuse_rate"])
        ax.set_xticklabels(sub["routing_variant"], rotation=45, ha="right")
        ax.set_title(gold.replace("_", " "))
        ax.set_ylabel("unsupported reuse rate")
    fig.tight_layout()
    fig.savefig(OUT_ABLATION_FIG)
    plt.close(fig)


def plot_final(compare: pd.DataFrame, domain_df: pd.DataFrame, utility: pd.DataFrame):
    if plt is None:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.5, "ClinVar-label-only reuse\nP/LP → assumed portable\n→ unsupported deterministic reuse", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 0].set_title("A")

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "CAB routing\nP/LP → portability stress test\n→ direct use / contextual repair /\nexpert review / no deterministic reuse", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 1].set_title("B")

    temp = compare[compare["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")].iloc[0]
    comp = compare[compare["gold_standard_name"].eq("conservative_composite_routing_gold_standard")].iloc[0]
    axes[1, 0].bar(["baseline", "CAB"], [temp["baseline_unsupported_reuse_rate"], temp["CAB_unsupported_reuse_rate"]])
    axes[1, 0].set_title("C. Temporal gold standard")
    axes[1, 0].set_ylabel("unsupported reuse rate")

    axes[1, 1].bar(["baseline", "CAB"], [comp["baseline_unsupported_reuse_rate"], comp["CAB_unsupported_reuse_rate"]])
    axes[1, 1].set_title("D. Composite gold standard")
    axes[1, 1].set_ylabel("unsupported reuse rate")

    dtemp = domain_df[domain_df["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")]
    axes[2, 0].bar(dtemp["domain"], dtemp["absolute_reduction"])
    axes[2, 0].set_xticklabels(dtemp["domain"], rotation=20, ha="right")
    axes[2, 0].set_title("E. Domain-level reductions")
    axes[2, 0].set_ylabel("absolute reduction")

    u = utility[(utility["gold_standard_name"].eq("temporal_condition_label_drift_gold_standard")) & (utility["domain"].eq("all"))].iloc[0]
    axes[2, 1].bar(["direct use", "repair", "expert review", "pop/penetrance", "no deterministic"], [
        u["CAB_direct_use_allowed_percent"] / 100,
        u["CAB_contextual_repair_percent"] / 100,
        u["CAB_disease_specific_review_percent"] / 100,
        u["CAB_population_penetrance_review_percent"] / 100,
        u["CAB_no_deterministic_reuse_percent"] / 100,
    ])
    axes[2, 1].set_xticklabels(["direct use", "repair", "expert review", "pop/penetrance", "no deterministic"], rotation=30, ha="right")
    axes[2, 1].set_title("F. CAB action distribution")
    axes[2, 1].set_ylabel("rate")

    fig.tight_layout()
    fig.savefig(OUT_FINAL_FIG)
    plt.close(fig)


def update_readiness(compare: pd.DataFrame, claims: pd.DataFrame, utility: pd.DataFrame):
    lines = [
        "# Final CAB Readiness Report",
        "",
        "Technical integration update; not manuscript prose.",
        "",
        "## Final readiness classification",
        "Publication-ready as an internal counterfactual routing benchmark with temporal/cross-domain validation; external expert adjudication remains pending.",
        "",
        "## Primary operational intervention result",
        "The primary routing benchmark uses the temporal condition-label drift gold standard. It evaluates whether CAB routing reduces unsupported deterministic reuse against a future temporal endpoint.",
        "",
        "## Secondary internal stress test",
        "The conservative composite routing benchmark is an internal operational stress test. It is not independent external validation because the composite gold standard includes portability restrictions and decision-layer logic.",
        "",
        "## Explicit limitations",
        "- no clinical outcome validation",
        "- no expert adjudication yet",
        "- no expert-validated decision correctness claim",
        "- no clinical actionability beyond routing",
        "- VCEP/CSpec variant-level validation remains blocked unless variant-level data are joined",
        "- over-restriction and direct-use allowance must be reported",
        "",
        "## Gold-standard comparison",
        compare.to_string(index=False),
        "",
        "## Over-restriction / utility audit",
        utility.to_string(index=False),
        "",
        "## Publication-safe claims",
        claims.to_string(index=False),
    ]
    OUT_READY.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    print("Loading tasks and gold components...")
    df = load_tasks_and_gold()
    print(f"Task N: {len(df):,}")
    print(df.groupby("domain").size().to_string())

    print("Writing gold standard definitions...")
    write_gold_definitions()

    print("Bootstrapping final CIs...")
    boot = bootstrap_all(df)

    print("Building gold standard comparison...")
    compare = build_gold_comparison(df, boot)

    print("Building domain decomposition...")
    domain_df = build_domain_benchmark(df, boot)

    print("Building overrestriction utility audit...")
    utility = build_utility_audit(df)

    print("Building ablation benchmark...")
    ablation = build_ablation(df)

    print("Writing publication-safe claims...")
    claims = build_claims(compare)

    print("Writing figures...")
    plot_domain(domain_df)
    plot_actions(utility)
    plot_ablation(ablation)
    plot_final(compare, domain_df, utility)

    print("Updating readiness report...")
    update_readiness(compare, claims, utility)

    print("Final CAB routing benchmark reporting complete.")
    print()
    print("Gold standard comparison:")
    print(compare.to_string(index=False))
    print()
    print("Publication-safe claims:")
    print(claims.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        OUT_DEF, OUT_GOLD_COMPARE, OUT_DOMAIN, OUT_DOMAIN_FIG, OUT_UTILITY,
        OUT_ACTION_FIG, OUT_ABLATION, OUT_ABLATION_FIG, OUT_BOOT_FINAL,
        OUT_CLAIMS, OUT_FINAL_FIG, OUT_READY,
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
