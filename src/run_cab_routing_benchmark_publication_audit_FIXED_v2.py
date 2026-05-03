#!/usr/bin/env python3
"""CAB routing benchmark publication audit FIXED v2.

Purpose
-------
Separate two publication-safe routing definitions that must not be mixed:

1. temporal_condition_gold
   Non-portable = future condition-label drift.
   This reproduces the original headline counterfactual routing benchmark:
   baseline label-only unsupported reuse tracks condition-label drift.

2. composite_routing_gold
   Non-portable = condition drift OR cross-environment drift OR low portability
   OR failure/regime topology OR decision-layer restriction.
   This is a conservative internal routing gold standard.

The previous audit mixed these by computing composite metrics while retaining the
old temporal-only headline claim. This script fixes that by outputting definition-
specific metrics and claims.

Guardrails:
- internal counterfactual routing benchmark only
- no clinical outcome claim
- no expert-validated decision correctness claim
- no clinical actionability beyond routing
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

OUT_DEF = QC / "cab_routing_benchmark_definition.md"
OUT_GOLD = TABLES / "cab_routing_benchmark_gold_standard_components.csv"
OUT_METRICS = TABLES / "routing_metric_summary_by_domain.csv"
OUT_ERR = TABLES / "routing_error_reduction_by_gold_standard.csv"
OUT_CLAIMS = TABLES / "routing_publication_safe_claims.csv"
OUT_BOOT = TABLES / "routing_benchmark_bootstrap_ci.csv"
OUT_FIG = FIGURES / "cab_routing_intervention_benchmark.svg"
OUT_READY = REPORTS / "final_cab_readiness_report.md"

RANDOM_STATE = 42
N_BOOT = 1000


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def bool_col(s):
    return s.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}).fillna(False).astype(bool)


def load_tasks():
    if not TASKS.exists():
        raise FileNotFoundError(f"Missing {TASKS}")
    df = pd.read_csv(TASKS, low_memory=False)
    required = ["domain", "assertion_id", "direct_single_model_reuse_allowed", "future_condition_label_drift", "future_cross_environment_drift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required task columns: {missing}")

    for c in [
        "direct_single_model_reuse_allowed", "cross_environment_reuse_allowed",
        "contextual_repair_required", "disease_specific_expert_review_required",
        "population_or_penetrance_review_required", "high_future_meaning_drift_risk",
        "high_future_cross_environment_drift_risk", "future_condition_label_drift",
        "future_cross_environment_drift", "future_any_meaning_drift", "self_loop_stable",
    ]:
        if c in df.columns:
            df[c] = bool_col(df[c])
        else:
            df[c] = False

    if "future_any_meaning_drift" not in df.columns:
        df["future_any_meaning_drift"] = False
    df["future_any_meaning_drift"] = df["future_any_meaning_drift"] | df["future_condition_label_drift"] | df["future_cross_environment_drift"]

    if "baseline_portability_score" not in df.columns:
        df["baseline_portability_score"] = np.nan
    df["baseline_portability_score"] = pd.to_numeric(df["baseline_portability_score"], errors="coerce")
    if "baseline_nonportability_score" not in df.columns:
        df["baseline_nonportability_score"] = 100 - df["baseline_portability_score"]
    df["baseline_nonportability_score"] = pd.to_numeric(df["baseline_nonportability_score"], errors="coerce")

    for c in ["baseline_regime_primary", "baseline_architecture_family", "gene", "environment_baseline"]:
        if c not in df.columns:
            df[c] = ""

    return df


def add_gold_standards(df):
    out = df.copy()
    reg = out["baseline_regime_primary"].astype(str).str.lower()
    arch = out["baseline_architecture_family"].astype(str).str.lower()

    low_portability = out["baseline_portability_score"].lt(50).fillna(False)
    failure_topology = (
        reg.str.contains("collision|nonportable|low|underresolved|nonspecific|moderate|penetrance|spectrum|recessive|biallelic", na=False)
        | arch.str.contains("collision|underresolved|overlap|spectrum|penetrance", na=False)
    )
    decision_layer = (
        out["contextual_repair_required"]
        | out["disease_specific_expert_review_required"]
        | out["population_or_penetrance_review_required"]
        | (~out["direct_single_model_reuse_allowed"])
    )

    out["baseline_direct_use_allowed"] = True
    out["cab_direct_use_allowed"] = out["direct_single_model_reuse_allowed"]

    out["gold_standard_component_condition_drift"] = out["future_condition_label_drift"]
    out["gold_standard_component_cross_environment_drift"] = out["future_cross_environment_drift"]
    out["gold_standard_component_low_portability"] = low_portability
    out["gold_standard_component_failure_topology"] = failure_topology
    out["gold_standard_component_decision_layer"] = decision_layer
    out["gold_standard_component_other"] = False

    out["gold_temporal_condition"] = out["gold_standard_component_condition_drift"]
    out["gold_temporal_condition_or_cross_environment"] = out["gold_standard_component_condition_drift"] | out["gold_standard_component_cross_environment_drift"]
    out["gold_any_meaning"] = out["future_any_meaning_drift"]
    out["gold_composite_routing"] = (
        out["gold_standard_component_condition_drift"]
        | out["gold_standard_component_cross_environment_drift"]
        | out["gold_standard_component_low_portability"]
        | out["gold_standard_component_failure_topology"]
        | out["gold_standard_component_decision_layer"]
    )

    out["unsupported_reuse_gold_standard"] = out["gold_composite_routing"]
    out["notes"] = "contains multiple gold definitions; temporal_condition reproduces headline benchmark; composite_routing is conservative internal routing gold"

    cols = [
        "assertion_id", "domain",
        "baseline_direct_use_allowed", "cab_direct_use_allowed",
        "unsupported_reuse_gold_standard",
        "gold_temporal_condition",
        "gold_temporal_condition_or_cross_environment",
        "gold_any_meaning",
        "gold_composite_routing",
        "gold_standard_component_condition_drift",
        "gold_standard_component_cross_environment_drift",
        "gold_standard_component_low_portability",
        "gold_standard_component_failure_topology",
        "gold_standard_component_decision_layer",
        "gold_standard_component_other",
        "notes",
    ]
    out[cols].to_csv(OUT_GOLD, index=False)
    return out


def confusion_metrics(direct_allowed, gold):
    pred_nonportable = ~direct_allowed.astype(bool)
    true_nonportable = gold.astype(bool)
    true_portable = ~true_nonportable

    tp = int((pred_nonportable & true_nonportable).sum())
    fn = int((~pred_nonportable & true_nonportable).sum())
    tn = int((~pred_nonportable & true_portable).sum())
    fp = int((pred_nonportable & true_portable).sum())
    n = tp + fn + tn + fp

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    direct_ppv = tn / (tn + fn) if (tn + fn) else np.nan
    restriction_precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * sensitivity * restriction_precision / (sensitivity + restriction_precision) if (sensitivity + restriction_precision) else np.nan

    return {
        "N": n,
        "true_direct_use_allowed": tn,
        "false_direct_use_unsupported_deterministic_reuse": fn,
        "true_restriction": tp,
        "false_restriction_overrestriction": fp,
        "unsupported_deterministic_reuse_rate": fn / n if n else np.nan,
        "overrestriction_rate": fp / n if n else np.nan,
        "direct_use_allowed_rate": (tn + fn) / n if n else np.nan,
        "sensitivity_recall_nonportability_detection": sensitivity,
        "specificity_allowing_portable_assertions": specificity,
        "precision_PPV_direct_use_decisions": direct_ppv,
        "NPV_restriction_decisions": restriction_precision,
        "F1_nonportability_detection": f1,
    }


def compute_metrics(df):
    gold_defs = {
        "temporal_condition_gold": "gold_temporal_condition",
        "temporal_condition_or_cross_environment_gold": "gold_temporal_condition_or_cross_environment",
        "any_meaning_gold": "gold_any_meaning",
        "composite_routing_gold": "gold_composite_routing",
    }
    rows = []
    reductions = []
    for gold_name, gold_col in gold_defs.items():
        for domain, sub in [("all", df)] + sorted(list(df.groupby("domain")), key=lambda x: x[0]):
            gold = sub[gold_col].astype(bool)
            base_m = confusion_metrics(sub["baseline_direct_use_allowed"], gold)
            cab_m = confusion_metrics(sub["cab_direct_use_allowed"], gold)
            for system, m in [("ClinVar_label_only_baseline", base_m), ("CAB", cab_m)]:
                row = {"gold_standard": gold_name, "domain": domain, "system": system}
                row.update(m)
                if system == "CAB":
                    row["repair_recall"] = float((sub["contextual_repair_required"] & gold).sum() / max(1, gold.sum()))
                    row["cross_environment_drift_capture"] = float((sub["high_future_cross_environment_drift_risk"] & sub["future_cross_environment_drift"]).sum() / max(1, sub["future_cross_environment_drift"].sum()))
                    row["condition_drift_capture"] = float((sub["high_future_meaning_drift_risk"] & sub["future_condition_label_drift"]).sum() / max(1, sub["future_condition_label_drift"].sum()))
                else:
                    row["repair_recall"] = 0.0
                    row["cross_environment_drift_capture"] = 0.0
                    row["condition_drift_capture"] = 0.0
                rows.append(row)
            base_rate = base_m["unsupported_deterministic_reuse_rate"]
            cab_rate = cab_m["unsupported_deterministic_reuse_rate"]
            reductions.append({
                "gold_standard": gold_name,
                "domain": domain,
                "N": len(sub),
                "baseline_unsupported_reuse_rate": base_rate,
                "cab_unsupported_reuse_rate": cab_rate,
                "absolute_reduction": base_rate - cab_rate,
                "relative_reduction": (base_rate - cab_rate) / base_rate if base_rate else np.nan,
            })
    metrics = pd.DataFrame(rows)
    err = pd.DataFrame(reductions)
    metrics.to_csv(OUT_METRICS, index=False)
    err.to_csv(OUT_ERR, index=False)
    return metrics, err


def bootstrap(df):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    gold_defs = {
        "temporal_condition_gold": "gold_temporal_condition",
        "composite_routing_gold": "gold_composite_routing",
    }
    domains = sorted(df["domain"].dropna().unique())
    for gold_name, gold_col in gold_defs.items():
        reps = []
        for _ in range(N_BOOT):
            sample_parts = []
            for d in domains:
                sub = df[df["domain"].eq(d)]
                idx = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
                sample_parts.append(df.loc[idx])
            boot = pd.concat(sample_parts, ignore_index=True)
            gold = boot[gold_col].astype(bool)
            base = (boot["baseline_direct_use_allowed"] & gold).mean()
            cab = (boot["cab_direct_use_allowed"] & gold).mean()
            reps.append({
                "baseline_unsupported_reuse_rate": base,
                "cab_unsupported_reuse_rate": cab,
                "absolute_reduction": base - cab,
                "relative_reduction": (base - cab) / base if base else np.nan,
            })
        b = pd.DataFrame(reps)
        for metric in b.columns:
            rows.append({
                "gold_standard": gold_name,
                "metric": metric,
                "estimate": b[metric].mean(),
                "ci95_low": b[metric].quantile(0.025),
                "ci95_high": b[metric].quantile(0.975),
                "bootstrap_replicates": N_BOOT,
                "stratification": "within_domain",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BOOT, index=False)
    return out


def pct(x):
    return f"{100*float(x):.2f}%" if pd.notna(x) else ""


def ci_text(boot, gold, metric):
    h = boot[(boot["gold_standard"].eq(gold)) & (boot["metric"].eq(metric))]
    if not len(h):
        return ""
    return f"{pct(h['ci95_low'].iloc[0])} to {pct(h['ci95_high'].iloc[0])}"


def write_claims(err, boot):
    rows = []
    for gold, label, strength in [
        ("temporal_condition_gold", "temporal condition-label drift gold standard", "temporal_endpoint_supported"),
        ("composite_routing_gold", "conservative composite internal routing gold standard", "internal_counterfactual_benchmark"),
    ]:
        h = err[(err["gold_standard"].eq(gold)) & (err["domain"].eq("all"))].iloc[0]
        n = int(h["N"])
        base_rate = float(h["baseline_unsupported_reuse_rate"])
        cab_rate = float(h["cab_unsupported_reuse_rate"])
        abs_red = float(h["absolute_reduction"])
        rel_red = float(h["relative_reduction"])
        base_n = round(base_rate * n)
        cab_n = round(cab_rate * n)
        rows.append({
            "claim_text": (
                f"Across {n:,} temporally aligned P/LP assertions in three domains, CAB reduced unsupported deterministic reuse "
                f"from {pct(base_rate)} under a ClinVar-label-only baseline to {pct(cab_rate)} using the {label}, "
                f"corresponding to a {pct(abs_red)} absolute reduction and {pct(rel_red)} relative reduction."
            ),
            "N": n,
            "numerator_denominator": f"baseline {base_n}/{n}; CAB {cab_n}/{n}",
            "percent": f"baseline {pct(base_rate)}; CAB {pct(cab_rate)}",
            "CI": (
                f"baseline {ci_text(boot, gold, 'baseline_unsupported_reuse_rate')}; "
                f"CAB {ci_text(boot, gold, 'cab_unsupported_reuse_rate')}; "
                f"absolute reduction {ci_text(boot, gold, 'absolute_reduction')}; "
                f"relative reduction {ci_text(boot, gold, 'relative_reduction')}"
            ),
            "statistic": f"absolute_reduction={abs_red:.6f}; relative_reduction={rel_red:.6f}",
            "source_table": "reports/tables/routing_metric_summary_by_domain.csv; reports/tables/routing_error_reduction_by_gold_standard.csv; reports/tables/routing_benchmark_bootstrap_ci.csv",
            "source_script": "src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py",
            "claim_strength": strength,
        })
    rows.append({
        "claim_text": "CAB routing provides operational routing support, not clinical outcome improvement or expert-validated decision correctness.",
        "N": int(err[err["domain"].eq("all")]["N"].max()) if len(err) else "",
        "numerator_denominator": "not applicable",
        "percent": "not applicable",
        "CI": "not applicable",
        "statistic": "scope limitation",
        "source_table": "reports/qc/cab_routing_benchmark_definition.md",
        "source_script": "src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py",
        "claim_strength": "external_expert_pending",
    })
    rows.append({
        "claim_text": "Claims of expert-validated routing, patient outcome improvement, or clinically actionable decision support are blocked until external expert adjudication exists.",
        "N": int(err[err["domain"].eq("all")]["N"].max()) if len(err) else "",
        "numerator_denominator": "not applicable",
        "percent": "not applicable",
        "CI": "not applicable",
        "statistic": "blocked claim rule",
        "source_table": "reports/tables/routing_publication_safe_claims.csv",
        "source_script": "src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py",
        "claim_strength": "external_expert_pending",
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CLAIMS, index=False)
    return out


def write_definition():
    lines = [
        "# CAB Routing Benchmark Definition",
        "",
        "Technical benchmark definition; not manuscript prose.",
        "",
        "## Key correction",
        "This benchmark reports two separate gold standards and does not mix them.",
        "",
        "## Temporal condition-label gold standard",
        "Unsupported deterministic reuse is defined by future condition-label drift. This reproduces the original headline counterfactual routing benchmark.",
        "",
        "## Conservative composite routing gold standard",
        "Unsupported deterministic reuse is defined by any of: future condition-label drift, future cross-environment drift, low baseline portability, failure/regime topology, or decision-layer restriction.",
        "",
        "## ClinVar-label-only baseline",
        "P/LP is treated as directly portable unless raw label conflict is detected. In this materialized task table, all P/LP assertions are direct-use allowed by baseline.",
        "",
        "## CAB routing",
        "CAB uses baseline portability regime, portability score, disease-model environment, gene/regime architecture, and population/penetrance or expert-review flags where available.",
        "",
        "## False-portable assertion",
        "An assertion allowed for direct deterministic reuse despite the selected internal gold standard marking it non-portable.",
        "",
        "## Internal vs external gold standard",
        "Both current gold standards are internal and rule-adjudicated. Neither is an external expert-adjudicated clinical truth set.",
        "",
        "## Limitations",
        "- counterfactual routing benchmark only",
        "- no clinical outcome improvement claim",
        "- no expert-validated decision correctness claim",
        "- no clinical actionability beyond routing",
        "- external expert adjudication remains pending",
    ]
    OUT_DEF.write_text("\n".join(lines), encoding="utf-8")


def plot_final(err):
    if plt is None or err.empty:
        return
    sub = err[(err["gold_standard"].eq("temporal_condition_gold")) & (err["domain"].ne("all"))].copy()
    comp = err[(err["gold_standard"].eq("composite_routing_gold")) & (err["domain"].ne("all"))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.6, "ClinVar-label-only\nP/LP → direct reuse", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 0].set_title("Panel A")

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.6, "CAB\nP/LP → portability stress test\n→ direct use / repair / review / restriction", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fill=False))
    axes[0, 1].set_title("Panel B")

    x = np.arange(len(sub))
    width = 0.35
    axes[1, 0].bar(x - width/2, sub["baseline_unsupported_reuse_rate"], width, label="baseline")
    axes[1, 0].bar(x + width/2, sub["cab_unsupported_reuse_rate"], width, label="CAB")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sub["domain"], rotation=20, ha="right")
    axes[1, 0].set_title("Panel C: temporal condition gold")
    axes[1, 0].set_ylabel("unsupported reuse rate")
    axes[1, 0].legend()

    x2 = np.arange(len(comp))
    axes[1, 1].bar(x2 - width/2, comp["baseline_unsupported_reuse_rate"], width, label="baseline")
    axes[1, 1].bar(x2 + width/2, comp["cab_unsupported_reuse_rate"], width, label="CAB")
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(comp["domain"], rotation=20, ha="right")
    axes[1, 1].set_title("Panel D: composite routing gold")
    axes[1, 1].set_ylabel("unsupported reuse rate")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)


def update_readiness(claims, boot):
    lines = [
        "# Final CAB Readiness Report",
        "",
        "Technical integration update; not manuscript prose.",
        "",
        "## CAB validation equivalent",
        "CAB's current validation equivalent is counterfactual routing intervention plus temporal/cross-domain validation.",
        "",
        "## Gold-standard correction",
        "The routing benchmark reports two separate internal gold standards: temporal condition-label drift and conservative composite routing. Claims must specify which gold standard is used.",
        "",
        "## Operational intervention outcome",
        "The routing benchmark provides an operational intervention outcome: reduced unsupported deterministic reuse in an internal counterfactual benchmark. It is not a clinical outcome.",
        "",
        "## Remaining missing piece",
        "External expert adjudication or adoption by a curation body remains pending.",
        "",
        "## Non-negotiable limitations",
        "- expert adjudication pending",
        "- VCEP/CSpec variant-level validation blocked unless variant-level data are joined",
        "- quarantined claims remain visible",
        "- no clinical outcome improvement claim",
        "- no expert-validated routing claim",
        "- no clinically actionable decision-system claim beyond routing",
        "",
        "## Publication-safe routing claims",
        claims.to_string(index=False),
        "",
        "## Bootstrap uncertainty",
        boot.to_string(index=False),
    ]
    OUT_READY.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    print("Loading routing task table...")
    tasks = load_tasks()
    print(f"Task rows: {len(tasks):,}")
    print(tasks.groupby("domain").size().to_string())

    print("Adding separate gold standards...")
    df = add_gold_standards(tasks)
    write_definition()

    print("Computing definition-specific routing metrics...")
    metrics, err = compute_metrics(df)

    print("Bootstrapping CIs...")
    boot = bootstrap(df)

    print("Writing publication-safe claims...")
    claims = write_claims(err, boot)

    print("Writing figure and readiness report...")
    plot_final(err)
    update_readiness(claims, boot)

    print("CAB routing benchmark publication audit fixed v2 complete.")
    print()
    print("Error reduction by gold standard:")
    print(err.to_string(index=False))
    print()
    print("Publication-safe claims:")
    print(claims.to_string(index=False))
    print()
    print("Key outputs:")
    for p in [OUT_DEF, OUT_GOLD, OUT_METRICS, OUT_ERR, OUT_BOOT, OUT_CLAIMS, OUT_FIG, OUT_READY]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
