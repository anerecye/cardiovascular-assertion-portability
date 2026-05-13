#!/usr/bin/env python3
"""Reviewer-requested repair analyses for the CAB manuscript.

This script adds narrowly scoped analyses requested after the main hardcore
evidence package:

1. 2x2 structural-functional overlap x disease-specific review with OR CI.
2. AlphaMissense selection/observability audit.
3. Domain-split continuous operating frontiers.
4. ClinVar label-drift decomposition for the real inherited-arrhythmia
   temporal alignment table.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

import build_cab_hardcore_evidence_upgrade as hardcore


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"


def read_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def woolf_ci(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05,
    correction: float | None = None,
) -> tuple[float, float, float]:
    """Return OR and Woolf log-normal CI.

    If ``correction`` is provided, add it to all four cells. The manuscript's
    prior OR=82.43 corresponds to a Haldane-Anscombe 0.5 correction on this
    nonzero but quasi-separated 2x2 table.
    """
    cells = np.array([a, b, c, d], dtype=float)
    if correction is not None:
        cells += correction
    elif np.any(cells == 0):
        cells += 0.5
    aa, bb, cc, dd = cells
    odds_ratio = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    z = 1.959963984540054
    lo = math.exp(math.log(odds_ratio) - z * se)
    hi = math.exp(math.log(odds_ratio) + z * se)
    return float(odds_ratio), float(lo), float(hi)


def structural_functional_2x2(df: pd.DataFrame) -> dict[str, float | int | str]:
    exposure = df["disease_architecture_regime"].astype(str).eq("structural_functional_overlap")
    outcome = read_bool(df["disease_specific_expert_review_required"])

    a = int((exposure & outcome).sum())
    b = int((exposure & ~outcome).sum())
    c = int((~exposure & outcome).sum())
    d = int((~exposure & ~outcome).sum())

    table = pd.DataFrame(
        [
            {
                "structural_functional_overlap": "yes",
                "disease_specific_review_yes": a,
                "disease_specific_review_no": b,
                "row_total": a + b,
                "review_rate": a / (a + b) if a + b else float("nan"),
            },
            {
                "structural_functional_overlap": "no",
                "disease_specific_review_yes": c,
                "disease_specific_review_no": d,
                "row_total": c + d,
                "review_rate": c / (c + d) if c + d else float("nan"),
            },
        ]
    )
    table.to_csv(TABLES / "structural_functional_overlap_disease_specific_review_2x2.csv", index=False)

    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    woolf_or, ci_low, ci_high = woolf_ci(a, b, c, d)
    ha_woolf_or, ha_ci_low, ha_ci_high = woolf_ci(a, b, c, d, correction=0.5)
    summary = {
        "analysis": "structural_functional_overlap_x_disease_specific_review",
        "N": int(a + b + c + d),
        "a_exposed_review_yes": a,
        "b_exposed_review_no": b,
        "c_unexposed_review_yes": c,
        "d_unexposed_review_no": d,
        "review_rate_exposed": a / (a + b) if a + b else float("nan"),
        "review_rate_unexposed": c / (c + d) if c + d else float("nan"),
        "fisher_exact_OR": float(fisher_or),
        "fisher_exact_p": float(fisher_p),
        "fisher_exact_OR_uncorrected": float(fisher_or),
        "woolf_OR_uncorrected": woolf_or,
        "woolf_CI95_low_uncorrected": ci_low,
        "woolf_CI95_high_uncorrected": ci_high,
        "reported_Haldane_Anscombe_Woolf_OR": ha_woolf_or,
        "reported_Haldane_Anscombe_Woolf_CI95_low": ha_ci_low,
        "reported_Haldane_Anscombe_Woolf_CI95_high": ha_ci_high,
        "ci_method": "Woolf log-normal CI; reported OR/CI use Haldane-Anscombe 0.5 correction on all four cells",
        "interpretation_boundary": (
            "Routing-rule enrichment for disease-specific review; use as operational/QC evidence, "
            "not as an independent biological effect size."
        ),
    }
    pd.DataFrame([summary]).to_csv(
        TABLES / "structural_functional_overlap_disease_specific_review_ci.csv", index=False
    )
    return summary


def q25(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.25))


def q75(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.75))


def alphamissense_selection_audit() -> pd.DataFrame:
    arr = pd.read_csv(ROOT / "data" / "processed" / "cab_cross_environment_drift.csv")
    qc = pd.read_csv(TABLES / "cab_alphamissense_hg38_join_qc.csv")
    qc_map = dict(zip(qc["metric"], qc["value"]))
    matched_n = int(qc_map.get("rows_high_confidence_join_and_hgvs_p_agreement", 214))
    total_n = int(qc_map.get("cab_rows", len(arr)))
    unmatched_n = total_n - matched_n

    arr["baseline_submitter_count"] = pd.to_numeric(arr.get("baseline_submitter_count"), errors="coerce")
    if arr["baseline_submitter_count"].isna().all() and "number_submitters_2023-01" in arr:
        arr["baseline_submitter_count"] = pd.to_numeric(arr["number_submitters_2023-01"], errors="coerce")
    arr["functional_class_norm"] = arr["functional_class"].fillna("missing").astype(str).str.lower()
    arr["am_feasible_missense_proxy"] = arr["functional_class_norm"].str.contains("missense", na=False)

    proxy_rows: list[dict[str, object]] = []
    for label, sub in [
        ("missense_AM_feasible_proxy", arr[arr["am_feasible_missense_proxy"]]),
        ("non_missense_or_unresolved_proxy", arr[~arr["am_feasible_missense_proxy"]]),
        ("full_arrhythmia_temporal_alignment", arr),
    ]:
        submitters = pd.to_numeric(sub["baseline_submitter_count"], errors="coerce").dropna()
        proxy_rows.append(
            {
                "stratum": label,
                "N": len(sub),
                "baseline_submitter_count_mean": float(submitters.mean()) if len(submitters) else float("nan"),
                "baseline_submitter_count_median": float(submitters.median()) if len(submitters) else float("nan"),
                "baseline_submitter_count_Q1": float(submitters.quantile(0.25)) if len(submitters) else float("nan"),
                "baseline_submitter_count_Q3": float(submitters.quantile(0.75)) if len(submitters) else float("nan"),
                "future_condition_label_drift_rate": float(read_bool(sub["future_condition_label_drift"]).mean()),
                "any_meaning_drift_rate": float(read_bool(sub["any_meaning_drift"]).mean()),
                "cross_environment_drift_rate": float(read_bool(sub["cross_environment_drift"]).mean()),
            }
        )
    proxy = pd.DataFrame(proxy_rows)
    proxy.to_csv(TABLES / "cab_alphamissense_selection_bias_functional_class_proxy.csv", index=False)

    missense_submitters = arr.loc[arr["am_feasible_missense_proxy"], "baseline_submitter_count"].dropna()
    other_submitters = arr.loc[~arr["am_feasible_missense_proxy"], "baseline_submitter_count"].dropna()
    if len(missense_submitters) and len(other_submitters):
        mw = mannwhitneyu(missense_submitters, other_submitters, alternative="two-sided")
        mw_p = float(mw.pvalue)
    else:
        mw_p = float("nan")

    audit = pd.DataFrame(
        [
            {
                "audit_axis": "high_confidence_AlphaMissense_observability",
                "matched_or_observed_N": matched_n,
                "unmatched_or_unobserved_N": unmatched_n,
                "comparison_metric": "row-level AM-score availability",
                "matched_value": "AM score available for high-confidence hg38/HGVS.p matched subset",
                "unmatched_value": "AM score absent or not high-confidence matched",
                "statistic": matched_n / total_n if total_n else float("nan"),
                "p_value": "",
                "interpretation": (
                    "Only 214/942 inherited-arrhythmia rows are eligible for the AlphaMissense comparator; "
                    "the comparator is therefore a restricted missense-subset analysis."
                ),
            },
            {
                "audit_axis": "ClinVar_submission_count_proxy",
                "matched_or_observed_N": int(len(missense_submitters)),
                "unmatched_or_unobserved_N": int(len(other_submitters)),
                "comparison_metric": "missense-feasible proxy vs non-missense/unresolved baseline submitter count",
                "matched_value": (
                    f"median {missense_submitters.median():.2f}, IQR "
                    f"{missense_submitters.quantile(0.25):.2f}-{missense_submitters.quantile(0.75):.2f}"
                )
                if len(missense_submitters)
                else "not available",
                "unmatched_value": (
                    f"median {other_submitters.median():.2f}, IQR "
                    f"{other_submitters.quantile(0.25):.2f}-{other_submitters.quantile(0.75):.2f}"
                )
                if len(other_submitters)
                else "not available",
                "statistic": float(missense_submitters.median() - other_submitters.median())
                if len(missense_submitters) and len(other_submitters)
                else float("nan"),
                "p_value": mw_p,
                "interpretation": (
                    "This is a row-level proxy screen because exact high-confidence matched row IDs were not "
                    "materialized in the current artifact set."
                ),
            },
            {
                "audit_axis": "AM_score_distribution_bias",
                "matched_or_observed_N": matched_n,
                "unmatched_or_unobserved_N": unmatched_n,
                "comparison_metric": "AlphaMissense score distribution in matched vs unmatched rows",
                "matched_value": "not recoverable from current aggregate model/QC tables",
                "unmatched_value": "not defined for rows without high-confidence AM match",
                "statistic": "",
                "p_value": "",
                "interpretation": (
                    "The manuscript should not claim that AM-score matched rows are score-distribution "
                    "representative until row-level AM scores and unmatched join candidates are archived."
                ),
            },
            {
                "audit_axis": "gnomAD_frequency_bias",
                "matched_or_observed_N": total_n,
                "unmatched_or_unobserved_N": "",
                "comparison_metric": "gnomAD allele frequency",
                "matched_value": "not available in local CAB artifacts",
                "unmatched_value": "not available in local CAB artifacts",
                "statistic": "",
                "p_value": "",
                "interpretation": (
                    "ClinVar submitter count is used as the available ascertainment/popularity proxy; "
                    "gnomAD-frequency bias remains an explicit claim boundary."
                ),
            },
        ]
    )
    audit.to_csv(TABLES / "cab_alphamissense_selection_bias_audit.csv", index=False)
    return audit


def endpoint_label(row: pd.Series) -> str:
    if bool(row["label_drift"]) and bool(row["cross_environment_drift_bool"]):
        return "real_environment_shift"
    if bool(row["label_drift"]) and bool(row["submitter_count_change_bool"]):
        return "submitter_change_same_variant_no_environment_shift"
    if bool(row["label_drift"]):
        return "MeSH_OMIM_or_condition_term_rename_no_environment_shift"
    return "no_condition_label_drift"


def clinvar_drift_decomposition() -> pd.DataFrame:
    arr = pd.read_csv(ROOT / "data" / "processed" / "cab_cross_environment_drift.csv")
    arr["condition_label_change_bool"] = read_bool(arr["condition_label_change"])
    arr["future_condition_label_drift_bool"] = read_bool(arr["future_condition_label_drift"])
    arr["label_drift"] = arr["condition_label_change_bool"] | arr["future_condition_label_drift_bool"]
    arr["cross_environment_drift_bool"] = read_bool(arr["cross_environment_drift"])
    arr["submitter_count_change_bool"] = read_bool(arr["submitter_count_change"])
    arr["drift_artifact_category"] = arr.apply(endpoint_label, axis=1)

    total_n = len(arr)
    drift_n = int(arr["label_drift"].sum())
    summary = (
        arr.groupby("drift_artifact_category", dropna=False)
        .agg(
            N=("variation_id", "size"),
            label_drift_N=("label_drift", "sum"),
            cross_environment_drift_N=("cross_environment_drift_bool", "sum"),
            submitter_count_change_N=("submitter_count_change_bool", "sum"),
        )
        .reset_index()
    )
    summary["rate_of_total_N"] = summary["N"] / total_n
    summary["rate_among_label_drift_N"] = summary["label_drift_N"] / drift_n if drift_n else np.nan
    summary["analysis_scope"] = (
        "Inherited-arrhythmia ClinVar temporal alignment only; cancer/cardiomyopathy replay endpoints "
        "do not contain row-level condition-term history in the current artifacts."
    )
    summary.to_csv(TABLES / "clinvar_label_drift_decomposition.csv", index=False)

    case_cols = [
        "variation_id",
        "gene",
        "variant_key",
        "baseline_condition_norm",
        "followup_condition_norm",
        "phenotype_list_2023-01",
        "phenotype_list_2026-04",
        "phenotype_ids_2023-01",
        "phenotype_ids_2026-04",
        "baseline_env",
        "followup_env",
        "baseline_submitter_count",
        "number_submitters_2026-04",
        "drift_artifact_category",
    ]
    existing = [c for c in case_cols if c in arr.columns]
    arr.loc[arr["label_drift"], existing].to_csv(TABLES / "clinvar_label_drift_decomposition_cases.csv", index=False)
    return summary


def mark_frontier_status(frontier: pd.DataFrame) -> pd.DataFrame:
    frontier = frontier.copy()
    frontier["frontier_status"] = "not_evaluated"
    for (group_name, axis_name), sub_idx in frontier.groupby(["domain_group", "threshold_axis"]).groups.items():
        sub = frontier.loc[sub_idx].copy()
        status: list[str] = []
        for idx, row in sub.iterrows():
            other = sub.drop(index=idx)
            dominated = (
                (other["unsupported_reuse"] <= row["unsupported_reuse"])
                & (other["overrestriction"] <= row["overrestriction"])
                & (other["review_burden"] <= row["review_burden"])
                & (
                    (other["unsupported_reuse"] < row["unsupported_reuse"])
                    | (other["overrestriction"] < row["overrestriction"])
                    | (other["review_burden"] < row["review_burden"])
                )
            ).any()
            status.append("dominated" if dominated else "frontier")
        frontier.loc[sub_idx, "frontier_status"] = status
    return frontier


def domain_split_frontier(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = "conservative_composite_non_portability"
    y = df[endpoint].astype(bool).astype(int).to_numpy()
    scores = hardcore.prepare_scores(df, [endpoint])
    regime_score = hardcore.category_rate_score(df["disease_architecture_regime"], y)
    combined = np.clip(
        0.45 * df["cab_risk_score"].to_numpy()
        + 0.35 * regime_score
        + 0.20 * scores[(endpoint, "gene+regime")],
        0,
        1,
    )
    score_map = {
        "direct-use threshold": combined,
        "repair/review threshold": df["cab_risk_score"].to_numpy(),
        "regime score threshold": regime_score,
        "allowed-risk threshold": scores[(endpoint, "gene+regime")],
    }

    groups = {
        "hereditary_cancer": df["domain"].eq("hereditary_cancer"),
        "cardiomyopathy_plus_inherited_arrhythmia": df["domain"].isin(["cardiomyopathy", "inherited_arrhythmia"]),
        "cardiomyopathy": df["domain"].eq("cardiomyopathy"),
        "inherited_arrhythmia": df["domain"].eq("inherited_arrhythmia"),
    }

    rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        sub = df.loc[mask].copy()
        for axis_name, score in score_map.items():
            score_sub = score[mask.to_numpy()]
            for threshold in np.linspace(0, 1, 101):
                direct = score_sub <= threshold
                metrics = hardcore.frontier_metrics(sub, direct, endpoint)
                rows.append(
                    {
                        "domain_group": group_name,
                        "endpoint": endpoint,
                        "N": len(sub),
                        "endpoint_rate": float(sub[endpoint].mean()) if len(sub) else float("nan"),
                        "threshold_axis": axis_name,
                        "threshold": threshold,
                        **metrics,
                    }
                )
        for mode, col in [
            ("CAB-Strict actual mode", "cab_strict_direct_use_allowed"),
            ("CAB-Balanced actual mode", "cab_balanced_direct_use_allowed"),
        ]:
            direct = read_bool(sub[col]).to_numpy()
            metrics = hardcore.frontier_metrics(sub, direct, endpoint)
            rows.append(
                {
                    "domain_group": group_name,
                    "endpoint": endpoint,
                    "N": len(sub),
                    "endpoint_rate": float(sub[endpoint].mean()) if len(sub) else float("nan"),
                    "threshold_axis": mode,
                    "threshold": float("nan"),
                    **metrics,
                }
            )

    frontier = mark_frontier_status(pd.DataFrame(rows))
    frontier.to_csv(TABLES / "cab_domain_split_operating_frontier.csv", index=False)

    compare_rows: list[dict[str, object]] = []
    for group_name in groups:
        sub = frontier[
            (frontier["domain_group"] == group_name)
            & (frontier["threshold_axis"] == "direct-use threshold")
        ].sort_values(["overrestriction", "unsupported_reuse"])
        auc = float(np.trapezoid(sub["unsupported_reuse"], sub["overrestriction"])) if len(sub) else float("nan")
        row: dict[str, object] = {
            "domain_group": group_name,
            "N": int(frontier.loc[frontier["domain_group"] == group_name, "N"].max()),
            "endpoint_rate": float(frontier.loc[frontier["domain_group"] == group_name, "endpoint_rate"].max()),
            "direct_use_threshold_frontier_AUC_unsupported_vs_overrestriction": auc,
        }
        for target_allowance in [0.10, 0.20, 0.30, 0.40]:
            idx = (sub["direct_use_allowance"] - target_allowance).abs().idxmin()
            row[f"unsupported_reuse_at_direct_allowance_{int(target_allowance * 100)}pct"] = float(
                sub.loc[idx, "unsupported_reuse"]
            )
            row[f"overrestriction_at_direct_allowance_{int(target_allowance * 100)}pct"] = float(
                sub.loc[idx, "overrestriction"]
            )
        for mode in ["CAB-Strict actual mode", "CAB-Balanced actual mode"]:
            mode_row = frontier[
                (frontier["domain_group"] == group_name) & (frontier["threshold_axis"] == mode)
            ].iloc[0]
            row[f"{mode}_unsupported_reuse"] = float(mode_row["unsupported_reuse"])
            row[f"{mode}_overrestriction"] = float(mode_row["overrestriction"])
            row[f"{mode}_direct_use_allowance"] = float(mode_row["direct_use_allowance"])
        compare_rows.append(row)
    comparison = pd.DataFrame(compare_rows)
    comparison.to_csv(TABLES / "cab_domain_split_operating_frontier_shape_comparison.csv", index=False)
    plot_domain_split_frontier(frontier)
    return frontier, comparison


def plot_domain_split_frontier(frontier: pd.DataFrame) -> None:
    focus = frontier[
        frontier["domain_group"].isin(["hereditary_cancer", "cardiomyopathy_plus_inherited_arrhythmia"])
        & frontier["threshold_axis"].isin(
            [
                "direct-use threshold",
                "repair/review threshold",
                "regime score threshold",
                "allowed-risk threshold",
                "CAB-Strict actual mode",
                "CAB-Balanced actual mode",
            ]
        )
    ].copy()

    palette = {
        "direct-use threshold": "#126a6a",
        "repair/review threshold": "#5b6f95",
        "regime score threshold": "#9a6b22",
        "allowed-risk threshold": "#8a4f8f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, group_name, title in [
        (axes[0], "hereditary_cancer", "Hereditary cancer"),
        (axes[1], "cardiomyopathy_plus_inherited_arrhythmia", "Cardiomyopathy + inherited arrhythmia"),
    ]:
        sub = focus[focus["domain_group"] == group_name]
        for axis_name, line in sub[~sub["threshold_axis"].str.contains("actual mode")].groupby("threshold_axis"):
            line = line.sort_values("overrestriction")
            ax.plot(
                line["overrestriction"],
                line["unsupported_reuse"],
                label=axis_name,
                color=palette.get(axis_name, "#444444"),
                linewidth=1.8,
                alpha=0.9,
            )
        for mode, marker, color in [
            ("CAB-Strict actual mode", "s", "#1f2937"),
            ("CAB-Balanced actual mode", "o", "#d13f31"),
        ]:
            point = sub[sub["threshold_axis"] == mode]
            if not point.empty:
                ax.scatter(
                    point["overrestriction"],
                    point["unsupported_reuse"],
                    marker=marker,
                    color=color,
                    s=70,
                    zorder=5,
                    label=mode.replace(" actual mode", ""),
                    edgecolor="white",
                    linewidth=0.7,
                )
        n = int(sub["N"].dropna().max()) if not sub.empty else 0
        rate = float(sub["endpoint_rate"].dropna().max()) if not sub.empty else float("nan")
        ax.set_title(f"{title}\nN={n:,}; endpoint rate={rate:.1%}", loc="left", fontweight="bold")
        ax.set_xlabel("Overrestriction")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Unsupported deterministic reuse")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Domain-split CAB operating frontier", x=0.05, y=0.98, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    fig.savefig(FIGURES / "cab_domain_split_operating_frontier.svg")
    fig.savefig(FIGURES / "cab_domain_split_operating_frontier.png", dpi=220)
    plt.close(fig)


def write_qc_summary(
    sf: dict[str, float | int | str],
    alpha_audit: pd.DataFrame,
    drift: pd.DataFrame,
    frontier_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Reviewer-requested CAB methodological repair analyses",
        "",
        "## Structural-functional overlap x disease-specific review",
        "",
        (
            f"2x2 table N={sf['N']:,}; Haldane-Anscombe/Woolf OR="
            f"{sf['reported_Haldane_Anscombe_Woolf_OR']:.2f}; "
            f"95% CI={sf['reported_Haldane_Anscombe_Woolf_CI95_low']:.2f}-"
            f"{sf['reported_Haldane_Anscombe_Woolf_CI95_high']:.2f}; "
            f"Fisher exact p={sf['fisher_exact_p']:.3g}."
        ),
        "",
        "This remains a routing-rule enrichment and is not an independent biological effect size.",
        "",
        "## AlphaMissense selection/observability audit",
        "",
        alpha_audit.to_markdown(index=False),
        "",
        "## ClinVar label-drift decomposition",
        "",
        drift.to_markdown(index=False),
        "",
        "## Domain-split operating frontier",
        "",
        frontier_comparison.to_markdown(index=False),
        "",
    ]
    (QC / "cab_reviewer_methodological_repairs.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    df = hardcore.load_benchmark_rows()
    sf = structural_functional_2x2(df)
    alpha_audit = alphamissense_selection_audit()
    drift = clinvar_drift_decomposition()
    _, frontier_comparison = domain_split_frontier(df)
    write_qc_summary(sf, alpha_audit, drift, frontier_comparison)
    print("Wrote reviewer-response analyses")


if __name__ == "__main__":
    main()
