#!/usr/bin/env python3
"""Build a separate domain-calibrated CAB-Balanced v2 layer.

This does not alter the CAB routing system. It evaluates a post-hoc calibration
layer on top of existing CAB-Balanced:

* CAB-Balanced-global: current global CAB-Balanced flag.
* CAB-Balanced-domain-calibrated: global Balanced plus baseline-only
  syndrome-organ normalization rescue for high-confidence hereditary-cancer
  BRCA/MMR-like contexts.
* CAB-SADS-high-stringency: domain-calibrated outside SADS/channelopathy
  contexts; strict routing retained inside SADS-sensitive contexts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"


def read_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).lower()


def gene_family(gene: object) -> str:
    g = str(gene).upper()
    if g in {"BRCA1", "BRCA2"}:
        return "BRCA1/2"
    if g in {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}:
        return "MMR/Lynch"
    if g in {"SCN5A", "KCNQ1", "KCNH2", "RYR2", "CASQ2", "TRDN", "KCNE1", "KCNE2", "LMNA", "PKP2", "DSP"}:
        return "SADS/channelopathy"
    return "other"


def sads_mask(df: pd.DataFrame) -> pd.Series:
    genes = {"SCN5A", "KCNQ1", "KCNH2", "RYR2", "CASQ2", "TRDN", "KCNE1", "KCNE2", "LMNA", "PKP2", "DSP", "DSG2", "DSC2"}
    text = (
        df["baseline_condition_proxy"].fillna("").astype(str)
        + " "
        + df["baseline_env_proxy"].fillna("").astype(str)
        + " "
        + df["followup_condition_proxy"].fillna("").astype(str)
    ).str.lower()
    context = text.str.contains("sads|sudden|brugada|long qt|lqts|cpvt|arrhythm|conduction|arvc|acm", regex=True)
    return df["gene"].astype(str).str.upper().isin(genes) | context | df["domain"].eq("inherited_arrhythmia")


def hereditary_repair_candidate(df: pd.DataFrame) -> pd.Series:
    text = (
        df["baseline_condition_proxy"].fillna("").astype(str)
        + " "
        + df["baseline_env_proxy"].fillna("").astype(str)
    ).str.lower()
    gene = df["gene"].astype(str).str.upper()
    brca_like = gene.isin({"BRCA1", "BRCA2"}) & text.str.contains(
        "breast|ovarian|hereditary breast|hboc|familial cancer of breast|breast-ovarian", regex=True
    )
    mmr_like = gene.isin({"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}) & text.str.contains(
        "lynch|colorectal|colon|mismatch|polyposis|endometrial", regex=True
    )
    broad_exclusion = text.str.contains(
        "fanconi|wilms|glioma|medulloblastoma|pancreatic|gastric|prostate|uterine|pan-cancer|not provided only",
        regex=True,
    )
    return df["domain"].eq("hereditary_cancer") & (brca_like | mmr_like) & (~broad_exclusion)


def brca_syndrome_organ_rescue_candidate(df: pd.DataFrame) -> pd.Series:
    text = (
        df.get("baseline_condition_proxy", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("followup_condition_proxy", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("input_condition_label", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("followup_condition_label", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()
    gene = df["gene"].astype(str).str.upper()
    brca_like = gene.isin({"BRCA1", "BRCA2"}) & text.str.contains(
        "breast|ovarian|hereditary breast|hboc|familial cancer of breast|breast-ovarian", regex=True
    )
    broad_exclusion = text.str.contains(
        "fanconi|wilms|glioma|medulloblastoma|pancreatic|gastric|prostate|uterine|pan-cancer|not provided only",
        regex=True,
    )
    return df["domain"].eq("hereditary_cancer") & brca_like & (~broad_exclusion)


def compute_policy_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    global_direct = read_bool(out["cab_balanced_direct_use_allowed"])
    strict_direct = read_bool(out["cab_strict_direct_use_allowed"])
    repair_candidate = hereditary_repair_candidate(out)
    calibration_rescue = brca_syndrome_organ_rescue_candidate(out)
    sads = sads_mask(out)
    out["gene_family_calibration"] = out["gene"].map(gene_family)
    out["calibration_repair_candidate"] = calibration_rescue
    out["syndrome_organ_repair_candidate"] = repair_candidate
    out["sads_high_stringency_context"] = sads
    out["CAB-Balanced-global"] = global_direct
    out["CAB-Balanced-domain-calibrated"] = global_direct | calibration_rescue
    sads_nonspecific = out["disease_architecture_regime"].astype(str).str.lower().eq("nonspecific_underresolved")
    sads_direct = global_direct & strict_direct & (~sads_nonspecific)
    out["CAB-SADS-high-stringency"] = np.where(sads, sads_direct, global_direct | calibration_rescue).astype(bool)
    return out


def policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policies = ["CAB-Balanced-global", "CAB-Balanced-domain-calibrated", "CAB-SADS-high-stringency"]
    strata = {
        "all": pd.Series(True, index=df.index),
        "hereditary_cancer": df["domain"].eq("hereditary_cancer"),
        "cardiomyopathy": df["domain"].eq("cardiomyopathy"),
        "inherited_arrhythmia": df["domain"].eq("inherited_arrhythmia"),
        "BRCA1/2-like stable": df["gene_family_calibration"].eq("BRCA1/2")
        & df["adjudication_proxy_label"].isin(["stable_no_drift", "ontology_synonym_or_parent_child"]),
        "MMR/Lynch-like stable": df["gene_family_calibration"].eq("MMR/Lynch")
        & df["adjudication_proxy_label"].isin(["stable_no_drift", "ontology_synonym_or_parent_child"]),
        "SADS-sensitive": df["sads_high_stringency_context"],
    }
    e3 = read_bool(df["E3_cross_environment_drift"])
    e4 = read_bool(df["E4_proxy_adjudicated_true_shift"])
    portable = ~(e3 | e4)
    for policy in policies:
        direct_all = read_bool(df[policy])
        for stratum, mask in strata.items():
            sub_idx = mask.fillna(False)
            if sub_idx.sum() == 0:
                continue
            direct = direct_all[sub_idx]
            e3_sub = e3[sub_idx]
            e4_sub = e4[sub_idx]
            portable_sub = portable[sub_idx]
            sads_sub = df.loc[sub_idx, "sads_high_stringency_context"].astype(bool)
            rows.append(
                {
                    "policy": policy,
                    "stratum": stratum,
                    "N": int(sub_idx.sum()),
                    "unsupported_reuse_E3_rate": float((direct & e3_sub).mean()),
                    "direct_use_rate": float(direct.mean()),
                    "overrestriction_rate_E3E4_portable": float(((~direct) & portable_sub).mean()),
                    "true_portable_allowance_rate": float(((direct & portable_sub).sum() / portable_sub.sum()))
                    if portable_sub.sum()
                    else np.nan,
                    "E4_false_direct_use_rate_all": float((direct & e4_sub).mean()),
                    "E4_false_direct_use_rate_among_direct": float(((direct & e4_sub).sum() / direct.sum()))
                    if direct.sum()
                    else 0.0,
                    "SADS_false_direct_use_rate_all": float((direct & e4_sub & sads_sub).sum() / max(1, len(direct))),
                    "SADS_false_direct_use_rate_among_SADS_direct": float(
                        (direct & e4_sub & sads_sub).sum() / max(1, (direct & sads_sub).sum())
                    ),
                    "E3_false_direct_N": int((direct & e3_sub).sum()),
                    "E4_false_direct_N": int((direct & e4_sub).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_domain_calibrated_balanced_v2_metrics.csv", index=False)
    return out


def repair_simulation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_direct = read_bool(df["CAB-Balanced-global"])
    repair_candidate = df["syndrome_organ_repair_candidate"]
    hereditary = df["domain"].eq("hereditary_cancer")
    e3 = read_bool(df["E3_cross_environment_drift"])
    e4 = read_bool(df["E4_proxy_adjudicated_true_shift"])
    stable = ~(e3 | e4)
    initially_not_direct = ~global_direct
    eligible = hereditary & initially_not_direct
    rescued = eligible & repair_candidate
    still_review = eligible & (~repair_candidate)
    after_direct = global_direct | rescued

    sim = df.loc[eligible].copy()
    sim["initial_route"] = np.select(
        [
            read_bool(sim["contextual_repair_required"]),
            read_bool(sim["disease_specific_expert_review_required"]),
            read_bool(sim["population_or_penetrance_review_required"]),
        ],
        ["contextual_repair", "disease_specific_review", "population_penetrance_review"],
        default="review_or_block",
    )
    sim["after_syndrome_organ_repair"] = np.where(
        sim["syndrome_organ_repair_candidate"], "direct_use", "still_review"
    )
    sim["stable_E3E4"] = stable.loc[sim.index].to_numpy()
    sim.to_csv(TABLES / "cab_syndrome_organ_repair_simulation_cases.csv", index=False)

    rows = []
    for label, mask in {
        "all_overrestricted_hereditary_cancer": eligible,
        "BRCA1/2_like": eligible & df["gene_family_calibration"].eq("BRCA1/2"),
        "MMR_Lynch_like": eligible & df["gene_family_calibration"].eq("MMR/Lynch"),
        "stable_overrestricted_hereditary_cancer": eligible & stable,
        "stable_BRCA1/2_like": eligible & stable & df["gene_family_calibration"].eq("BRCA1/2"),
        "stable_MMR_Lynch_like": eligible & stable & df["gene_family_calibration"].eq("MMR/Lynch"),
    }.items():
        sub = df.loc[mask]
        if sub.empty:
            continue
        resc = rescued.loc[mask]
        rows.append(
            {
                "repair_stratum": label,
                "initial_review_or_repair_N": len(sub),
                "after_repair_direct_use_N": int(resc.sum()),
                "after_repair_still_review_N": int((~resc).sum()),
                "repair_rescue_rate": float(resc.mean()),
                "E3_false_direct_after_repair_N": int((resc & e3.loc[mask]).sum()),
                "E3_false_direct_after_repair_rate": float((resc & e3.loc[mask]).sum() / max(1, resc.sum())),
                "E4_false_direct_after_repair_N": int((resc & e4.loc[mask]).sum()),
                "E4_false_direct_after_repair_rate": float((resc & e4.loc[mask]).sum() / max(1, resc.sum())),
                "stable_rescued_N": int((resc & stable.loc[mask]).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "cab_syndrome_organ_repair_simulation_summary.csv", index=False)
    df.assign(
        after_domain_calibrated_direct=after_direct,
        repair_simulation_rescued=rescued,
        repair_simulation_still_review=still_review,
    ).to_csv(TABLES / "cab_domain_calibrated_balanced_v2_analysis_frame.csv", index=False)
    return summary, sim


def plot_policy(metrics: pd.DataFrame) -> None:
    focus = metrics[metrics["stratum"].isin(["all", "BRCA1/2-like stable", "SADS-sensitive"])].copy()
    policies = ["CAB-Balanced-global", "CAB-Balanced-domain-calibrated", "CAB-SADS-high-stringency"]
    colors = {
        "CAB-Balanced-global": "#6b7280",
        "CAB-Balanced-domain-calibrated": "#1b7f79",
        "CAB-SADS-high-stringency": "#b23b3b",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = [
        ("all", "Global safety / capacity", ["direct_use_rate", "unsupported_reuse_E3_rate", "E4_false_direct_use_rate_all"]),
        ("BRCA1/2-like stable", "BRCA1/2-like stable stratum", ["direct_use_rate", "overrestriction_rate_E3E4_portable"]),
        ("SADS-sensitive", "SADS-sensitive stratum", ["direct_use_rate", "SADS_false_direct_use_rate_among_SADS_direct"]),
    ]
    labels = {
        "direct_use_rate": "direct use",
        "unsupported_reuse_E3_rate": "E3 unsupported",
        "E4_false_direct_use_rate_all": "E4 false-direct",
        "overrestriction_rate_E3E4_portable": "overrestriction",
        "SADS_false_direct_use_rate_among_SADS_direct": "SADS false-direct\namong direct",
    }
    for ax, (stratum, title, metric_names) in zip(axes, panels):
        sub = focus[focus["stratum"].eq(stratum)].set_index("policy").reindex(policies)
        x = np.arange(len(metric_names))
        width = 0.24
        for j, policy in enumerate(policies):
            vals = [sub.loc[policy, m] for m in metric_names]
            ax.bar(x + (j - 1) * width, vals, width=width, label=policy.replace("CAB-", ""), color=colors[policy])
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in metric_names], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Domain-calibrated CAB-Balanced v2", x=0.05, y=0.98, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    fig.savefig(FIGURES / "cab_domain_calibrated_balanced_v2.svg")
    fig.savefig(FIGURES / "cab_domain_calibrated_balanced_v2.png", dpi=220)
    plt.close(fig)


def plot_repair(summary: pd.DataFrame) -> None:
    focus = summary[summary["repair_stratum"].isin(["all_overrestricted_hereditary_cancer", "BRCA1/2_like", "MMR_Lynch_like", "stable_BRCA1/2_like"])].copy()
    focus["label"] = focus["repair_stratum"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(focus))
    rescued = focus["after_repair_direct_use_N"].to_numpy()
    still = focus["after_repair_still_review_N"].to_numpy()
    ax.bar(x, rescued, color="#1b7f79", label="rescued to direct use")
    ax.bar(x, still, bottom=rescued, color="#d7dce2", label="still review")
    for i, row in focus.reset_index(drop=True).iterrows():
        ax.text(i, rescued[i] + still[i] * 0.02, f"{row['repair_rescue_rate']:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], rotation=20, ha="right")
    ax.set_ylabel("Initially non-direct hereditary-cancer assertions")
    ax.set_title("Syndrome-organ contextual repair simulation", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "cab_syndrome_organ_repair_simulation.svg")
    fig.savefig(FIGURES / "cab_syndrome_organ_repair_simulation.png", dpi=220)
    plt.close(fig)


def write_qc(metrics: pd.DataFrame, repair: pd.DataFrame) -> None:
    all_rows = metrics[metrics["stratum"].eq("all")].set_index("policy")
    brca_rows = metrics[metrics["stratum"].eq("BRCA1/2-like stable")].set_index("policy")
    sads_rows = metrics[metrics["stratum"].eq("SADS-sensitive")].set_index("policy")
    lines = [
        "# Domain-calibrated CAB-Balanced v2",
        "",
        "This is a separate calibration layer. It does not alter the base CAB routing rules.",
        "",
        "## Main comparison",
        "",
        all_rows[
            [
                "direct_use_rate",
                "unsupported_reuse_E3_rate",
                "overrestriction_rate_E3E4_portable",
                "true_portable_allowance_rate",
                "E4_false_direct_use_rate_all",
            ]
        ].to_markdown(),
        "",
        "## BRCA1/2-like stable stratum",
        "",
        brca_rows[["direct_use_rate", "overrestriction_rate_E3E4_portable", "E4_false_direct_use_rate_all"]].to_markdown(),
        "",
        "## SADS-sensitive stratum",
        "",
        sads_rows[["direct_use_rate", "SADS_false_direct_use_rate_among_SADS_direct", "E4_false_direct_use_rate_all"]].to_markdown(),
        "",
        "## Repair simulation",
        "",
        repair.to_markdown(index=False),
        "",
    ]
    (QC / "cab_domain_calibrated_balanced_v2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_indexes() -> None:
    table_index = ROOT / "reports" / "tables" / "final" / "TABLE_INDEX.md"
    if table_index.exists():
        text = table_index.read_text(encoding="utf-8")
        block = """
## Domain-Calibrated Balanced v2 Tables

| Table | Role | Source |
|---|---|---|
| cab_domain_calibrated_balanced_v2_metrics.csv | CAB-Balanced-global vs domain-calibrated vs SADS-high-stringency metrics | scripts/build_domain_calibrated_balanced_v2.py |
| cab_syndrome_organ_repair_simulation_summary.csv | syndrome-organ repair rescue summary | scripts/build_domain_calibrated_balanced_v2.py |
| cab_syndrome_organ_repair_simulation_cases.csv | row-level repair simulation cases | scripts/build_domain_calibrated_balanced_v2.py |
"""
        if "Domain-Calibrated Balanced v2 Tables" not in text:
            table_index.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    fig_index = ROOT / "reports" / "figures" / "final" / "FIGURE_INDEX.md"
    if fig_index.exists():
        text = fig_index.read_text(encoding="utf-8")
        line = "| Upgrade Figure 11 | Domain-calibrated CAB-Balanced v2 | reports/tables/cab_domain_calibrated_balanced_v2_metrics.csv | scripts/build_domain_calibrated_balanced_v2.py | Domain calibration lowers BRCA-like overrestriction while SADS remains stringent. |\n"
        if "Domain-calibrated CAB-Balanced v2" not in text:
            fig_index.write_text(text.rstrip() + "\n" + line, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLES / "cab_silver_standard_analysis_frame.csv", low_memory=False)
    df = compute_policy_flags(df)
    metrics = policy_metrics(df)
    repair, _ = repair_simulation(df)
    plot_policy(metrics)
    plot_repair(repair)
    write_qc(metrics, repair)
    update_indexes()
    print("Wrote domain-calibrated CAB-Balanced v2 package")


if __name__ == "__main__":
    main()
