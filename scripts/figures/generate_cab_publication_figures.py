#!/usr/bin/env python3
"""Generate publication-grade CAB manuscript figures, source tables, and captions.

Run from the repository root:

    python scripts/figures/generate_cab_publication_figures.py

The script is intentionally locked to existing benchmark/report artifacts. It
does not compute new scientific endpoints; it exports publication panels from
the locked tables already present in reports/tables, reports/packages, and
data/processed.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
PACKAGES = ROOT / "reports" / "packages"
FIG_DIR = ROOT / "reports" / "figures" / "final"
SRC_DIR = ROOT / "reports" / "figure_source_tables"
CAPTION_DIR = ROOT / "reports" / "figure_captions"
FINAL_TABLE_DIR = ROOT / "reports" / "tables" / "final"

PALETTE = {
    "inherited_arrhythmia": "#1b9e77",
    "cardiomyopathy": "#4c78a8",
    "hereditary_cancer": "#c44e52",
    "neutral": "#4d4d4d",
    "light": "#f4f6f8",
    "mid": "#9aa4ad",
    "strict": "#5b6ee1",
    "balanced": "#e07b39",
    "clinvar": "#6f6f6f",
    "repair": "#d95f02",
    "review": "#7570b3",
    "direct": "#1b9e77",
}

DOMAIN_LABELS = {
    "inherited_arrhythmia": "Inherited arrhythmia",
    "cardiomyopathy": "Cardiomyopathy",
    "hereditary_cancer": "Hereditary cancer",
}

MODE_LABELS = {
    "ClinVar-label-only": "ClinVar-label-only",
    "CAB-Strict": "CAB-Strict",
    "CAB-Balanced": "CAB-Balanced",
}

FIGURES: list[dict[str, str]] = []


def ensure_dirs() -> None:
    for d in [FIG_DIR, SRC_DIR, CAPTION_DIR, FINAL_TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def as_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def pct(value: object) -> float:
    return as_float(value) * 100.0


def fmt_pct(value: object, digits: int = 2) -> str:
    return f"{pct(value):.{digits}f}%"


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.svg", format="svg", facecolor="white", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", format="png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_caption(stem: str, text: str) -> None:
    caption_text = text.strip() + "\n"
    CAPTION_DIR.joinpath(f"{stem}_caption.md").write_text(caption_text, encoding="utf-8")
    if stem == "Figure1_CAB_framework_overview":
        CAPTION_DIR.joinpath("Figure1_caption.md").write_text(caption_text, encoding="utf-8")


def register_figure(
    figure_number: str,
    title: str,
    panels: str,
    source_tables: str,
    generation_script: str,
    primary_claim: str,
    major_caveat: str,
) -> None:
    FIGURES.append(
        {
            "figure_number": figure_number,
            "title": title,
            "panels": panels,
            "source_tables": source_tables,
            "generation_script": generation_script,
            "primary_claim": primary_claim,
            "major_caveat": major_caveat,
        }
    )


def style_axis(ax: plt.Axes, title: str = "") -> None:
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9dee3", linewidth=0.8, alpha=0.7)
    ax.tick_params(labelsize=8)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str = "#ffffff",
    edge: str = "#4d4d4d",
    fontsize: int = 8,
    radius: float = 0.025,
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return box


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#4d4d4d") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=3, shrinkB=3),
    )


def source_three_domain() -> pd.DataFrame:
    return read_csv(TABLES / "three_domain_portability_summary.csv")


def source_routing_temporal() -> pd.DataFrame:
    df = read_csv(TABLES / "routing_metrics_all_modes_all_endpoints.csv")
    keep = df[
        df["endpoint"].eq("temporal_condition_label_drift")
        & df["cab_mode"].isin(["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"])
    ].copy()
    order = {"ClinVar-label-only": 0, "CAB-Strict": 1, "CAB-Balanced": 2}
    keep["mode_order"] = keep["cab_mode"].map(order)
    return keep.sort_values("mode_order").drop(columns=["mode_order"])


def parse_rolling_manifest() -> pd.DataFrame:
    text = (PACKAGES / "cab_10yr_predictor_repair_package_manifest.md").read_text(encoding="utf-8")
    auroc = {}
    enrich = {}
    for key in ["random", "gene-only", "regime-only", "gene+regime", "all-baseline predictor"]:
        m = re.search(rf"- {re.escape(key)}: ~([0-9.]+)", text)
        if m:
            auroc[key] = float(m.group(1))
    enrichment_block = text.split("Top-10% review queue enrichment:", 1)[1]
    for key in ["random", "gene-only", "regime-only", "gene+regime", "full baseline"]:
        m = re.search(rf"- {re.escape(key)}: ~([0-9.]+)", enrichment_block)
        if m:
            enrich[key] = float(m.group(1))
    delta = re.search(r"mean delta AUROC: ~\+([0-9.]+)", text)
    ci = re.search(r"paired bootstrap CI across held-out origins: ([0-9.]+) to ([0-9.]+)", text)
    rows = []
    model_pairs = [
        ("random", "Random"),
        ("gene-only", "Gene-only"),
        ("regime-only", "Regime-only"),
        ("gene+regime", "Gene + regime"),
        ("all-baseline predictor", "Full baseline"),
    ]
    for key, label in model_pairs:
        rows.append(
            {
                "model": label,
                "locked_manifest_key": key,
                "held_out_AUROC": auroc.get(key, math.nan),
                "AUROC_CI95": "not reported in locked manifest",
                "top10_review_enrichment": enrich.get("full baseline" if key == "all-baseline predictor" else key, math.nan),
                "comparison": "gene-only vs gene+regime" if key == "gene+regime" else "",
                "delta_AUROC": float(delta.group(1)) if key == "gene+regime" and delta else math.nan,
                "delta_CI95_low": float(ci.group(1)) if key == "gene+regime" and ci else math.nan,
                "delta_CI95_high": float(ci.group(2)) if key == "gene+regime" and ci else math.nan,
                "validation_type": "historical prospective emulation / temporal backtest",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SRC_DIR / "Figure6_source_data.csv", index=False)
    return out


def write_final_tables() -> None:
    bench = source_three_domain().copy()
    table1 = pd.DataFrame(
        {
            "domain": bench["domain"].map(DOMAIN_LABELS),
            "N": bench["aligned_N"],
            "condition_label_drift": bench["condition_label_change_rate"].map(fmt_pct),
            "cross_environment_drift": bench["cross_environment_drift_rate"].map(fmt_pct),
            "classification_change": bench["classification_change_rate"].map(fmt_pct),
            "self_loop_stability": bench["self_loop_stable_rate"].map(fmt_pct),
        }
    )
    table1.to_csv(FINAL_TABLE_DIR / "Table1_benchmark_cohort.csv", index=False)

    support = read_csv(TABLES / "disease_architecture_regime_support_levels.csv")
    regimes = read_csv(TABLES / "disease_architecture_portability_regimes_final.csv")
    regimes["regime_key"] = regimes["regime_name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    support["regime_key"] = support["regime_name"]
    merged = support.merge(
        regimes[["regime_key", "dominant_routing_action", "meaning_travel_rule", "publication_safe_interpretation"]],
        on="regime_key",
        how="left",
    )
    table2 = pd.DataFrame(
        {
            "regime": merged["regime_label"],
            "N": merged["N"],
            "support_level": merged["support_level"],
            "dominant_routing": merged["dominant_routing_action"].fillna(""),
            "drift_behavior": merged["meaning_travel_rule"].fillna(merged["key_enrichment"]),
            "claim_strength": merged["allowed_claim"],
        }
    )
    table2.to_csv(FINAL_TABLE_DIR / "Table2_regime_summary.csv", index=False)

    routing = source_routing_temporal()
    table3 = pd.DataFrame(
        {
            "mode": routing["cab_mode"],
            "unsupported_reuse": routing["unsupported_reuse_rate"].map(fmt_pct),
            "overrestriction": routing["overrestriction_rate"].map(fmt_pct),
            "direct_use_allowance": routing["direct_use_allowed_rate"].map(fmt_pct),
            "true_portable_allowance": routing["true_portable_allowed_rate"].map(fmt_pct),
        }
    )
    table3.to_csv(FINAL_TABLE_DIR / "Table3_routing_frontier.csv", index=False)

    taxonomy = read_csv(TABLES / "phenotype_domain_discordance_taxonomy.csv")
    table4 = taxonomy.rename(
        columns={
            "discordance_taxon": "discordance_taxonomy",
            "N": "discordance_taxonomy_N",
            "percent": "discordance_taxonomy_percent",
        }
    ).copy()
    table4.insert(0, "source_matched", 26725)
    table4.insert(1, "meaning_accepted", 26421)
    table4.insert(2, "meaning_rejected", 304)
    table4["discordance_taxonomy_percent"] = table4["discordance_taxonomy_percent"].map(lambda x: fmt_pct(x))
    table4.to_csv(FINAL_TABLE_DIR / "Table4_identity_meaning_concordance.csv", index=False)

    rolling = parse_rolling_manifest()
    table5 = rolling.rename(
        columns={
            "held_out_AUROC": "AUROC",
            "AUROC_CI95": "CI",
            "top10_review_enrichment": "review_enrichment_top10",
            "delta_CI95_low": "comparison_delta_CI95_low",
            "delta_CI95_high": "comparison_delta_CI95_high",
        }
    )
    table5.to_csv(FINAL_TABLE_DIR / "Table5_predictive_modeling.csv", index=False)

    ladder = read_csv(TABLES / "cab_external_validation_proxy_ladder.csv")
    wanted = ["eMERGE", "ClinGen", "PGP", "DiscovEHR", "LOVD", "PhysioNet"]
    mask = ladder["resource"].str.contains("|".join(wanted), case=False, na=False)
    table6 = ladder.loc[mask].copy()
    table6 = table6.rename(
        columns={
            "what it can validate": "supports",
            "what it cannot validate": "does_not_support",
        }
    )[["resource", "supports", "does_not_support", "claim_strength"]]
    table6.to_csv(FINAL_TABLE_DIR / "Table6_external_evidence_proxy_ladder.csv", index=False)


def figure1() -> None:
    routing = source_routing_temporal()
    source_rows = [
        {"panel": "A", "item": "variant identity", "description": "source-level variant identity"},
        {"panel": "A", "item": "P/LP assertion", "description": "public pathogenic/likely pathogenic assertion"},
        {"panel": "A", "item": "disease-model environment", "description": "clinical inference context"},
        {"panel": "A", "item": "disease architecture regime", "description": "architecture-aware portability grammar"},
        {"panel": "A", "item": "CAB routing outcome", "description": "direct use, repair, review, PRF-needed, or no deterministic reuse"},
        {"panel": "B", "item": "source matched", "N": 26725},
        {"panel": "B", "item": "meaning accepted", "N": 26421},
        {"panel": "B", "item": "meaning rejected", "N": 304, "example": "ARR_977320"},
    ]
    for _, row in routing.iterrows():
        source_rows.append(
            {
                "panel": "D",
                "item": row["cab_mode"],
                "unsupported_reuse": row["unsupported_reuse_rate"],
                "direct_use_allowed": row["direct_use_allowed_rate"],
            }
        )
    pd.DataFrame(source_rows).to_csv(SRC_DIR / "Figure1_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
    for ax in axes.flat:
        ax.set_axis_off()

    ax = axes[0, 0]
    panel_label(ax, "A")
    ax.set_title("Variant assertion to CAB routing", loc="left", fontsize=11, fontweight="bold")
    labels = [
        ("Variant\nidentity", "#e8f3ef"),
        ("P/LP\nassertion", "#fff3e8"),
        ("Clinical inference\nenvironment", "#eef2ff"),
        ("Disease architecture\nregime", "#f7eef8"),
        ("CAB routing\noutcome", "#edf7fb"),
    ]
    xs = [0.02, 0.22, 0.43, 0.65, 0.84]
    for i, ((txt, face), x) in enumerate(zip(labels, xs)):
        draw_box(ax, (x, 0.46), 0.15, 0.22, txt, face=face, fontsize=8)
        if i:
            arrow(ax, (xs[i - 1] + 0.15, 0.57), (x, 0.57))
    ax.text(0.02, 0.18, "Question: can this assertion be deterministically reused in this new disease-model environment?", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Identity and meaning are separate layers", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.05, 0.64), 0.28, 0.16, "26,725\nsource matched", "#eef2f5")
    draw_box(ax, (0.39, 0.64), 0.24, 0.16, "gene\nconcordant", "#eef2f5")
    draw_box(ax, (0.71, 0.72), 0.24, 0.14, "26,421\nmeaning accepted", "#e8f3ef")
    draw_box(ax, (0.71, 0.43), 0.24, 0.14, "304\nmeaning rejected", "#fdebec", edge="#c44e52")
    arrow(ax, (0.33, 0.72), (0.39, 0.72))
    arrow(ax, (0.63, 0.72), (0.71, 0.79), PALETTE["direct"])
    arrow(ax, (0.63, 0.68), (0.71, 0.50), PALETTE["repair"])
    ax.text(0.05, 0.18, "ARR_977320: KCNQ1 source/gene match accepted;\nSilver-Russell syndrome phenotype-domain discordant;\nmeaning reuse rejected.", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("Disease architecture governs meaning mobility", loc="left", fontsize=11, fontweight="bold")
    mappings = [
        ("phenotype anchored", "stable", PALETTE["direct"]),
        ("modifier / penetrance boundary", "conditional", "#b07aa1"),
        ("underresolved", "repair", PALETTE["repair"]),
        ("structural overlap", "disease-specific review", PALETTE["review"]),
    ]
    y = 0.74
    for left, right, color in mappings:
        draw_box(ax, (0.05, y), 0.36, 0.12, left, "#ffffff", color, fontsize=8)
        draw_box(ax, (0.58, y), 0.34, 0.12, right, "#ffffff", color, fontsize=8)
        arrow(ax, (0.41, y + 0.06), (0.58, y + 0.06), color)
        y -= 0.18
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_axis_on()
    style_axis(ax, "CAB routing frontier")
    colors = [PALETTE["clinvar"], PALETTE["strict"], PALETTE["balanced"]]
    for (_, row), color in zip(routing.iterrows(), colors):
        ax.scatter(pct(row["unsupported_reuse_rate"]), pct(row["direct_use_allowed_rate"]), s=120, color=color, edgecolor="white", zorder=3)
        ax.text(
            pct(row["unsupported_reuse_rate"]) + 1.3,
            pct(row["direct_use_allowed_rate"]) + 1.3,
            row["cab_mode"],
            fontsize=8,
        )
    ax.set_xlabel("Unsupported reuse (%)", fontsize=9)
    ax.set_ylabel("Direct-use allowance (%)", fontsize=9)
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 106)

    fig.suptitle("Contextual Assertion Biology framework", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "Figure1_CAB_framework_overview")
    write_caption(
        "Figure1_CAB_framework_overview",
        """# Figure 1. CAB conceptual framework

CAB evaluates whether a public pathogenic/likely pathogenic assertion can be reused with the same disease-model meaning after context transfer. Panels show the assertion-to-routing workflow, the separation of source identity from disease meaning using ARR_977320 as a schematic example, disease-architecture rules for meaning mobility, and the CAB operating frontier. CAB measures assertion portability, not variant pathogenicity, penetrance, sudden death risk, or clinical utility.""",
    )
    register_figure(
        "Figure 1",
        "CAB conceptual framework",
        "A: workflow; B: identity versus meaning; C: architecture grammar; D: routing frontier",
        "reports/figure_source_tables/Figure1_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure1",
        "CAB separates variant/source identity, disease meaning, phenotype realization, and routing.",
        "CAB is not a pathogenicity classifier or clinical decision-support system.",
    )


def figure2() -> None:
    df = source_three_domain()
    source = df.copy()
    source.to_csv(SRC_DIR / "Figure2_source_data.csv", index=False)
    colors = [PALETTE[d] for d in df["domain"]]
    labels = [DOMAIN_LABELS[d] for d in df["domain"]]

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    gs = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2]),
    ]

    ax = axes[0]
    panel_label(ax, "A")
    style_axis(ax, "Cohort composition")
    ax.bar(labels, df["aligned_N"].astype(float), color=colors)
    ax.set_ylabel("Assertions")
    ax.tick_params(axis="x", rotation=25)
    ax.text(0.02, 0.92, "Total N = 26,725", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1]
    panel_label(ax, "B")
    style_axis(ax, "Condition-label drift")
    ax.bar(labels, df["condition_label_change_rate"].astype(float) * 100, color=colors)
    ax.set_ylim(0, 45)
    ax.set_ylabel("% assertions")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[2]
    panel_label(ax, "C")
    style_axis(ax, "Cross-environment drift")
    ax.bar(labels, df["cross_environment_drift_rate"].astype(float) * 100, color=colors)
    ax.set_ylim(0, 20)
    ax.set_ylabel("% assertions")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[3]
    panel_label(ax, "D")
    style_axis(ax, "Classification stability versus meaning drift")
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df["classification_change_rate"].astype(float) * 100, width, label="Classification change", color="#9aa4ad")
    ax.bar(x, df["condition_label_change_rate"].astype(float) * 100, width, label="Condition-label drift", color="#4c78a8")
    ax.bar([i + width for i in x], df["any_meaning_drift_rate"].astype(float) * 100, width, label="Any meaning drift", color="#c44e52")
    ax.set_xticks(list(x), labels, rotation=15)
    ax.set_ylabel("% assertions")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left")

    ax = axes[4]
    panel_label(ax, "E")
    style_axis(ax, "Self-loop stability")
    ax.bar(labels, df["self_loop_stable_rate"].astype(float) * 100, color=colors)
    ax.set_ylim(75, 95)
    ax.set_ylabel("% stable self-loop")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Temporal portability benchmark (ClinVar January 2023 to April 2026)", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure2_temporal_portability_benchmark")
    write_caption(
        "Figure2_temporal_portability_benchmark",
        """# Figure 2. Temporal portability benchmark

The CAB benchmark contains 26,725 temporally aligned ClinVar assertions from January 2023 and April 2026 across inherited arrhythmia, cardiomyopathy, and hereditary cancer. Condition-label drift was 38.75%, 38.65%, and 36.43%, while cross-environment drift was 15.50%, 9.86%, and 16.19%, respectively. Classification change remained low relative to disease-meaning drift, showing that classification stability is not equivalent to meaning stability.""",
    )
    register_figure(
        "Figure 2",
        "Temporal portability benchmark",
        "A: cohort composition; B: condition-label drift; C: cross-environment drift; D: classification versus meaning drift; E: self-loop stability",
        "reports/figure_source_tables/Figure2_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure2",
        "Disease meaning can drift even when ClinVar classification is stable.",
        "Temporal alignment is retrospective/historical and not patient outcome validation.",
    )


def figure3() -> None:
    support = read_csv(TABLES / "disease_architecture_regime_support_levels.csv")
    enrich = read_csv(TABLES / "disease_architecture_regime_enrichment_tests.csv")
    source = support.merge(enrich[["regime", "hypothesis", "OR", "CI95_low", "CI95_high", "result"]], left_on="regime_name", right_on="regime", how="left")
    source.to_csv(SRC_DIR / "Figure3_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    ax = axes[0, 0]
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("Regime map", loc="left", fontsize=11, fontweight="bold")
    labels = support["regime_label"].tolist()
    ys = list(reversed([0.08 + i * 0.105 for i in range(len(labels))]))
    for y, label, level in zip(ys, labels, support["support_level"]):
        face = "#e8f3ef" if "strong" in level else ("#fff3e8" if "underpowered" in level else "#f2f2f2")
        draw_box(ax, (0.05, y), 0.84, 0.075, label, face=face, fontsize=7, radius=0.015)
    ax.text(0.05, 0.01, "Trigger-dependent and genotype-first categories are underpowered or unsampled in this benchmark.", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    panel_label(ax, "B")
    style_axis(ax, "Regime support levels")
    support_counts = support.groupby("support_level").size().reset_index(name="count")
    ax.barh(support_counts["support_level"], support_counts["count"], color=["#1b9e77", "#9aa4ad", "#e07b39"][: len(support_counts)])
    ax.set_xlabel("Regime count")
    ax.grid(axis="x", color="#d9dee3")

    ax = axes[1, 0]
    ax.set_axis_off()
    panel_label(ax, "C")
    ax.set_title("Meaning mobility grammar", loc="left", fontsize=11, fontweight="bold")
    grammar = [
        ("stable self-loop", "reuse only in concordant disease model", "#e8f3ef"),
        ("conditional transfer", "risk or context travels with explicit constraints", "#fff3e8"),
        ("boundary crossing", "source identity travels, disease meaning may not", "#fdebec"),
        ("repair-required", "contextual repair or disease-specific review", "#eef2ff"),
    ]
    y = 0.76
    for term, desc, face in grammar:
        draw_box(ax, (0.05, y), 0.28, 0.12, term, face=face, fontsize=8)
        draw_box(ax, (0.43, y), 0.48, 0.12, desc, face="#ffffff", fontsize=8)
        arrow(ax, (0.33, y + 0.06), (0.43, y + 0.06))
        y -= 0.18
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    panel_label(ax, "D")
    style_axis(ax, "Routing enrichment, not patient-level effect size")
    selected = enrich[enrich["OR"].astype(float).round(2).isin([2.76, 82.43, 3.07, 39870.69])].copy()
    selected["ORf"] = selected["OR"].astype(float)
    selected = selected.sort_values("ORf")
    y_pos = range(len(selected))
    ax.errorbar(
        selected["ORf"],
        list(y_pos),
        xerr=[
            selected["ORf"] - selected["CI95_low"].astype(float),
            selected["CI95_high"].astype(float) - selected["ORf"],
        ],
        fmt="o",
        color="#4c78a8",
        ecolor="#9aa4ad",
        capsize=3,
    )
    ax.axvline(1, color="#4d4d4d", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(list(y_pos), selected["regime"].str.replace("_", " ").tolist(), fontsize=8)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.grid(axis="x", color="#d9dee3")

    fig.suptitle("Disease architecture regimes", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure3_disease_architecture_regimes")
    write_caption(
        "Figure3_disease_architecture_regimes",
        """# Figure 3. Disease architecture regimes

CAB represents portability as a disease-architecture problem. Empirically supported regimes include phenotype-anchored, modifier/penetrance-boundary, nonspecific/underresolved, structural-functional overlap, and syndrome-organ boundary classes. Trigger-dependent latent and genotype-first absent-phenotype categories are retained as framework categories but are underpowered or unsampled here. Odds ratios describe routing enrichment within the benchmark, not patient-level effect sizes.""",
    )
    register_figure(
        "Figure 3",
        "Disease architecture regimes",
        "A: regime map; B: support levels; C: mobility grammar; D: enrichment forest plot",
        "reports/figure_source_tables/Figure3_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure3",
        "Portable disease meaning follows disease architecture.",
        "Routing enrichments are not patient-level effect sizes; some regimes are underpowered or unsampled.",
    )


def figure4() -> None:
    taxonomy = read_csv(TABLES / "phenotype_domain_discordance_taxonomy.csv")
    sens = read_csv(TABLES / "identity_meaning_discordance_sensitivity_core.csv")
    case = read_csv(TABLES / "clinvar_identity_vs_meaning_concordance.csv")
    case = case[case["assertion_id"].eq("ARR_977320")]
    source = pd.concat(
        [
            taxonomy.assign(source_section="taxonomy"),
            sens.assign(source_section="sensitivity"),
            case.assign(source_section="ARR_977320"),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(SRC_DIR / "Figure4_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    ax = axes[0, 0]
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("Source matching flow", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.05, 0.58), 0.25, 0.18, "26,725\nsource matched", "#eef2f5")
    draw_box(ax, (0.42, 0.70), 0.25, 0.16, "26,421\nmeaning accepted", "#e8f3ef")
    draw_box(ax, (0.42, 0.40), 0.25, 0.16, "304\nmeaning rejected", "#fdebec", "#c44e52")
    arrow(ax, (0.30, 0.67), (0.42, 0.78), PALETTE["direct"])
    arrow(ax, (0.30, 0.63), (0.42, 0.48), PALETTE["repair"])
    ax.text(0.05, 0.20, "All rows retain source identity; disease meaning reuse is rejected for phenotype-domain discordance.", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    panel_label(ax, "B")
    style_axis(ax, "Discordance taxonomy")
    tax = taxonomy.sort_values("N", key=lambda s: s.astype(float))
    ax.barh(tax["discordance_taxon"].str.replace("_", " "), tax["N"].astype(float), color="#4c78a8")
    ax.set_xlabel("Rows")
    ax.grid(axis="x", color="#d9dee3")

    ax = axes[1, 0]
    ax.set_axis_off()
    panel_label(ax, "C")
    ax.set_title("ARR_977320 case inset", loc="left", fontsize=11, fontweight="bold")
    case_row = case.iloc[0]
    text = (
        "ARR_977320 / ClinVar 977320\n"
        f"Local gene: {case_row['local_gene']}\n"
        f"ClinVar gene field: {case_row['clinvar_gene_symbol']}\n"
        f"ClinVar phenotype: {case_row['clinvar_phenotype_list']}\n\n"
        "Source and gene concordance accepted.\n"
        "Phenotype-domain concordance failed.\n"
        "Meaning reuse routed to repair/review."
    )
    draw_box(ax, (0.06, 0.18), 0.82, 0.64, text, "#ffffff", "#c44e52", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    panel_label(ax, "D")
    style_axis(ax, "Sensitivity: excluding 304 rows")
    metrics = ["condition_label_drift_rate", "cross_environment_drift_rate", "any_meaning_drift_rate", "CAB_Balanced_direct_use_allowed_rate"]
    labels2 = ["Condition drift", "Cross-env drift", "Any meaning drift", "Balanced direct use"]
    full = sens[sens["subset"].eq("full_benchmark")].iloc[0]
    excl = sens[sens["subset"].eq("excluding_304_meaning_rejected")].iloc[0]
    x = range(len(metrics))
    ax.bar([i - 0.18 for i in x], [pct(full[m]) for m in metrics], 0.36, color="#9aa4ad", label="Full")
    ax.bar([i + 0.18 for i in x], [pct(excl[m]) for m in metrics], 0.36, color="#1b9e77", label="Excluding 304")
    ax.set_xticks(list(x), labels2, rotation=20)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Identity versus meaning concordance", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure4_identity_vs_meaning")
    write_caption(
        "Figure4_identity_vs_meaning",
        """# Figure 4. Identity versus meaning concordance

All 26,725 benchmark assertions were source matched, but 304 source-matched rows were meaning rejected because phenotype-domain concordance failed. The ARR_977320 example illustrates a KCNQ1 source/gene match whose ClinVar phenotype label is Silver-Russell syndrome 1, so deterministic inherited-arrhythmia meaning reuse is rejected. Excluding the 304 rows minimally changes aggregate benchmark rates.""",
    )
    register_figure(
        "Figure 4",
        "Identity versus meaning concordance",
        "A: source matching flow; B: discordance taxonomy; C: ARR_977320 inset; D: sensitivity analysis",
        "reports/figure_source_tables/Figure4_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure4",
        "Source identity is necessary but insufficient for deterministic disease-meaning portability.",
        "Meaning rejection does not invalidate ClinVar records and does not reclassify variants.",
    )


def figure5() -> None:
    routing = source_routing_temporal()
    routing.to_csv(SRC_DIR / "Figure5_source_data.csv", index=False)
    colors = [PALETTE["clinvar"], PALETTE["strict"], PALETTE["balanced"]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
    metrics = [
        ("unsupported_reuse_rate", "Unsupported reuse", "A", (0, 42)),
        ("direct_use_allowed_rate", "Direct-use allowance", "B", (0, 105)),
        ("overrestriction_rate", "Overrestriction", "C", (0, 65)),
    ]
    for ax, (col, title, lab, ylim), color_override in zip(axes.flat[:3], metrics, [None, None, None]):
        panel_label(ax, lab)
        style_axis(ax, title)
        ax.bar(routing["cab_mode"], routing[col].astype(float) * 100, color=colors)
        ax.set_ylim(*ylim)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    panel_label(ax, "D")
    style_axis(ax, "Operating frontier")
    for (_, row), color in zip(routing.iterrows(), colors):
        ax.scatter(pct(row["unsupported_reuse_rate"]), pct(row["direct_use_allowed_rate"]), s=140, color=color, edgecolor="white", zorder=3)
        ax.text(pct(row["unsupported_reuse_rate"]) + 1, pct(row["direct_use_allowed_rate"]) + 1, row["cab_mode"], fontsize=8)
    ax.set_xlabel("Unsupported reuse (%)")
    ax.set_ylabel("Direct-use allowance (%)")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 106)

    fig.suptitle("CAB routing frontier", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure5_routing_frontier")
    write_caption(
        "Figure5_routing_frontier",
        """# Figure 5. CAB routing frontier

CAB defines an operating frontier rather than a single binary classifier. In the temporal condition-label drift benchmark, CAB-Strict reduced unsupported deterministic reuse from 36.92% to 2.42% but allowed direct use for 8.09% of assertions. CAB-Balanced reduced unsupported reuse to 7.46% while allowing direct use for 27.31% of assertions and 31.48% of true portable assertions. External expert adjudication remains pending.""",
    )
    register_figure(
        "Figure 5",
        "CAB routing frontier",
        "A: unsupported reuse; B: direct-use allowance; C: overrestriction; D: frontier plot",
        "reports/figure_source_tables/Figure5_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure5",
        "CAB routing trades off unsupported reuse against direct-use allowance.",
        "Routing correctness is internally benchmarked; external expert adjudication remains pending.",
    )


def figure6() -> None:
    rolling = parse_rolling_manifest()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
    comp = rolling[rolling["model"].isin(["Gene-only", "Regime-only", "Gene + regime", "Full baseline"])]

    ax = axes[0, 0]
    panel_label(ax, "A")
    style_axis(ax, "Held-out AUROC comparison")
    ax.bar(comp["model"], comp["held_out_AUROC"].astype(float), color=["#9aa4ad", "#4c78a8", "#1b9e77", "#c44e52"])
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    panel_label(ax, "B")
    style_axis(ax, "Incremental value")
    row = rolling[rolling["model"].eq("Gene + regime")].iloc[0]
    delta = float(row["delta_AUROC"])
    low = float(row["delta_CI95_low"])
    high = float(row["delta_CI95_high"])
    ax.bar(["Gene-only to gene+regime"], [delta], color="#1b9e77")
    ax.errorbar([0], [delta], yerr=[[delta - low], [high - delta]], fmt="none", ecolor="#4d4d4d", capsize=5)
    ax.axhline(0, color="#4d4d4d", linewidth=1)
    ax.set_ylabel("Delta AUROC")
    ax.set_ylim(0, 0.20)
    ax.text(0, delta + 0.025, "+0.1383\n95% CI 0.1066-0.1627", ha="center", fontsize=8)

    ax = axes[1, 0]
    panel_label(ax, "C")
    style_axis(ax, "Top-10% review enrichment")
    ax.bar(rolling["model"], rolling["top10_review_enrichment"].astype(float), color=["#9aa4ad", "#9aa4ad", "#4c78a8", "#1b9e77", "#c44e52"])
    ax.set_ylabel("Enrichment")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    ax.set_axis_off()
    panel_label(ax, "D")
    ax.set_title("Rolling-origin schematic", loc="left", fontsize=11, fontweight="bold")
    xs = [0.08, 0.27, 0.46, 0.65, 0.84]
    years = ["2015", "2018", "2021", "2024", "2026"]
    for x, year in zip(xs, years):
        ax.plot([x, x], [0.35, 0.55], color="#4d4d4d", lw=1)
        ax.text(x, 0.30, year, ha="center", fontsize=8)
    ax.plot([xs[0], xs[-1]], [0.45, 0.45], color="#4d4d4d", lw=1.4)
    arrow(ax, (0.20, 0.62), (0.36, 0.62), PALETTE["strict"])
    arrow(ax, (0.40, 0.62), (0.56, 0.62), PALETTE["strict"])
    arrow(ax, (0.60, 0.62), (0.76, 0.62), PALETTE["strict"])
    ax.text(0.08, 0.72, "baseline-only features -> later snapshot endpoints", fontsize=8)
    draw_box(
        ax,
        (0.10, 0.05),
        0.78,
        0.16,
        "Temporal backtesting / historical prospective emulation only.",
        "#fdebec",
        "#c44e52",
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Predictive modeling and portability forecasting", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure6_predictive_modeling")
    write_caption(
        "Figure6_predictive_modeling",
        """# Figure 6. Predictive modeling and portability forecasting

Rolling-origin historical prospective emulation showed held-out AUROCs of 0.5781 for gene-only, 0.6998 for regime-only, 0.7164 for gene+regime, and 0.7419 for the full baseline predictor. Gene+regime improved over gene-only by +0.1383 AUROC (paired bootstrap CI 0.1066-0.1627). This is temporal backtesting only and is not prospective clinical validation.""",
    )
    register_figure(
        "Figure 6",
        "Predictive modeling and portability forecasting",
        "A: AUROC comparison; B: incremental value; C: top-10% review enrichment; D: rolling-origin schematic",
        "reports/figure_source_tables/Figure6_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure6",
        "Disease architecture adds temporal forecasting information beyond gene identity.",
        "Historical prospective emulation is not true prospective clinical validation.",
    )


def figure7() -> None:
    alpha = read_csv(TABLES / "cab_alphamissense_model_comparison.csv")
    alpha.to_csv(SRC_DIR / "Figure7_source_data.csv", index=False)
    endpoint = alpha[alpha["endpoint"].eq("future_condition_label_drift")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white")
    ax = axes[0]
    panel_label(ax, "A")
    style_axis(ax, "Single-source comparators")
    single = endpoint[endpoint["model"].isin(["AlphaMissense-only", "CAB-only", "gene-only"])]
    ax.bar(single["model"], single["AUROC"].astype(float), color=["#9aa4ad", "#1b9e77", "#4c78a8"])
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    panel_label(ax, "B")
    style_axis(ax, "Combined models")
    combo = endpoint[endpoint["model"].isin(["gene+AlphaMissense", "CAB+AlphaMissense", "gene+CAB+AlphaMissense"])]
    ax.bar(combo["model"], combo["AUROC"].astype(float), color=["#4c78a8", "#1b9e77", "#c44e52"])
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[2]
    ax.set_axis_off()
    panel_label(ax, "C")
    ax.set_title("Conceptual distinction", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.08, 0.58), 0.34, 0.22, "AlphaMissense\nprotein damage", "#eef2f5", fontsize=9)
    draw_box(ax, (0.58, 0.58), 0.34, 0.22, "CAB\nassertion portability", "#e8f3ef", fontsize=9)
    ax.text(0.08, 0.26, "Molecular deleteriousness", fontsize=8)
    ax.text(0.58, 0.26, "Disease-model meaning reuse", fontsize=8)
    ax.text(0.20, 0.08, "Portability is not reducible to molecular deleteriousness.", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("AlphaMissense comparator", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "Figure7_alphamissense_comparator")
    write_caption(
        "Figure7_alphamissense_comparator",
        """# Figure 7. AlphaMissense comparator

In the high-confidence inherited-arrhythmia missense subset, CAB-only and gene-aware CAB models outperformed AlphaMissense-only for assertion portability endpoints. The comparison is a mechanistic negative-control style analysis: protein-level deleteriousness and disease-model assertion portability are related but non-equivalent quantities.""",
    )
    register_figure(
        "Figure 7",
        "AlphaMissense comparator",
        "A: AlphaMissense-only versus CAB-only versus gene-only; B: combined models; C: conceptual distinction",
        "reports/figure_source_tables/Figure7_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure7",
        "Portability is not reducible to molecular deleteriousness.",
        "AlphaMissense comparison is subset-limited and not clinical validation.",
    )


def figure8() -> None:
    use_cases = read_csv(TABLES / "sads_cab_portability_use_cases.csv")
    use_cases.to_csv(SRC_DIR / "Figure8_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    ax = axes[0, 0]
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("Context transfer chain", loc="left", fontsize=11, fontweight="bold")
    chain = ["arrhythmia\nassertion", "postmortem\ninterpretation", "family-risk\ninterpretation", "genotype-first\nphenotype-negative\nrelative"]
    xs = [0.05, 0.31, 0.57, 0.82]
    for i, (x, txt) in enumerate(zip(xs, chain)):
        draw_box(ax, (x, 0.50), 0.16, 0.22, txt, "#ffffff", "#4c78a8", fontsize=8)
        if i:
            arrow(ax, (xs[i - 1] + 0.16, 0.61), (x, 0.61))
    ax.text(0.05, 0.20, "Each transfer can change disease-model meaning.", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.set_axis_off()
    panel_label(ax, "B")
    ax.set_title("CAB routing", loc="left", fontsize=11, fontweight="bold")
    actions = ["direct reuse", "context review", "disease-specific review", "PRF-needed", "no deterministic reuse"]
    y = 0.78
    for action in actions:
        face = "#e8f3ef" if action == "direct reuse" else "#fff3e8" if "review" in action else "#fdebec"
        draw_box(ax, (0.15, y), 0.70, 0.105, action, face, fontsize=8)
        y -= 0.14
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    ax.set_axis_off()
    panel_label(ax, "C")
    ax.set_title("Forbidden overclaim boundary", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.07, 0.62), 0.28, 0.14, "variant\npathogenicity", "#eef2f5")
    draw_box(ax, (0.39, 0.62), 0.22, 0.14, "cause\nof death", "#fdebec", "#c44e52")
    draw_box(ax, (0.65, 0.62), 0.28, 0.14, "family-member\nrisk prediction", "#fdebec", "#c44e52")
    ax.text(0.36, 0.68, "!=", fontsize=13, fontweight="bold")
    ax.text(0.62, 0.68, "!=", fontsize=13, fontweight="bold")
    draw_box(ax, (0.10, 0.18), 0.80, 0.18, "CAB does not infer cause of death or predict family-member risk.", "#ffffff", "#c44e52", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.set_axis_off()
    panel_label(ax, "D")
    ax.set_title("Brugada/SADS portability boundary", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.08, 0.64), 0.35, 0.15, "SCN5A / Brugada\nassertion", "#eef2ff", fontsize=8)
    draw_box(ax, (0.57, 0.64), 0.35, 0.15, "SADS/postmortem\nmeaning", "#fff3e8", fontsize=8)
    arrow(ax, (0.43, 0.71), (0.57, 0.71), "#c44e52")
    ax.text(0.24, 0.48, "trigger context\nphenotype context\ndisease-specific curation\npenetrance framing", fontsize=8, ha="center")
    draw_box(ax, (0.08, 0.10), 0.84, 0.18, "SADS is a portability stress-test/use-case, not a validated CAB clinical endpoint.", "#fdebec", "#c44e52", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("SADS/postmortem portability stress-test", fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "Figure8_SADS_portability")
    write_caption(
        "Figure8_SADS_portability",
        """# Figure 8. SADS/postmortem portability stress-test

SADS is treated as a high-value portability stress-test because assertions may be transferred from inherited-arrhythmia classification into postmortem causal interpretation, family-risk interpretation, and genotype-first phenotype-negative relatives. CAB routes these transfers to direct reuse, contextual review, disease-specific review, PRF-needed, or no deterministic reuse. CAB does not infer cause of death, predict sudden death, predict family-member risk, or validate SADS risk.""",
    )
    register_figure(
        "Figure 8",
        "SADS/postmortem portability stress-test",
        "A: context transfer chain; B: CAB routing; C: forbidden overclaim boundary; D: Brugada/SADS boundary",
        "reports/figure_source_tables/Figure8_source_data.csv",
        "scripts/figures/generate_cab_publication_figures.py::figure8",
        "SADS is a stress-test for portability theory across postmortem and family-risk contexts.",
        "CAB does not infer cause of death, predict sudden death, or predict family-member risk.",
    )


def supplementary_figures() -> None:
    supplementary_cardiomyopathy_leakage()
    supplementary_rolling_origin()
    supplementary_routing_sensitivity()
    supplementary_regime_forest()
    supplementary_alphamissense_qc()
    supplementary_identity_repair()
    supplementary_external_evidence()
    supplementary_sads_grammar()


def supplementary_cardiomyopathy_leakage() -> None:
    audit = read_csv(TABLES / "cardiomyopathy_regime_leakage_audit.csv")
    models = read_csv(TABLES / "cardiomyopathy_model_comparison_baseline_only.csv")
    src = pd.concat([audit.assign(source_section="leakage_audit"), models.assign(source_section="model_comparison")], ignore_index=True, sort=False)
    src.to_csv(SRC_DIR / "SupplementaryFigure1_source_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
    ax = axes[0]
    style_axis(ax, "Leakage audit actions")
    counts = audit.groupby("action").size().sort_values()
    ax.barh(counts.index, counts.values, color="#4c78a8")
    ax.set_xlabel("Feature count")
    ax.grid(axis="x", color="#d9dee3")
    ax = axes[1]
    style_axis(ax, "Baseline-only cardiomyopathy AUROCs")
    subset = models[models["endpoint"].eq("condition_label_change") & models["model"].isin(["M1_gene_only", "M2_baseline_regime_only", "M4_gene_plus_baseline_regime"])]
    ax.bar(subset["model"], subset["AUROC"].astype(float), color=["#9aa4ad", "#4c78a8", "#1b9e77"])
    ax.set_ylim(0.5, 0.78)
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Supplementary Figure 1. Cardiomyopathy leakage repair", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "SupplementaryFigure1_cardiomyopathy_leakage_repair")
    write_caption("SupplementaryFigure1_cardiomyopathy_leakage_repair", "# Supplementary Figure 1. Cardiomyopathy leakage repair\n\nFeature audit and baseline-only cardiomyopathy model comparison used to prevent follow-up/end-point leakage in portability modeling.")
    register_figure("Supplementary Figure 1", "Cardiomyopathy leakage repair", "Feature leakage audit and baseline-only AUROC comparison", "reports/figure_source_tables/SupplementaryFigure1_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_cardiomyopathy_leakage", "Cardiomyopathy portability modeling uses baseline-only features after leakage repair.", "This is leakage QC, not independent validation.")


def supplementary_rolling_origin() -> None:
    plan = read_csv(TABLES / "cab_10yr_rolling_origin_plan.csv")
    rolling = parse_rolling_manifest()
    src = pd.concat([plan.assign(source_section="origin_plan"), rolling.assign(source_section="manifest_metrics")], ignore_index=True, sort=False)
    src.to_csv(SRC_DIR / "SupplementaryFigure2_source_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
    ax = axes[0]
    style_axis(ax, "Rolling-origin windows")
    annual = plan[plan["notes"].eq("annual rolling origin")]
    ax.bar(range(len(annual)), annual["horizon_months"].astype(float), color="#4c78a8")
    ax.set_xticks(range(len(annual)), annual["origin_id"], rotation=90, fontsize=7)
    ax.set_ylabel("Horizon months")
    ax = axes[1]
    style_axis(ax, "Historical backtest AUROC")
    comp = rolling[rolling["model"].isin(["Gene-only", "Regime-only", "Gene + regime", "Full baseline"])]
    ax.bar(comp["model"], comp["held_out_AUROC"].astype(float), color=["#9aa4ad", "#4c78a8", "#1b9e77", "#c44e52"])
    ax.set_ylim(0.45, 0.80)
    ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Supplementary Figure 2. Rolling-origin validation design", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "SupplementaryFigure2_rolling_origin_validation_design")
    write_caption("SupplementaryFigure2_rolling_origin_validation_design", "# Supplementary Figure 2. Rolling-origin validation design\n\nAnnual rolling-origin plan and locked manifest-level AUROC summaries for the historical prospective emulation.")
    register_figure("Supplementary Figure 2", "Rolling-origin validation design", "Window design and AUROC summary", "reports/figure_source_tables/SupplementaryFigure2_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_rolling_origin", "Rolling-origin analysis emulates prospective temporal evaluation.", "The package is temporal backtesting, not true prospective clinical validation.")


def supplementary_routing_sensitivity() -> None:
    metrics = read_csv(TABLES / "routing_metrics_all_modes_all_endpoints.csv")
    boot = read_csv(TABLES / "routing_operating_modes_bootstrap_ci.csv")
    src = pd.concat([metrics.assign(source_section="metrics"), boot.assign(source_section="bootstrap")], ignore_index=True, sort=False)
    src.to_csv(SRC_DIR / "SupplementaryFigure3_source_data.csv", index=False)
    subset = metrics[metrics["cab_mode"].isin(["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"])]
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    style_axis(ax, "Unsupported reuse by endpoint and mode")
    endpoints = subset["endpoint"].drop_duplicates().tolist()
    modes = ["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"]
    x = range(len(endpoints))
    width = 0.24
    for i, mode in enumerate(modes):
        vals = [pct(subset[(subset["endpoint"].eq(e)) & (subset["cab_mode"].eq(mode))]["unsupported_reuse_rate"].iloc[0]) for e in endpoints]
        ax.bar([j + (i - 1) * width for j in x], vals, width, label=mode, color=[PALETTE["clinvar"], PALETTE["strict"], PALETTE["balanced"]][i])
    ax.set_xticks(list(x), [e.replace("_", "\n") for e in endpoints], fontsize=8)
    ax.set_ylabel("Unsupported reuse (%)")
    ax.legend(frameon=False, fontsize=8, ncols=3)
    fig.suptitle("Supplementary Figure 3. Routing sensitivity analyses", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "SupplementaryFigure3_routing_sensitivity_analyses")
    write_caption("SupplementaryFigure3_routing_sensitivity_analyses", "# Supplementary Figure 3. Routing sensitivity analyses\n\nUnsupported deterministic reuse across benchmark endpoints for ClinVar-label-only, CAB-Strict, and CAB-Balanced operating modes.")
    register_figure("Supplementary Figure 3", "Routing sensitivity analyses", "Unsupported reuse across endpoints and modes", "reports/figure_source_tables/SupplementaryFigure3_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_routing_sensitivity", "CAB modes behave consistently as alternative operating points.", "Endpoint definitions are internal benchmark endpoints.")


def supplementary_regime_forest() -> None:
    enrich = read_csv(TABLES / "disease_architecture_regime_enrichment_tests.csv")
    enrich.to_csv(SRC_DIR / "SupplementaryFigure4_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    style_axis(ax, "Regime enrichment forest plot")
    df = enrich.copy()
    df["ORf"] = df["OR"].astype(float)
    df = df.sort_values("ORf")
    ypos = range(len(df))
    ax.errorbar(
        df["ORf"],
        list(ypos),
        xerr=[df["ORf"] - df["CI95_low"].astype(float), df["CI95_high"].astype(float) - df["ORf"]],
        fmt="o",
        color="#4c78a8",
        ecolor="#9aa4ad",
        capsize=3,
    )
    ax.axvline(1, color="#4d4d4d", linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(list(ypos), df["regime"].str.replace("_", " "), fontsize=8)
    ax.set_xlabel("Odds ratio (log scale)")
    fig.suptitle("Supplementary Figure 4. Regime enrichment forest plot", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "SupplementaryFigure4_regime_enrichment_forest_plots")
    write_caption("SupplementaryFigure4_regime_enrichment_forest_plots", "# Supplementary Figure 4. Regime enrichment forest plots\n\nAll locked disease-architecture enrichment tests. Odds ratios describe routing/end-point enrichment, not patient-level effects.")
    register_figure("Supplementary Figure 4", "Regime enrichment forest plots", "All enrichment tests with log-scaled odds ratios", "reports/figure_source_tables/SupplementaryFigure4_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_regime_forest", "Regime enrichments structure routing behavior.", "Several categories are low-N or unsampled.")


def supplementary_alphamissense_qc() -> None:
    qc = read_csv(TABLES / "cab_alphamissense_hg38_join_qc.csv")
    qc.to_csv(SRC_DIR / "SupplementaryFigure5_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    style_axis(ax, "AlphaMissense matching QC")
    rows = qc[qc["metric"].isin(["cab_rows", "alpha_raw_matches_before_resolution", "rows_joined_by_hg38_coordinate", "rows_high_confidence_join_and_hgvs_p_agreement", "join_status__not_found_in_AlphaMissense_hg38"])]
    ax.barh(rows["metric"].str.replace("_", " "), rows["value"].astype(float), color="#4c78a8")
    ax.set_xlabel("Rows")
    ax.grid(axis="x", color="#d9dee3")
    fig.suptitle("Supplementary Figure 5. AlphaMissense matching QC", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "SupplementaryFigure5_alphamissense_matching_QC")
    write_caption("SupplementaryFigure5_alphamissense_matching_QC", "# Supplementary Figure 5. AlphaMissense matching QC\n\nHigh-confidence hg38 coordinate and HGVS protein agreement filter for the inherited-arrhythmia AlphaMissense comparator subset.")
    register_figure("Supplementary Figure 5", "AlphaMissense matching QC", "Rows retained through high-confidence matching", "reports/figure_source_tables/SupplementaryFigure5_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_alphamissense_qc", "AlphaMissense comparator is restricted to high-confidence matched missense rows.", "Subset restriction limits generalization.")


def supplementary_identity_repair() -> None:
    audit = read_csv(TABLES / "external_clinvar_join_final_audit.csv")
    audit.to_csv(SRC_DIR / "SupplementaryFigure6_source_data.csv", index=False)
    row = audit.iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.5), facecolor="white")
    ax.set_axis_off()
    panel = [
        ("Matched before", row["matched_before"]),
        ("ARR attempted", row["arr_rows_attempted"]),
        ("ARR accepted", row["arr_resolution_accepted"]),
        ("Meaning rejected", row["meaning_match_rejected"]),
        ("Matched after", row["matched_after"]),
    ]
    xs = [0.05, 0.25, 0.45, 0.65, 0.84]
    for i, ((label, value), x) in enumerate(zip(panel, xs)):
        color = "#fdebec" if "rejected" in label.lower() else "#eef2f5"
        edge = "#c44e52" if "rejected" in label.lower() else "#4d4d4d"
        draw_box(ax, (x, 0.45), 0.14, 0.22, f"{label}\n{value}", color, edge, fontsize=8)
        if i:
            arrow(ax, (xs[i - 1] + 0.14, 0.56), (x, 0.56))
    ax.text(0.05, 0.20, "Final match rate: 1.0; phenotype-domain discordance flagged for 304 rows.", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("Supplementary Figure 6. ClinVar identity repair pipeline", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "SupplementaryFigure6_ClinVar_identity_repair_pipeline")
    write_caption("SupplementaryFigure6_ClinVar_identity_repair_pipeline", "# Supplementary Figure 6. ClinVar identity repair pipeline\n\nClinVar source identity repair achieved complete source matching while preserving a separate meaning-rejection flag for phenotype-domain discordance.")
    register_figure("Supplementary Figure 6", "ClinVar identity repair pipeline", "Source matching and phenotype-domain discordance flags", "reports/figure_source_tables/SupplementaryFigure6_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_identity_repair", "Identity repair and meaning portability are separate QC layers.", "Source repair does not imply deterministic disease-meaning portability.")


def supplementary_external_evidence() -> None:
    ladder = read_csv(TABLES / "cab_external_validation_proxy_ladder.csv")
    ladder.to_csv(SRC_DIR / "SupplementaryFigure7_source_data.csv", index=False)
    subset = ladder[ladder["resource"].str.contains("eMERGE|ClinGen|PGP|DiscovEHR|LOVD|PhysioNet", case=False, na=False)].copy()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_axis_off()
    y = 0.86
    for _, row in subset.iterrows():
        draw_box(ax, (0.05, y), 0.25, 0.075, row["resource"].split(" / ")[0][:34], "#eef2f5", fontsize=7)
        draw_box(ax, (0.36, y), 0.24, 0.075, row["claim_strength"], "#e8f3ef", fontsize=7)
        draw_box(ax, (0.66, y), 0.27, 0.075, "proxy support only", "#fff3e8", fontsize=7)
        y -= 0.105
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("Supplementary Figure 7. External evidence stack", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "SupplementaryFigure7_external_evidence_stack")
    write_caption("SupplementaryFigure7_external_evidence_stack", "# Supplementary Figure 7. External evidence stack\n\nExternal resources provide proxy support, feasibility checks, or expert-curation comparators. They do not validate CAB clinical outcomes or automated clinical deployment.")
    register_figure("Supplementary Figure 7", "External evidence stack", "Proxy evidence resources and claim strengths", "reports/figure_source_tables/SupplementaryFigure7_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_external_evidence", "External evidence supports framing and feasibility, not clinical validation.", "No external resource validates CAB as clinical decision support.")


def supplementary_sads_grammar() -> None:
    sads = read_csv(TABLES / "sads_cab_portability_use_cases.csv")
    sads.to_csv(SRC_DIR / "SupplementaryFigure8_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    style_axis(ax, "SADS portability grammar")
    prf_counts = sads.groupby("PRF_required").size().reset_index(name="count")
    ax.bar(prf_counts["PRF_required"], prf_counts["count"], color=["#4c78a8", "#e07b39", "#c44e52"][: len(prf_counts)])
    ax.set_xlabel("PRF required")
    ax.set_ylabel("Use cases")
    ax.text(0.02, 0.90, "Stress-test use-case only; no cause-of-death or family-risk prediction.", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle("Supplementary Figure 8. SADS portability grammar", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "SupplementaryFigure8_SADS_portability_grammar")
    write_caption("SupplementaryFigure8_SADS_portability_grammar", "# Supplementary Figure 8. SADS portability grammar\n\nSADS use cases map inherited-arrhythmia assertions into postmortem, family-risk, disease-specific, trigger-dependent, and genotype-first contexts. These are portability checks, not risk predictions.")
    register_figure("Supplementary Figure 8", "SADS portability grammar", "PRF requirement distribution across SADS use cases", "reports/figure_source_tables/SupplementaryFigure8_source_data.csv", "scripts/figures/generate_cab_publication_figures.py::supplementary_sads_grammar", "SADS requires explicit portability grammar across non-equivalent contexts.", "CAB does not infer cause of death or predict sudden death.")


def write_figure_index() -> None:
    rows = []
    for f in FIGURES:
        rows.append(
            "| {figure_number} | {title} | {panels} | {source_tables} | {generation_script} | {primary_claim} | {major_caveat} |".format(
                **{k: str(v).replace("\n", " ") for k, v in f.items()}
            )
        )
    text = [
        "# CAB Figure Index",
        "",
        "| Figure | Title | Panels | Source tables | Generation script | Primary claim | Major caveat |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "All figures are exported as SVG, PDF, and PNG with white backgrounds. Source tables are exported in reports/figure_source_tables/.",
    ]
    (FIG_DIR / "FIGURE_INDEX.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4d4d4d",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )
    write_final_tables()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    supplementary_figures()
    write_figure_index()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote source tables to {SRC_DIR}")
    print(f"Wrote captions to {CAPTION_DIR}")
    print(f"Wrote final tables to {FINAL_TABLE_DIR}")


if __name__ == "__main__":
    main()
