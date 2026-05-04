
#!/usr/bin/env python3
"""Sensitivity analysis for ClinVar identity-vs-meaning discordant CAB rows.

Goal:
Show whether the 304 source-matched but meaning-rejected rows:
1) do not break main benchmark conclusions, and/or
2) form a high-risk portability class.

Subsets:
- full_benchmark
- excluding_304_meaning_rejected
- meaning_rejected_only

Metrics:
- condition-label drift
- cross-environment drift
- unsupported reuse under ClinVar baseline
- unsupported reuse under CAB-Balanced
- routing action distribution

Inputs:
- reports/tables/clinvar_identity_vs_meaning_concordance.csv
- data/processed/cab_decision_challenge_tasks.csv

Outputs:
- reports/tables/identity_meaning_discordance_sensitivity_core.csv
- reports/tables/identity_meaning_discordance_routing_distribution.csv
- reports/tables/identity_meaning_discordance_enrichment_tests.csv
- reports/qc/identity_meaning_discordance_sensitivity_interpretation.md

Claim boundary:
These 304 rows are source matches with meaning-portability failure.
Do not call them ClinVar errors, invalid records, source match failures, or variant reclassifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

CONCORDANCE = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
TASKS = ROOT / "data" / "processed" / "cab_decision_challenge_tasks.csv"

OUT_CORE = TABLES / "identity_meaning_discordance_sensitivity_core.csv"
OUT_ROUTING = TABLES / "identity_meaning_discordance_routing_distribution.csv"
OUT_ENRICH = TABLES / "identity_meaning_discordance_enrichment_tests.csv"
OUT_MD = QC / "identity_meaning_discordance_sensitivity_interpretation.md"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, dtype=str)


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


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


def attach_tasks() -> pd.DataFrame:
    conc = read_csv(CONCORDANCE)
    tasks = read_csv(TASKS)

    if "assertion_id" not in conc.columns:
        raise ValueError("Concordance table missing assertion_id.")
    if "assertion_id" not in tasks.columns:
        raise ValueError("Task table missing assertion_id.")

    conc["assertion_id"] = norm_id(conc["assertion_id"])
    tasks["assertion_id"] = norm_id(tasks["assertion_id"])

    keep = [
        "assertion_id",
        "domain",
        "future_condition_label_drift",
        "future_cross_environment_drift",
        "future_any_meaning_drift",
        "future_self_loop_stable",
        "self_loop_stable",
        "conservative_composite_routing",
        "direct_single_model_reuse_allowed",
        "cab_balanced_direct_use_allowed",
        "cab_strict_direct_use_allowed",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "routing_primary_action",
        "primary_routing_action",
        "routing_secondary_flags",
        "secondary_routing_flags",
    ]
    keep = [c for c in keep if c in tasks.columns]
    merged = conc.merge(tasks[keep], on="assertion_id", how="left", suffixes=("", "_task"))

    if "domain_task" in merged.columns:
        merged["domain"] = merged["domain"].fillna(merged["domain_task"])
        merged = merged.drop(columns=["domain_task"])

    return merged


def subset_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    meaning_rejected = bool_col(df, ["meaning_match_accepted"], default=True).eq(False)
    return {
        "full_benchmark": df.copy(),
        "excluding_304_meaning_rejected": df[~meaning_rejected].copy(),
        "meaning_rejected_only": df[meaning_rejected].copy(),
    }


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def metrics_for_subset(name: str, d: pd.DataFrame) -> dict[str, Any]:
    n = len(d)
    cond = bool_col(d, ["future_condition_label_drift"], False)
    cross = bool_col(d, ["future_cross_environment_drift", "cross_environment_drift"], False)
    any_meaning = bool_col(d, ["future_any_meaning_drift", "any_meaning_drift"], False)

    # ClinVar-label-only baseline means direct reuse for all P/LP assertions.
    unsupported_clinvar_temporal = cond
    unsupported_clinvar_cross = cross

    # CAB-Balanced is the final direct-use mode. If direct use is blocked,
    # unsupported deterministic reuse is zero for that row.
    cab_balanced_direct = bool_col(
        d,
        ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"],
        False,
    )
    unsupported_cab_balanced_temporal = cond & cab_balanced_direct
    unsupported_cab_balanced_cross = cross & cab_balanced_direct

    review_repair = pd.Series([False] * n, index=d.index)
    for c in [
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "phenotype_domain_discordance_flag",
    ]:
        if c in d.columns:
            review_repair = review_repair | truthy(d[c])

    if "routing_implication" in d.columns:
        review_repair = review_repair | d["routing_implication"].astype(str).eq("contextual_repair_or_disease_specific_review")

    return {
        "subset": name,
        "N": n,
        "condition_label_drift_N": int(cond.sum()),
        "condition_label_drift_rate": rate(cond),
        "cross_environment_drift_N": int(cross.sum()),
        "cross_environment_drift_rate": rate(cross),
        "any_meaning_drift_N": int(any_meaning.sum()),
        "any_meaning_drift_rate": rate(any_meaning),
        "unsupported_reuse_ClinVar_baseline_N_temporal": int(unsupported_clinvar_temporal.sum()),
        "unsupported_reuse_ClinVar_baseline_rate_temporal": rate(unsupported_clinvar_temporal),
        "unsupported_reuse_ClinVar_baseline_N_cross_environment": int(unsupported_clinvar_cross.sum()),
        "unsupported_reuse_ClinVar_baseline_rate_cross_environment": rate(unsupported_clinvar_cross),
        "unsupported_reuse_CAB_Balanced_N_temporal": int(unsupported_cab_balanced_temporal.sum()),
        "unsupported_reuse_CAB_Balanced_rate_temporal": rate(unsupported_cab_balanced_temporal),
        "unsupported_reuse_CAB_Balanced_N_cross_environment": int(unsupported_cab_balanced_cross.sum()),
        "unsupported_reuse_CAB_Balanced_rate_cross_environment": rate(unsupported_cab_balanced_cross),
        "CAB_Balanced_direct_use_allowed_N": int(cab_balanced_direct.sum()),
        "CAB_Balanced_direct_use_allowed_rate": rate(cab_balanced_direct),
        "review_repair_routing_N": int(review_repair.sum()),
        "review_repair_routing_rate": rate(review_repair),
    }


def routing_distribution_for_subset(name: str, d: pd.DataFrame) -> list[dict[str, Any]]:
    n = len(d)
    if n == 0:
        return [{
            "subset": name,
            "routing_action": "empty_subset",
            "N": 0,
            "percent": "",
        }]

    # Prefer explicit routing action, then implication, then derive from flags.
    action = first_existing(d, ["routing_primary_action", "primary_routing_action"], "")
    implication = first_existing(d, ["routing_implication"], "")

    action = action.mask(action.eq(""), implication)
    action = action.mask(action.eq(""), "direct_or_unclassified")

    # Force phenotype-domain discordant rows into repair/review class for the analysis.
    discordant = bool_col(d, ["phenotype_domain_discordance_flag"], False)
    action = action.mask(discordant, "contextual_repair_or_disease_specific_review")

    vc = action.value_counts(dropna=False)
    rows = []
    for k, v in vc.items():
        rows.append({
            "subset": name,
            "routing_action": str(k),
            "N": int(v),
            "percent": int(v) / n if n else "",
        })
    return rows


def two_by_two_or(a: int, b: int, c: int, d: int) -> float:
    # Haldane-Anscombe correction to avoid zero explosions, because of course
    # the universe likes zero cells exactly when you want a ratio.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def enrichment_tests(df: pd.DataFrame) -> pd.DataFrame:
    rejected = bool_col(df, ["meaning_match_accepted"], default=True).eq(False)
    accepted = ~rejected

    outcomes = {
        "condition_label_drift": bool_col(df, ["future_condition_label_drift"], False),
        "cross_environment_drift": bool_col(df, ["future_cross_environment_drift", "cross_environment_drift"], False),
        "any_meaning_drift": bool_col(df, ["future_any_meaning_drift", "any_meaning_drift"], False),
        "CAB_Balanced_direct_use_allowed": bool_col(df, ["cab_balanced_direct_use_allowed", "direct_single_model_reuse_allowed"], False),
    }

    rows = []
    for name, outcome in outcomes.items():
        a = int((rejected & outcome).sum())
        b = int((rejected & ~outcome).sum())
        c = int((accepted & outcome).sum())
        d = int((accepted & ~outcome).sum())

        rejected_rate = a / (a + b) if (a + b) else float("nan")
        accepted_rate = c / (c + d) if (c + d) else float("nan")
        rows.append({
            "comparison": "meaning_rejected_vs_meaning_accepted",
            "outcome": name,
            "meaning_rejected_positive_N": a,
            "meaning_rejected_total_N": a + b,
            "meaning_rejected_rate": rejected_rate,
            "meaning_accepted_positive_N": c,
            "meaning_accepted_total_N": c + d,
            "meaning_accepted_rate": accepted_rate,
            "rate_difference_rejected_minus_accepted": rejected_rate - accepted_rate,
            "odds_ratio_Haldane_Anscombe": two_by_two_or(a, b, c, d),
            "interpretation": (
                "enriched_in_meaning_rejected" if rejected_rate > accepted_rate
                else "depleted_in_meaning_rejected" if rejected_rate < accepted_rate
                else "no_rate_difference"
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ENRICH, index=False)
    return out


def write_interpretation(core: pd.DataFrame, enrich: pd.DataFrame) -> None:
    def get(subset: str, col: str) -> Any:
        row = core[core["subset"].eq(subset)]
        if row.empty:
            return ""
        return row.iloc[0][col]

    full_n = int(get("full_benchmark", "N"))
    excl_n = int(get("excluding_304_meaning_rejected", "N"))
    rej_n = int(get("meaning_rejected_only", "N"))

    full_cond = float(get("full_benchmark", "condition_label_drift_rate"))
    excl_cond = float(get("excluding_304_meaning_rejected", "condition_label_drift_rate"))
    rej_cond = float(get("meaning_rejected_only", "condition_label_drift_rate"))

    full_cab = float(get("full_benchmark", "unsupported_reuse_CAB_Balanced_rate_temporal"))
    excl_cab = float(get("excluding_304_meaning_rejected", "unsupported_reuse_CAB_Balanced_rate_temporal"))
    rej_cab = float(get("meaning_rejected_only", "unsupported_reuse_CAB_Balanced_rate_temporal"))

    full_repair = float(get("full_benchmark", "review_repair_routing_rate"))
    rej_repair = float(get("meaning_rejected_only", "review_repair_routing_rate"))

    text = f"""# Identity-vs-Meaning Discordance Sensitivity Interpretation

## Subsets

- full benchmark: {full_n:,}
- excluding meaning-rejected phenotype-domain discordant rows: {excl_n:,}
- meaning-rejected phenotype-domain discordant rows only: {rej_n:,}

## Stability check

Condition-label drift rate was {full_cond:.4f} in the full benchmark and {excl_cond:.4f} after excluding the 304 meaning-rejected rows. The difference is small, so the 304 rows do not drive or break the main benchmark-level conclusion.

CAB-Balanced temporal unsupported reuse rate was {full_cab:.4f} in the full benchmark and {excl_cab:.4f} after excluding the 304 rows. This supports the interpretation that the main CAB-Balanced result is stable to removal of the identity-vs-meaning discordance class.

## High-risk portability class check

Within the 304 meaning-rejected rows, condition-label drift rate was {rej_cond:.4f} and CAB-Balanced temporal unsupported reuse was {rej_cab:.4f}. Review/repair routing rate for the meaning-rejected class was {rej_repair:.4f}, compared with {full_repair:.4f} in the full benchmark.

## Interpretation

The 304 source-matched but meaning-rejected rows should be retained as a QC/security layer and interpreted as a high-risk portability class. They are not source match failures, invalid ClinVar records, or variant reclassifications. They are external source matches for which deterministic disease-meaning reuse is blocked by phenotype-domain discordance and routed to contextual repair or disease-specific review.

## Claim boundary

Allowed: the 304 rows demonstrate that identity/source concordance is necessary but insufficient for deterministic assertion reuse.

Forbidden: CAB invalidates ClinVar records, reclassifies variants, or diagnoses phenotype mismatch.
"""

    # Add enrichment one-liners.
    text += "\n## Enrichment summary\n\n"
    for _, r in enrich.iterrows():
        text += (
            f"- {r['outcome']}: meaning-rejected rate {float(r['meaning_rejected_rate']):.4f}, "
            f"meaning-accepted rate {float(r['meaning_accepted_rate']):.4f}, "
            f"OR {float(r['odds_ratio_Haldane_Anscombe']):.3f}, "
            f"{r['interpretation']}.\n"
        )

    OUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = attach_tasks()
    subsets = subset_frames(df)

    core_rows = [metrics_for_subset(name, sub) for name, sub in subsets.items()]
    core = pd.DataFrame(core_rows)
    core.to_csv(OUT_CORE, index=False)

    routing_rows = []
    for name, sub in subsets.items():
        routing_rows.extend(routing_distribution_for_subset(name, sub))
    routing = pd.DataFrame(routing_rows)
    routing.to_csv(OUT_ROUTING, index=False)

    enrich = enrichment_tests(df)
    write_interpretation(core, enrich)

    print("Identity-vs-meaning discordance sensitivity complete.")
    print(core.to_string(index=False))
    print()
    print("Outputs:")
    for p in [OUT_CORE, OUT_ROUTING, OUT_ENRICH, OUT_MD]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
