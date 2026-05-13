#!/usr/bin/env python3
"""Analyze completed CAB expert-endpoint adjudication verdicts.

The script expects real specialist labels in:
reports/adjudication/cab_expert_endpoint_verdict_template.csv

It intentionally refuses to treat blank templates as validation results. With
--allow-pending, it writes pending-status placeholders for pipeline checks.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "reports" / "adjudication"
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

VERDICTS = ADJ / "cab_expert_endpoint_verdict_template.csv"
KEY = ADJ / "cab_expert_endpoint_temporal_endpoint_key.csv"
SADS_VERDICTS = ADJ / "cab_sads_cpvt_expert_endpoint_verdict_template.csv"
SADS_KEY = ADJ / "cab_sads_cpvt_expert_endpoint_prediction_key.csv"


def parse_yes_no(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "portable"}:
        return 1.0
    if text in {"no", "n", "0", "false", "not_portable", "nonportable"}:
        return 0.0
    return np.nan


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(score)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(score)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], score[ok]))


def safe_brier(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(score)
    if ok.sum() == 0:
        return float("nan")
    return float(brier_score_loss(y[ok], np.clip(score[ok], 0, 1)))


def pairwise_kappa(verdicts: pd.DataFrame) -> float:
    wide = verdicts.pivot_table(
        index="blinded_case_id",
        columns="reviewer_id",
        values="portable_binary",
        aggfunc="first",
    )
    kappas = []
    for a, b in combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        if pair.empty:
            continue
        pa = float((pair[a] == pair[b]).mean())
        p_yes_a = float(pair[a].mean())
        p_yes_b = float(pair[b].mean())
        pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
        if pe < 1:
            kappas.append((pa - pe) / (1 - pe))
    return float(np.nanmean(kappas)) if kappas else float("nan")


def consensus_from_verdicts(verdicts: pd.DataFrame) -> pd.DataFrame:
    verdicts = verdicts.copy()
    verdicts["portable_binary"] = verdicts["portable_without_additional_interpretation"].map(parse_yes_no)
    valid = verdicts.dropna(subset=["portable_binary"])
    grouped = valid.groupby("blinded_case_id")
    consensus = grouped.agg(
        expert_reviewer_N_completed=("portable_binary", "size"),
        portable_yes_N=("portable_binary", "sum"),
        portable_rate=("portable_binary", "mean"),
        domain=("case_domain", "first"),
        gene=("gene", "first"),
        target_context=("target_context", "first"),
        arrhythmia_high_value_gene=("arrhythmia_high_value_gene", "first"),
        sads_cpvt_priority=("sads_cpvt_priority", "first"),
    ).reset_index()
    consensus["expert_nonportable_consensus"] = consensus["portable_rate"] < 0.5
    consensus["majority_strength"] = (consensus["portable_rate"] - 0.5).abs() * 2
    return consensus


def evaluate_core(consensus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = pd.read_csv(KEY)
    df = consensus.merge(key, on=["blinded_case_id", "domain", "gene"], how="left")
    y = df["expert_nonportable_consensus"].astype(int).to_numpy()
    cab_nonportable = (~df["CAB_routing_action"].astype(str).str.contains("direct", case=False, na=False)).astype(float).to_numpy()
    clinvar_drift = df["conservative_composite_non_portability"].astype(str).str.lower().isin({"1", "true", "yes"}).astype(float).to_numpy()
    condition_drift = df["condition_label_drift"].astype(str).str.lower().isin({"1", "true", "yes"}).astype(float).to_numpy()

    rows = []
    for model, score in [
        ("CAB routing nonportable", cab_nonportable),
        ("ClinVar conservative composite drift", clinvar_drift),
        ("ClinVar condition-label drift", condition_drift),
    ]:
        rows.append(
            {
                "model_or_proxy": model,
                "endpoint": "expert_nonportable_consensus",
                "N": int(np.isfinite(score).sum()),
                "positive_N": int(y.sum()),
                "AUROC": safe_auc(y, score),
                "AUPRC": safe_auprc(y, score),
                "Brier": safe_brier(y, score),
                "mean_score_positive": float(np.nanmean(score[y == 1])) if (y == 1).any() else float("nan"),
                "mean_score_negative": float(np.nanmean(score[y == 0])) if (y == 0).any() else float("nan"),
            }
        )
    metrics = pd.DataFrame(rows)

    regime = (
        df.groupby("CAB_regime")
        .agg(
            N=("expert_nonportable_consensus", "size"),
            expert_nonportable_N=("expert_nonportable_consensus", "sum"),
            expert_nonportable_rate=("expert_nonportable_consensus", "mean"),
            median_majority_strength=("majority_strength", "median"),
        )
        .reset_index()
    )
    return metrics, regime


def evaluate_sads() -> pd.DataFrame:
    if not SADS_VERDICTS.exists() or not SADS_KEY.exists():
        return pd.DataFrame()
    verdicts = pd.read_csv(SADS_VERDICTS)
    consensus = consensus_from_verdicts(verdicts)
    if consensus.empty:
        return pd.DataFrame()
    key = pd.read_csv(SADS_KEY)
    df = consensus.merge(key, on=["blinded_case_id", "gene"], how="left")
    return pd.DataFrame(
        [
            {
                "stratum": "SADS_CPVT_priority_addendum",
                "N_consensus_cases": len(df),
                "expert_nonportable_N": int(df["expert_nonportable_consensus"].sum()),
                "expert_nonportable_rate": float(df["expert_nonportable_consensus"].mean()),
                "median_majority_strength": float(df["majority_strength"].median()),
                "cases_completed_threshold_40": len(df) >= 40,
            }
        ]
    )


def write_pending() -> None:
    pd.DataFrame(
        [
            {
                "status": "pending_real_expert_verdicts",
                "required_input": str(VERDICTS.relative_to(ROOT)),
                "message": "No yes/no expert labels found; validation metrics intentionally not computed.",
            }
        ]
    ).to_csv(TABLES / "cab_expert_endpoint_validation_results.csv", index=False)
    (QC / "cab_expert_endpoint_validation_results.md").write_text(
        "# CAB Expert Endpoint Validation Results\n\nStatus: pending real specialist verdicts.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    verdicts = pd.read_csv(VERDICTS)
    verdicts["portable_binary"] = verdicts["portable_without_additional_interpretation"].map(parse_yes_no)
    if verdicts["portable_binary"].notna().sum() == 0:
        if args.allow_pending:
            write_pending()
            print("No real verdicts yet; wrote pending validation status")
            return
        raise SystemExit("No real expert yes/no verdicts found. Use --allow-pending only for pipeline checks.")

    consensus = consensus_from_verdicts(verdicts)
    metrics, regime = evaluate_core(consensus)
    sads = evaluate_sads()
    reliability = pd.DataFrame(
        [
            {
                "N_cases_with_consensus": len(consensus),
                "N_verdict_rows_completed": int(verdicts["portable_binary"].notna().sum()),
                "mean_completed_reviewers_per_case": float(consensus["expert_reviewer_N_completed"].mean()),
                "median_majority_strength": float(consensus["majority_strength"].median()),
                "pairwise_kappa_mean": pairwise_kappa(verdicts),
            }
        ]
    )

    metrics.to_csv(TABLES / "cab_expert_endpoint_validation_results.csv", index=False)
    regime.to_csv(TABLES / "cab_expert_regime_calibration_results.csv", index=False)
    reliability.to_csv(TABLES / "cab_expert_endpoint_interrater_reliability.csv", index=False)
    sads.to_csv(TABLES / "cab_expert_sads_cpvt_validation_results.csv", index=False)
    (QC / "cab_expert_endpoint_validation_results.md").write_text(
        "# CAB Expert Endpoint Validation Results\n\n"
        f"Completed consensus cases: {len(consensus)}.\n\n"
        "Primary comparison table: reports/tables/cab_expert_endpoint_validation_results.csv\n",
        encoding="utf-8",
    )
    print("Wrote CAB expert endpoint validation results")


if __name__ == "__main__":
    main()
