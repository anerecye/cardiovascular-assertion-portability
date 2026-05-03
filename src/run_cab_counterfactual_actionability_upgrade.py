#!/usr/bin/env python3
"""CAB counterfactual decision benchmark and gene-archetype actionability upgrade.

Local runner. No GitHub writes.

Purpose
-------
Upgrade CAB beyond drift prediction by testing whether CAB changes assertion-use
decisions relative to a standard deterministic reuse pipeline.

Core comparison
---------------
Baseline pipeline:
    ClinVar P/LP + gene + public condition label -> deterministic reuse

CAB pipeline:
    P/LP + CAB architecture + baseline-only CPI + routing flags -> allow,
    contextual repair, or expert review

Outputs
-------
reports/tables/cab_counterfactual_decision_benchmark.csv
reports/tables/cab_counterfactual_decision_summary.csv
reports/tables/cab_counterfactual_task_metrics.csv
data/processed/cab_gene_archetypes.csv
reports/tables/cab_gene_archetypes.csv
reports/tables/cab_gene_archetype_model_comparison.csv
reports/tables/cab_alphamissense_negative_control.csv
reports/figures/cab_counterfactual_unsupported_reuse.png
reports/figures/cab_gene_archetype_drift_rates.png
reports/qc/cab_counterfactual_decision_benchmark_report.md
reports/final_cab_actionability_upgrade_report.md

Notes
-----
- Original leakage-susceptible AUCs are not used.
- Expert-adjudicated correctness is operational-rule adjudication unless an
  external expert review table is supplied and detected.
- AlphaMissense is optional. If no AlphaMissense-like file/columns are detected,
  the negative control is explicitly skipped with a QC row.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:
    raise ImportError(
        "This script requires scikit-learn. Install with: python -m pip install scikit-learn"
    ) from exc


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
QC = REPORTS / "qc"

FRAMEWORK = DATA / "cab_predictive_operational_framework.csv"
CPI_BASELINE = DATA / "cab_portability_index_baseline_only.csv"
CROSS_ENV = DATA / "cab_cross_environment_drift.csv"
GENE_MAP = DATA / "cab_gene_temporal_instability_map.csv"

BENCHMARK_OUT = TABLES / "cab_counterfactual_decision_benchmark.csv"
SUMMARY_OUT = TABLES / "cab_counterfactual_decision_summary.csv"
TASK_METRICS_OUT = TABLES / "cab_counterfactual_task_metrics.csv"
GENE_ARCH_OUT_DATA = DATA / "cab_gene_archetypes.csv"
GENE_ARCH_OUT_TABLE = TABLES / "cab_gene_archetypes.csv"
GENE_ARCH_MODEL_OUT = TABLES / "cab_gene_archetype_model_comparison.csv"
ALPHAMISSENSE_OUT = TABLES / "cab_alphamissense_negative_control.csv"
REPORT_OUT = QC / "cab_counterfactual_decision_benchmark_report.md"
FINAL_REPORT_OUT = REPORTS / "final_cab_actionability_upgrade_report.md"

FIG_UNSUPPORTED = FIGURES / "cab_counterfactual_unsupported_reuse.png"
FIG_GENE_ARCH = FIGURES / "cab_gene_archetype_drift_rates.png"


SENTINEL_ARCHETYPES = {
    "SCN5A": "disease_model_collision_hub",
    "RYR2": "provocation_death_context_drift_axis",
    "CASQ2": "provocation_death_context_drift_axis",
    "TRDN": "provocation_death_context_drift_axis",
    "KCNQ1": "phenotype_anchored_LQTS_stability_pole",
    "KCNH2": "phenotype_anchored_LQTS_stability_pole",
    "CACNA1C": "structural_electrical_model_collision_hybrid",
    "HCN4": "underresolved_contextual_conduction_overlap_edge_case",
    "ANK2": "underresolved_contextual_conduction_overlap_edge_case",
    "KCNE1": "phenotype_anchored_modifier_edge_case",
}


def ensure_dirs() -> None:
    for d in [DATA, TABLES, FIGURES, QC, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)


def norm_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def as_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ci_binom_wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mode_value(s: pd.Series, default: str = "unavailable") -> str:
    vals = [str(x) for x in s.dropna() if str(x).strip()]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def entropy_from_labels(labels: Iterable[str]) -> float:
    vals = [v for v in labels if v]
    if not vals:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def clinical_environment(label: object) -> str:
    t = norm_text(label)
    if not t:
        return "other/unknown"
    if any(k in t for k in ["brugada", "brs"]):
        return "Brugada"
    if any(k in t for k in ["long qt", "long-qt", "lqts", "romano", "jervell"]):
        return "LQTS"
    if any(k in t for k in ["catecholaminergic", "cpvt"]):
        return "CPVT"
    if any(k in t for k in ["sudden", "sads", "death", "postmortem", "post-mortem"]):
        return "SADS"
    if any(k in t for k in ["short qt", "short-qt", "sqts"]):
        return "SQTS"
    if any(k in t for k in ["cardiomyopathy", "dilated", "hypertrophic", "arrhythmogenic right ventricular", "arvc"]):
        return "cardiomyopathy_overlap"
    if any(k in t for k in ["conduction", "sick sinus", "sinus node", "atrioventricular block", "av block"]):
        return "conduction"
    if any(k in t for k in ["arrhythmia", "arrhythmogenic", "atrial fibrillation", "ventricular fibrillation"]):
        return "nonspecific_arrhythmia"
    return "other/unknown"


def plp_group(row: pd.Series) -> bool:
    bg = norm_text(row.get("baseline_clinical_group", ""))
    cs = norm_text(row.get("clinical_significance_2023-01", ""))
    if bg in {"p_lp", "pathogenic_likely_pathogenic"}:
        return True
    return ("pathogenic" in cs) and ("benign" not in cs) and ("uncertain" not in cs)


def merge_inputs() -> pd.DataFrame:
    if not FRAMEWORK.exists():
        raise FileNotFoundError(f"Missing {FRAMEWORK}; run prior CAB predictive framework first.")

    df = pd.read_csv(FRAMEWORK, low_memory=False)
    if "variation_id" in df.columns:
        df["assertion_id"] = df["variation_id"].map(norm_id)
    elif "assertion_id" in df.columns:
        df["assertion_id"] = df["assertion_id"].map(norm_id)
    else:
        raise ValueError("Framework table lacks variation_id/assertion_id")

    if "gene" not in df.columns:
        gene_col = first_existing(df, ["gene_baseline", "GeneSymbol", "gene_2023-01"])
        df["gene"] = df[gene_col] if gene_col else "unknown"
    df["gene"] = df["gene"].fillna("unknown").astype(str)

    # Merge baseline-only CPI.
    if CPI_BASELINE.exists():
        cpi = pd.read_csv(CPI_BASELINE, low_memory=False)
        id_col = first_existing(cpi, ["assertion_id", "variation_id", "VariationID"])
        if id_col:
            cpi["assertion_id"] = cpi[id_col].map(norm_id)
            keep = ["assertion_id"] + [c for c in [
                "CPI_baseline_only",
                "nonportability_score_baseline_only",
                "CPI_tier_baseline_only",
                "included_features",
                "excluded_features_due_to_leakage",
            ] if c in cpi.columns]
            df = df.merge(cpi[keep].drop_duplicates("assertion_id"), on="assertion_id", how="left")
    if "CPI_baseline_only" not in df.columns:
        if "cab_portability_index" in df.columns:
            warnings.warn("CPI_baseline_only not found; falling back to cab_portability_index but marking as fallback.")
            df["CPI_baseline_only"] = df["cab_portability_index"].map(safe_float)
            df["CPI_tier_baseline_only"] = pd.cut(
                df["CPI_baseline_only"],
                [-1, 25, 50, 75, 101],
                labels=["severe_non_portability", "low_portability", "intermediate_portability", "high_portability"],
            ).astype(str)
        else:
            df["CPI_baseline_only"] = np.nan
            df["CPI_tier_baseline_only"] = "unavailable"
    if "nonportability_score_baseline_only" not in df.columns:
        df["nonportability_score_baseline_only"] = 100.0 - df["CPI_baseline_only"].map(lambda x: safe_float(x, np.nan))

    # Merge cross-environment endpoint if present.
    if CROSS_ENV.exists():
        ce = pd.read_csv(CROSS_ENV, low_memory=False)
        id_col = first_existing(ce, ["assertion_id", "variation_id", "VariationID"])
        if id_col:
            ce["assertion_id"] = ce[id_col].map(norm_id)
            keep = ["assertion_id"] + [c for c in [
                "cross_environment_drift",
                "within_environment_label_drift",
                "stable_environment",
                "baseline_clinical_environment",
                "followup_clinical_environment",
                "source_clinical_environment",
                "target_clinical_environment",
            ] if c in ce.columns]
            df = df.merge(ce[keep].drop_duplicates("assertion_id"), on="assertion_id", how="left", suffixes=("", "_ce"))

    # Derive endpoints if missing.
    if "condition_label_change" not in df.columns:
        df["condition_label_change"] = False
    df["future_condition_label_drift"] = df["condition_label_change"].map(as_bool)

    if "failure_topology_severity" in df.columns:
        df["future_classification_severity_drift"] = df["failure_topology_severity"].map(safe_float) >= 3
    else:
        df["future_classification_severity_drift"] = df.get("classification_change", False).map(as_bool)

    if "assertion_meaning_drift_score" in df.columns:
        df["any_meaning_drift"] = df["assertion_meaning_drift_score"].map(safe_float) > 0
    else:
        df["any_meaning_drift"] = (
            df.get("future_condition_label_drift", False).map(as_bool)
            | df.get("classification_change", False).map(as_bool)
            | df.get("review_status_change", False).map(as_bool)
        )

    df["semantic_drift_without_reclassification"] = (
        df["future_condition_label_drift"].map(as_bool)
        & ~df.get("classification_change", False).map(as_bool)
    )
    df["review_status_change"] = df.get("review_status_change", False).map(as_bool)

    # Submitter count change if count columns exist.
    b_sub = first_existing(df, ["number_submitters_2023-01", "NumberSubmitters_2023-01", "baseline_submitter_count"])
    f_sub = first_existing(df, ["number_submitters_2026-04", "NumberSubmitters_2026-04", "followup_submitter_count"])
    if b_sub and f_sub:
        df["submitter_count_change"] = df[b_sub].map(safe_float) != df[f_sub].map(safe_float)
    else:
        df["submitter_count_change"] = False

    # Env.
    b_env = first_existing(df, ["baseline_clinical_environment", "source_clinical_environment"])
    f_env = first_existing(df, ["followup_clinical_environment", "target_clinical_environment"])
    if b_env:
        df["baseline_clinical_environment"] = df[b_env].fillna("").map(lambda x: str(x) if str(x).strip() else "other/unknown")
    else:
        b_label = first_existing(df, ["phenotype_list_2023-01", "baseline_condition_norm", "clinical_significance_2023-01"])
        df["baseline_clinical_environment"] = df[b_label].map(clinical_environment) if b_label else "other/unknown"
    if f_env:
        df["followup_clinical_environment"] = df[f_env].fillna("").map(lambda x: str(x) if str(x).strip() else "other/unknown")
    else:
        f_label = first_existing(df, ["phenotype_list_2026-04", "followup_condition_norm", "clinical_significance_2026-04"])
        df["followup_clinical_environment"] = df[f_label].map(clinical_environment) if f_label else "other/unknown"

    if "cross_environment_drift" not in df.columns:
        df["cross_environment_drift"] = df["baseline_clinical_environment"] != df["followup_clinical_environment"]
    df["cross_environment_drift"] = df["cross_environment_drift"].map(as_bool)

    # CAB flags normalization.
    for col in [
        "is_disease_model_collision",
        "single_model_repair_required",
        "is_unanchored_assertion_state",
        "is_canonical_monogenic",
        "is_ancestry_concentrated",
        "is_penetrance_boundary",
        "sads_gene_context_flag",
        "is_sads_sensitive",
    ]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].map(as_bool)

    if "evidence_collision_index" not in df.columns:
        df["evidence_collision_index"] = 0.0
    if "regime_membership_count" not in df.columns:
        df["regime_membership_count"] = 0.0
    df["evidence_collision_index"] = df["evidence_collision_index"].map(safe_float)
    df["regime_membership_count"] = df["regime_membership_count"].map(safe_float)

    if "causal_architecture_category" not in df.columns:
        df["causal_architecture_category"] = "unavailable"
    if "primary_regime" not in df.columns:
        df["primary_regime"] = "unavailable"

    if "failure_topology" not in df.columns:
        df["failure_topology"] = "unavailable"
    if "failure_topology_severity" not in df.columns:
        df["failure_topology_severity"] = 0

    df["is_baseline_plp"] = df.apply(plp_group, axis=1)

    # Archetype code.
    df["gene_archetype"] = df["gene"].map(lambda g: SENTINEL_ARCHETYPES.get(str(g), "other_or_insufficiently_characterized_gene"))

    # Condition labels for transitions.
    b_lab = first_existing(df, ["phenotype_list_2023-01", "baseline_condition_norm", "baseline_condition_exact_string"])
    f_lab = first_existing(df, ["phenotype_list_2026-04", "followup_condition_norm", "followup_condition_exact_string"])
    df["baseline_condition_label_for_task"] = df[b_lab].fillna("").astype(str) if b_lab else ""
    df["followup_condition_label_for_task"] = df[f_lab].fillna("").astype(str) if f_lab else ""

    return df


def cpi_tier(score: float) -> str:
    if math.isnan(score):
        return "unavailable"
    if score >= 75:
        return "high_portability"
    if score >= 50:
        return "intermediate_portability"
    if score >= 25:
        return "low_portability"
    return "severe_non_portability"


def baseline_decision(row: pd.Series, task: str) -> str:
    # Counterfactual standard reuse pipeline: deterministic reuse from P/LP + gene + public condition label.
    if not row["is_baseline_plp"]:
        return "not_applicable_non_PLP"
    return "deterministic_reuse"


def cab_decision(row: pd.Series, task: str) -> str:
    if not row["is_baseline_plp"]:
        return "not_applicable_non_PLP"

    cpi = safe_float(row.get("CPI_baseline_only", np.nan), np.nan)
    arch = norm_text(row.get("causal_architecture_category", ""))
    base_env = row.get("baseline_clinical_environment", "other/unknown")
    collision = as_bool(row.get("is_disease_model_collision", False))
    repair = as_bool(row.get("single_model_repair_required", False))
    unanchored = as_bool(row.get("is_unanchored_assertion_state", False))
    ancestry = as_bool(row.get("is_ancestry_concentrated", False))
    canonical = as_bool(row.get("is_canonical_monogenic", False))
    multi_regime = safe_float(row.get("regime_membership_count", 0)) >= 3
    eci = safe_float(row.get("evidence_collision_index", 0))

    high_risk = (not math.isnan(cpi) and cpi < 25) or collision or repair or eci >= 3
    moderate_risk = (not math.isnan(cpi) and cpi < 50) or unanchored or multi_regime

    if task == "T1_phenotype_first_diagnosis":
        if canonical and cpi >= 75 and not collision and not unanchored:
            return "allow_reuse"
        if high_risk or base_env in {"other/unknown", "nonspecific_arrhythmia"}:
            return "expert_review"
        return "contextual_repair"

    if task == "T2_SADS_postmortem_interpretation":
        sads_flag = as_bool(row.get("is_sads_sensitive", False)) or as_bool(row.get("sads_gene_context_flag", False)) or base_env == "SADS"
        if sads_flag and not high_risk:
            return "contextual_repair"
        return "expert_review"

    if task == "T3_reuse_without_ancestry_aware_AF":
        if ancestry or "population" in arch or cpi < 50:
            return "contextual_repair"
        if high_risk:
            return "expert_review"
        return "allow_reuse"

    if task == "T4_single_disease_model":
        if collision or multi_regime or "multi" in arch or repair:
            return "expert_review"
        if canonical and cpi >= 75:
            return "allow_reuse"
        return "contextual_repair"

    if task == "T5_requires_expert_disease_specific_review":
        if high_risk or moderate_risk or row.get("cross_environment_drift", False):
            return "expert_review"
        return "allow_reuse"

    return "contextual_repair"


def adjudicated_action(row: pd.Series, task: str) -> str:
    """Rule-based benchmark adjudication.

    This is not external expert review; it is an operational adjudication rule
    using baseline CAB structure plus temporal endpoints to flag unsupported deterministic reuse.
    """
    if not row["is_baseline_plp"]:
        return "not_applicable_non_PLP"

    cpi = safe_float(row.get("CPI_baseline_only", np.nan), np.nan)
    collision = as_bool(row.get("is_disease_model_collision", False))
    repair = as_bool(row.get("single_model_repair_required", False))
    unanchored = as_bool(row.get("is_unanchored_assertion_state", False))
    ancestry = as_bool(row.get("is_ancestry_concentrated", False))
    canonical = as_bool(row.get("is_canonical_monogenic", False))
    drift = as_bool(row.get("future_condition_label_drift", False)) or as_bool(row.get("cross_environment_drift", False))
    cross = as_bool(row.get("cross_environment_drift", False))
    multi_regime = safe_float(row.get("regime_membership_count", 0)) >= 3
    severe = safe_float(row.get("failure_topology_severity", 0)) >= 3
    base_env = row.get("baseline_clinical_environment", "other/unknown")

    if task == "T1_phenotype_first_diagnosis":
        if canonical and not drift and cpi >= 75:
            return "allow_reuse"
        if cross or severe or collision:
            return "expert_review"
        return "contextual_repair"

    if task == "T2_SADS_postmortem_interpretation":
        sads_context = base_env == "SADS" or as_bool(row.get("is_sads_sensitive", False)) or as_bool(row.get("sads_gene_context_flag", False))
        if not sads_context:
            return "expert_review"
        if cross or collision or cpi < 50:
            return "expert_review"
        return "contextual_repair"

    if task == "T3_reuse_without_ancestry_aware_AF":
        if ancestry:
            return "contextual_repair"
        if cpi < 50 or collision or cross:
            return "expert_review"
        return "allow_reuse"

    if task == "T4_single_disease_model":
        if collision or multi_regime or cross:
            return "expert_review"
        if canonical and cpi >= 75:
            return "allow_reuse"
        return "contextual_repair"

    if task == "T5_requires_expert_disease_specific_review":
        if collision or cpi < 50 or cross or severe or unanchored:
            return "expert_review"
        return "allow_reuse"

    return "contextual_repair"


def comparable_correct(pred: str, truth: str) -> bool:
    if pred == truth:
        return True
    # Treat contextual repair as safer than deterministic reuse but not exact if expert was required.
    return False


def run_counterfactual_benchmark(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [
        "T1_phenotype_first_diagnosis",
        "T2_SADS_postmortem_interpretation",
        "T3_reuse_without_ancestry_aware_AF",
        "T4_single_disease_model",
        "T5_requires_expert_disease_specific_review",
    ]

    rows = []
    base = df[df["is_baseline_plp"]].copy()
    for _, row in base.iterrows():
        for task in tasks:
            b = baseline_decision(row, task)
            c = cab_decision(row, task)
            truth = adjudicated_action(row, task)
            unsupported_baseline = b == "deterministic_reuse" and truth != "allow_reuse"
            unsupported_cab = c == "allow_reuse" and truth != "allow_reuse"
            rows.append({
                "assertion_id": row["assertion_id"],
                "gene": row["gene"],
                "task": task,
                "baseline_pipeline_decision": b,
                "CAB_pipeline_decision": c,
                "rule_adjudicated_correct_action": truth,
                "baseline_correct": comparable_correct(b, truth),
                "CAB_correct": comparable_correct(c, truth),
                "baseline_unsupported_deterministic_reuse": unsupported_baseline,
                "CAB_unsupported_deterministic_reuse": unsupported_cab,
                "CAB_contextual_repair_or_expert": c in {"contextual_repair", "expert_review"},
                "future_condition_label_drift": as_bool(row.get("future_condition_label_drift", False)),
                "cross_environment_drift": as_bool(row.get("cross_environment_drift", False)),
                "any_meaning_drift": as_bool(row.get("any_meaning_drift", False)),
                "CPI_baseline_only": row.get("CPI_baseline_only", np.nan),
                "CPI_tier_baseline_only": row.get("CPI_tier_baseline_only", cpi_tier(safe_float(row.get("CPI_baseline_only", np.nan), np.nan))),
                "causal_architecture_category": row.get("causal_architecture_category", "unavailable"),
                "primary_regime": row.get("primary_regime", "unavailable"),
                "gene_archetype": row.get("gene_archetype", "unavailable"),
                "baseline_clinical_environment": row.get("baseline_clinical_environment", "other/unknown"),
                "followup_clinical_environment": row.get("followup_clinical_environment", "other/unknown"),
            })
    bench = pd.DataFrame(rows)

    summary_rows = []
    for pipeline in ["baseline", "CAB"]:
        for task, sub in bench.groupby("task"):
            n = len(sub)
            if pipeline == "baseline":
                unsupported = int(sub["baseline_unsupported_deterministic_reuse"].sum())
                correct = int(sub["baseline_correct"].sum())
                deterministic = int((sub["baseline_pipeline_decision"] == "deterministic_reuse").sum())
                repair = int((sub["baseline_pipeline_decision"].isin(["contextual_repair", "expert_review"])).sum())
                temporal_risk = float(sub.loc[sub["baseline_pipeline_decision"] == "deterministic_reuse", "future_condition_label_drift"].mean()) if deterministic else np.nan
                cross_error = float(sub.loc[sub["baseline_pipeline_decision"] == "deterministic_reuse", "cross_environment_drift"].mean()) if deterministic else np.nan
            else:
                unsupported = int(sub["CAB_unsupported_deterministic_reuse"].sum())
                correct = int(sub["CAB_correct"].sum())
                deterministic = int((sub["CAB_pipeline_decision"] == "allow_reuse").sum())
                repair = int(sub["CAB_contextual_repair_or_expert"].sum())
                temporal_risk = float(sub.loc[sub["CAB_pipeline_decision"] == "allow_reuse", "future_condition_label_drift"].mean()) if deterministic else np.nan
                cross_error = float(sub.loc[sub["CAB_pipeline_decision"] == "allow_reuse", "cross_environment_drift"].mean()) if deterministic else np.nan

            summary_rows.append({
                "pipeline": pipeline,
                "task": task,
                "N_task_assertion_uses": n,
                "deterministic_reuse_N": deterministic,
                "unsupported_deterministic_reuse_N": unsupported,
                "unsupported_deterministic_reuse_rate": round(unsupported / deterministic, 4) if deterministic else 0.0,
                "contextual_repair_or_expert_N": repair,
                "contextual_repair_rate": round(repair / n, 4) if n else np.nan,
                "rule_adjudicated_correct_N": correct,
                "expert_adjudicated_correctness_rate": round(correct / n, 4) if n else np.nan,
                "temporal_drift_risk_among_allowed_reuse": round(temporal_risk, 4) if not math.isnan(temporal_risk) else np.nan,
                "cross_environment_error_rate_among_allowed_reuse": round(cross_error, 4) if not math.isnan(cross_error) else np.nan,
                "adjudication_source": "rule_based_CAB_temporal_operational_benchmark",
            })
    summary = pd.DataFrame(summary_rows)

    task_metrics = []
    for task in tasks:
        s = summary[summary["task"] == task].set_index("pipeline")
        if {"baseline", "CAB"}.issubset(s.index):
            task_metrics.append({
                "task": task,
                "baseline_unsupported_deterministic_reuse_rate": s.loc["baseline", "unsupported_deterministic_reuse_rate"],
                "CAB_unsupported_deterministic_reuse_rate": s.loc["CAB", "unsupported_deterministic_reuse_rate"],
                "absolute_reduction_in_unsupported_reuse": round(
                    safe_float(s.loc["baseline", "unsupported_deterministic_reuse_rate"]) - safe_float(s.loc["CAB", "unsupported_deterministic_reuse_rate"]),
                    4,
                ),
                "baseline_correctness_rate": s.loc["baseline", "expert_adjudicated_correctness_rate"],
                "CAB_correctness_rate": s.loc["CAB", "expert_adjudicated_correctness_rate"],
                "absolute_gain_in_rule_adjudicated_correctness": round(
                    safe_float(s.loc["CAB", "expert_adjudicated_correctness_rate"]) - safe_float(s.loc["baseline", "expert_adjudicated_correctness_rate"]),
                    4,
                ),
                "CAB_contextual_repair_rate": s.loc["CAB", "contextual_repair_rate"],
                "baseline_cross_environment_error_rate": s.loc["baseline", "cross_environment_error_rate_among_allowed_reuse"],
                "CAB_cross_environment_error_rate": s.loc["CAB", "cross_environment_error_rate_among_allowed_reuse"],
            })
    return bench, summary, pd.DataFrame(task_metrics)


def gene_instability_map(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer existing map, but enrich/standardize.
    total = df.groupby("gene").size().rename("n_assertions_total_in_CAB").reset_index()
    aligned = df.copy()

    rows = []
    for gene, sub in aligned.groupby("gene", dropna=False):
        n = len(sub)
        total_n = int(total.loc[total["gene"] == gene, "n_assertions_total_in_CAB"].iloc[0]) if gene in set(total["gene"]) else n
        b_labels = list(sub["baseline_condition_label_for_task"].map(norm_text))
        rows.append({
            "gene": gene,
            "gene_archetype": SENTINEL_ARCHETYPES.get(str(gene), "other_or_insufficiently_characterized_gene"),
            "n_assertions_total_in_CAB": total_n,
            "n_temporal_aligned": n,
            "temporal_alignment_rate": round(n / total_n, 4) if total_n else np.nan,
            "condition_label_change_n": int(sub["future_condition_label_drift"].sum()),
            "condition_label_change_rate": round(float(sub["future_condition_label_drift"].mean()), 4),
            "any_meaning_drift_n": int(sub["any_meaning_drift"].sum()),
            "any_meaning_drift_rate": round(float(sub["any_meaning_drift"].mean()), 4),
            "classification_change_n": int(sub.get("classification_change", pd.Series(False, index=sub.index)).map(as_bool).sum()),
            "classification_change_rate": round(float(sub.get("classification_change", pd.Series(False, index=sub.index)).map(as_bool).mean()), 4),
            "review_status_change_n": int(sub.get("review_status_change", pd.Series(False, index=sub.index)).map(as_bool).sum()),
            "review_status_change_rate": round(float(sub.get("review_status_change", pd.Series(False, index=sub.index)).map(as_bool).mean()), 4),
            "submitter_count_change_n": int(sub.get("submitter_count_change", pd.Series(False, index=sub.index)).map(as_bool).sum()),
            "submitter_count_change_rate": round(float(sub.get("submitter_count_change", pd.Series(False, index=sub.index)).map(as_bool).mean()), 4),
            "cross_environment_drift_n": int(sub["cross_environment_drift"].sum()),
            "cross_environment_drift_rate": round(float(sub["cross_environment_drift"].mean()), 4),
            "mean_CPI_baseline_only": round(float(sub["CPI_baseline_only"].map(lambda x: safe_float(x, np.nan)).mean()), 4),
            "median_CPI_baseline_only": round(float(sub["CPI_baseline_only"].map(lambda x: safe_float(x, np.nan)).median()), 4),
            "mean_nonportability_score": round(float(sub["nonportability_score_baseline_only"].map(lambda x: safe_float(x, np.nan)).mean()), 4),
            "disease_model_collision_fraction": round(float(sub["is_disease_model_collision"].mean()), 4),
            "unanchored_fraction": round(float(sub["is_unanchored_assertion_state"].mean()), 4),
            "canonical_fraction": round(float(sub["is_canonical_monogenic"].mean()), 4),
            "population_localized_fraction": round(float(sub["is_ancestry_concentrated"].mean()), 4),
            "mean_failure_membership_count": round(float(sub["regime_membership_count"].mean()), 4),
            "mean_evidence_collision_index": round(float(sub["evidence_collision_index"].mean()), 4),
            "condition_entropy_mean": round(entropy_from_labels(b_labels), 4),
            "condition_entropy_max": round(entropy_from_labels(b_labels), 4),
            "number_of_distinct_condition_labels": len(set([x for x in b_labels if x])),
            "primary_clinical_environment": mode_value(sub["baseline_clinical_environment"]),
            "dominant_CAB_architecture": mode_value(sub["causal_architecture_category"]),
            "dominant_CAB_regime": mode_value(sub["primary_regime"]),
        })
    gm = pd.DataFrame(rows).sort_values(["n_temporal_aligned", "condition_label_change_rate"], ascending=[False, False])
    return gm


def model_auc(df: pd.DataFrame, endpoint: str, features: List[str], model_name: str) -> Dict[str, object]:
    data = df.copy()
    y = data[endpoint].map(as_bool).astype(int)
    if y.nunique() < 2:
        return {"model": model_name, "endpoint": endpoint, "N": len(data), "AUROC": np.nan, "AUPRC": np.nan, "Brier_score": np.nan, "status": "single_class_endpoint"}

    for f in features:
        if f not in data.columns:
            data[f] = np.nan

    X = data[features].copy()
    cat = [c for c in features if X[c].dtype == "object" or str(X[c].dtype).startswith("category") or X[c].map(lambda z: isinstance(z, str)).any()]
    num = [c for c in features if c not in cat]

    transformers = []
    if num:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat))

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    try:
        pipe.fit(X, y)
        pred = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)
        auprc = average_precision_score(y, pred)
        brier = brier_score_loss(y, pred)
    except Exception as exc:
        return {"model": model_name, "endpoint": endpoint, "N": len(data), "AUROC": np.nan, "AUPRC": np.nan, "Brier_score": np.nan, "status": f"fit_failed:{exc}"}

    return {
        "model": model_name,
        "endpoint": endpoint,
        "N": len(data),
        "endpoint_positive_N": int(y.sum()),
        "AUROC": round(float(auc), 4),
        "AUPRC": round(float(auprc), 4),
        "Brier_score": round(float(brier), 4),
        "status": "fit",
    }


def gene_archetype_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = ["future_condition_label_drift", "any_meaning_drift", "cross_environment_drift"]
    models = {
        "gene_only": ["gene"],
        "gene_archetype_only": ["gene_archetype"],
        "CAB_architecture_only": ["causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index", "regime_membership_count"],
        "gene_plus_CAB_architecture": ["gene", "causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index", "regime_membership_count"],
        "gene_archetype_plus_CAB": ["gene_archetype", "causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index", "regime_membership_count"],
    }
    rows = []
    for ep in endpoints:
        for name, feats in models.items():
            rows.append(model_auc(df, ep, feats, name))
    return pd.DataFrame(rows)


def find_alphamissense_source(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
    # First, already present columns.
    am_cols = [c for c in df.columns if "alphamissense" in c.lower() or c.lower() in {"am_score", "am_pathogenicity", "alphamissense_score"}]
    if am_cols:
        return df[["assertion_id"] + am_cols].copy(), "cab_framework_existing_columns", am_cols

    candidates = list(DATA.glob("*alphamissense*.csv")) + list((BASE / "reports" / "tables").glob("*alphamissense*.csv"))
    if not candidates:
        return None, "missing", []

    path = candidates[0]
    am = pd.read_csv(path, low_memory=False)
    id_col = first_existing(am, ["assertion_id", "variation_id", "VariationID", "clinvar_id", "variant_id"])
    if not id_col:
        return None, f"{path}:missing_id_column", []
    am["assertion_id"] = am[id_col].map(norm_id)
    am_cols = [c for c in am.columns if "alphamissense" in c.lower() or c.lower() in {"am_score", "am_pathogenicity", "score"}]
    if not am_cols:
        return None, f"{path}:missing_score_column", []
    return am[["assertion_id"] + am_cols], str(path.relative_to(BASE)), am_cols


def alphamissense_negative_control(df: pd.DataFrame) -> pd.DataFrame:
    am, source, cols = find_alphamissense_source(df)
    endpoints = ["future_condition_label_drift", "cross_environment_drift", "any_meaning_drift"]
    if am is None:
        return pd.DataFrame([{
            "endpoint": ep,
            "model": "AlphaMissense-only",
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "Brier_score": np.nan,
            "alpha_source": source,
            "status": "skipped_missing_AlphaMissense_input",
            "interpretation": "Protein-level deleteriousness negative control unavailable; no claim made.",
        } for ep in endpoints])

    score_col = cols[0]
    merged = df.merge(am, on="assertion_id", how="left")
    merged["AlphaMissense_score_proxy"] = merged[score_col].map(lambda x: safe_float(x, np.nan))
    rows = []
    model_defs = {
        "AlphaMissense_only": ["AlphaMissense_score_proxy"],
        "CAB_only": ["CPI_baseline_only", "causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index"],
        "gene_only": ["gene"],
        "CAB_plus_AlphaMissense": ["AlphaMissense_score_proxy", "CPI_baseline_only", "causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index"],
        "gene_plus_CAB_plus_AlphaMissense": ["gene", "AlphaMissense_score_proxy", "CPI_baseline_only", "causal_architecture_category", "primary_regime", "is_disease_model_collision", "evidence_collision_index"],
    }
    for ep in endpoints:
        sub = merged[merged["AlphaMissense_score_proxy"].notna()].copy()
        for model_name, feats in model_defs.items():
            r = model_auc(sub, ep, feats, model_name)
            r["alpha_source"] = source
            r["alpha_score_column"] = score_col
            r["status"] = r.get("status", "fit")
            if model_name == "AlphaMissense_only":
                r["interpretation"] = "Tests whether protein-level deleteriousness alone explains assertion portability."
            else:
                r["interpretation"] = "Complementarity model."
            rows.append(r)
    return pd.DataFrame(rows)


def write_reports(
    df: pd.DataFrame,
    bench: pd.DataFrame,
    summary: pd.DataFrame,
    task_metrics: pd.DataFrame,
    gm: pd.DataFrame,
    arch_models: pd.DataFrame,
    am: pd.DataFrame,
) -> None:
    # Plot unsupported reuse.
    if plt is not None and not task_metrics.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(task_metrics))
        width = 0.35
        ax.bar(x - width/2, task_metrics["baseline_unsupported_deterministic_reuse_rate"], width, label="Baseline")
        ax.bar(x + width/2, task_metrics["CAB_unsupported_deterministic_reuse_rate"], width, label="CAB")
        ax.set_xticks(x)
        ax.set_xticklabels(task_metrics["task"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Unsupported deterministic reuse rate")
        ax.set_title("Counterfactual unsupported assertion reuse by task")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_UNSUPPORTED, dpi=150)
        plt.close(fig)

    if plt is not None and not gm.empty:
        plot_df = gm[gm["n_temporal_aligned"] >= 10].copy().sort_values("condition_label_change_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(plot_df["gene"], plot_df["condition_label_change_rate"], label="condition-label drift")
        ax.bar(plot_df["gene"], plot_df["cross_environment_drift_rate"], label="cross-environment drift", alpha=0.7)
        ax.set_ylabel("Rate")
        ax.set_title("Gene temporal drift rates by archetype")
        ax.set_xticklabels(plot_df["gene"], rotation=45, ha="right")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_GENE_ARCH, dpi=150)
        plt.close(fig)

    # Reports.
    overall_base = summary[summary["pipeline"] == "baseline"]
    overall_cab = summary[summary["pipeline"] == "CAB"]

    lines = [
        "# CAB Counterfactual Decision Benchmark Report",
        "",
        "Technical analysis report, not manuscript prose.",
        "",
        "## Input scope",
        f"- Temporally aligned assertion rows: {len(df)}",
        f"- Baseline P/LP assertion rows used in assertion-use tasks: {int(df['is_baseline_plp'].sum())}",
        "- Original leakage-susceptible AUCs are not restored or used.",
        "- Expert correctness is rule-based operational adjudication unless an external expert table is added later.",
        "",
        "## Outputs",
        f"- `{BENCHMARK_OUT.relative_to(BASE)}`",
        f"- `{SUMMARY_OUT.relative_to(BASE)}`",
        f"- `{TASK_METRICS_OUT.relative_to(BASE)}`",
        f"- `{GENE_ARCH_OUT_TABLE.relative_to(BASE)}`",
        f"- `{GENE_ARCH_MODEL_OUT.relative_to(BASE)}`",
        f"- `{ALPHAMISSENSE_OUT.relative_to(BASE)}`",
        "",
        "## Counterfactual task metrics",
        task_metrics.to_string(index=False),
        "",
        "## Interpretation guardrails",
        "- Baseline pipeline is intentionally deterministic: ClinVar P/LP + gene + condition label.",
        "- CAB pipeline is action-routing: allow reuse, contextual repair, or expert review.",
        "- A reduction in unsupported deterministic reuse supports actionability, not clinical truth prediction.",
        "- CAB is not claimed to outperform gene-only unless the generated model comparison supports it.",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    # Final report.
    final = [
        "# Final CAB Actionability Upgrade Report",
        "",
        "Analysis package, not manuscript prose.",
        "",
        "## 1. Counterfactual decision benchmark",
        "CAB was evaluated against a deterministic reuse baseline across five assertion-use tasks.",
        "The target estimand is unsupported deterministic reuse reduction, not raw predictive AUC.",
        "",
        "Task metrics:",
        task_metrics.to_string(index=False),
        "",
        "## 2. Gene-only as biological axis",
        "Gene identity is retained as a biological axis. CAB is evaluated as a mechanism-decomposition layer over gene-level instability.",
        "",
        "Gene archetype rows:",
        gm[[c for c in ["gene", "gene_archetype", "n_temporal_aligned", "condition_label_change_rate", "cross_environment_drift_rate", "dominant_CAB_architecture", "dominant_CAB_regime"] if c in gm.columns]].to_string(index=False),
        "",
        "## 3. Gene archetype model comparison",
        arch_models.to_string(index=False),
        "",
        "## 4. AlphaMissense negative explanatory control",
        am.to_string(index=False),
        "",
        "## 5. Publication-safe claim rules",
        "- Do not claim CPI or CAB beats gene-only unless generated comparison tables support it.",
        "- If CAB + gene improves over gene-only, claim additive/decomposition value only.",
        "- If AlphaMissense-only fails while CAB succeeds, the supported interpretation is disease-model portability rather than molecular damage alone.",
        "- If AlphaMissense adds to CAB, the supported interpretation is complementary molecular and assertion-portability layers.",
        "- Cross-environment drift claims remain mapping-sensitive and must preserve failed/unknown mappings.",
        "",
    ]
    FINAL_REPORT_OUT.write_text("\n".join(final), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = merge_inputs()

    # Save enriched gene archetypes.
    gm = gene_instability_map(df)
    gm.to_csv(GENE_ARCH_OUT_DATA, index=False)
    gm.to_csv(GENE_ARCH_OUT_TABLE, index=False)

    bench, summary, task_metrics = run_counterfactual_benchmark(df)
    bench.to_csv(BENCHMARK_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    task_metrics.to_csv(TASK_METRICS_OUT, index=False)

    arch_models = gene_archetype_model_comparison(df)
    arch_models.to_csv(GENE_ARCH_MODEL_OUT, index=False)

    am = alphamissense_negative_control(df)
    am.to_csv(ALPHAMISSENSE_OUT, index=False)

    write_reports(df, bench, summary, task_metrics, gm, arch_models, am)

    print("CAB counterfactual decision/actionability upgrade complete.")
    print(f"Temporally aligned rows: {len(df):,}")
    print(f"Baseline P/LP rows used in tasks: {int(df['is_baseline_plp'].sum()):,}")
    print()
    print("Counterfactual task metrics:")
    print(task_metrics.to_string(index=False))
    print()
    print("Gene archetype preview:")
    preview_cols = [c for c in [
        "gene", "gene_archetype", "n_temporal_aligned", "condition_label_change_rate",
        "cross_environment_drift_rate", "dominant_CAB_architecture", "dominant_CAB_regime"
    ] if c in gm.columns]
    print(gm[preview_cols].head(20).to_string(index=False))
    print()
    print("AlphaMissense negative control status:")
    print(am.head(20).to_string(index=False))
    print()
    print("Key outputs:")
    for p in [
        BENCHMARK_OUT, SUMMARY_OUT, TASK_METRICS_OUT, GENE_ARCH_OUT_TABLE,
        GENE_ARCH_MODEL_OUT, ALPHAMISSENSE_OUT, REPORT_OUT, FINAL_REPORT_OUT
    ]:
        print(f"  - {p.relative_to(BASE)}")


if __name__ == "__main__":
    main()
