#!/usr/bin/env python3
"""Build de-circularized modifier/penetrance-boundary evidence package.

The old modifier/penetrance OR is retained only as a routing-rule
quasi-separation diagnostic. Publication-facing evidence is rebuilt around
temporal endpoints and held-out prediction summaries.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"
PACKAGE_ZIP = ROOT / "reports" / "packages" / "cab_10yr_predictor_repair_package.zip"

DOMAINS = ["hereditary_cancer", "cardiomyopathy", "inherited_arrhythmia"]
BENCHMARK_ORIGIN = "benchmark_2023-01_to_2026-04"
MODIFIER_REGIME = "modifier_penetrance_boundary"
RNG_SEED = 20260512
BOOTSTRAPS = 1000
DELTA_BOOTSTRAPS = 500
PERMUTATIONS = 2000

ENDPOINTS = [
    ("future_cross_environment_drift", "primary"),
    ("condition_label_drift", "secondary"),
    ("any_meaning_drift", "secondary"),
    ("semantic_drift_without_reclassification", "secondary"),
    ("self_loop_stable_inverse", "secondary_inverse"),
    ("classification_stable_meaning_drift", "secondary"),
]

PACKAGE_MODEL_MAP = {
    "gene-only": "gene-only",
    "regime-only": "all-regimes-only",
    "gene+regime": "gene+all-regimes",
    "all-baseline-regularized": "full-baseline-predictor",
}


@dataclass
class TestResult:
    a: int
    b: int
    c: int
    d: int
    exposed_n: int
    unexposed_n: int
    exposed_rate: float
    unexposed_rate: float
    crude_or: float
    ci_low: float
    ci_high: float
    fisher_p: float


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, QC]:
        path.mkdir(parents=True, exist_ok=True)


def read_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def safe_float(value: object) -> float:
    try:
        if value == "" or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def ci_str(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "not_estimated"
    return f"{low:.6g} to {high:.6g}"


def metric_or_blank(value: float) -> float | str:
    if value is None or math.isnan(float(value)):
        return ""
    return round(float(value), 6)


def load_benchmark_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for domain in DOMAINS:
        baseline = pd.read_csv(ROOT / "benchmark" / domain / "baseline_assertions.csv")
        endpoints = pd.read_csv(ROOT / "benchmark" / domain / "temporal_endpoints.csv")
        with (ROOT / "benchmark" / domain / "expected_metrics.json").open("r", encoding="utf-8") as f:
            expected = json.load(f)
        merged = baseline.merge(endpoints, on=["assertion_id", "domain"], how="inner")
        merged["origin_id"] = BENCHMARK_ORIGIN
        merged["baseline_snapshot"] = expected.get("baseline_snapshot", "2023-01")
        merged["followup_snapshot"] = expected.get("followup_snapshot", "2026-04")
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True, sort=False)
    regime = pd.read_csv(ROOT / "data" / "processed" / "assertion_disease_architecture_regime_map_final.csv")
    keep = [
        "assertion_id",
        "domain",
        "disease_architecture_regime",
        "PRF_required",
        "dominant_routing_action",
        "mapping_confidence",
        "mapping_reason",
    ]
    df = df.merge(regime[keep], on=["assertion_id", "domain"], how="left", validate="one_to_one")

    df["modifier_penetrance_boundary"] = df["disease_architecture_regime"].eq(MODIFIER_REGIME)
    df["condition_label_drift"] = read_bool(df["future_condition_label_drift"])
    df["any_meaning_drift"] = read_bool(df["future_any_meaning_drift"])
    df["future_cross_environment_drift"] = read_bool(df["future_cross_environment_drift"])
    df["semantic_drift_without_reclassification"] = read_bool(df["semantic_drift_without_reclassification"])
    df["future_classification_change"] = read_bool(df["future_classification_change"])
    df["self_loop_stable"] = ~(df["any_meaning_drift"] | df["future_classification_change"])
    df["self_loop_stable_inverse"] = ~df["self_loop_stable"]
    df["classification_stable_meaning_drift"] = df["any_meaning_drift"] & ~df["future_classification_change"]
    df["routing_defined_PRF_needed"] = df["PRF_required"].astype(str).str.lower().eq("yes")
    df["primary_routing_action_population_penetrance_review_PRF_needed"] = (
        df["dominant_routing_action"].astype(str).str.lower().str.contains("population/penetrance", regex=False)
    )
    df["population_or_penetrance_review_required_baseline_flag"] = read_bool(
        df["population_or_penetrance_review_required"]
    )
    return df


def table_counts(df: pd.DataFrame, endpoint: str) -> TestResult:
    exposed = df["modifier_penetrance_boundary"].astype(bool)
    y = df[endpoint].astype(bool)
    a = int((exposed & y).sum())
    b = int((exposed & ~y).sum())
    c = int((~exposed & y).sum())
    d = int((~exposed & ~y).sum())
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    crude_or = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    low = math.exp(math.log(crude_or) - 1.96 * se)
    high = math.exp(math.log(crude_or) + 1.96 * se)
    try:
        fisher_p = float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)
    except Exception:
        fisher_p = float("nan")
    exposed_n = a + b
    unexposed_n = c + d
    return TestResult(
        a=a,
        b=b,
        c=c,
        d=d,
        exposed_n=exposed_n,
        unexposed_n=unexposed_n,
        exposed_rate=a / exposed_n if exposed_n else float("nan"),
        unexposed_rate=c / unexposed_n if unexposed_n else float("nan"),
        crude_or=crude_or,
        ci_low=low,
        ci_high=high,
        fisher_p=fisher_p,
    )


def penalized_logistic_or(df: pd.DataFrame, endpoint: str) -> tuple[float, str]:
    x = df[["modifier_penetrance_boundary"]].astype(int).to_numpy()
    y = df[endpoint].astype(bool).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan"), "not_estimated_endpoint_constant"
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    model.fit(x, y)
    return float(math.exp(model.coef_[0][0])), "L2-penalized logistic fallback; Firth unavailable"


def mantel_haenszel_or(df: pd.DataFrame, endpoint: str, strata_cols: list[str]) -> tuple[float, int]:
    numerator = 0.0
    denominator = 0.0
    informative = 0
    for _, sub in df.groupby(strata_cols, dropna=False):
        if sub["modifier_penetrance_boundary"].nunique() < 2:
            continue
        y = sub[endpoint].astype(bool)
        exposed = sub["modifier_penetrance_boundary"].astype(bool)
        a = float((exposed & y).sum())
        b = float((exposed & ~y).sum())
        c = float((~exposed & y).sum())
        d = float((~exposed & ~y).sum())
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        n = a + b + c + d
        if n:
            numerator += (a * d) / n
            denominator += (b * c) / n
            informative += 1
    if denominator == 0:
        return float("nan"), informative
    return numerator / denominator, informative


def bootstrap_or_ci(df: pd.DataFrame, endpoint: str, cluster_col: str, n_boot: int = BOOTSTRAPS) -> tuple[float, float, str]:
    clusters = list(df[cluster_col].dropna().astype(str).unique())
    if len(clusters) < 2:
        return float("nan"), float("nan"), f"not_estimated_{len(clusters)}_{cluster_col}_cluster"
    rng = np.random.default_rng(RNG_SEED)
    counts_by_cluster: dict[str, np.ndarray] = {}
    for cluster, sub in df.groupby(df[cluster_col].astype(str), dropna=False):
        exposed = sub["modifier_penetrance_boundary"].astype(bool)
        y = sub[endpoint].astype(bool)
        counts_by_cluster[str(cluster)] = np.array(
            [
                int((exposed & y).sum()),
                int((exposed & ~y).sum()),
                int((~exposed & y).sum()),
                int((~exposed & ~y).sum()),
            ],
            dtype=float,
        )
    vals: list[float] = []
    for _ in range(n_boot):
        sample_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        a, b, c, d = np.sum([counts_by_cluster[c] for c in sample_clusters], axis=0)
        val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan"), float("nan"), "not_estimated_no_finite_bootstrap"
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), f"{n_boot}_{cluster_col}_cluster_bootstrap"


def build_decircularized_evidence(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint, role in ENDPOINTS:
        counts = table_counts(df, endpoint)
        pen_or, pen_status = penalized_logistic_or(df, endpoint)
        mh_domain, mh_domain_strata = mantel_haenszel_or(df, endpoint, ["domain"])
        mh_gene_domain, mh_gene_domain_strata = mantel_haenszel_or(df, endpoint, ["domain", "gene"])
        gene_low, gene_high, gene_status = bootstrap_or_ci(df, endpoint, "gene")
        origin_low, origin_high, origin_status = bootstrap_or_ci(df, endpoint, "origin_id")
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_role": role,
                "source": "three_domain_benchmark_2023-01_to_2026-04",
                "N": len(df),
                "modifier_N": counts.exposed_n,
                "non_modifier_N": counts.unexposed_n,
                "modifier_endpoint_N": counts.a,
                "non_modifier_endpoint_N": counts.c,
                "modifier_endpoint_rate": counts.exposed_rate,
                "non_modifier_endpoint_rate": counts.unexposed_rate,
                "rate_difference_modifier_minus_nonmodifier": counts.exposed_rate - counts.unexposed_rate,
                "crude_OR_Haldane_Anscombe": counts.crude_or,
                "crude_OR_CI95_low": counts.ci_low,
                "crude_OR_CI95_high": counts.ci_high,
                "firth_logistic_OR": "",
                "firth_status": "not_available_in_current_python_stack",
                "penalized_logistic_OR": pen_or,
                "penalized_logistic_status": pen_status,
                "domain_stratified_Mantel_Haenszel_OR": mh_domain,
                "domain_strata_informative_N": mh_domain_strata,
                "gene_domain_stratified_Mantel_Haenszel_OR": mh_gene_domain,
                "gene_domain_strata_informative_N": mh_gene_domain_strata,
                "gene_cluster_bootstrap_CI95_low": gene_low,
                "gene_cluster_bootstrap_CI95_high": gene_high,
                "gene_cluster_bootstrap_status": gene_status,
                "origin_cluster_bootstrap_CI95_low": origin_low,
                "origin_cluster_bootstrap_CI95_high": origin_high,
                "origin_cluster_bootstrap_status": origin_status,
                "Fisher_exact_p_descriptive_only": counts.fisher_p,
                "publication_interpretation": (
                    "eligible independent temporal endpoint; interpret as temporal portability evidence, "
                    "not routing-defined biological effect-size magnitude"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "modifier_penetrance_decircularized_evidence.csv", index=False)
    return out


def metric_at_top_fraction(y: np.ndarray, score: np.ndarray, frac: float = 0.10) -> tuple[float, float, float]:
    n = len(y)
    k = max(1, int(math.ceil(n * frac)))
    order = np.argsort(-score, kind="mergesort")[:k]
    positives = y.sum()
    precision = float(y[order].mean()) if k else float("nan")
    recall = float(y[order].sum() / positives) if positives else float("nan")
    prevalence = float(y.mean()) if n else float("nan")
    enrichment = precision / prevalence if prevalence else float("nan")
    return precision, recall, enrichment


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))
    p10, r10, enrich = metric_at_top_fraction(y, score, 0.10)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": float(brier_score_loss(y, score)),
        "precision@top10%": p10,
        "recall@top10%": r10,
        "top10%_enrichment": enrich,
    }


def matrix_for_model(train: pd.DataFrame, test: pd.DataFrame, categorical: list[str], numeric: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    if categorical:
        train_cat = pd.get_dummies(train[categorical].fillna("missing").astype(str), columns=categorical)
        test_cat = pd.get_dummies(test[categorical].fillna("missing").astype(str), columns=categorical)
        train_cat, test_cat = train_cat.align(test_cat, join="left", axis=1, fill_value=0)
        train_parts.append(train_cat.astype(float))
        test_parts.append(test_cat.astype(float))
    if numeric:
        train_num = train[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        test_num = test[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        means = train_num.mean(axis=0)
        stds = train_num.std(axis=0).replace(0, 1).fillna(1)
        train_parts.append((train_num - means) / stds)
        test_parts.append((test_num - means) / stds)
    if not train_parts:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    return pd.concat(train_parts, axis=1), pd.concat(test_parts, axis=1)


def cv_predict(df: pd.DataFrame, endpoint: str, categorical: list[str], numeric: list[str]) -> np.ndarray:
    y = df[endpoint].astype(bool).astype(int).to_numpy()
    pred = np.zeros(len(df), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    for train_idx, test_idx in cv.split(df, y):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            pred[test_idx] = y_train.mean()
            continue
        x_train, x_test = matrix_for_model(train, test, categorical, numeric)
        if x_train.shape[1] == 0:
            pred[test_idx] = y_train.mean()
            continue
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        model.fit(x_train, y_train)
        pred[test_idx] = model.predict_proba(x_test)[:, 1]
    return pred


def prepare_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "cab_strict_direct_use_allowed",
        "cab_balanced_direct_use_allowed",
        "direct_single_model_reuse_allowed",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
        "modifier_penetrance_boundary",
    ]:
        out[col] = read_bool(out[col]).astype(int)
    for col in ["submitter_count", "cab_portability_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def cluster_delta_ci(
    df: pd.DataFrame,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    baseline: str,
    augmented: str,
    cluster_col: str = "gene",
    metric: str = "AUROC",
) -> tuple[float, float, str]:
    clusters = list(df[cluster_col].dropna().astype(str).unique())
    if len(clusters) < 2:
        return float("nan"), float("nan"), "not_estimated"
    rng = np.random.default_rng(RNG_SEED)
    grouped_indices = {
        cluster: df.index[df[cluster_col].astype(str).eq(cluster)].to_numpy()
        for cluster in clusters
    }
    vals: list[float] = []
    for _ in range(DELTA_BOOTSTRAPS):
        sample_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped_indices[c] for c in sample_clusters])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        m_aug = metrics_from_scores(yy, scores[augmented][idx])[metric]
        m_base = metrics_from_scores(yy, scores[baseline][idx])[metric]
        if math.isfinite(m_aug) and math.isfinite(m_base):
            vals.append(m_aug - m_base)
    if not vals:
        return float("nan"), float("nan"), "not_estimated_no_finite_bootstrap"
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), f"{DELTA_BOOTSTRAPS}_{cluster_col}_cluster_bootstrap"


def read_package_csv(member: str) -> pd.DataFrame:
    with zipfile.ZipFile(PACKAGE_ZIP) as zf:
        with zf.open(f"cab_10yr_predictor_repair_package/reports/tables/{member}") as f:
            return pd.read_csv(f)


def origin_bootstrap_delta(
    by_origin: pd.DataFrame,
    baseline_model: str,
    augmented_model: str,
    metric: str,
) -> tuple[float, float, str]:
    pivot = by_origin.pivot(index="origin_id", columns="model_normalized", values=metric)
    if baseline_model not in pivot or augmented_model not in pivot:
        return float("nan"), float("nan"), "not_available"
    pivot = pivot[[baseline_model, augmented_model]].dropna()
    if len(pivot) < 2:
        return float("nan"), float("nan"), f"not_estimated_{len(pivot)}_origins"
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    deltas = pivot[augmented_model] - pivot[baseline_model]
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(deltas.to_numpy(), size=len(deltas), replace=True)
        vals.append(float(np.mean(sample)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), f"{BOOTSTRAPS}_paired_origin_bootstrap"


def build_incremental_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    endpoint = "future_cross_environment_drift"

    package = read_package_csv("predictor_temporal_nested_results.csv")
    package = package[
        (package["endpoint"].eq(endpoint))
        & (package["domain"].eq("all"))
        & (package["model"].isin(PACKAGE_MODEL_MAP))
        & (package["status"].eq("evaluated"))
    ].copy()
    package["model_normalized"] = package["model"].map(PACKAGE_MODEL_MAP)
    package = package.rename(
        columns={
            "precision@10%": "precision@top10%",
            "recall@10%": "recall@top10%",
            "enrichment_over_random": "top10%_enrichment",
            "Brier_score": "Brier",
        }
    )
    package_metrics = ["AUROC", "AUPRC", "Brier", "precision@top10%", "recall@top10%", "top10%_enrichment"]
    for model_name, sub in package.groupby("model_normalized", sort=False):
        row = {
            "row_type": "model_performance",
            "source": "rolling_origin_predictor_repair_package",
            "endpoint": endpoint,
            "model": model_name,
            "baseline_model": "",
            "augmented_model": "",
            "N": int(sub["N"].sum()),
            "origins_evaluated": sub["origin_id"].nunique(),
            "validation_type": "rolling-origin held-out temporal backtest",
            "availability_note": "available in package aggregate outputs",
        }
        for metric in package_metrics:
            row[metric] = float(sub[metric].mean())
        rows.append(row)

    for missing_model in ["modifier-flag-only", "gene+modifier-flag"]:
        rows.append(
            {
                "row_type": "model_performance",
                "source": "rolling_origin_predictor_repair_package",
                "endpoint": endpoint,
                "model": missing_model,
                "baseline_model": "",
                "augmented_model": "",
                "N": "",
                "origins_evaluated": package["origin_id"].nunique(),
                "validation_type": "rolling-origin held-out temporal backtest",
                "availability_note": "not available because package excludes row-level modifier feature matrices",
            }
        )

    comparisons = [
        ("gene-only", "gene+all-regimes", "primary_gene_only_to_gene_all_regimes"),
        ("gene-only", "all-regimes-only", "regime_only_vs_gene_only"),
        ("gene+all-regimes", "full-baseline-predictor", "full_baseline_vs_gene_all_regimes"),
    ]
    for baseline, augmented, label in comparisons:
        pivot = package.pivot(index="origin_id", columns="model_normalized", values=package_metrics)
        if baseline not in package["model_normalized"].values or augmented not in package["model_normalized"].values:
            continue
        row = {
            "row_type": "incremental_delta",
            "source": "rolling_origin_predictor_repair_package",
            "endpoint": endpoint,
            "model": label,
            "baseline_model": baseline,
            "augmented_model": augmented,
            "N": "",
            "origins_evaluated": package["origin_id"].nunique(),
            "validation_type": "rolling-origin held-out temporal backtest",
            "availability_note": "paired by held-out origin",
        }
        for metric in package_metrics:
            per_origin = package.pivot(index="origin_id", columns="model_normalized", values=metric)
            if baseline in per_origin and augmented in per_origin:
                deltas = (per_origin[augmented] - per_origin[baseline]).dropna()
                row[f"delta_{metric}"] = float(deltas.mean()) if len(deltas) else ""
                low, high, status = origin_bootstrap_delta(package, baseline, augmented, metric)
                row[f"delta_{metric}_CI95_low"] = low
                row[f"delta_{metric}_CI95_high"] = high
                row[f"delta_{metric}_CI_status"] = status
        rows.append(row)

        for left_out in sorted(package["origin_id"].unique()):
            rest = package[~package["origin_id"].eq(left_out)]
            loo = {
                "source": "rolling_origin_predictor_repair_package",
                "left_out_origin": left_out,
                "comparison": label,
                "baseline_model": baseline,
                "augmented_model": augmented,
                "remaining_origins": rest["origin_id"].nunique(),
            }
            for metric in package_metrics:
                per_origin = rest.pivot(index="origin_id", columns="model_normalized", values=metric)
                if baseline in per_origin and augmented in per_origin:
                    loo[f"delta_{metric}"] = float((per_origin[augmented] - per_origin[baseline]).dropna().mean())
            loo_rows.append(loo)

    pred_df = prepare_prediction_frame(df)
    model_specs = {
        "gene-only": (["gene"], []),
        "modifier-flag-only": ([], ["modifier_penetrance_boundary"]),
        "all-regimes-only": (["disease_architecture_regime"], []),
        "gene+modifier-flag": (["gene"], ["modifier_penetrance_boundary"]),
        "gene+all-regimes": (["gene", "disease_architecture_regime"], []),
        "full-baseline-predictor": (
            ["gene", "domain", "baseline_environment", "disease_architecture_regime", "classification", "review_status"],
            [
                "submitter_count",
                "cab_portability_score",
                "cab_strict_direct_use_allowed",
                "cab_balanced_direct_use_allowed",
                "direct_single_model_reuse_allowed",
                "contextual_repair_required",
                "disease_specific_expert_review_required",
                "population_or_penetrance_review_required",
            ],
        ),
    }
    y = pred_df[endpoint].astype(bool).astype(int).to_numpy()
    scores: dict[str, np.ndarray] = {}
    for model_name, (cats, nums) in model_specs.items():
        scores[model_name] = cv_predict(pred_df, endpoint, cats, nums)
        metrics = metrics_from_scores(y, scores[model_name])
        row = {
            "row_type": "model_performance",
            "source": "three_domain_benchmark_blocked_5fold_cv",
            "endpoint": endpoint,
            "model": model_name,
            "baseline_model": "",
            "augmented_model": "",
            "N": len(pred_df),
            "origins_evaluated": 1,
            "validation_type": "baseline-snapshot temporal endpoint, 5-fold CV for feature-family contrast",
            "availability_note": "row-level modifier feature available; not a rolling-origin package metric",
        }
        row.update(metrics)
        rows.append(row)

    benchmark_comparisons = [
        ("gene-only", "gene+modifier-flag", "primary_gene_only_to_gene_modifier_flag"),
        ("gene-only", "gene+all-regimes", "primary_gene_only_to_gene_all_regimes"),
        ("gene-only", "all-regimes-only", "regime_only_vs_gene_only"),
        ("gene+all-regimes", "full-baseline-predictor", "full_baseline_vs_gene_all_regimes"),
    ]
    for baseline, augmented, label in benchmark_comparisons:
        row = {
            "row_type": "incremental_delta",
            "source": "three_domain_benchmark_blocked_5fold_cv",
            "endpoint": endpoint,
            "model": label,
            "baseline_model": baseline,
            "augmented_model": augmented,
            "N": len(pred_df),
            "origins_evaluated": 1,
            "validation_type": "baseline-snapshot temporal endpoint, 5-fold CV for feature-family contrast",
            "availability_note": "origin-paired CI unavailable for single benchmark origin; gene-cluster CI reported",
        }
        for metric in package_metrics:
            base_m = metrics_from_scores(y, scores[baseline])[metric]
            aug_m = metrics_from_scores(y, scores[augmented])[metric]
            row[f"delta_{metric}"] = aug_m - base_m
            if metric == "AUROC":
                low, high, status = cluster_delta_ci(pred_df, y, scores, baseline, augmented, "gene", metric)
                row[f"delta_{metric}_CI95_low"] = low
                row[f"delta_{metric}_CI95_high"] = high
                row[f"delta_{metric}_CI_status"] = status
            else:
                row[f"delta_{metric}_CI95_low"] = ""
                row[f"delta_{metric}_CI95_high"] = ""
                row[f"delta_{metric}_CI_status"] = "not_bootstrapped_for_benchmark_secondary_metric"
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "modifier_penetrance_incremental_prediction.csv", index=False)
    loo_out = pd.DataFrame(loo_rows)
    loo_out.to_csv(TABLES / "modifier_penetrance_leave_one_origin_out.csv", index=False)
    figure_payload = {
        "package": package,
        "benchmark_scores": scores,
        "benchmark_y": y,
        "benchmark_df": pred_df,
    }
    return out, loo_out, figure_payload


def matched_permutation(df: pd.DataFrame, endpoint: str) -> tuple[dict[str, object], np.ndarray]:
    blocks = []
    for key, sub in df.groupby(["domain", "gene", "origin_id"], dropna=False):
        if sub["modifier_penetrance_boundary"].nunique() < 2:
            continue
        blocks.append((key, sub.copy()))
    if not blocks:
        return {
            "endpoint": endpoint,
            "strata_N": 0,
            "status": "no_matched_blocks",
        }, np.array([])

    exposed_pos = exposed_n = unexposed_pos = unexposed_n = 0
    block_diffs = []
    block_arrays: list[tuple[np.ndarray, int]] = []
    for _, sub in blocks:
        exposed = sub["modifier_penetrance_boundary"].astype(bool)
        y = sub[endpoint].astype(bool)
        e_n = int(exposed.sum())
        u_n = int((~exposed).sum())
        e_pos = int((exposed & y).sum())
        u_pos = int((~exposed & y).sum())
        exposed_pos += e_pos
        unexposed_pos += u_pos
        exposed_n += e_n
        unexposed_n += u_n
        block_diffs.append((e_pos / e_n if e_n else 0.0) - (u_pos / u_n if u_n else 0.0))
        block_arrays.append((y.to_numpy(dtype=int), e_n))
    observed = (exposed_pos / exposed_n) - (unexposed_pos / unexposed_n)
    cmh_or, cmh_strata = mantel_haenszel_or(pd.concat([sub for _, sub in blocks]), endpoint, ["domain", "gene", "origin_id"])
    rng = np.random.default_rng(RNG_SEED)
    null = np.zeros(PERMUTATIONS, dtype=float)
    for i in range(PERMUTATIONS):
        e_pos_perm = 0
        total_pos_perm = 0
        for y_array, e_n in block_arrays:
            total_pos_perm += int(y_array.sum())
            chosen = rng.choice(len(y_array), size=e_n, replace=False)
            e_pos_perm += int(y_array[chosen].sum())
        u_pos_perm = total_pos_perm - e_pos_perm
        null[i] = (e_pos_perm / exposed_n) - (u_pos_perm / unexposed_n)
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (PERMUTATIONS + 1)
    result = {
        "endpoint": endpoint,
        "source": "three_domain_benchmark_2023-01_to_2026-04",
        "strata_N": len(blocks),
        "modifier_N_in_matched_blocks": exposed_n,
        "non_modifier_N_in_matched_blocks": unexposed_n,
        "modifier_endpoint_rate_matched": exposed_pos / exposed_n,
        "non_modifier_endpoint_rate_matched": unexposed_pos / unexposed_n,
        "matched_endpoint_rate_difference": observed,
        "mean_block_rate_difference": float(np.mean(block_diffs)),
        "Cochran_Mantel_Haenszel_OR": cmh_or,
        "CMH_informative_strata_N": cmh_strata,
        "permutation_p_value_within_gene_domain_origin_blocks": p,
        "permutation_iterations": PERMUTATIONS,
        "permutation_null_mean": float(np.mean(null)),
        "permutation_null_CI95_low": float(np.percentile(null, 2.5)),
        "permutation_null_CI95_high": float(np.percentile(null, 97.5)),
        "status": "matched_blocks_evaluated",
    }
    return result, null


def build_matched_and_permutation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    matched_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    primary_null = np.array([])
    for endpoint, _ in ENDPOINTS:
        result, null = matched_permutation(df, endpoint)
        matched_rows.append(result)
        permutation_rows.append(
            {
                "endpoint": endpoint,
                "source": result.get("source", "three_domain_benchmark_2023-01_to_2026-04"),
                "observed_matched_rate_difference": result.get("matched_endpoint_rate_difference", ""),
                "null_mean": result.get("permutation_null_mean", ""),
                "null_CI95_low": result.get("permutation_null_CI95_low", ""),
                "null_CI95_high": result.get("permutation_null_CI95_high", ""),
                "permutation_p_value": result.get("permutation_p_value_within_gene_domain_origin_blocks", ""),
                "permutation_iterations": result.get("permutation_iterations", ""),
                "block_definition": "domain + gene + origin_id",
                "status": result.get("status", ""),
            }
        )
        if endpoint == "future_cross_environment_drift":
            primary_null = null
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(TABLES / "modifier_penetrance_matched_gene_domain_tests.csv", index=False)
    perms = pd.DataFrame(permutation_rows)
    perms.to_csv(TABLES / "modifier_penetrance_permutation_tests.csv", index=False)
    return matched, perms, primary_null


def build_quarantine(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    old = pd.read_csv(TABLES / "disease_architecture_regime_enrichment_tests.csv")
    old_row = old[old["regime"].eq(MODIFIER_REGIME)].iloc[0].to_dict()
    rows.append(
        {
            "row_type": "old_OR_quarantine",
            "endpoint_or_action": "population_penetrance_review / PRF-needed routing",
            "label": "routing-defined endpoint; quasi-separation; not independent biological effect size",
            "N": int(old_row.get("N", 16125)),
            "modifier_N": int(old_row.get("N", 16125)),
            "modifier_positive_N": int(old_row.get("endpoint_positives", 16125)),
            "modifier_positive_rate": old_row.get("regime_rate", 1.0),
            "non_modifier_positive_rate": old_row.get("background_rate", ""),
            "OR": old_row.get("OR", ""),
            "CI95_low": old_row.get("CI95_low", ""),
            "CI95_high": old_row.get("CI95_high", ""),
            "p_value_descriptive_only": old_row.get("p_value", ""),
            "publication_use": "QC only; do not report as biological effect size",
            "source": "reports/tables/disease_architecture_regime_enrichment_tests.csv",
        }
    )

    for endpoint in [
        "routing_defined_PRF_needed",
        "primary_routing_action_population_penetrance_review_PRF_needed",
        "population_or_penetrance_review_required_baseline_flag",
    ]:
        counts = table_counts(df, endpoint)
        rows.append(
            {
                "row_type": "negative_control_circularity_audit",
                "endpoint_or_action": endpoint,
                "label": "routing-defined endpoint; expected quasi-separation",
                "N": len(df),
                "modifier_N": counts.exposed_n,
                "modifier_positive_N": counts.a,
                "modifier_positive_rate": counts.exposed_rate,
                "non_modifier_N": counts.unexposed_n,
                "non_modifier_positive_N": counts.c,
                "non_modifier_positive_rate": counts.unexposed_rate,
                "OR": counts.crude_or,
                "CI95_low": counts.ci_low,
                "CI95_high": counts.ci_high,
                "p_value_descriptive_only": counts.fisher_p,
                "publication_use": "QC only; confirms routing circularity hazard",
                "source": "three_domain_benchmark_baseline_routing_fields",
            }
        )

    mod = df[df["modifier_penetrance_boundary"]]
    for action, n in mod["dominant_routing_action"].value_counts(dropna=False).items():
        rows.append(
            {
                "row_type": "modifier_routing_distribution",
                "endpoint_or_action": action,
                "label": "routing distribution within modifier/penetrance-boundary class",
                "N": len(mod),
                "modifier_N": len(mod),
                "modifier_positive_N": int(n),
                "modifier_positive_rate": int(n) / len(mod) if len(mod) else float("nan"),
                "publication_use": "QC/supporting routing distribution only",
                "source": "data/processed/assertion_disease_architecture_regime_map_final.csv",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "modifier_penetrance_quasi_separation_quarantine.csv", index=False)
    return out


def lookup_delta(prediction: pd.DataFrame, source: str, model: str, metric: str = "AUROC") -> tuple[float, float, float]:
    row = prediction[
        prediction["row_type"].eq("incremental_delta")
        & prediction["source"].eq(source)
        & prediction["model"].eq(model)
    ]
    if row.empty:
        return float("nan"), float("nan"), float("nan")
    r = row.iloc[0]
    return (
        safe_float(r.get(f"delta_{metric}", "")),
        safe_float(r.get(f"delta_{metric}_CI95_low", "")),
        safe_float(r.get(f"delta_{metric}_CI95_high", "")),
    )


def build_qc_markdown(
    evidence: pd.DataFrame,
    prediction: pd.DataFrame,
    matched: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> None:
    primary = evidence[evidence["endpoint"].eq("future_cross_environment_drift")].iloc[0]
    matched_primary = matched[matched["endpoint"].eq("future_cross_environment_drift")].iloc[0]
    pkg_delta, pkg_low, pkg_high = lookup_delta(
        prediction,
        "rolling_origin_predictor_repair_package",
        "primary_gene_only_to_gene_all_regimes",
    )
    bench_mod_delta, bench_mod_low, bench_mod_high = lookup_delta(
        prediction,
        "three_domain_benchmark_blocked_5fold_cv",
        "primary_gene_only_to_gene_modifier_flag",
    )
    old_or = quarantine[quarantine["row_type"].eq("old_OR_quarantine")].iloc[0]
    text = f"""# Modifier/Penetrance Claim Repair

## Status

The old modifier/penetrance-boundary OR is quarantined in QC, not removed.

- Old routing-defined OR: {safe_float(old_or["OR"]):.2f}
- Modifier/penetrance-boundary N: {int(old_or["modifier_N"]):,}
- Label: routing-defined endpoint; quasi-separation; not independent biological effect size
- Publication use: QC only

## Replacement Evidence

Primary independent endpoint: `future_cross_environment_drift`.

Three-domain benchmark result, baseline 2023-01 to follow-up 2026-04:

- Modifier endpoint rate: {primary["modifier_endpoint_rate"]:.4f}
- Non-modifier endpoint rate: {primary["non_modifier_endpoint_rate"]:.4f}
- Crude Haldane-Anscombe OR: {primary["crude_OR_Haldane_Anscombe"]:.4f} ({primary["crude_OR_CI95_low"]:.4f} to {primary["crude_OR_CI95_high"]:.4f})
- Penalized logistic fallback OR: {primary["penalized_logistic_OR"]:.4f}
- Gene-cluster bootstrap CI for OR: {ci_str(primary["gene_cluster_bootstrap_CI95_low"], primary["gene_cluster_bootstrap_CI95_high"])}

Matched anti-circularity test within domain-gene-origin blocks:

- Matched strata: {int(matched_primary["strata_N"])}
- Matched rate difference: {matched_primary["matched_endpoint_rate_difference"]:.4f}
- CMH OR: {matched_primary["Cochran_Mantel_Haenszel_OR"]:.4f}
- Within-block permutation p-value: {matched_primary["permutation_p_value_within_gene_domain_origin_blocks"]:.6g}

Incremental prediction:

- Rolling-origin package gene-only to gene + all regimes delta AUROC: {pkg_delta:.4f} ({pkg_low:.4f} to {pkg_high:.4f}, paired by origin)
- Benchmark row-level gene-only to gene + modifier flag delta AUROC: {bench_mod_delta:.4f} ({bench_mod_low:.4f} to {bench_mod_high:.4f}, gene-cluster bootstrap)

## Boundary

The rolling-origin predictor repair package provides aggregate held-out origin metrics for gene-only, regime-only, gene + regime, and full-baseline predictors. It does not include row-level modifier feature matrices, so `gene + modifier flag` is evaluated in the three-domain benchmark snapshot with cross-validated row-level predictions and gene-cluster uncertainty, not as an origin-paired package metric.

## Publication-Safe Wording

Modifier/penetrance-boundary is a conditional-liability portability class supported by its N, routing behavior, matched within-gene/domain temporal tests, and independent temporal prediction contrasts. It is not supported by the magnitude of the routing-defined OR.

## Forbidden Wording

- Do not state that OR = 39,870.69 is a biological effect size.
- Do not use population/penetrance review, PRF-needed routing, or primary routing action as the primary endpoint.
- Do not present the quarantined OR in the main figure as biological enrichment.
"""
    (QC / "modifier_penetrance_claim_repair.md").write_text(text, encoding="utf-8")


def build_figure(
    evidence: pd.DataFrame,
    prediction: pd.DataFrame,
    matched: pd.DataFrame,
    null: np.ndarray,
    quarantine: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    old_or = safe_float(quarantine[quarantine["row_type"].eq("old_OR_quarantine")].iloc[0]["OR"])
    ax_a.barh(["Quarantined OR"], [math.log10(old_or)], color="#8f8f8f")
    ax_a.set_xlabel("log10(OR)")
    ax_a.set_title("A. Quarantined Routing OR", loc="left", fontweight="bold")
    ax_a.text(
        0.02,
        0.18,
        "Routing-rule quasi-separation;\nnot biological effect size.",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff2cc", "edgecolor": "#b38f00"},
    )
    ax_a.set_xlim(0, max(5.2, math.log10(old_or) + 0.4))
    ax_a.grid(axis="x", alpha=0.25)

    deltas = []
    labels = []
    colors = []
    for source, model, label, color in [
        (
            "rolling_origin_predictor_repair_package",
            "primary_gene_only_to_gene_all_regimes",
            "Gene -> gene + all regimes\n(rolling origins)",
            "#2f6fbb",
        ),
        (
            "three_domain_benchmark_blocked_5fold_cv",
            "primary_gene_only_to_gene_modifier_flag",
            "Gene -> gene + modifier\n(benchmark CV)",
            "#c44e52",
        ),
        (
            "rolling_origin_predictor_repair_package",
            "full_baseline_vs_gene_all_regimes",
            "Gene + regimes -> full baseline\n(rolling origins)",
            "#55a868",
        ),
    ]:
        delta, low, high = lookup_delta(prediction, source, model)
        deltas.append((delta, low, high))
        labels.append(label)
        colors.append(color)
    y_pos = np.arange(len(labels))
    means = [d[0] for d in deltas]
    xerr_low = [max(0, d[0] - d[1]) if math.isfinite(d[1]) else 0 for d in deltas]
    xerr_high = [max(0, d[2] - d[0]) if math.isfinite(d[2]) else 0 for d in deltas]
    ax_b.barh(y_pos, means, color=colors, alpha=0.88)
    ax_b.errorbar(means, y_pos, xerr=[xerr_low, xerr_high], fmt="none", ecolor="black", capsize=3, lw=1)
    ax_b.set_yticks(y_pos, labels)
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_xlabel("Delta AUROC over comparator")
    ax_b.set_title("B. Independent Temporal Prediction", loc="left", fontweight="bold")
    pkg_perf = prediction[
        prediction["row_type"].eq("model_performance")
        & prediction["source"].eq("rolling_origin_predictor_repair_package")
    ]
    top10_gene_regime = safe_float(
        pkg_perf[pkg_perf["model"].eq("gene+all-regimes")]["top10%_enrichment"].iloc[0]
    )
    top10_full = safe_float(
        pkg_perf[pkg_perf["model"].eq("full-baseline-predictor")]["top10%_enrichment"].iloc[0]
    )
    loo_path = TABLES / "modifier_penetrance_leave_one_origin_out.csv"
    loo_note = ""
    if loo_path.exists():
        loo = pd.read_csv(loo_path)
        primary_loo = loo[loo["comparison"].eq("primary_gene_only_to_gene_all_regimes")]
        if not primary_loo.empty:
            loo_note = (
                f"\nLOO delta AUROC: {primary_loo['delta_AUROC'].min():.3f}"
                f" to {primary_loo['delta_AUROC'].max():.3f}"
            )
    ax_b.text(
        0.98,
        0.05,
        f"Top-10% enrichment:\ngene+regime {top10_gene_regime:.2f}x; full {top10_full:.2f}x{loo_note}",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f3f6ee", "edgecolor": "#8aa05f"},
    )
    ax_b.grid(axis="x", alpha=0.25)

    primary = matched[matched["endpoint"].eq("future_cross_environment_drift")].iloc[0]
    observed = safe_float(primary["matched_endpoint_rate_difference"])
    if len(null):
        ax_c.hist(null, bins=35, color="#8172b2", alpha=0.75, density=True)
        ax_c.axvline(observed, color="#c44e52", lw=2, label="observed")
        ax_c.axvline(0, color="black", lw=0.8)
        ax_c.legend(frameon=False)
    ax_c.set_xlabel("Matched rate difference")
    ax_c.set_ylabel("Null density")
    ax_c.set_title("C. Gene-Domain-Origin Permutation", loc="left", fontweight="bold")
    ax_c.text(
        0.02,
        0.95,
        f"p = {safe_float(primary['permutation_p_value_within_gene_domain_origin_blocks']):.3g}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
    )

    ax_d.axis("off")
    ax_d.set_title("D. Publication-Safe Interpretation", loc="left", fontweight="bold")
    interpretation = (
        "Modifier/penetrance-boundary is a conditional-liability portability class.\n\n"
        "Support comes from N, routing behavior, matched temporal tests, and held-out\n"
        "forecasting contrasts. The routing-defined OR magnitude stays in QC."
    )
    ax_d.text(0.02, 0.78, interpretation, ha="left", va="top", fontsize=11)
    primary_ev = evidence[evidence["endpoint"].eq("future_cross_environment_drift")].iloc[0]
    ax_d.text(
        0.02,
        0.36,
        (
            f"Primary endpoint OR: {primary_ev['crude_OR_Haldane_Anscombe']:.3f}\n"
            f"Matched rate difference: {observed:.3f}\n"
            f"Modifier class N: {int(primary_ev['modifier_N']):,}"
        ),
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eef5f9", "edgecolor": "#6a8fb3"},
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES / "modifier_penetrance_decircularized_evidence.svg", format="svg")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load_benchmark_rows()
    evidence = build_decircularized_evidence(df)
    prediction, _, figure_payload = build_incremental_prediction(df)
    matched, _, null = build_matched_and_permutation(df)
    quarantine = build_quarantine(df)
    build_qc_markdown(evidence, prediction, matched, quarantine)
    build_figure(evidence, prediction, matched, null, quarantine)

    print("Wrote modifier/penetrance de-circularized evidence package:")
    for path in [
        TABLES / "modifier_penetrance_decircularized_evidence.csv",
        TABLES / "modifier_penetrance_incremental_prediction.csv",
        TABLES / "modifier_penetrance_matched_gene_domain_tests.csv",
        TABLES / "modifier_penetrance_leave_one_origin_out.csv",
        TABLES / "modifier_penetrance_permutation_tests.csv",
        TABLES / "modifier_penetrance_quasi_separation_quarantine.csv",
        QC / "modifier_penetrance_claim_repair.md",
        FIGURES / "modifier_penetrance_decircularized_evidence.svg",
    ]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
