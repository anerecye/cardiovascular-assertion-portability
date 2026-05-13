#!/usr/bin/env python3
"""Build silver-standard CAB endpoint and robustness stress tests.

This package focuses on reviewer-facing anti-circularity and anti-overfitting
questions:

* proxy adjudication of condition drift using independent condition IDs,
  environment mappings, and ClinGen/VCEP coverage artifacts available locally;
* strict endpoint hierarchy and main-claim recomputation;
* leave-one-gene/family/environment/domain-out validation;
* ontology-only baselines;
* calibration, decision curves, submitter-stratified analyses;
* stable-domain negative controls, direct-use safety, overrestriction, SADS,
  ablation, AUPRC/enrichment, and rule-selected case studies.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import build_cab_hardcore_evidence_upgrade as hardcore


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
QC = ROOT / "reports" / "qc"
ADJ = ROOT / "reports" / "adjudication"

RNG = np.random.default_rng(20260513)
BOOT = 300


def read_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[_/,-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(value: object) -> set[str]:
    text = norm_text(value)
    if not text:
        return set()
    stop = {"of", "the", "and", "or", "type", "disease", "syndrome", "familial", "primary"}
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1 and t not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def string_similarity(a: object, b: object) -> float:
    aa = norm_text(a)
    bb = norm_text(b)
    if not aa and not bb:
        return float("nan")
    return float(SequenceMatcher(None, aa, bb).ratio())


def parse_ids(value: object, prefix: str) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value)
    if prefix == "MONDO":
        return set(re.findall(r"MONDO:MONDO:(\d+)", text))
    if prefix == "OMIM":
        return set(re.findall(r"OMIM:(PS)?(\d+)", text))
    if prefix == "HPO":
        return set(re.findall(r"(?:Human Phenotype Ontology:)?HP:(\d+)", text))
    return set()


def parse_omim_family(value: object) -> set[str]:
    raw = parse_ids(value, "OMIM")
    fam: set[str] = set()
    for item in raw:
        if isinstance(item, tuple):
            is_ps, ident = item
            fam.add(f"PS{ident}" if is_ps else ident)
        else:
            fam.add(str(item))
    return fam


def model_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_train: list[pd.DataFrame] = []
    parts_test: list[pd.DataFrame] = []
    if categorical:
        tr = pd.get_dummies(train[categorical].fillna("missing").astype(str), columns=categorical)
        te = pd.get_dummies(test[categorical].fillna("missing").astype(str), columns=categorical)
        tr, te = tr.align(te, join="left", axis=1, fill_value=0)
        parts_train.append(tr.astype(float))
        parts_test.append(te.astype(float))
    if numeric:
        trn = train[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        ten = test[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        mean = trn.mean(axis=0)
        std = trn.std(axis=0).replace(0, 1).fillna(1)
        parts_train.append((trn - mean) / std)
        parts_test.append((ten - mean) / std)
    if not parts_train:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    return pd.concat(parts_train, axis=1), pd.concat(parts_test, axis=1)


def fit_predict_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    endpoint: str,
    categorical: list[str],
    numeric: list[str],
) -> np.ndarray:
    y_train = train[endpoint].astype(int).to_numpy()
    if len(test) == 0:
        return np.array([])
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(np.mean(y_train)) if len(y_train) else 0.0, len(test))
    x_train, x_test = model_matrix(train, test, categorical, numeric)
    if x_train.shape[1] == 0:
        return np.repeat(float(np.mean(y_train)), len(test))
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def oof_predict(df: pd.DataFrame, endpoint: str, categorical: list[str], numeric: list[str]) -> np.ndarray:
    y = df[endpoint].astype(int).to_numpy()
    pred = np.zeros(len(df), dtype=float)
    if len(np.unique(y)) < 2:
        pred[:] = float(np.mean(y)) if len(y) else 0.0
        return pred
    min_class = int(np.bincount(y).min())
    splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=20260513)
    for tr_idx, te_idx in cv.split(df, y):
        pred[te_idx] = fit_predict_split(df.iloc[tr_idx], df.iloc[te_idx], endpoint, categorical, numeric)
    return pred


def safe_auc(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pred))


def metrics(y: np.ndarray, pred: np.ndarray, budget: float = 0.10) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "Brier": float("nan"),
            "precision_at_top10": float("nan"),
            "recall_at_top10": float("nan"),
            "lift_at_top10": float("nan"),
            "number_needed_to_review_top10": float("nan"),
        }
    k = max(1, int(math.ceil(len(y) * budget)))
    order = np.argsort(-pred)
    top = order[:k]
    prevalence = float(y.mean())
    precision = float(y[top].mean()) if k else float("nan")
    recall = float(y[top].sum() / y.sum()) if y.sum() else float("nan")
    lift = precision / prevalence if prevalence > 0 else float("nan")
    nnr = 1 / precision if precision > 0 else float("inf")
    return {
        "AUROC": safe_auc(y, pred),
        "AUPRC": float(average_precision_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "Brier": float(brier_score_loss(y, np.clip(pred, 0, 1))) if len(np.unique(y)) > 1 else float("nan"),
        "precision_at_top10": precision,
        "recall_at_top10": recall,
        "lift_at_top10": lift,
        "number_needed_to_review_top10": nnr,
    }


def bootstrap_delta(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int = BOOT) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    observed = safe_auc(y, p2) - safe_auc(y, p1)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(safe_auc(y[idx], p2[idx]) - safe_auc(y[idx], p1[idx]))
    if not vals:
        return observed, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(observed), float(lo), float(hi)


def precision_at_budget(y: np.ndarray, pred: np.ndarray, budget: float) -> tuple[float, float, float, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    k = max(1, int(math.ceil(len(y) * budget)))
    idx = np.argsort(-pred)[:k]
    precision = float(y[idx].mean()) if len(idx) else float("nan")
    recall = float(y[idx].sum() / y.sum()) if y.sum() else float("nan")
    lift = precision / float(y.mean()) if y.mean() > 0 else float("nan")
    nnr = 1 / precision if precision > 0 else float("inf")
    return precision, recall, lift, nnr


def add_condition_environment_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variation_id_clean"] = out["variation_id"].astype(str).str.replace(r"^[A-Z]+_", "", regex=True)
    out["baseline_condition_proxy"] = out["input_condition_label"].fillna("")
    out["followup_condition_proxy"] = out.get("followup_condition_label", "").fillna("")
    out["baseline_env_proxy"] = out["baseline_environment"].fillna("unknown")
    out["followup_env_proxy"] = out["baseline_environment"].fillna("unknown")
    out["baseline_condition_ids_proxy"] = ""
    out["followup_condition_ids_proxy"] = ""

    cancer_path = ROOT / "data" / "processed" / "cancer_condition_environment_map.csv"
    if cancer_path.exists():
        cancer = pd.read_csv(cancer_path)
        cancer_cols = {
            "condition_label_baseline": "cancer_baseline_condition",
            "condition_label_followup": "cancer_followup_condition",
            "environment_baseline": "cancer_baseline_env",
            "environment_followup": "cancer_followup_env",
        }
        cancer = cancer.rename(columns=cancer_cols)
        out = out.merge(
            cancer[["assertion_id", *cancer_cols.values()]],
            on="assertion_id",
            how="left",
        )
        for dst, src in [
            ("baseline_condition_proxy", "cancer_baseline_condition"),
            ("followup_condition_proxy", "cancer_followup_condition"),
            ("baseline_env_proxy", "cancer_baseline_env"),
            ("followup_env_proxy", "cancer_followup_env"),
        ]:
            out[dst] = out[src].combine_first(out[dst])

    cm_path = ROOT / "data" / "processed" / "cardiomyopathy_temporal_endpoints_v2.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path)
        cm = cm.rename(
            columns={
                "condition_label_followup": "cm_followup_condition",
                "followup_environment_v2": "cm_followup_env",
            }
        )
        out = out.merge(
            cm[["assertion_id", "cm_followup_condition", "cm_followup_env"]],
            on="assertion_id",
            how="left",
        )
        out["followup_condition_proxy"] = out["cm_followup_condition"].combine_first(out["followup_condition_proxy"])
        out["followup_env_proxy"] = out["cm_followup_env"].combine_first(out["followup_env_proxy"])

    arr_path = ROOT / "data" / "processed" / "cab_cross_environment_drift.csv"
    if arr_path.exists():
        arr = pd.read_csv(arr_path, dtype={"variation_id": str})
        arr["assertion_id"] = "ARR_" + arr["variation_id"].astype(str)
        arr = arr.rename(
            columns={
                "baseline_condition_norm": "arr_baseline_condition",
                "followup_condition_norm": "arr_followup_condition",
                "baseline_env": "arr_baseline_env",
                "followup_env": "arr_followup_env",
                "phenotype_ids_2023-01": "arr_baseline_ids",
                "phenotype_ids_2026-04": "arr_followup_ids",
                "phenotype_list_2023-01": "arr_baseline_phenotype_list",
                "phenotype_list_2026-04": "arr_followup_phenotype_list",
                "gene_disease_validity_score": "arr_gene_disease_validity_score",
            }
        )
        keep = [
            "assertion_id",
            "arr_baseline_condition",
            "arr_followup_condition",
            "arr_baseline_env",
            "arr_followup_env",
            "arr_baseline_ids",
            "arr_followup_ids",
                "arr_baseline_phenotype_list",
                "arr_followup_phenotype_list",
                "arr_gene_disease_validity_score",
                "submitter_count_change",
                "condition_label_change",
                "cross_environment_drift",
                "within_environment_label_drift",
            ]
        arr = arr.rename(
            columns={
                "submitter_count_change": "arr_submitter_count_change",
                "condition_label_change": "arr_condition_label_change",
                "cross_environment_drift": "arr_cross_environment_drift",
                "within_environment_label_drift": "arr_within_environment_label_drift",
            }
        )
        keep = [
            {
                "submitter_count_change": "arr_submitter_count_change",
                "condition_label_change": "arr_condition_label_change",
                "cross_environment_drift": "arr_cross_environment_drift",
                "within_environment_label_drift": "arr_within_environment_label_drift",
            }.get(c, c)
            for c in keep
        ]
        out = out.merge(arr[keep], on="assertion_id", how="left")
        for dst, src in [
            ("baseline_condition_proxy", "arr_baseline_condition"),
            ("followup_condition_proxy", "arr_followup_condition"),
            ("baseline_env_proxy", "arr_baseline_env"),
            ("followup_env_proxy", "arr_followup_env"),
            ("baseline_condition_ids_proxy", "arr_baseline_ids"),
            ("followup_condition_ids_proxy", "arr_followup_ids"),
        ]:
            out[dst] = out[src].combine_first(out[dst])
        out["gene_disease_validity_score_proxy"] = pd.to_numeric(
            out.get("arr_gene_disease_validity_score"), errors="coerce"
        )
        if "arr_submitter_count_change" in out.columns:
            out["submitter_count_change"] = read_bool(out["submitter_count_change"]) | read_bool(
                out["arr_submitter_count_change"]
            )
        arr_mask = out["domain"].eq("inherited_arrhythmia") & out["arr_cross_environment_drift"].notna()
        if arr_mask.any():
            for bool_col in [
                "future_cross_environment_drift",
                "cross_environment_drift",
                "future_condition_label_drift",
                "condition_label_drift",
            ]:
                if bool_col in out.columns:
                    out[bool_col] = read_bool(out[bool_col]).astype(object)
            out.loc[arr_mask, "future_cross_environment_drift"] = read_bool(
                out.loc[arr_mask, "arr_cross_environment_drift"]
            ).to_numpy()
            out.loc[arr_mask, "cross_environment_drift"] = read_bool(
                out.loc[arr_mask, "arr_cross_environment_drift"]
            ).to_numpy()
            out.loc[arr_mask, "future_condition_label_drift"] = read_bool(
                out.loc[arr_mask, "arr_condition_label_change"]
            ).to_numpy()
            out.loc[arr_mask, "condition_label_drift"] = read_bool(
                out.loc[arr_mask, "arr_condition_label_change"]
            ).to_numpy()
    else:
        out["gene_disease_validity_score_proxy"] = np.nan

    vcep_path = TABLES / "cab_clingen_vcep_comparator_mapping.csv"
    if vcep_path.exists():
        vcep = pd.read_csv(vcep_path, usecols=["assertion_id", "domain", "VCEP_covered_gene", "VCEP_or_CSpec_resource"])
        out = out.merge(vcep.drop_duplicates(["assertion_id", "domain"]), on=["assertion_id", "domain"], how="left")
    else:
        out["VCEP_covered_gene"] = False
        out["VCEP_or_CSpec_resource"] = ""
    out["VCEP_covered_gene"] = read_bool(out["VCEP_covered_gene"])

    identity_path = TABLES / "clinvar_identity_vs_meaning_concordance.csv"
    if identity_path.exists():
        ident = pd.read_csv(identity_path)
        out = out.merge(
            ident[["assertion_id", "domain", "meaning_match_accepted", "phenotype_domain_discordance_flag"]],
            on=["assertion_id", "domain"],
            how="left",
            suffixes=("", "_identity"),
        )
    out["meaning_match_accepted"] = read_bool(out.get("meaning_match_accepted", pd.Series(True, index=out.index))).fillna(
        True
    )
    out["phenotype_domain_discordance_flag"] = read_bool(
        out.get("phenotype_domain_discordance_flag", pd.Series(False, index=out.index))
    )

    for col in ["baseline_condition_proxy", "followup_condition_proxy", "baseline_env_proxy", "followup_env_proxy"]:
        out[col] = out[col].fillna("").astype(str)

    out["baseline_followup_string_similarity"] = [
        string_similarity(a, b) for a, b in zip(out["baseline_condition_proxy"], out["followup_condition_proxy"])
    ]
    out["condition_token_overlap"] = [
        jaccard(token_set(a), token_set(b))
        for a, b in zip(out["baseline_condition_proxy"], out["followup_condition_proxy"])
    ]
    out["condition_specificity_score"] = [
        max(len(token_set(a)), len(token_set(b)))
        for a, b in zip(out["baseline_condition_proxy"], out["followup_condition_proxy"])
    ]
    out["baseline_condition_specificity_score"] = [len(token_set(a)) for a in out["baseline_condition_proxy"]]
    out["baseline_env_specificity_score"] = [len(token_set(a)) for a in out["baseline_env_proxy"]]
    out["baseline_condition_has_not_provided"] = out["baseline_condition_proxy"].str.contains(
        "not provided|not specified|unknown", case=False, na=False
    ).astype(int)
    out["baseline_condition_multi_label_count"] = out["baseline_condition_proxy"].fillna("").astype(str).str.count(";") + 1
    return out


def build_proxy_adjudication(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        base_ids = row.get("baseline_condition_ids_proxy", "")
        foll_ids = row.get("followup_condition_ids_proxy", "")
        mondo_b = parse_ids(base_ids, "MONDO")
        mondo_f = parse_ids(foll_ids, "MONDO")
        hpo_b = parse_ids(base_ids, "HPO")
        hpo_f = parse_ids(foll_ids, "HPO")
        omim_b = parse_omim_family(base_ids)
        omim_f = parse_omim_family(foll_ids)

        same_env = norm_text(row["baseline_env_proxy"]) == norm_text(row["followup_env_proxy"])
        cross_env = bool(row.get("cross_environment_drift", False))
        if "future_cross_environment_drift" in row:
            cross_env = cross_env or bool(row["future_cross_environment_drift"])
        condition_drift = bool(row.get("condition_label_drift", False))
        if "future_condition_label_drift" in row:
            condition_drift = condition_drift or bool(row["future_condition_label_drift"])
        submitter_delta = bool(row.get("submitter_count_change", False))
        sim = float(row["baseline_followup_string_similarity"]) if not pd.isna(row["baseline_followup_string_similarity"]) else 0
        tok = float(row["condition_token_overlap"]) if not pd.isna(row["condition_token_overlap"]) else 0
        same_mondo = bool(mondo_b and mondo_f and mondo_b.intersection(mondo_f))
        same_omim_family = bool(omim_b and omim_f and omim_b.intersection(omim_f))
        hpo_overlap = jaccard(hpo_b, hpo_f)

        if same_mondo:
            mondo_distance = 0
        elif same_env and (same_omim_family or sim >= 0.86 or tok >= 0.60):
            mondo_distance = 1
        elif not same_env or cross_env:
            mondo_distance = 3
        else:
            mondo_distance = 2

        clingen_same_pair = ""
        if bool(row.get("VCEP_covered_gene", False)) or not pd.isna(row.get("gene_disease_validity_score_proxy", np.nan)):
            clingen_same_pair = "yes" if same_env else "no"

        if condition_drift or cross_env:
            if cross_env and not (same_mondo or same_omim_family):
                label = "true_environment_shift"
            elif same_env and submitter_delta and not cross_env:
                label = "submitter_noise"
            elif same_env and (same_mondo or same_omim_family or sim >= 0.80 or tok >= 0.45):
                label = "ontology_synonym_or_parent_child"
            elif cross_env and (same_mondo or same_omim_family):
                label = "uncertain"
            else:
                label = "uncertain"
        else:
            label = "stable_no_drift"

        rows.append(
            {
                "assertion_id": row["assertion_id"],
                "VariationID": row["variation_id_clean"],
                "domain": row["domain"],
                "gene": row["gene"],
                "baseline_condition": row["baseline_condition_proxy"],
                "followup_condition": row["followup_condition_proxy"],
                "baseline_env": row["baseline_env_proxy"],
                "followup_env": row["followup_env_proxy"],
                "CAB_drift": bool(cross_env),
                "condition_label_drift": bool(condition_drift),
                "MONDO_distance": mondo_distance,
                "HPO_overlap": hpo_overlap,
                "OMIM_same_disease_family": same_omim_family,
                "ClinGen_same_GD_pair": clingen_same_pair,
                "VCEP_covered_gene": bool(row.get("VCEP_covered_gene", False)),
                "string_similarity": sim,
                "condition_token_overlap": tok,
                "submitter_count_change": submitter_delta,
                "adjudication_proxy_label": label,
            }
        )
    proxy = pd.DataFrame(rows)
    proxy.to_csv(TABLES / "cab_proxy_adjudication_layer.csv", index=False)

    summary = (
        proxy.groupby(["adjudication_proxy_label", "domain"], dropna=False)
        .size()
        .reset_index(name="N")
        .sort_values(["adjudication_proxy_label", "domain"])
    )
    summary["rate_within_domain"] = summary["N"] / summary.groupby("domain")["N"].transform("sum")
    summary.to_csv(TABLES / "cab_proxy_adjudication_summary.csv", index=False)
    return proxy, summary


def add_proxy_endpoints(df: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        proxy[
            [
                "assertion_id",
                "domain",
                "MONDO_distance",
                "HPO_overlap",
                "OMIM_same_disease_family",
                "ClinGen_same_GD_pair",
                "string_similarity",
                "condition_token_overlap",
                "adjudication_proxy_label",
            ]
        ],
        on=["assertion_id", "domain"],
        how="left",
    )
    # The analysis frame already has string/condition-overlap features. If the
    # proxy merge created x/y suffixed copies, keep the pre-merge values and
    # fill from proxy values only where needed.
    for col in ["string_similarity", "condition_token_overlap"]:
        if col not in out.columns:
            left = f"{col}_x"
            right = f"{col}_y"
            if left in out.columns and right in out.columns:
                out[col] = out[left].combine_first(out[right])
            elif left in out.columns:
                out[col] = out[left]
            elif right in out.columns:
                out[col] = out[right]
    out["proxy_true_environment_shift"] = out["adjudication_proxy_label"].eq("true_environment_shift")
    out["proxy_true_plus_uncertain"] = out["adjudication_proxy_label"].isin(["true_environment_shift", "uncertain"])
    out["E1_crude_condition_label_drift"] = read_bool(out["future_condition_label_drift"])
    out["E2_normalized_condition_label_drift"] = out["E1_crude_condition_label_drift"] & ~out[
        "adjudication_proxy_label"
    ].isin(["ontology_synonym_or_parent_child", "submitter_noise"])
    out["E3_cross_environment_drift"] = read_bool(out["future_cross_environment_drift"])
    out["E4_proxy_adjudicated_true_shift"] = out["proxy_true_environment_shift"]
    out["MONDO_distance"] = pd.to_numeric(out["MONDO_distance"], errors="coerce").fillna(2)
    out["HPO_overlap"] = pd.to_numeric(out["HPO_overlap"], errors="coerce").fillna(0)
    out["string_similarity"] = pd.to_numeric(out["string_similarity"], errors="coerce").fillna(0)
    out["condition_token_overlap"] = pd.to_numeric(out["condition_token_overlap"], errors="coerce").fillna(0)
    out["condition_specificity_score"] = pd.to_numeric(out["condition_specificity_score"], errors="coerce").fillna(0)
    out["ontology_shift_score"] = (
        0.35 * (out["MONDO_distance"].clip(0, 3) / 3.0)
        + 0.25 * (1 - out["string_similarity"].clip(0, 1))
        + 0.20 * (1 - out["condition_token_overlap"].clip(0, 1))
        + 0.20 * (1 - out["HPO_overlap"].clip(0, 1))
    )
    return out


MODEL_SPECS = {
    "gene-only": (["gene"], []),
    "regime-only": (["disease_architecture_regime"], []),
    "gene+regime": (["gene", "disease_architecture_regime"], []),
    "metadata-only": (["review_status"], ["submitter_count"]),
    "ontology-only combined": (
        [],
        ["MONDO_distance", "HPO_overlap", "string_similarity", "condition_token_overlap", "condition_specificity_score"],
    ),
    "gene+ontology": (
        ["gene"],
        ["MONDO_distance", "HPO_overlap", "string_similarity", "condition_token_overlap", "condition_specificity_score"],
    ),
    "gene+ontology+CAB": (
        ["gene", "disease_architecture_regime"],
        ["MONDO_distance", "HPO_overlap", "string_similarity", "condition_token_overlap", "condition_specificity_score"],
    ),
    "full baseline": (
        ["gene", "domain", "baseline_environment", "disease_architecture_regime", "review_status", "classification"],
        ["submitter_count", "cab_portability_score"],
    ),
    "full+ontology": (
        ["gene", "domain", "baseline_environment", "disease_architecture_regime", "review_status", "classification"],
        [
            "submitter_count",
            "cab_portability_score",
            "MONDO_distance",
            "HPO_overlap",
            "string_similarity",
            "condition_token_overlap",
            "condition_specificity_score",
        ],
    ),
}


def evaluate_models_oof(df: pd.DataFrame, endpoints: list[str], specs: dict[str, tuple[list[str], list[str]]]) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    preds: dict[tuple[str, str], np.ndarray] = {}
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        for name, (cats, nums) in specs.items():
            pred = oof_predict(df, endpoint, cats, nums)
            preds[(endpoint, name)] = pred
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": name,
                    "N": len(df),
                    "positive_N": int(y.sum()),
                    "positive_rate": float(y.mean()) if len(y) else float("nan"),
                    **metrics(y, pred),
                }
            )
    return pd.DataFrame(rows), preds


def recompute_main_claims(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    endpoints = ["proxy_true_environment_shift", "proxy_true_plus_uncertain", "E3_cross_environment_drift"]
    specs = {
        "gene-only": MODEL_SPECS["gene-only"],
        "regime-only": MODEL_SPECS["regime-only"],
        "gene+regime": MODEL_SPECS["gene+regime"],
        "full baseline": MODEL_SPECS["full baseline"],
        "gene+ontology": MODEL_SPECS["gene+ontology"],
        "gene+ontology+CAB": MODEL_SPECS["gene+ontology+CAB"],
    }
    result, preds = evaluate_models_oof(df, endpoints, specs)
    delta_rows = []
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        for base, alt in [("gene-only", "gene+regime"), ("gene+ontology", "gene+ontology+CAB")]:
            d, lo, hi = bootstrap_delta(y, preds[(endpoint, base)], preds[(endpoint, alt)])
            delta_rows.append(
                {
                    "endpoint": endpoint,
                    "comparison": f"{base} -> {alt}",
                    "delta_AUROC": d,
                    "CI95_low": lo,
                    "CI95_high": hi,
                }
            )
    out = result.merge(pd.DataFrame(delta_rows), on="endpoint", how="left")
    result.to_csv(TABLES / "cab_proxy_adjudicated_main_claim_models.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(TABLES / "cab_proxy_adjudicated_main_claim_deltas.csv", index=False)
    return result, preds


def gene_family(gene: object) -> str:
    g = str(gene).upper()
    families = {
        "BRCA1_BRCA2": {"BRCA1", "BRCA2"},
        "MMR": {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"},
        "SCN5A": {"SCN5A"},
        "LMNA": {"LMNA"},
        "DESMOSOMAL": {"PKP2", "DSP", "DSG2", "DSC2", "JUP"},
        "SARCOMERE": {"MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3"},
        "CPVT_CALCIUM": {"RYR2", "CASQ2", "TRDN", "CALM1", "CALM2", "CALM3"},
        "LQTS_POTASSIUM": {"KCNQ1", "KCNH2", "KCNE1", "KCNE2", "KCNJ2"},
        "TP53": {"TP53"},
        "CHEK2_ATM": {"CHEK2", "ATM"},
    }
    for name, members in families.items():
        if g in members:
            return name
    return f"OTHER_{g}"


def evaluate_holdout(df: pd.DataFrame, group_col: str, groups: list[str], endpoint: str) -> pd.DataFrame:
    rows = []
    specs = {
        "gene-only": MODEL_SPECS["gene-only"],
        "regime-only": MODEL_SPECS["regime-only"],
        "gene+regime": MODEL_SPECS["gene+regime"],
    }
    for group in groups:
        test_mask = df[group_col].astype(str).eq(str(group))
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        if len(test) < 25 or len(train) < 25:
            continue
        y = test[endpoint].astype(int).to_numpy()
        preds = {}
        for name, (cats, nums) in specs.items():
            preds[name] = fit_predict_split(train, test, endpoint, cats, nums)
        d, lo, hi = bootstrap_delta(y, preds["gene-only"], preds["gene+regime"])
        rows.append(
            {
                "Split": f"{group_col}={group}",
                "N_train": len(train),
                "N_test": len(test),
                "endpoint": endpoint,
                "gene-only_AUROC": safe_auc(y, preds["gene-only"]),
                "regime-only_AUROC": safe_auc(y, preds["regime-only"]),
                "gene+regime_AUROC": safe_auc(y, preds["gene+regime"]),
                "delta_gene_to_gene_regime": d,
                "bootstrap_CI95_low": lo,
                "bootstrap_CI95_high": hi,
                "test_positive_rate": float(y.mean()) if len(y) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def leave_out_validation(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gene_family"] = work["gene"].map(gene_family)
    work["disease_environment"] = work["baseline_env_proxy"].fillna("unknown").astype(str)
    endpoints = ["E3_cross_environment_drift", "proxy_true_environment_shift", "proxy_true_plus_uncertain"]
    frames = []
    top_genes = work["gene"].value_counts()
    gene_groups = sorted(set(top_genes[top_genes >= 75].index).union({"BRCA1", "BRCA2", "SCN5A", "LMNA", "PKP2", "DSP", "MYH7", "MYBPC3", "MLH1", "MSH2", "MSH6", "PMS2"}))
    family_groups = [g for g, n in work["gene_family"].value_counts().items() if n >= 50 and not str(g).startswith("OTHER_")]
    env_groups = [g for g, n in work["disease_environment"].value_counts().items() if n >= 100]
    domain_groups = sorted(work["domain"].unique())
    for endpoint in endpoints:
        frames.append(evaluate_holdout(work, "gene", gene_groups, endpoint))
        frames.append(evaluate_holdout(work, "gene_family", family_groups, endpoint))
        frames.append(evaluate_holdout(work, "disease_environment", env_groups, endpoint))
        frames.append(evaluate_holdout(work, "domain", domain_groups, endpoint))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLES / "cab_leave_gene_family_environment_domain_out_validation.csv", index=False)
    return out


def ontology_baseline_comparator(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = ["E1_crude_condition_label_drift", "E2_normalized_condition_label_drift", "E3_cross_environment_drift", "E4_proxy_adjudicated_true_shift"]
    specs = {
        "string similarity only": ([], ["string_similarity"]),
        "MONDO graph distance only": ([], ["MONDO_distance"]),
        "HPO overlap only": ([], ["HPO_overlap"]),
        "condition specificity only": ([], ["condition_specificity_score"]),
        "ontology-only combined": MODEL_SPECS["ontology-only combined"],
        "CAB regime only": MODEL_SPECS["regime-only"],
        "gene + ontology": MODEL_SPECS["gene+ontology"],
        "gene + ontology + CAB": MODEL_SPECS["gene+ontology+CAB"],
    }
    out, preds = evaluate_models_oof(df, endpoints, specs)
    deltas = []
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        d, lo, hi = bootstrap_delta(y, preds[(endpoint, "gene + ontology")], preds[(endpoint, "gene + ontology + CAB")])
        deltas.append({"endpoint": endpoint, "comparison": "gene + ontology -> gene + ontology + CAB", "delta_AUROC": d, "CI95_low": lo, "CI95_high": hi})
    out.to_csv(TABLES / "cab_ontology_only_baseline_comparator.csv", index=False)
    pd.DataFrame(deltas).to_csv(TABLES / "cab_ontology_only_incremental_cab_deltas.csv", index=False)
    return out


def baseline_only_ontology_forecasting_comparator(df: pd.DataFrame) -> pd.DataFrame:
    """Forecasting comparator that avoids follow-up label-pair leakage."""
    endpoints = ["E3_cross_environment_drift", "E4_proxy_adjudicated_true_shift", "proxy_true_plus_uncertain"]
    specs = {
        "baseline condition specificity only": ([], ["baseline_condition_specificity_score"]),
        "baseline environment only": (["baseline_env_proxy"], []),
        "baseline ontology-like combined": (
            ["baseline_env_proxy"],
            [
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
        "CAB regime only": MODEL_SPECS["regime-only"],
        "gene + baseline ontology": (
            ["gene", "baseline_env_proxy"],
            [
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
        "gene + baseline ontology + CAB": (
            ["gene", "baseline_env_proxy", "disease_architecture_regime"],
            [
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
    }
    out, preds = evaluate_models_oof(df, endpoints, specs)
    deltas = []
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        d, lo, hi = bootstrap_delta(
            y,
            preds[(endpoint, "gene + baseline ontology")],
            preds[(endpoint, "gene + baseline ontology + CAB")],
        )
        deltas.append(
            {
                "endpoint": endpoint,
                "comparison": "gene + baseline ontology -> gene + baseline ontology + CAB",
                "delta_AUROC": d,
                "CI95_low": lo,
                "CI95_high": hi,
            }
        )
    out.to_csv(TABLES / "cab_baseline_only_ontology_forecasting_comparator.csv", index=False)
    pd.DataFrame(deltas).to_csv(
        TABLES / "cab_baseline_only_ontology_incremental_cab_deltas.csv", index=False
    )
    return out


def negative_control_stable_domains(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gene_family"] = work["gene"].map(gene_family)
    stable = (
        (~work["E3_cross_environment_drift"])
        & (~work["E2_normalized_condition_label_drift"])
        & (~read_bool(work["future_classification_change"]))
    )
    high_conf = stable & (
        work["disease_architecture_regime"].astype(str).str.contains("phenotype|monogenic|syndrome", case=False, na=False)
        | work["gene_family"].isin(["BRCA1_BRCA2", "MMR", "SARCOMERE"])
        | (work["VCEP_covered_gene"])
    )
    rows = []
    for label, mask in [
        ("all_high_confidence_stable", high_conf),
        ("BRCA1_BRCA2_HBOC_like_stable", high_conf & work["gene_family"].eq("BRCA1_BRCA2")),
        ("MMR_Lynch_like_stable", high_conf & work["gene_family"].eq("MMR")),
        ("sarcomere_cardiomyopathy_stable", high_conf & work["gene_family"].eq("SARCOMERE")),
        ("VCEP_covered_stable", high_conf & work["VCEP_covered_gene"]),
        ("MONDO_parent_child_or_synonym_stable", work["adjudication_proxy_label"].eq("ontology_synonym_or_parent_child")),
    ]:
        sub = work.loc[mask]
        if sub.empty:
            continue
        for mode, col in [("CAB-Strict", "cab_strict_direct_use_allowed"), ("CAB-Balanced", "cab_balanced_direct_use_allowed")]:
            direct = read_bool(sub[col])
            rows.append(
                {
                    "negative_control_stratum": label,
                    "mode": mode,
                    "N": len(sub),
                    "false_alarm_rate_review_or_block": float((~direct).mean()),
                    "direct_use_preserved_rate": float(direct.mean()),
                    "cross_env_drift_rate": float(sub["E3_cross_environment_drift"].mean()),
                    "proxy_true_shift_rate": float(sub["proxy_true_environment_shift"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_negative_control_stable_domain_audit.csv", index=False)
    return out


def calibration_and_decision_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = "E3_cross_environment_drift"
    specs = {
        "gene+regime": MODEL_SPECS["gene+regime"],
        "gene+baseline-ontology+CAB": (
            ["gene", "baseline_env_proxy", "disease_architecture_regime"],
            [
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
        "full baseline no follow-up ontology": (
            ["gene", "domain", "baseline_env_proxy", "disease_architecture_regime", "review_status", "classification"],
            [
                "submitter_count",
                "cab_portability_score",
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
    }
    model_rows, preds = evaluate_models_oof(df, [endpoint], specs)
    y = df[endpoint].astype(int).to_numpy()
    cal_rows = []
    decile_rows = []
    curve_rows = []
    for name in specs:
        pred = np.clip(preds[(endpoint, name)], 1e-6, 1 - 1e-6)
        # Calibration intercept/slope: logit(observed) ~ intercept + slope*logit(pred)
        logit_pred = np.log(pred / (1 - pred))
        try:
            fit = sm.Logit(y, sm.add_constant(logit_pred)).fit(disp=False, maxiter=200)
            intercept = float(fit.params[0])
            slope = float(fit.params[1])
        except Exception:
            intercept = float("nan")
            slope = float("nan")
        bins = pd.qcut(pred, q=10, labels=False, duplicates="drop")
        ece = 0.0
        for b in sorted(pd.Series(bins).dropna().unique()):
            idx = np.asarray(bins == b)
            observed = float(y[idx].mean()) if idx.sum() else float("nan")
            expected = float(pred[idx].mean()) if idx.sum() else float("nan")
            ece += (idx.sum() / len(y)) * abs(observed - expected)
            decile_rows.append(
                {
                    "model": name,
                    "endpoint": endpoint,
                    "risk_decile": int(b) + 1,
                    "N": int(idx.sum()),
                    "mean_predicted_risk": expected,
                    "observed_drift_rate": observed,
                    "lift_vs_baseline": observed / float(y.mean()) if y.mean() else float("nan"),
                }
            )
        cal_rows.append(
            {
                "model": name,
                "endpoint": endpoint,
                "N": len(df),
                "positive_rate": float(y.mean()),
                **metrics(y, pred),
                "expected_calibration_error": float(ece),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        for threshold in np.linspace(0.01, 0.50, 50):
            selected = pred >= threshold
            tp = float((selected & (y == 1)).sum()) / len(y)
            fp = float((selected & (y == 0)).sum()) / len(y)
            net_benefit = tp - fp * (threshold / (1 - threshold))
            curve_rows.append(
                {
                    "model": name,
                    "endpoint": endpoint,
                    "review_threshold": threshold,
                    "selected_rate": float(selected.mean()),
                    "net_benefit": net_benefit,
                }
            )
    cal = pd.DataFrame(cal_rows)
    dec = pd.DataFrame(decile_rows)
    curve = pd.DataFrame(curve_rows)
    cal.to_csv(TABLES / "cab_calibration_metrics.csv", index=False)
    dec.to_csv(TABLES / "cab_risk_decile_calibration.csv", index=False)
    curve.to_csv(TABLES / "cab_decision_curve_net_benefit.csv", index=False)
    plot_risk_deciles(dec)
    return cal, dec


def plot_risk_deciles(dec: pd.DataFrame) -> None:
    focus = dec[
        dec["model"].isin(
            ["gene+regime", "gene+baseline-ontology+CAB", "full baseline no follow-up ontology"]
        )
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for name, sub in focus.groupby("model"):
        ax.plot(sub["risk_decile"], sub["observed_drift_rate"], marker="o", label=name)
    baseline = (focus["observed_drift_rate"] * focus["N"]).sum() / focus["N"].sum()
    if np.isfinite(baseline):
        ax.axhline(baseline, color="#555", linestyle="--", linewidth=1, label="mean observed")
    ax.set_xlabel("Predicted risk decile")
    ax.set_ylabel("Observed future cross-environment drift")
    ax.set_title("CAB risk calibration by decile", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "cab_risk_decile_calibration.svg")
    fig.savefig(FIGURES / "cab_risk_decile_calibration.png", dpi=220)
    plt.close(fig)


def submitter_stratified(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["single_submitter"] = pd.to_numeric(work["submitter_count"], errors="coerce").fillna(0) <= 1
    work["review_star_category"] = work["review_status"].fillna("missing").astype(str)
    work["submitter_count_change_bool"] = read_bool(work["submitter_count_change"])
    endpoint = "E3_cross_environment_drift"
    specs = {
        "gene+regime+submitter_metadata": (
            ["gene", "disease_architecture_regime", "review_star_category", "domain"],
            ["submitter_count", "submitter_count_change_bool"],
        )
    }
    rows = []
    for label, mask in [
        ("single_submitter", work["single_submitter"]),
        ("multiple_submitters", ~work["single_submitter"]),
        ("submitter_count_increase_or_change", work["submitter_count_change_bool"]),
        ("stable_submitter_count", ~work["submitter_count_change_bool"]),
    ]:
        sub = work.loc[mask].copy()
        if len(sub) < 50:
            continue
        out, _ = evaluate_models_oof(sub, [endpoint], specs)
        out["stratum"] = label
        rows.append(out)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(TABLES / "cab_submitter_stratified_models.csv", index=False)

    # Matched drift/non-drift analysis by gene, environment, review status, submitter count bin, domain.
    work["submitter_bin"] = pd.cut(pd.to_numeric(work["submitter_count"], errors="coerce").fillna(0), [-1, 1, 3, 10, 9999], labels=["1", "2-3", "4-10", "10+"])
    y = work[endpoint].astype(bool)
    rates = work.groupby("disease_architecture_regime")[endpoint].mean().to_dict()
    global_rate = float(work[endpoint].mean())
    work["regime_empirical_risk"] = work["disease_architecture_regime"].map(rates).fillna(global_rate)
    controls = work.loc[~y].copy()
    pairs = []
    for _, case in work.loc[y].iterrows():
        candidates = controls[
            (controls["gene"] == case["gene"])
            & (controls["baseline_env_proxy"] == case["baseline_env_proxy"])
            & (controls["domain"] == case["domain"])
            & (controls["review_status"].fillna("") == str(case["review_status"]))
            & (controls["submitter_bin"].astype(str) == str(case["submitter_bin"]))
        ]
        if candidates.empty:
            candidates = controls[(controls["gene"] == case["gene"]) & (controls["domain"] == case["domain"])]
        if candidates.empty:
            continue
        control = candidates.sample(n=1, random_state=int(RNG.integers(1, 1_000_000))).iloc[0]
        pairs.append(
            {
                "case_assertion_id": case["assertion_id"],
                "control_assertion_id": control["assertion_id"],
                "gene": case["gene"],
                "domain": case["domain"],
                "baseline_env": case["baseline_env_proxy"],
                "case_regime": case["disease_architecture_regime"],
                "control_regime": control["disease_architecture_regime"],
                "case_regime_empirical_risk": case["regime_empirical_risk"],
                "control_regime_empirical_risk": control["regime_empirical_risk"],
                "risk_difference": case["regime_empirical_risk"] - control["regime_empirical_risk"],
            }
        )
    matched = pd.DataFrame(pairs)
    matched.to_csv(TABLES / "cab_submitter_matched_drift_nondrift_analysis.csv", index=False)
    return result, matched


def external_curated_subset(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    sads_genes = {"SCN5A", "KCNQ1", "KCNH2", "RYR2", "LMNA", "PKP2", "DSP", "CASQ2", "TRDN", "KCNE1", "KCNE2"}
    subset = work[(work["VCEP_covered_gene"]) | (work["gene"].isin(sads_genes))].copy()
    if len(subset) > 500:
        subset = subset.sample(n=500, random_state=20260513)
    def agree(row: pd.Series) -> str:
        if bool(row["cab_balanced_direct_use_allowed"]) and not bool(row["E3_cross_environment_drift"]):
            return "external_proxy_agrees_direct_use"
        if bool(row["disease_specific_expert_review_required"]) or bool(row["E3_cross_environment_drift"]):
            return "external_proxy_requires_disease_specific_interpretation"
        return "ambiguous"
    subset["external_proxy_agreement_label"] = subset.apply(agree, axis=1)
    out = subset[
        [
            "assertion_id",
            "domain",
            "gene",
            "baseline_condition_proxy",
            "followup_condition_proxy",
            "baseline_env_proxy",
            "followup_env_proxy",
            "disease_architecture_regime",
            "VCEP_covered_gene",
            "VCEP_or_CSpec_resource",
            "cab_balanced_direct_use_allowed",
            "disease_specific_expert_review_required",
            "population_or_penetrance_review_required",
            "external_proxy_agreement_label",
        ]
    ].copy()
    out.to_csv(TABLES / "cab_external_curated_subset_agreement.csv", index=False)
    summary = out.groupby(["external_proxy_agreement_label", "domain"]).size().reset_index(name="N")
    summary.to_csv(TABLES / "cab_external_curated_subset_agreement_summary.csv", index=False)
    return out


def sads_validation_set(df: pd.DataFrame) -> pd.DataFrame:
    genes = ["SCN5A", "KCNQ1", "KCNH2", "RYR2", "LMNA", "PKP2", "DSP", "CASQ2", "TRDN", "KCNE1", "KCNE2"]
    contexts = ["LQTS", "Brugada", "CPVT", "DCM", "ACM/ARVC", "conduction disease", "SADS", "SUDC", "sudden death unspecified", "phenotype-negative cascade testing"]
    sub = df[df["gene"].isin(genes)].copy()
    if sub.empty:
        out = pd.DataFrame()
    else:
        sub["context_bucket"] = sub["baseline_env_proxy"].apply(lambda x: next((c for c in contexts if c.lower().split("/")[0] in norm_text(x)), "other/unspecified"))
        rows = []
        for gene, gsub in sub.groupby("gene"):
            for ctx, csub in gsub.groupby("context_bucket"):
                rows.append(
                    {
                        "gene": gene,
                        "context": ctx,
                        "N": len(csub),
                        "clinvar_label_only_wrongly_reused_N": int(csub["E3_cross_environment_drift"].sum()),
                        "clinvar_label_only_wrongly_reused_rate": float(csub["E3_cross_environment_drift"].mean()),
                        "CAB_routes_to_review_N": int((~read_bool(csub["cab_balanced_direct_use_allowed"])).sum()),
                        "CAB_routes_to_review_rate": float((~read_bool(csub["cab_balanced_direct_use_allowed"])).mean()),
                        "CAB_direct_use_remains_N": int(read_bool(csub["cab_balanced_direct_use_allowed"]).sum()),
                        "CAB_direct_use_remains_rate": float(read_bool(csub["cab_balanced_direct_use_allowed"]).mean()),
                        "proxy_true_environment_shift_N": int(csub["proxy_true_environment_shift"].sum()),
                    }
                )
        out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_sads_molecular_autopsy_special_validation_set.csv", index=False)
    return out


def component_ablation(df: pd.DataFrame) -> pd.DataFrame:
    endpoint = "E3_cross_environment_drift"
    full_cats = ["gene", "domain", "baseline_environment", "disease_architecture_regime", "review_status"]
    full_nums = [
        "submitter_count",
        "cab_portability_score",
        "MONDO_distance",
        "HPO_overlap",
        "string_similarity",
        "condition_specificity_score",
        "structural_functional_overlap_flag",
        "syndrome_organ_boundary_flag",
        "modifier_penetrance_flag",
    ]
    work = df.copy()
    work["structural_functional_overlap_flag"] = work["disease_architecture_regime"].astype(str).eq("structural_functional_overlap").astype(int)
    work["syndrome_organ_boundary_flag"] = work["disease_architecture_regime"].astype(str).str.contains("syndrome|organ", case=False, na=False).astype(int)
    work["modifier_penetrance_flag"] = work["disease_architecture_regime"].astype(str).str.contains("modifier|penetrance", case=False, na=False).astype(int)
    specs = {"full model": (full_cats, full_nums)}
    removals = {
        "minus gene": (["domain", "baseline_environment", "disease_architecture_regime", "review_status"], full_nums),
        "minus domain": (["gene", "baseline_environment", "disease_architecture_regime", "review_status"], full_nums),
        "minus baseline environment": (["gene", "domain", "disease_architecture_regime", "review_status"], full_nums),
        "minus regime": (["gene", "domain", "baseline_environment", "review_status"], [x for x in full_nums if not x.endswith("_flag")]),
        "minus metadata": (["gene", "domain", "baseline_environment", "disease_architecture_regime"], [x for x in full_nums if x not in ["submitter_count"]]),
        "minus ontology": (full_cats, [x for x in full_nums if x not in ["MONDO_distance", "HPO_overlap", "string_similarity", "condition_specificity_score"]]),
        "minus condition specificity": (full_cats, [x for x in full_nums if x != "condition_specificity_score"]),
        "minus overlap flags": (full_cats, [x for x in full_nums if not x.endswith("_flag")]),
    }
    specs.update(removals)
    result, preds = evaluate_models_oof(work, [endpoint], specs)
    y = work[endpoint].astype(int).to_numpy()
    full_pred = preds[(endpoint, "full model")]
    rows = []
    full_metrics = metrics(y, full_pred)
    for name in specs:
        pred = preds[(endpoint, name)]
        mm = metrics(y, pred)
        rows.append(
            {
                "Component removed": "none" if name == "full model" else name.replace("minus ", ""),
                "model": name,
                "endpoint": endpoint,
                "AUROC": mm["AUROC"],
                "AUROC_drop": full_metrics["AUROC"] - mm["AUROC"],
                "AUPRC": mm["AUPRC"],
                "AUPRC_drop": full_metrics["AUPRC"] - mm["AUPRC"],
                "top10_enrichment": mm["lift_at_top10"],
                "top10_enrichment_drop": full_metrics["lift_at_top10"] - mm["lift_at_top10"],
                "unsupported_reuse_increase_proxy": mm["precision_at_top10"] - full_metrics["precision_at_top10"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_component_ablation_study.csv", index=False)
    return out


def baseline_only_component_ablation(df: pd.DataFrame) -> pd.DataFrame:
    endpoint = "E3_cross_environment_drift"
    work = df.copy()
    work["structural_functional_overlap_flag"] = work["disease_architecture_regime"].astype(str).eq(
        "structural_functional_overlap"
    ).astype(int)
    work["syndrome_organ_boundary_flag"] = work["disease_architecture_regime"].astype(str).str.contains(
        "syndrome|organ", case=False, na=False
    ).astype(int)
    work["modifier_penetrance_flag"] = work["disease_architecture_regime"].astype(str).str.contains(
        "modifier|penetrance", case=False, na=False
    ).astype(int)
    cats = ["gene", "domain", "baseline_env_proxy", "disease_architecture_regime", "review_status"]
    nums = [
        "submitter_count",
        "cab_portability_score",
        "baseline_condition_specificity_score",
        "baseline_env_specificity_score",
        "baseline_condition_has_not_provided",
        "baseline_condition_multi_label_count",
        "structural_functional_overlap_flag",
        "syndrome_organ_boundary_flag",
        "modifier_penetrance_flag",
    ]
    specs = {
        "baseline-only full model": (cats, nums),
        "minus gene": ([c for c in cats if c != "gene"], nums),
        "minus domain": ([c for c in cats if c != "domain"], nums),
        "minus baseline environment": ([c for c in cats if c != "baseline_env_proxy"], nums),
        "minus regime": ([c for c in cats if c != "disease_architecture_regime"], [n for n in nums if not n.endswith("_flag")]),
        "minus metadata": ([c for c in cats if c != "review_status"], [n for n in nums if n != "submitter_count"]),
        "minus baseline ontology": (
            cats,
            [
                n
                for n in nums
                if n
                not in [
                    "baseline_condition_specificity_score",
                    "baseline_env_specificity_score",
                    "baseline_condition_has_not_provided",
                    "baseline_condition_multi_label_count",
                ]
            ],
        ),
        "minus overlap flags": (cats, [n for n in nums if not n.endswith("_flag")]),
    }
    _, preds = evaluate_models_oof(work, [endpoint], specs)
    y = work[endpoint].astype(int).to_numpy()
    full = metrics(y, preds[(endpoint, "baseline-only full model")])
    rows = []
    for name in specs:
        mm = metrics(y, preds[(endpoint, name)])
        rows.append(
            {
                "Component removed": "none" if name == "baseline-only full model" else name.replace("minus ", ""),
                "model": name,
                "endpoint": endpoint,
                "AUROC": mm["AUROC"],
                "AUROC_drop": full["AUROC"] - mm["AUROC"],
                "AUPRC": mm["AUPRC"],
                "AUPRC_drop": full["AUPRC"] - mm["AUPRC"],
                "top10_enrichment": mm["lift_at_top10"],
                "top10_enrichment_drop": full["lift_at_top10"] - mm["lift_at_top10"],
                "top10_precision": mm["precision_at_top10"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_baseline_only_component_ablation_study.csv", index=False)
    return out


def auprc_enrichment(df: pd.DataFrame, preds: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    endpoints = ["E1_crude_condition_label_drift", "E2_normalized_condition_label_drift", "E3_cross_environment_drift", "E4_proxy_adjudicated_true_shift"]
    specs = {
        "gene-only": MODEL_SPECS["gene-only"],
        "regime-only": MODEL_SPECS["regime-only"],
        "gene+regime": MODEL_SPECS["gene+regime"],
        "gene+baseline-ontology+CAB": (
            ["gene", "baseline_env_proxy", "disease_architecture_regime"],
            [
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
        "full baseline no follow-up ontology": (
            ["gene", "domain", "baseline_env_proxy", "disease_architecture_regime", "review_status", "classification"],
            [
                "submitter_count",
                "cab_portability_score",
                "baseline_condition_specificity_score",
                "baseline_env_specificity_score",
                "baseline_condition_has_not_provided",
                "baseline_condition_multi_label_count",
            ],
        ),
    }
    model_out, model_preds = evaluate_models_oof(df, endpoints, specs)
    rows = []
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        for model in specs:
            pred = model_preds[(endpoint, model)]
            for budget in [0.01, 0.05, 0.10, 0.20, 0.30]:
                precision, recall, lift, nnr = precision_at_budget(y, pred, budget)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "model": model,
                        "budget": budget,
                        "AUPRC": float(average_precision_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
                        "precision_at_budget": precision,
                        "recall_at_budget": recall,
                        "lift_at_budget": lift,
                        "number_needed_to_review": nnr,
                        "net_prevented_unsupported_reuses_per_100_reviewed": precision * 100,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_auprc_enrichment_review_utility.csv", index=False)
    return out


def direct_use_and_overrestriction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for mode, col in [
        ("ClinVar-label-only", None),
        ("CAB-Strict", "cab_strict_direct_use_allowed"),
        ("CAB-Balanced", "cab_balanced_direct_use_allowed"),
    ]:
        direct = pd.Series(True, index=df.index) if col is None else read_bool(df[col])
        sub = df.loc[direct]
        rows.append(
            {
                "mode": mode,
                "direct_use_N": len(sub),
                "direct_use_rate": float(direct.mean()),
                "later_condition_label_drift_rate": float(sub["E1_crude_condition_label_drift"].mean()) if len(sub) else float("nan"),
                "cross_environment_drift_rate": float(sub["E3_cross_environment_drift"].mean()) if len(sub) else float("nan"),
                "classification_change_rate": float(read_bool(sub["future_classification_change"]).mean()) if len(sub) else float("nan"),
                "meaning_rejected_rate": float((~sub["meaning_match_accepted"]).mean()) if len(sub) else float("nan"),
                "submitter_churn_rate": float(read_bool(sub["submitter_count_change"]).mean()) if len(sub) else float("nan"),
            }
        )
    safety = pd.DataFrame(rows)
    safety.to_csv(TABLES / "cab_direct_use_safety_analysis.csv", index=False)

    stable = (~df["E3_cross_environment_drift"]) & (~df["E2_normalized_condition_label_drift"])
    over_rows = []
    for mode, col in [("CAB-Strict", "cab_strict_direct_use_allowed"), ("CAB-Balanced", "cab_balanced_direct_use_allowed")]:
        direct = read_bool(df[col])
        over = df.loc[(~direct) & stable].copy()
        rescue = read_bool(over["contextual_repair_required"]) & stable.loc[over.index]
        over_rows.append(
            {
                "mode": mode,
                "overrestricted_stable_N": len(over),
                "overrestricted_stable_rate": float(((~direct) & stable).mean()),
                "top_regimes": "; ".join(over["disease_architecture_regime"].value_counts().head(5).index.astype(str)),
                "top_genes": "; ".join(over["gene"].value_counts().head(5).index.astype(str)),
                "top_domains": "; ".join(over["domain"].value_counts().head(5).index.astype(str)),
                "contextual_repair_rescue_N": int(rescue.sum()),
                "contextual_repair_rescue_rate_among_overrestricted": float(rescue.mean()) if len(over) else float("nan"),
            }
        )
    over_out = pd.DataFrame(over_rows)
    over_out.to_csv(TABLES / "cab_overrestriction_and_repair_audit.csv", index=False)
    return safety, over_out


def strict_endpoint_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "E1_crude_condition_label_drift",
        "E2_normalized_condition_label_drift",
        "E3_cross_environment_drift",
        "E4_proxy_adjudicated_true_shift",
    ]
    specs = {
        "gene-only": MODEL_SPECS["gene-only"],
        "gene+regime": MODEL_SPECS["gene+regime"],
        "gene+ontology+CAB": MODEL_SPECS["gene+ontology+CAB"],
    }
    out, preds = evaluate_models_oof(df, endpoints, specs)
    deltas = []
    for endpoint in endpoints:
        y = df[endpoint].astype(int).to_numpy()
        d, lo, hi = bootstrap_delta(y, preds[(endpoint, "gene-only")], preds[(endpoint, "gene+regime")])
        deltas.append({"endpoint": endpoint, "comparison": "gene-only -> gene+regime", "delta_AUROC": d, "CI95_low": lo, "CI95_high": hi})
    out.to_csv(TABLES / "cab_strict_endpoint_hierarchy_models.csv", index=False)
    pd.DataFrame(deltas).to_csv(TABLES / "cab_strict_endpoint_hierarchy_deltas.csv", index=False)
    return out


def case_studies(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    endpoint = "E3_cross_environment_drift"
    pred = oof_predict(work, endpoint, *MODEL_SPECS["gene+ontology+CAB"])
    work["case_risk_score"] = pred
    categories: list[tuple[str, pd.DataFrame]] = [
        ("top-risk", work.sort_values("case_risk_score", ascending=False)),
        ("false-positive", work[(work["case_risk_score"] >= np.quantile(pred, 0.90)) & (~work[endpoint])].sort_values("case_risk_score", ascending=False)),
        ("false-negative", work[(work["case_risk_score"] < np.quantile(pred, 0.50)) & (work[endpoint])].sort_values("case_risk_score")),
        ("direct-use stable", work[read_bool(work["cab_balanced_direct_use_allowed"]) & (~work[endpoint])].sort_values("cab_portability_score", ascending=False)),
        ("repaired/contextual", work[read_bool(work["contextual_repair_required"])].sort_values("case_risk_score", ascending=False)),
        ("SADS/channelopathy", work[work["gene"].isin(["SCN5A", "KCNQ1", "KCNH2", "RYR2", "LMNA", "PKP2", "DSP"])].sort_values("case_risk_score", ascending=False)),
        ("hereditary-cancer boundary", work[work["gene"].isin(["BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "TP53", "CHEK2", "ATM"])].sort_values("case_risk_score", ascending=False)),
    ]
    rows = []
    seen = set()
    for category, sub in categories:
        taken = 0
        for _, row in sub.iterrows():
            if row["assertion_id"] in seen:
                continue
            rows.append(
                {
                    "case_category": category,
                    "assertion_id": row["assertion_id"],
                    "VariationID": row["variation_id_clean"],
                    "gene": row["gene"],
                    "domain": row["domain"],
                    "baseline_condition": row["baseline_condition_proxy"],
                    "followup_condition": row["followup_condition_proxy"],
                    "baseline_env": row["baseline_env_proxy"],
                    "followup_env": row["followup_env_proxy"],
                    "CAB_regime": row["disease_architecture_regime"],
                    "CAB_balanced_direct_use": bool(row["cab_balanced_direct_use_allowed"]),
                    "proxy_label": row["adjudication_proxy_label"],
                    "future_cross_environment_drift": bool(row["E3_cross_environment_drift"]),
                    "case_risk_score": row["case_risk_score"],
                    "selection_rule": category,
                }
            )
            seen.add(row["assertion_id"])
            taken += 1
            if taken >= 2:
                break
        if len(rows) >= 8:
            break
    if len(rows) < 8:
        priority = [
            "SCN5A",
            "KCNQ1",
            "KCNH2",
            "LMNA",
            "PKP2",
            "DSP",
            "BRCA1",
            "BRCA2",
            "MLH1",
            "MSH2",
            "TP53",
            "CHEK2",
            "ATM",
        ]
        for gene in priority:
            sub = work[(work["gene"].eq(gene)) & (~work["assertion_id"].isin(seen))].sort_values(
                "case_risk_score", ascending=False
            )
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "case_category": "predefined-gene-fill",
                    "assertion_id": row["assertion_id"],
                    "VariationID": row["variation_id_clean"],
                    "gene": row["gene"],
                    "domain": row["domain"],
                    "baseline_condition": row["baseline_condition_proxy"],
                    "followup_condition": row["followup_condition_proxy"],
                    "baseline_env": row["baseline_env_proxy"],
                    "followup_env": row["followup_env_proxy"],
                    "CAB_regime": row["disease_architecture_regime"],
                    "CAB_balanced_direct_use": bool(row["cab_balanced_direct_use_allowed"]),
                    "proxy_label": row["adjudication_proxy_label"],
                    "future_cross_environment_drift": bool(row["E3_cross_environment_drift"]),
                    "case_risk_score": row["case_risk_score"],
                    "selection_rule": f"predefined gene priority: {gene}",
                }
            )
            seen.add(row["assertion_id"])
            if len(rows) >= 8:
                break
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_rule_selected_case_studies.csv", index=False)
    return out


def prospective_temporal_hardening_inventory() -> pd.DataFrame:
    inv_path = TABLES / "clinvar_historical_snapshot_inventory.csv"
    inv = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()
    desired = ["2023-01", "2023-07", "2024-01", "2024-07", "2025-01", "2025-07", "2026-04"]
    rows = []
    for snap in desired:
        matched = inv[inv["snapshot_date"].astype(str).eq(snap)] if not inv.empty else pd.DataFrame()
        if matched.empty:
            status = "not_materialized_locally"
            source = ""
        else:
            status = str(matched.iloc[0].get("file_available", "unknown"))
            source = str(matched.iloc[0].get("source_url_or_path", ""))
        rows.append(
            {
                "requested_snapshot": snap,
                "local_or_remote_status": status,
                "source": source,
                "can_run_temporal_panel_now": status in {"yes", "local"},
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "cab_prospective_temporal_hardening_inventory.csv", index=False)
    return out


def write_summary(generated: list[str]) -> None:
    lines = [
        "# CAB Silver-Standard Stress-Test Package",
        "",
        "This package implements proxy adjudication and robustness analyses requested for reviewer-hardening.",
        "",
        "## Key Claim Boundary",
        "",
        "The proxy adjudication layer uses locally available condition identifiers, environment mappings, ClinGen/VCEP coverage artifacts, and string/ontology heuristics. It is stronger than raw ClinVar label drift, but it is not a substitute for real blinded expert adjudication or full MONDO/HPO graph traversal with archived ontology releases.",
        "",
        "## Generated artifacts",
        "",
    ]
    lines += [f"- {path}" for path in generated]
    lines += [
        "",
        "## Main endpoint hierarchy",
        "",
        "- E1: crude condition-label drift.",
        "- E2: normalized condition-label drift after proxy synonym/submitter-noise repair.",
        "- E3: cross-environment drift.",
        "- E4: proxy-adjudicated true disease-model shift.",
        "",
    ]
    (QC / "cab_silver_standard_stress_test_summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_indexes() -> None:
    fig_index = ROOT / "reports" / "figures" / "final" / "FIGURE_INDEX.md"
    if fig_index.exists():
        text = fig_index.read_text(encoding="utf-8")
        line = "| Upgrade Figure 10 | Risk decile calibration | reports/tables/cab_risk_decile_calibration.csv | scripts/build_cab_silver_standard_stress_tests.py | CAB risk scores are evaluated for calibration, not only AUROC. |\n"
        if "Risk decile calibration" not in text:
            fig_index.write_text(text.rstrip() + "\n" + line, encoding="utf-8")
    table_index = ROOT / "reports" / "tables" / "final" / "TABLE_INDEX.md"
    if table_index.exists():
        text = table_index.read_text(encoding="utf-8")
        block = """
## Silver-Standard Stress-Test Tables

| Table | Role | Source |
|---|---|---|
| cab_proxy_adjudication_layer.csv | proxy adjudication of drifted and stable rows using condition IDs/environment mappings | scripts/build_cab_silver_standard_stress_tests.py |
| cab_proxy_adjudicated_main_claim_models.csv | main-claim models recomputed on proxy strict endpoints | scripts/build_cab_silver_standard_stress_tests.py |
| cab_leave_gene_family_environment_domain_out_validation.csv | leave-one-gene/family/environment/domain-out validation | scripts/build_cab_silver_standard_stress_tests.py |
| cab_ontology_only_baseline_comparator.csv | ontology-only baseline comparator | scripts/build_cab_silver_standard_stress_tests.py |
| cab_baseline_only_ontology_forecasting_comparator.csv | baseline-only ontology-like forecasting comparator without follow-up label leakage | scripts/build_cab_silver_standard_stress_tests.py |
| cab_negative_control_stable_domain_audit.csv | false-alarm audit in high-confidence stable strata | scripts/build_cab_silver_standard_stress_tests.py |
| cab_calibration_metrics.csv | Brier, ECE, calibration slope/intercept | scripts/build_cab_silver_standard_stress_tests.py |
| cab_risk_decile_calibration.csv | risk decile calibration source table | scripts/build_cab_silver_standard_stress_tests.py |
| cab_submitter_stratified_models.csv | submitter-stratified model robustness | scripts/build_cab_silver_standard_stress_tests.py |
| cab_external_curated_subset_agreement.csv | partial external curated-subset agreement proxy | scripts/build_cab_silver_standard_stress_tests.py |
| cab_sads_molecular_autopsy_special_validation_set.csv | SADS/molecular-autopsy special validation set | scripts/build_cab_silver_standard_stress_tests.py |
| cab_component_ablation_study.csv | ablation of gene, domain, ontology, metadata, and CAB regime components | scripts/build_cab_silver_standard_stress_tests.py |
| cab_baseline_only_component_ablation_study.csv | baseline-only ablation avoiding post-hoc ontology leakage | scripts/build_cab_silver_standard_stress_tests.py |
| cab_auprc_enrichment_review_utility.csv | AUPRC and review-budget enrichment across endpoint hierarchy | scripts/build_cab_silver_standard_stress_tests.py |
| cab_direct_use_safety_analysis.csv | direct-use safety analysis for ClinVar-label-only, CAB-Strict, and CAB-Balanced | scripts/build_cab_silver_standard_stress_tests.py |
| cab_overrestriction_and_repair_audit.csv | overrestriction and contextual-repair rescue audit | scripts/build_cab_silver_standard_stress_tests.py |
| cab_rule_selected_case_studies.csv | rule-selected case studies, not cherry-picked | scripts/build_cab_silver_standard_stress_tests.py |
"""
        if "Silver-Standard Stress-Test Tables" not in text:
            table_index.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def main() -> None:
    for path in [TABLES, FIGURES, QC, ADJ]:
        path.mkdir(parents=True, exist_ok=True)
    df = hardcore.load_benchmark_rows()
    df = add_condition_environment_context(df)
    proxy, _ = build_proxy_adjudication(df)
    df = add_proxy_endpoints(df, proxy)
    df.to_csv(TABLES / "cab_silver_standard_analysis_frame.csv", index=False)

    generated = [
        "reports/tables/cab_silver_standard_analysis_frame.csv",
        "reports/tables/cab_proxy_adjudication_layer.csv",
        "reports/tables/cab_proxy_adjudication_summary.csv",
    ]
    recompute_main_claims(df)
    generated += [
        "reports/tables/cab_proxy_adjudicated_main_claim_models.csv",
        "reports/tables/cab_proxy_adjudicated_main_claim_deltas.csv",
    ]
    leave_out_validation(df)
    ontology_baseline_comparator(df)
    baseline_only_ontology_forecasting_comparator(df)
    negative_control_stable_domains(df)
    calibration_and_decision_curves(df)
    submitter_stratified(df)
    external_curated_subset(df)
    sads_validation_set(df)
    component_ablation(df)
    baseline_only_component_ablation(df)
    auprc_enrichment(df, {})
    strict_endpoint_hierarchy(df)
    direct_use_and_overrestriction(df)
    case_studies(df)
    prospective_temporal_hardening_inventory()
    generated += [
        "reports/tables/cab_leave_gene_family_environment_domain_out_validation.csv",
        "reports/tables/cab_ontology_only_baseline_comparator.csv",
        "reports/tables/cab_baseline_only_ontology_forecasting_comparator.csv",
        "reports/tables/cab_baseline_only_ontology_incremental_cab_deltas.csv",
        "reports/tables/cab_negative_control_stable_domain_audit.csv",
        "reports/tables/cab_calibration_metrics.csv",
        "reports/tables/cab_risk_decile_calibration.csv",
        "reports/figures/cab_risk_decile_calibration.svg",
        "reports/tables/cab_submitter_stratified_models.csv",
        "reports/tables/cab_submitter_matched_drift_nondrift_analysis.csv",
        "reports/tables/cab_external_curated_subset_agreement.csv",
        "reports/tables/cab_sads_molecular_autopsy_special_validation_set.csv",
        "reports/tables/cab_component_ablation_study.csv",
        "reports/tables/cab_baseline_only_component_ablation_study.csv",
        "reports/tables/cab_auprc_enrichment_review_utility.csv",
        "reports/tables/cab_strict_endpoint_hierarchy_models.csv",
        "reports/tables/cab_direct_use_safety_analysis.csv",
        "reports/tables/cab_overrestriction_and_repair_audit.csv",
        "reports/tables/cab_rule_selected_case_studies.csv",
        "reports/tables/cab_prospective_temporal_hardening_inventory.csv",
    ]
    write_summary(generated)
    update_indexes()
    print("Wrote CAB silver-standard stress-test package")


if __name__ == "__main__":
    main()
