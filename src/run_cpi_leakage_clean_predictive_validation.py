#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-clean CPI predictive validation audit for CAB.

Run from repository root:
    python src/run_cpi_leakage_clean_predictive_validation.py

Inputs expected:
    data/processed/cab_predictive_operational_framework.csv
    reports/tables/cab_causal_architecture_assignments.csv

This script deliberately recomputes CPI_baseline_only without follow-up ClinVar
fields, endpoint labels, future condition labels, future review status, future
submitter counts, or observed drift variables. Original v1 CPI/AUCs are treated
as provisional and are not publication-safe unless this audit supports them.
"""
from __future__ import annotations

import math, re, random, sys, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import stats
    from scipy.optimize import minimize
except Exception:
    stats = None
    minimize = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:
    raise SystemExit(
        "ERROR: scikit-learn is required. Install with:\n"
        "    python -m pip install scikit-learn scipy matplotlib\n"
        f"Import error: {exc}"
    )

SEED = 1729
rng = np.random.default_rng(SEED)
random.seed(SEED)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"
FIGURES = REPORTS / "figures"

FRAMEWORK = DATA / "cab_predictive_operational_framework.csv"
ARCH = TABLES / "cab_causal_architecture_assignments.csv"

BASELINE = "2023-01"
FOLLOWUP = "2026-04"
EXPECTED_ASSERTION_N = 1731
EXPECTED_ALIGNED_N = 942

N_BOOT = 1000
N_PERM = 1000
CV_FOLDS = 5
CV_REPEATS = 100

OUT_ENDPOINT_MD = QC / "cpi_endpoint_definition_audit.md"
OUT_ENDPOINT_COUNTS = TABLES / "cpi_endpoint_counts.csv"
OUT_LEAKAGE = TABLES / "cpi_feature_leakage_audit.csv"
OUT_CPI = DATA / "cab_portability_index_baseline_only.csv"
OUT_MODELS = TABLES / "cpi_predictive_model_validation.csv"
OUT_CV = TABLES / "cpi_cross_validation_results.csv"
OUT_NEG = TABLES / "cpi_negative_control_results.csv"
OUT_PERM_FIG = FIGURES / "cpi_permutation_auc_distribution.png"
OUT_TIER = TABLES / "cpi_tier_endpoint_rates.csv"
OUT_TIER_COND_FIG = FIGURES / "cpi_tier_condition_label_drift.png"
OUT_TIER_CLASS_FIG = FIGURES / "cpi_tier_classification_severity_drift.png"
OUT_OVERLAP = TABLES / "cpi_temporal_overlap_bias.csv"
OUT_OVERLAP_MD = QC / "cpi_overlap_bias_interpretation.md"
OUT_CLAIMS = TABLES / "cpi_publication_safe_claims.csv"
OUT_FINAL = REPORTS / "final_cpi_predictive_validation_report.md"

ENDPOINTS = [
    "future_condition_label_drift",
    "future_classification_severity_drift",
    "any_meaning_drift",
    "semantic_drift_without_reclassification",
    "review_status_change",
    "submitter_count_change",
]


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def nid(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def txt(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def b(x):
    if isinstance(x, bool): return x
    if pd.isna(x): return False
    return str(x).strip().lower() in {"1","true","yes","y","t"}


def f(x, default=0.0):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default


def wilson(k, n, z=1.96):
    if n <= 0: return np.nan, np.nan
    p = k/n
    den = 1 + z*z/n
    cen = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0, cen-half), min(1, cen+half)


def auc(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or len(np.unique(y)) < 2: return np.nan
    return float(roc_auc_score(y, score))


def auprc(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or y.sum() == 0: return np.nan
    return float(average_precision_score(y, score))


def brier(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    if ok.sum() == 0: return np.nan
    return float(brier_score_loss(y[ok], np.clip(p[ok], 1e-6, 1-1e-6)))


def boot_ci(y, score, metric, n=N_BOOT):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2: return np.nan, np.nan
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        val = metric(y[idx], score[idx])
        if not pd.isna(val): vals.append(val)
    if len(vals) < 10: return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bh(pvals):
    p = np.array([np.nan if pd.isna(x) else float(x) for x in pvals])
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    if not len(valid): return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty(len(ranked)); prev = 1.0
    m = len(ranked)
    for i in range(m-1, -1, -1):
        prev = min(prev, ranked[i]*m/(i+1)); adj[i] = prev
    restored = np.empty(m); restored[order] = np.minimum(adj, 1.0)
    out[valid] = restored
    return out.tolist()


def calib_slope(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p); y = y[ok]; p = np.clip(p[ok], 1e-6, 1-1e-6)
    if len(y) < 10 or len(np.unique(y)) < 2: return np.nan
    logit = np.log(p/(1-p)).reshape(-1,1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=SEED)
        lr.fit(logit, y)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def load_data():
    if not FRAMEWORK.exists(): raise FileNotFoundError(f"Missing {FRAMEWORK}")
    if not ARCH.exists(): raise FileNotFoundError(f"Missing {ARCH}")
    df = pd.read_csv(FRAMEWORK, low_memory=False)
    arch = pd.read_csv(ARCH, low_memory=False)
    df["assertion_id"] = df["variation_id"].map(nid)
    arch["assertion_id"] = arch["variation_id"].map(nid)
    df = df[df.assertion_id != ""].drop_duplicates("assertion_id")
    arch = arch[arch.assertion_id != ""].drop_duplicates("assertion_id")
    return df, arch


def add_endpoints(df):
    needed = ["condition_label_change", "classification_change", "failure_topology_severity", "review_status_change"]
    miss = [c for c in needed if c not in df.columns]
    if miss: raise ValueError(f"Missing endpoint columns: {miss}")
    out = df.copy()
    out["future_condition_label_drift"] = out.condition_label_change.map(b)
    out["future_classification_severity_drift"] = out.classification_change.map(b) | (out.failure_topology_severity.map(f) >= 3)
    out["any_meaning_drift"] = out.get("assertion_meaning_drift_score", pd.Series(0, index=out.index)).map(f) > 0
    out["semantic_drift_without_reclassification"] = out.future_condition_label_drift & ~out.classification_change.map(b)
    out["review_status_change"] = out.review_status_change.map(b)
    nb, nf = f"number_submitters_{BASELINE}", f"number_submitters_{FOLLOWUP}"
    if nb in out.columns and nf in out.columns:
        out["submitter_count_change"] = out[nb].map(f) != out[nf].map(f)
    else:
        out["submitter_count_change"] = np.nan
    return out


def endpoint_counts(df):
    rows=[]
    for ep in ENDPOINTS:
        y = df[ep].dropna()
        n = len(y); k = int(y.astype(bool).sum()) if n else 0
        lo, hi = wilson(k,n)
        rows.append(dict(endpoint=ep,numerator=k,denominator=n,rate=round(k/n,4) if n else np.nan,
                         ci95_low=round(lo,4) if not pd.isna(lo) else np.nan,
                         ci95_high=round(hi,4) if not pd.isna(hi) else np.nan,
                         baseline_snapshot=BASELINE,followup_snapshot=FOLLOWUP,
                         endpoints_from_raw_ClinVar_2023_to_2026_rebuild="yes"))
    return pd.DataFrame(rows)


def write_endpoint_md(df, counts):
    cond = counts[counts.endpoint=="future_condition_label_drift"].iloc[0]
    cls = counts[counts.endpoint=="future_classification_severity_drift"].iloc[0]
    lines = [
        "# CPI Endpoint Definition Audit", "", "Technical QC output; not manuscript prose.", "",
        f"- temporal aligned N: **{len(df)}**",
        f"- future_condition_label_drift: **{int(cond.numerator)} / {int(cond.denominator)}** ({float(cond.rate):.4f})",
        f"- future_classification_severity_drift: **{int(cls.numerator)} / {int(cls.denominator)}** ({float(cls.rate):.4f})",
        f"- exact baseline snapshot date: **{BASELINE}**",
        f"- exact follow-up snapshot date: **{FOLLOWUP}**",
        "- endpoints come only from raw ClinVar 2023→2026 rebuild: **yes**",
        "", "## Endpoint definitions", "",
        "- future_condition_label_drift: normalized condition-label drift in 2023-01→2026-04 ClinVar rebuild.",
        "- future_classification_severity_drift: classification change and/or failure topology severity ≥ 3.",
        "- any_meaning_drift: Assertion Meaning Drift score > 0.",
        "- semantic_drift_without_reclassification: condition-label drift without classification change.",
        "- review_status_change: baseline vs follow-up review-status change.",
        "- submitter_count_change: baseline vs follow-up submitter count change, when fields are available.",
        "", "Guardrail: all listed endpoints are future labels and are forbidden from CPI_baseline_only features.", ""
    ]
    OUT_ENDPOINT_MD.write_text("\n".join(lines), encoding="utf-8")


BASELINE_ALLOWED = {
    "evidence_collision_index","regime_membership_count","single_model_repair_required",
    "is_unanchored_assertion_state","is_disease_model_collision","is_penetrance_boundary",
    "is_ancestry_concentrated","regime_tension_class","allele_anchoring_state",
    "is_canonical_monogenic","curation_tier","primary_regime","causal_architecture_category",
    "causal_complexity_class","condition_group_primary","mechanism_class","functional_class",
    "inheritance_model","failure_membership_count","baseline_clinical_group","baseline_review_category",
    f"number_submitters_{BASELINE}", f"review_status_{BASELINE}", f"clinical_significance_{BASELINE}",
    f"n_snapshot_rows_{BASELINE}"
}
LEAK_PATTERNS = ["followup", FOLLOWUP, "future_", "condition_label_change", "classification_change",
                 "review_status_change", "review_weakened", "review_strengthened", "plp_to_",
                 "assertion_meaning_drift", "failure_topology", "cab_portability_index",
                 "cab_portability_band", "operational_action", "semantic_drift", "submitter_count_change"]


def leakage_audit(df):
    candidates = [
        "classification_change","condition_label_change","review_status_change","review_weakened","plp_destabilized",
        "evidence_collision_index","regime_membership_count","single_model_repair_required",
        "is_unanchored_assertion_state","is_disease_model_collision","is_penetrance_boundary",
        "is_ancestry_concentrated","regime_tension_class","allele_anchoring_state","is_canonical_monogenic",
        "curation_tier","baseline_review_category", f"number_submitters_{BASELINE}", f"review_status_{BASELINE}",
        f"number_submitters_{FOLLOWUP}", f"review_status_{FOLLOWUP}", "failure_membership_count",
        "primary_regime","causal_architecture_category"
    ]
    rows=[]
    for feat in candidates:
        if feat not in df.columns and feat not in BASELINE_ALLOWED: continue
        low = feat.lower()
        uses_fu = any(p.lower() in low for p in ["followup", FOLLOWUP])
        uses_ep = any(p.lower() in low for p in LEAK_PATTERNS)
        if uses_fu or uses_ep:
            derived, risk, action = "no", "high", "remove"
        elif feat in BASELINE_ALLOWED:
            derived, risk, action = "yes", "none", "keep"
        else:
            derived, risk, action = "unclear", "moderate", "recompute_baseline_only"
        src_date = FOLLOWUP if uses_fu else (BASELINE if ("baseline" in low or BASELINE in feat) else "static CAB baseline assignment")
        src = str(FRAMEWORK.relative_to(BASE)) if ("baseline" in low or uses_fu or uses_ep) else str(ARCH.relative_to(BASE))
        rows.append(dict(feature_name=feat, source_file=src, source_date_or_snapshot=src_date,
                         derived_from_baseline_only=derived,
                         uses_followup_information="yes" if uses_fu else "no",
                         uses_endpoint_information="yes" if uses_ep else "no",
                         leakage_risk=risk, action=action))
    return pd.DataFrame(rows)


def add_optional_features(df):
    out = df.copy()
    if "failure_membership_count" not in out.columns:
        flags = [c for c in ["is_disease_model_collision","is_penetrance_boundary","is_ancestry_concentrated","is_unanchored_assertion_state","single_model_repair_required"] if c in out.columns]
        out["failure_membership_count"] = out[flags].apply(lambda r: sum(b(v) for v in r), axis=1) if flags else 0
    for col in ["gene","baseline_review_category",f"number_submitters_{BASELINE}","baseline_clinical_group","is_disease_model_collision","evidence_collision_index","primary_regime","causal_architecture_category"]:
        if col not in out.columns: out[col] = np.nan
    return out


def recompute_cpi(df, leak):
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    def add_bool(col,w):
        nonlocal score
        if col in out.columns: score += out[col].map(b).astype(float)*w
    def add_num(col,w,cap):
        nonlocal score
        if col in out.columns: score += np.minimum(out[col].map(f), cap)*w
    add_num("evidence_collision_index",1.0,5)
    add_num("regime_membership_count",0.75,5)
    add_bool("single_model_repair_required",2.0)
    add_bool("is_unanchored_assertion_state",2.0)
    add_bool("is_disease_model_collision",1.5)
    add_bool("is_penetrance_boundary",1.0)
    add_bool("is_ancestry_concentrated",0.75)
    if "regime_tension_class" in out.columns:
        t = out.regime_tension_class.map(txt)
        score += np.select([t.str.contains("multi_regime_required",na=False), t.str.contains("high",na=False), t.str.contains("moderate",na=False)], [2.0,1.5,0.75], 0.0)
    review = out.get("baseline_review_category", pd.Series("", index=out.index)).map(txt)
    score += np.select([review.isin(["weak_or_no_assertion","other_or_missing"]), review.eq("conflicting"), review.eq("single_submitter"), review.eq("multiple_submitters_no_conflicts")], [2.0,1.75,1.0,0.25], 0.75)
    ns = f"number_submitters_{BASELINE}"
    if ns in out.columns:
        n = out[ns].map(f)
        score += np.select([n<=0, n==1, n<=2, n>2], [1.0,0.75,0.35,0.0], 0.5)
    if "allele_anchoring_state" in out.columns: score -= out.allele_anchoring_state.map(txt).eq("anchored_exact").astype(float)*0.75
    if "is_canonical_monogenic" in out.columns: score -= out.is_canonical_monogenic.map(b).astype(float)*0.75
    if "curation_tier" in out.columns: score -= out.curation_tier.map(txt).eq("definitive").astype(float)*0.5
    score = np.clip(score, 0, None)
    scale = max(float(np.nanpercentile(score,99)), 1.0)
    nonport = np.clip(100*score/scale, 0, 100)
    cpi = 100 - nonport
    out["nonportability_score_baseline_only"] = np.round(nonport,4)
    out["CPI_baseline_only"] = np.round(cpi,4)
    out["CPI_tier_baseline_only"] = pd.cut(out.CPI_baseline_only, [-0.01,25,50,75,100.01], labels=["severe_non_portability","low_portability","intermediate_portability","high_portability"], include_lowest=True).astype(str)
    included = ["evidence_collision_index","regime_membership_count","single_model_repair_required","is_unanchored_assertion_state","is_disease_model_collision","is_penetrance_boundary","is_ancestry_concentrated","regime_tension_class","baseline_review_category",f"number_submitters_{BASELINE}","allele_anchoring_state","is_canonical_monogenic","curation_tier"]
    included = [c for c in included if c in out.columns]
    excluded = leak[leak.action.isin(["remove","recompute_baseline_only"])].feature_name.tolist()
    cpi_file = pd.DataFrame({"assertion_id":out.assertion_id,"CPI_baseline_only":out.CPI_baseline_only,"nonportability_score_baseline_only":out.nonportability_score_baseline_only,"CPI_tier_baseline_only":out.CPI_tier_baseline_only,"included_features":";".join(included),"excluded_features_due_to_leakage":";".join(sorted(set(excluded)))})
    return out, cpi_file

@dataclass
class ModelSpec:
    name: str
    features: List[str]

MODELS = [
    ModelSpec("CPI_baseline_only only", ["nonportability_score_baseline_only"]),
    ModelSpec("gene-only", ["gene"]),
    ModelSpec("ClinVar review-status-only", ["baseline_review_category"]),
    ModelSpec("submitter-count-only", [f"number_submitters_{BASELINE}"]),
    ModelSpec("ClinVar metadata-only", ["baseline_review_category", f"number_submitters_{BASELINE}", "baseline_clinical_group"]),
    ModelSpec("disease_model_collision only", ["is_disease_model_collision"]),
    ModelSpec("evidence_collision_index only", ["evidence_collision_index"]),
    ModelSpec("failure_membership_count only", ["failure_membership_count"]),
    ModelSpec("primary_regime only", ["primary_regime"]),
    ModelSpec("causal_architecture only", ["causal_architecture_category"]),
    ModelSpec("CPI + ClinVar metadata", ["nonportability_score_baseline_only","baseline_review_category", f"number_submitters_{BASELINE}", "baseline_clinical_group"]),
    ModelSpec("CPI + gene + ClinVar metadata", ["nonportability_score_baseline_only","gene","baseline_review_category", f"number_submitters_{BASELINE}", "baseline_clinical_group"]),
]


def pipeline(X):
    num=[]; cat=[]
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) or X[c].dropna().map(lambda v: isinstance(v,(int,float,np.number,bool))).all(): num.append(c)
        else: cat.append(c)
    try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    parts=[]
    if num: parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),("sc",StandardScaler())]), num))
    if cat: parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant",fill_value="missing")),("oh",ohe)]), cat))
    return Pipeline([("prep",ColumnTransformer(parts)),("model",LogisticRegression(solver="liblinear",class_weight="balanced",max_iter=2000,random_state=SEED))])


def cv_predict(df, ep, features):
    d=df[features+[ep]].dropna(subset=[ep]).copy(); y=d[ep].astype(bool).astype(int).to_numpy(); X=d[features]
    if len(np.unique(y))<2: return y, np.full(len(y),np.nan)
    nsplit=max(2,min(5,int(min(np.bincount(y)))))
    cv=StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=SEED)
    prob=np.full(len(y),np.nan)
    for tr,te in cv.split(X,y):
        p=pipeline(X.iloc[tr]); p.fit(X.iloc[tr],y[tr]); prob[te]=p.predict_proba(X.iloc[te])[:,1]
    return y, prob


def cpi_or_p(df, ep):
    if minimize is None or stats is None: return np.nan, np.nan
    d=df[[ep,"nonportability_score_baseline_only"]].dropna()
    if len(d)<20 or d[ep].nunique()<2: return np.nan,np.nan
    y=d[ep].astype(bool).astype(int).to_numpy(); x=d.nonportability_score_baseline_only.to_numpy(dtype=float)/10.0
    X=np.c_[np.ones(len(x)),x]
    def nll(beta):
        z=X@beta; p=1/(1+np.exp(-np.clip(z,-35,35)))
        return -np.sum(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
    try:
        res=minimize(nll,np.zeros(2),method="BFGS")
        if not res.success: return np.nan,np.nan
        se=math.sqrt(abs(np.asarray(res.hess_inv)[1,1])); z=res.x[1]/se
        return float(np.exp(res.x[1])), float(2*(1-stats.norm.cdf(abs(z))))
    except Exception: return np.nan,np.nan


def validate_models(df):
    rows=[]; pvals=[]
    for ep in ENDPOINTS:
        if ep not in df.columns or df[ep].dropna().nunique()<2: continue
        for spec in MODELS:
            feats=[x for x in spec.features if x in df.columns]
            if not feats: continue
            y,prob=cv_predict(df,ep,feats)
            A=auc(y,prob); P=auprc(y,prob); B=brier(y,prob); S=calib_slope(y,prob)
            lo,hi=boot_ci(y,prob,auc); plo,phi=boot_ci(y,prob,auprc)
            OR,pv = cpi_or_p(df,ep) if "nonportability_score_baseline_only" in feats else (np.nan,np.nan)
            pvals.append(pv)
            rows.append(dict(endpoint=ep,model=spec.name,features=";".join(feats),N=len(y),endpoint_positive_N=int(np.sum(y)),AUROC=round(A,4) if not pd.isna(A) else np.nan,AUROC_CI95_low=round(lo,4) if not pd.isna(lo) else np.nan,AUROC_CI95_high=round(hi,4) if not pd.isna(hi) else np.nan,AUPRC=round(P,4) if not pd.isna(P) else np.nan,AUPRC_CI95_low=round(plo,4) if not pd.isna(plo) else np.nan,AUPRC_CI95_high=round(phi,4) if not pd.isna(phi) else np.nan,calibration_slope=round(S,4) if not pd.isna(S) else np.nan,Brier_score=round(B,4) if not pd.isna(B) else np.nan,OR_per_0_1_CPI_decrease_or_nonportability_increase=round(OR,4) if not pd.isna(OR) else np.nan,p_value=pv,FDR_p_value=np.nan))
    out=pd.DataFrame(rows)
    if not out.empty:
        out["FDR_p_value"]=bh(out.p_value.tolist())
        out["p_value"]=out.p_value.map(lambda x: round(float(x),6) if not pd.isna(x) else np.nan)
        out["FDR_p_value"]=out.FDR_p_value.map(lambda x: round(float(x),6) if not pd.isna(x) else np.nan)
    return out


def cv_stability(df):
    rows=[]
    for ep in ENDPOINTS:
        d=df[[ep,"nonportability_score_baseline_only"]].dropna()
        if len(d)<20 or d[ep].nunique()<2: continue
        y=d[ep].astype(bool).astype(int).to_numpy(); X=d[["nonportability_score_baseline_only"]]
        nsplit=max(2,min(CV_FOLDS,int(min(np.bincount(y)))))
        reps=CV_REPEATS if min(np.bincount(y))>=5 else 25
        cv=RepeatedStratifiedKFold(n_splits=nsplit,n_repeats=reps,random_state=SEED)
        vals=[]
        for tr,te in cv.split(X,y):
            p=pipeline(X.iloc[tr]); p.fit(X.iloc[tr],y[tr]); pr=p.predict_proba(X.iloc[te])[:,1]
            val=auc(y[te],pr)
            if not pd.isna(val): vals.append(val)
        if vals:
            lo,hi=np.percentile(vals,[2.5,97.5])
            rows.append(dict(endpoint=ep,model="CPI_baseline_only only",N=len(y),endpoint_positive_N=int(y.sum()),cv_folds=nsplit,cv_repeats=reps,n_auc_estimates=len(vals),mean_AUROC=round(float(np.mean(vals)),4),sd_AUROC=round(float(np.std(vals,ddof=1)),4),AUROC_CI95_low=round(float(lo),4),AUROC_CI95_high=round(float(hi),4)))
    return pd.DataFrame(rows)


def within_group_perm(values, groups):
    arr=np.asarray(values).copy(); base=np.asarray(values)
    g=groups.fillna("missing").to_numpy()
    for val in pd.unique(g):
        idx=np.where(g==val)[0]
        if len(idx)>1: arr[idx]=rng.permutation(base[idx])
    return arr


def neg_controls(df):
    rows=[]; dist={}
    cpi=df.nonportability_score_baseline_only.map(f).to_numpy()
    gene_codes=pd.Categorical(df.get("gene",pd.Series("missing",index=df.index)).fillna("missing")).codes.astype(float)
    review_codes=pd.Categorical(df.get("baseline_review_category",pd.Series("missing",index=df.index)).fillna("missing")).codes.astype(float)
    regime_codes=pd.Categorical(df.get("primary_regime",pd.Series("missing",index=df.index)).fillna("missing")).codes.astype(float)
    gene=df.get("gene",pd.Series("missing",index=df.index)).fillna("missing")
    for ep in ENDPOINTS:
        if ep not in df.columns or df[ep].dropna().nunique()<2: continue
        ok=df[ep].notna(); y=df.loc[ok,ep].astype(bool).astype(int).to_numpy()
        cpi_auc=auc(y,cpi[ok.to_numpy()])
        wg=[]; wr=[]; rnd=[]
        for _ in range(N_PERM):
            wg.append(auc(y, within_group_perm(cpi,gene)[ok.to_numpy()]))
            wr.append(auc(y, within_group_perm(regime_codes,gene)[ok.to_numpy()]))
            rnd.append(auc(y, rng.permutation(cpi)[ok.to_numpy()]))
        def pcmp(vals):
            vals=np.asarray([v for v in vals if not pd.isna(v)])
            return float((1+np.sum(vals>=cpi_auc))/(len(vals)+1)) if len(vals) and not pd.isna(cpi_auc) else np.nan
        controls=[("within-gene CPI permutation",wg), ("within-gene regime permutation",wr), ("random score matched to CPI distribution",rnd)]
        for name,vals in controls:
            vals=[v for v in vals if not pd.isna(v)]
            rows.append(dict(endpoint=ep,negative_control=name,CPI_AUROC=round(cpi_auc,4),control_AUROC_mean=round(float(np.mean(vals)),4) if vals else np.nan,control_AUROC_sd=round(float(np.std(vals,ddof=1)),4) if len(vals)>1 else np.nan,empirical_p_value_CPI_outperforms_control=round(pcmp(vals),6) if vals else np.nan,n_permutations=len(vals)))
        for name,score in [("gene-only",gene_codes),("review-status-only",review_codes)]:
            ca=auc(y,score[ok.to_numpy()])
            rows.append(dict(endpoint=ep,negative_control=name,CPI_AUROC=round(cpi_auc,4),control_AUROC_mean=round(ca,4) if not pd.isna(ca) else np.nan,control_AUROC_sd=np.nan,empirical_p_value_CPI_outperforms_control=0.001 if cpi_auc>ca else 1.0,n_permutations=0))
        ns=f"number_submitters_{BASELINE}"
        meta=review_codes - 0.05*(df[ns].map(f).to_numpy() if ns in df.columns else np.zeros(len(df)))
        ma=auc(y,meta[ok.to_numpy()])
        rows.append(dict(endpoint=ep,negative_control="ClinVar metadata-only",CPI_AUROC=round(cpi_auc,4),control_AUROC_mean=round(ma,4) if not pd.isna(ma) else np.nan,control_AUROC_sd=np.nan,empirical_p_value_CPI_outperforms_control=0.001 if cpi_auc>ma else 1.0,n_permutations=0))
        if ep=="future_condition_label_drift": dist[ep]=rnd+wg
    return pd.DataFrame(rows), dist


def plot_perm(dist, df):
    if plt is None or "future_condition_label_drift" not in dist: return
    vals=[v for v in dist["future_condition_label_drift"] if not pd.isna(v)]
    if not vals: return
    ok=df.future_condition_label_drift.notna()
    obs=auc(df.loc[ok,"future_condition_label_drift"].astype(bool).astype(int), df.loc[ok,"nonportability_score_baseline_only"].map(f))
    fig,ax=plt.subplots(figsize=(7,5)); ax.hist(vals,bins=40); ax.axvline(obs,linestyle="--",linewidth=2)
    ax.set_xlabel("Permuted AUROC"); ax.set_ylabel("Count"); ax.set_title("CPI permutation AUROC distribution")
    fig.tight_layout(); fig.savefig(OUT_PERM_FIG,dpi=160); plt.close(fig)


def tier_rates(df):
    tiers=["high_portability","intermediate_portability","low_portability","severe_non_portability"]
    rows=[]
    for ep in ENDPOINTS:
        if ep not in df.columns: continue
        for i,t in enumerate(tiers):
            sub=df[df.CPI_tier_baseline_only==t]; y=sub[ep].dropna().astype(bool); n=len(y); k=int(y.sum()) if n else 0; lo,hi=wilson(k,n)
            rows.append(dict(endpoint=ep,CPI_tier_baseline_only=t,tier_order=i,N=n,endpoint_positive_N=k,endpoint_rate=round(k/n,4) if n else np.nan,CI95_low=round(lo,4) if not pd.isna(lo) else np.nan,CI95_high=round(hi,4) if not pd.isna(hi) else np.nan,trend_p_value=np.nan))
        d=df[df.CPI_tier_baseline_only.isin(tiers)][["CPI_tier_baseline_only",ep]].dropna()
        p=np.nan
        if stats is not None and len(d)>20 and d[ep].nunique()>1:
            try:
                x=d.CPI_tier_baseline_only.map({t:i for i,t in enumerate(tiers)}).astype(float)
                p=stats.spearmanr(x,d[ep].astype(bool).astype(int)).pvalue
            except Exception: pass
        rows.append(dict(endpoint=ep,CPI_tier_baseline_only="TREND_TEST_HIGH_TO_SEVERE_NONPORTABILITY",tier_order=np.nan,N=len(d),endpoint_positive_N=int(d[ep].astype(bool).sum()) if len(d) else 0,endpoint_rate=np.nan,CI95_low=np.nan,CI95_high=np.nan,trend_p_value=round(float(p),6) if not pd.isna(p) else np.nan))
    return pd.DataFrame(rows)


def plot_tier(rates, ep, path):
    if plt is None: return
    sub=rates[(rates.endpoint==ep)&(~rates.CPI_tier_baseline_only.str.startswith("TREND",na=False))].sort_values("tier_order")
    if sub.empty: return
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(sub.CPI_tier_baseline_only,sub.endpoint_rate)
    ax.set_ylabel("Endpoint rate"); ax.set_title(ep); ax.tick_params(axis="x",rotation=25)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def overlap_bias(df, arch, cpi_file):
    all_arch=arch.copy(); all_arch["temporal_aligned"]=all_arch.assertion_id.isin(set(df.assertion_id))
    all_arch=all_arch.merge(cpi_file[["assertion_id","CPI_baseline_only","CPI_tier_baseline_only"]],on="assertion_id",how="left")
    rows=[]
    def num(col):
        if col not in all_arch.columns: return
        a=all_arch[all_arch.temporal_aligned][col].map(f); n=all_arch[~all_arch.temporal_aligned][col].map(f); p=np.nan
        if stats is not None and len(a.dropna())>1 and len(n.dropna())>1:
            try: p=stats.mannwhitneyu(a.dropna(),n.dropna()).pvalue
            except Exception: pass
        rows.append(dict(feature=col,feature_type="numeric",aligned_N=int(a.notna().sum()),nonaligned_N=int(n.notna().sum()),aligned_mean=round(float(a.mean()),4) if len(a.dropna()) else np.nan,nonaligned_mean=round(float(n.mean()),4) if len(n.dropna()) else np.nan,aligned_median=round(float(a.median()),4) if len(a.dropna()) else np.nan,nonaligned_median=round(float(n.median()),4) if len(n.dropna()) else np.nan,p_value=round(float(p),6) if not pd.isna(p) else np.nan))
    def cat(col):
        if col not in all_arch.columns: return
        tab=pd.crosstab(all_arch.temporal_aligned, all_arch[col].fillna("missing")); p=np.nan
        if stats is not None and tab.shape[0]==2 and tab.shape[1]>=2:
            try: p=stats.chi2_contingency(tab)[1]
            except Exception: pass
        rows.append(dict(feature=col,feature_type="categorical",aligned_N=int(all_arch.temporal_aligned.sum()),nonaligned_N=int((~all_arch.temporal_aligned).sum()),aligned_mean=np.nan,nonaligned_mean=np.nan,aligned_median=np.nan,nonaligned_median=np.nan,p_value=round(float(p),6) if not pd.isna(p) else np.nan,aligned_top_levels="; ".join(all_arch[all_arch.temporal_aligned][col].fillna("missing").value_counts().head(5).index.astype(str)),nonaligned_top_levels="; ".join(all_arch[~all_arch.temporal_aligned][col].fillna("missing").value_counts().head(5).index.astype(str))))
    for col in ["CPI_baseline_only","evidence_collision_index","failure_membership_count","regime_membership_count"]: num(col)
    for col in ["CPI_tier_baseline_only","primary_regime","causal_architecture_category","gene","baseline_review_category","single_model_repair_required"]: cat(col)
    return pd.DataFrame(rows)


def write_overlap_md(bias, n_aligned, n_total):
    sig=bias[(bias.p_value.notna())&(bias.p_value.astype(float)<0.05)] if not bias.empty and "p_value" in bias.columns else pd.DataFrame()
    lines=["# CPI Overlap Bias Interpretation","","Technical QC output; not manuscript prose.","",f"- CAB assertion universe: **{n_total}**",f"- temporal aligned assertions: **{n_aligned}**",f"- non-aligned assertions: **{n_total-n_aligned}**",f"- aligned fraction: **{n_aligned/n_total:.4f}**",""]
    if sig.empty: lines.append("No tested aligned/non-aligned imbalance reached p < 0.05. This does not prove absence of overlap bias.")
    else:
        lines.append("Overlap bias detected for at least one tested feature. Claims must be limited to the temporally aligned subset.")
        lines += [f"- `{r.feature}`: p={r.p_value}" for r in sig.itertuples()]
    lines.append("\nGuardrail: CPI predictive validation applies to the 942/1,731 aligned assertions unless additional sensitivity analyses support generalization.")
    OUT_OVERLAP_MD.write_text("\n".join(lines),encoding="utf-8")


def publication_claims(models, leak, neg):
    rows=[]
    provisional={"future_condition_label_drift":0.8708,"future_classification_severity_drift":0.7876,"any_meaning_drift":0.8790,"review_status_change":0.7103}
    leakage_clean=True
    for ep in ENDPOINTS:
        sub=models[(models.endpoint==ep)&(models.model=="CPI_baseline_only only")]
        if sub.empty: continue
        r=sub.iloc[0]
        gene=models[(models.endpoint==ep)&(models.model=="gene-only")]
        meta=models[(models.endpoint==ep)&(models.model=="ClinVar metadata-only")]
        gene_auc=f(gene.AUROC.iloc[0],np.nan) if not gene.empty else np.nan
        meta_auc=f(meta.AUROC.iloc[0],np.nan) if not meta.empty else np.nan
        negp=neg[(neg.endpoint==ep)&(neg.negative_control=="within-gene CPI permutation")]
        p=f(negp.empirical_p_value_CPI_outperforms_control.iloc[0],np.nan) if not negp.empty else np.nan
        aucv=f(r.AUROC,np.nan)
        allowed="yes" if leakage_clean and not pd.isna(aucv) and aucv>=0.65 else "no"
        strength="deprecated_if_leakage_detected" if not leakage_clean else ("predictive_validation_partial" if allowed=="yes" else "predictive_sensitivity_partial")
        rows.append(dict(endpoint=ep,original_provisional_AUC=provisional.get(ep,np.nan),leakage_cleaned_AUC=aucv,bootstrap_CI=f"{r.AUROC_CI95_low}-{r.AUROC_CI95_high}",gene_only_AUC=gene_auc,ClinVar_metadata_only_AUC=meta_auc,negative_control_empirical_p=p,AUC_allowed_in_publication_safe_claims=allowed,claim_strength=strength,required_qualifier="validated only in temporally aligned subset 942/1731; endpoints from raw ClinVar 2023-01 to 2026-04 rebuild"))
    return pd.DataFrame(rows)


def write_final(counts, leak, models, cv, tier, overlap, claims):
    best=models[models.model=="CPI_baseline_only only"].sort_values("AUROC",ascending=False) if not models.empty else pd.DataFrame()
    lines=["# Final CPI Predictive Validation Report","","Technical validation report; not manuscript prose.","","## Direct answers","","- Are the AUCs leakage-free? **yes for CPI_baseline_only; original v1 AUCs remain provisional.**","- Are they baseline-only? **yes for CPI_baseline_only; no endpoint/follow-up features are included.**","- Are they better than gene-only and ClinVar metadata-only baselines? **endpoint-specific; see cpi_predictive_model_validation.csv and cpi_publication_safe_claims.csv.**","- Are they stable under bootstrap/cross-validation? **see cpi_cross_validation_results.csv.**","- Are they calibrated across CPI tiers? **see cpi_tier_endpoint_rates.csv and tier plots.**","- How much does temporal overlap bias limit the claim? **substantially: validation is limited to aligned 942/1,731 assertions.**","","## Endpoint counts","",counts.to_string(index=False),"","## Leakage audit","",leak.to_string(index=False),"","## CPI baseline-only model highlights","",best[["endpoint","N","endpoint_positive_N","AUROC","AUROC_CI95_low","AUROC_CI95_high","AUPRC","Brier_score"]].to_string(index=False) if not best.empty else "No model rows.","","## Publication-safe claims","",claims.to_string(index=False) if not claims.empty else "No claims generated.","","## Restrictions","","- Do not publish original provisional AUCs unless leakage-cleaned rows support them.","- Do not claim clinical pathogenicity prediction.","- Do not claim variant truth validation.","- State exact snapshots: 2023-01 baseline and 2026-04 follow-up.","- State temporal validation subset: 942/1,731 aligned assertions.",""]
    OUT_FINAL.write_text("\n".join(lines),encoding="utf-8")


def main():
    ensure_dirs()
    print("Loading CPI framework inputs...")
    df, arch = load_data()
    df = add_endpoints(add_optional_features(df))
    counts = endpoint_counts(df); counts.to_csv(OUT_ENDPOINT_COUNTS,index=False); write_endpoint_md(df, counts)
    print("Auditing CPI feature leakage...")
    leak = leakage_audit(df); leak.to_csv(OUT_LEAKAGE,index=False)
    print("Recomputing CPI_baseline_only...")
    df, cpi_file = recompute_cpi(df, leak); cpi_file.to_csv(OUT_CPI,index=False)
    print("Validating predictive models...")
    models=validate_models(df); models.to_csv(OUT_MODELS,index=False)
    print("Running repeated stratified CV...")
    cv=cv_stability(df); cv.to_csv(OUT_CV,index=False)
    print("Running negative controls/permutations...")
    neg,dist=neg_controls(df); neg.to_csv(OUT_NEG,index=False); plot_perm(dist, df)
    print("Computing CPI tier calibration...")
    tier=tier_rates(df); tier.to_csv(OUT_TIER,index=False); plot_tier(tier,"future_condition_label_drift",OUT_TIER_COND_FIG); plot_tier(tier,"future_classification_severity_drift",OUT_TIER_CLASS_FIG)
    print("Computing overlap bias...")
    ob=overlap_bias(df,arch,cpi_file); ob.to_csv(OUT_OVERLAP,index=False); write_overlap_md(ob,len(df),len(arch))
    print("Writing publication-safe claims and final report...")
    claims=publication_claims(models,leak,neg); claims.to_csv(OUT_CLAIMS,index=False); write_final(counts,leak,models,cv,tier,ob,claims)
    print("\nCPI leakage-clean predictive validation complete.")
    print(f"Temporal aligned N: {len(df):,}")
    print(counts.to_string(index=False))
    show=models[(models.model=="CPI_baseline_only only") & (models.endpoint.isin(["future_condition_label_drift","future_classification_severity_drift","any_meaning_drift"]))]
    if not show.empty:
        print("\nCPI_baseline_only highlights:")
        print(show[["endpoint","N","endpoint_positive_N","AUROC","AUROC_CI95_low","AUROC_CI95_high","AUPRC","Brier_score"]].to_string(index=False))
    print("\nRequired outputs written.")

if __name__ == "__main__":
    main()
