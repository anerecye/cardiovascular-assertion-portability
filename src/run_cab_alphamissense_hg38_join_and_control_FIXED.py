#!/usr/bin/env python3
"""Join CAB aligned variants to AlphaMissense hg38 and run missense-only controls.

This runner assumes AlphaMissense_hg38.tsv.gz has already been downloaded to:
data/raw/alphamissense/v3/AlphaMissense_hg38.tsv.gz

It performs:
1. genomic join by CHROM/POS/REF/ALT from CAB variant_key,
2. QC against ClinVar HGVS.p mapping candidates,
3. missense-only negative explanatory control models.

No publication claim is made unless high-confidence joins exist.
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

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

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"

CAB_FRAMEWORK = DATA / "cab_predictive_operational_framework.csv"
CROSS_ENV = DATA / "cab_cross_environment_drift.csv"
CPI_BASELINE = DATA / "cab_portability_index_baseline_only.csv"
MAPPING_CANDIDATES = TABLES / "cab_alphamissense_mapping_candidates.csv"
ALPHAMISSENSE_HG38 = RAW / "alphamissense" / "v3" / "AlphaMissense_hg38.tsv.gz"
README = RAW / "alphamissense" / "v3" / "README.md"

JOIN_OUT = TABLES / "cab_alphamissense_hg38_join.csv"
JOIN_QC_OUT = TABLES / "cab_alphamissense_hg38_join_qc.csv"
MODEL_OUT = TABLES / "cab_alphamissense_model_comparison.csv"
GENE_BIAS_OUT = TABLES / "cab_alphamissense_gene_bias_sensitivity.csv"
NEG_CONTROL_OUT = TABLES / "cab_alphamissense_negative_control.csv"
REPORT_OUT = QC / "cab_alphamissense_negative_control_report.md"

CHUNKSIZE = 1_000_000
RANDOM_STATE = 42
N_BOOT = 300


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def parse_variant_key(v) -> Tuple[str, float, str, str]:
    if pd.isna(v):
        return "", np.nan, "", ""
    parts = str(v).strip().split(":")
    if len(parts) != 4:
        return "", np.nan, "", ""
    chrom, pos, ref, alt = parts
    chrom = chrom.strip()
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    try:
        pos_i = int(pos)
    except Exception:
        return chrom, np.nan, ref.strip().upper(), alt.strip().upper()
    return chrom, float(pos_i), ref.strip().upper(), alt.strip().upper()


def parse_protein_variant(pv) -> Tuple[str, float, str]:
    if pd.isna(pv):
        return "", np.nan, ""
    m = re.match(r"^([A-Z*])([0-9]+)([A-Z*])$", str(pv).strip())
    if not m:
        return "", np.nan, ""
    return m.group(1), float(int(m.group(2))), m.group(3)


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def binarize_class(value: object) -> str:
    s = "" if pd.isna(value) else str(value).strip().lower()
    if "pathogenic" in s:
        return "likely_pathogenic"
    if "benign" in s:
        return "likely_benign"
    if "ambiguous" in s:
        return "ambiguous"
    return "unknown"


def load_base() -> pd.DataFrame:
    if not CAB_FRAMEWORK.exists():
        raise FileNotFoundError(CAB_FRAMEWORK)
    if not MAPPING_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Missing {MAPPING_CANDIDATES}; run run_cab_alphamissense_mapping_prep.py first."
        )
    if not ALPHAMISSENSE_HG38.exists():
        raise FileNotFoundError(ALPHAMISSENSE_HG38)

    cab = pd.read_csv(CAB_FRAMEWORK, low_memory=False)
    cab["variation_id"] = cab["variation_id"].map(norm_id)

    # Add baseline-only CPI if available.
    if CPI_BASELINE.exists():
        cpi = pd.read_csv(CPI_BASELINE, low_memory=False)
        cpi_id_col = "assertion_id" if "assertion_id" in cpi.columns else "variation_id"
        cpi[cpi_id_col] = cpi[cpi_id_col].map(norm_id)
        keep = [cpi_id_col] + [c for c in ["CPI_baseline_only", "nonportability_score_baseline_only", "CPI_tier_baseline_only"] if c in cpi.columns]
        cpi = cpi[keep].drop_duplicates(cpi_id_col)
        cab = cab.merge(cpi, left_on="variation_id", right_on=cpi_id_col, how="left")
        if cpi_id_col != "variation_id" and cpi_id_col in cab.columns:
            cab = cab.drop(columns=[cpi_id_col])

    if "CPI_baseline_only" not in cab.columns:
        cab["CPI_baseline_only"] = cab.get("cab_portability_index", np.nan)
    cab["nonportability_score_baseline_only"] = 100 - cab["CPI_baseline_only"].map(lambda x: safe_float(x, 50))

    # Add cross-environment endpoint if available.
    if CROSS_ENV.exists():
        ce = pd.read_csv(CROSS_ENV, low_memory=False)
        ce["variation_id"] = ce["variation_id"].map(norm_id)
        ce_keep = [c for c in ["variation_id", "cross_environment_drift", "within_environment_label_drift", "baseline_environment", "followup_environment"] if c in ce.columns]
        cab = cab.merge(ce[ce_keep].drop_duplicates("variation_id"), on="variation_id", how="left")
    if "cross_environment_drift" not in cab.columns:
        cab["cross_environment_drift"] = False

    # Add mapping candidates.
    mapdf = pd.read_csv(MAPPING_CANDIDATES, low_memory=False)
    mapdf["variation_id"] = mapdf["variation_id"].map(norm_id)
    map_keep = [c for c in [
        "variation_id", "clinvar_name_for_mapping", "transcript_candidate", "hgvs_p_normalized",
        "protein_ref_aa", "protein_position", "protein_alt_aa",
        "alphamissense_mapping_feasible", "alphamissense_mapping_status"
    ] if c in mapdf.columns]
    cab = cab.merge(mapdf[map_keep].drop_duplicates("variation_id"), on="variation_id", how="left", suffixes=("", "_mapping"))

    parsed = cab["variant_key"].map(parse_variant_key)
    cab["am_CHROM"] = [x[0] for x in parsed]
    cab["am_POS"] = [x[1] for x in parsed]
    cab["am_REF"] = [x[2] for x in parsed]
    cab["am_ALT"] = [x[3] for x in parsed]
    cab["am_join_key"] = (
        cab["am_CHROM"].astype(str)
        + ":"
        + cab["am_POS"].fillna(-1).astype(int).astype(str)
        + ":"
        + cab["am_REF"].astype(str)
        + ":"
        + cab["am_ALT"].astype(str)
    )
    return cab


def read_alpha_subset(join_keys: set[str]) -> pd.DataFrame:
    """Read only CAB-matching rows from AlphaMissense hg38.

    AlphaMissense_hg38.tsv.gz v3 is headerless in the downloaded Zenodo file.
    Column order from README / observed rows:
    CHROM POS REF ALT genome uniprot_id transcript_id protein_variant am_pathogenicity am_class
    """
    required = ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id", "transcript_id", "protein_variant", "am_pathogenicity", "am_class"]

    # Detect whether file has a header. The actual v3 Zenodo hg38 file is usually headerless.
    with gzip.open(ALPHAMISSENSE_HG38, "rt", encoding="utf-8", errors="replace") as fh:
        first_data_line = ""
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            first_data_line = line.rstrip("\n")
            break

    first_fields = first_data_line.split("\t")
    has_header = first_fields[:4] == ["CHROM", "POS", "REF", "ALT"]

    chunks = []
    read_kwargs = dict(
        sep="\t",
        compression="gzip",
        comment="#",
        chunksize=CHUNKSIZE,
        low_memory=False,
    )

    if has_header:
        iterator = pd.read_csv(ALPHAMISSENSE_HG38, usecols=required, **read_kwargs)
    else:
        iterator = pd.read_csv(ALPHAMISSENSE_HG38, names=required, header=None, **read_kwargs)

    for chunk in iterator:
        # Normalize.
        chunk["CHROM"] = chunk["CHROM"].astype(str)
        chunk["POS"] = pd.to_numeric(chunk["POS"], errors="coerce").astype("Int64")
        chunk["REF"] = chunk["REF"].astype(str).str.upper()
        chunk["ALT"] = chunk["ALT"].astype(str).str.upper()
        chunk["am_join_key"] = (
            chunk["CHROM"].astype(str)
            + ":"
            + chunk["POS"].astype(str)
            + ":"
            + chunk["REF"]
            + ":"
            + chunk["ALT"]
        )
        sub = chunk[chunk["am_join_key"].isin(join_keys)].copy()
        if len(sub):
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=required + ["am_join_key"])
    return pd.concat(chunks, ignore_index=True)

def resolve_alpha_matches(alpha: pd.DataFrame) -> pd.DataFrame:
    if alpha.empty:
        return alpha

    # Parse protein_variant.
    parsed = alpha["protein_variant"].map(parse_protein_variant)
    alpha["am_ref_aa"] = [x[0] for x in parsed]
    alpha["am_protein_position"] = [x[1] for x in parsed]
    alpha["am_alt_aa"] = [x[2] for x in parsed]

    # If multiple entries per genomic key exist, keep all for QC and then choose highest pathogenicity
    # as conservative single score, but mark multi-match.
    alpha["am_pathogenicity"] = pd.to_numeric(alpha["am_pathogenicity"], errors="coerce")
    alpha["am_class_normalized"] = alpha["am_class"].map(binarize_class)
    alpha["n_alpha_matches_for_key"] = alpha.groupby("am_join_key")["am_join_key"].transform("size")

    chosen = (
        alpha.sort_values(["am_join_key", "am_pathogenicity"], ascending=[True, False])
        .drop_duplicates("am_join_key", keep="first")
        .copy()
    )
    chosen["alpha_multi_match"] = chosen["n_alpha_matches_for_key"] > 1
    rename = {
        "uniprot_id": "alphamissense_uniprot_id",
        "transcript_id": "alphamissense_transcript_id",
        "protein_variant": "alphamissense_protein_variant",
        "am_pathogenicity": "alphamissense_score",
        "am_class": "alphamissense_class",
    }
    return chosen.rename(columns=rename)


def join_alpha(cab: pd.DataFrame, alpha_resolved: pd.DataFrame) -> pd.DataFrame:
    if alpha_resolved.empty:
        out = cab.copy()
        for c in [
            "alphamissense_uniprot_id", "alphamissense_transcript_id", "alphamissense_protein_variant",
            "alphamissense_score", "alphamissense_class", "am_ref_aa", "am_protein_position",
            "am_alt_aa", "n_alpha_matches_for_key", "alpha_multi_match"
        ]:
            out[c] = np.nan
    else:
        cols = [c for c in [
            "am_join_key", "alphamissense_uniprot_id", "alphamissense_transcript_id",
            "alphamissense_protein_variant", "alphamissense_score", "alphamissense_class",
            "am_class_normalized", "am_ref_aa", "am_protein_position", "am_alt_aa",
            "n_alpha_matches_for_key", "alpha_multi_match"
        ] if c in alpha_resolved.columns]
        out = cab.merge(alpha_resolved[cols], on="am_join_key", how="left")

    out["alphamissense_joined"] = out["alphamissense_score"].notna()
    out["alphamissense_join_status"] = np.select(
        [
            out["alphamissense_joined"] & out.get("alpha_multi_match", pd.Series(False, index=out.index)).fillna(False).astype(bool),
            out["alphamissense_joined"],
            out["am_CHROM"].eq("") | out["am_POS"].isna() | out["am_REF"].eq("") | out["am_ALT"].eq(""),
        ],
        [
            "joined_multi_match_highest_score_selected",
            "joined_high_confidence_genomic",
            "missing_or_invalid_variant_key",
        ],
        default="not_found_in_AlphaMissense_hg38",
    )

    # Compare ClinVar HGVS.p candidate with AlphaMissense protein_variant where possible.
    out["protein_variant_agrees_with_clinvar_hgvs_p"] = False
    for idx, row in out.iterrows():
        if not bool(row.get("alphamissense_joined", False)):
            continue
        ref = str(row.get("protein_ref_aa", "")).strip()
        alt = str(row.get("protein_alt_aa", "")).strip()
        pos = safe_float(row.get("protein_position"), np.nan)
        aref = str(row.get("am_ref_aa", "")).strip()
        aalt = str(row.get("am_alt_aa", "")).strip()
        apos = safe_float(row.get("am_protein_position"), np.nan)
        if ref and alt and not math.isnan(pos) and ref == aref and alt == aalt and int(pos) == int(apos):
            out.at[idx, "protein_variant_agrees_with_clinvar_hgvs_p"] = True

    out["alphamissense_analysis_eligible"] = (
        out["alphamissense_joined"].astype(bool)
        & out["protein_variant_agrees_with_clinvar_hgvs_p"].astype(bool)
    )
    return out


def prepare_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["future_condition_label_drift"] = out.get("condition_label_change", False).map(as_bool)
    out["any_meaning_drift"] = out.get("assertion_meaning_drift_score", 0).map(lambda x: safe_float(x, 0) > 0)
    out["cross_environment_drift"] = out.get("cross_environment_drift", False).map(as_bool)
    return out


def fit_model(df: pd.DataFrame, endpoint: str, features: list[str], model_name: str) -> Dict[str, object]:
    y = df[endpoint].astype(bool).astype(int)
    if y.nunique() < 2 or len(df) < 20:
        return {
            "endpoint": endpoint, "model": model_name, "N": len(df), "positive_N": int(y.sum()),
            "AUROC": np.nan, "AUROC_CI95_low": np.nan, "AUROC_CI95_high": np.nan,
            "AUPRC": np.nan, "Brier_score": np.nan, "log_loss": np.nan,
            "status": "skipped_insufficient_endpoint_or_N",
        }

    X = df[features].copy()
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in features if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))

    pre = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)),
    ])
    try:
        model.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        ll = log_loss(y, p, labels=[0, 1])

        # CV predictions.
        min_class = int(y.value_counts().min())
        n_splits = min(5, max(2, min_class))
        cv_auc = np.nan
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            try:
                p_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
                cv_auc = roc_auc_score(y, p_cv)
            except Exception:
                cv_auc = np.nan

        rng = np.random.default_rng(RANDOM_STATE)
        boots = []
        idx_all = np.arange(len(df))
        for _ in range(N_BOOT):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if len(np.unique(y.iloc[idx])) < 2:
                continue
            try:
                boots.append(roc_auc_score(y.iloc[idx], p[idx]))
            except Exception:
                continue
        lo, hi = (np.nan, np.nan)
        if boots:
            lo, hi = np.percentile(boots, [2.5, 97.5])

        return {
            "endpoint": endpoint, "model": model_name, "N": len(df), "positive_N": int(y.sum()),
            "AUROC": round(float(auc), 4),
            "AUROC_CI95_low": round(float(lo), 4) if not math.isnan(lo) else np.nan,
            "AUROC_CI95_high": round(float(hi), 4) if not math.isnan(hi) else np.nan,
            "cross_validated_AUROC": round(float(cv_auc), 4) if not math.isnan(cv_auc) else np.nan,
            "AUPRC": round(float(auprc), 4),
            "Brier_score": round(float(brier), 4),
            "log_loss": round(float(ll), 4),
            "status": "fit",
        }
    except Exception as e:
        return {
            "endpoint": endpoint, "model": model_name, "N": len(df), "positive_N": int(y.sum()),
            "AUROC": np.nan, "AUROC_CI95_low": np.nan, "AUROC_CI95_high": np.nan,
            "AUPRC": np.nan, "Brier_score": np.nan, "log_loss": np.nan,
            "status": f"fit_failed: {type(e).__name__}: {str(e)[:160]}",
        }


def run_models(joined: pd.DataFrame) -> pd.DataFrame:
    eligible = joined[joined["alphamissense_analysis_eligible"]].copy()
    if eligible.empty:
        rows = []
        for endpoint in ["future_condition_label_drift", "cross_environment_drift", "any_meaning_drift"]:
            rows.append({
                "endpoint": endpoint, "model": "AlphaMissense-only", "N": 0, "positive_N": 0,
                "AUROC": np.nan, "AUPRC": np.nan, "Brier_score": np.nan,
                "status": "skipped_no_high_confidence_AlphaMissense_joins",
            })
        return pd.DataFrame(rows)

    # Ensure feature aliases.
    eligible["alphamissense_class"] = eligible["alphamissense_class"].fillna("unknown").astype(str)
    eligible["gene"] = eligible["gene"].fillna("unknown").astype(str)
    eligible["causal_architecture_category"] = eligible["causal_architecture_category"].fillna("unknown").astype(str)
    eligible["primary_regime"] = eligible["primary_regime"].fillna("unknown").astype(str)
    eligible["baseline_review_category"] = eligible.get("baseline_review_category", pd.Series("unknown", index=eligible.index)).fillna("unknown").astype(str)
    eligible["CPI_baseline_only"] = eligible["CPI_baseline_only"].map(lambda x: safe_float(x, 50))
    eligible["evidence_collision_index"] = eligible.get("evidence_collision_index", pd.Series(0, index=eligible.index)).map(lambda x: safe_float(x, 0))
    eligible["regime_membership_count"] = eligible.get("regime_membership_count", pd.Series(0, index=eligible.index)).map(lambda x: safe_float(x, 0))

    model_specs = {
        "AlphaMissense-only": ["alphamissense_score", "alphamissense_class"],
        "CAB-only": ["CPI_baseline_only", "causal_architecture_category", "primary_regime", "evidence_collision_index", "regime_membership_count"],
        "gene-only": ["gene"],
        "gene+AlphaMissense": ["gene", "alphamissense_score", "alphamissense_class"],
        "CAB+AlphaMissense": ["CPI_baseline_only", "causal_architecture_category", "primary_regime", "evidence_collision_index", "regime_membership_count", "alphamissense_score", "alphamissense_class"],
        "gene+CAB+AlphaMissense": ["gene", "CPI_baseline_only", "causal_architecture_category", "primary_regime", "evidence_collision_index", "regime_membership_count", "alphamissense_score", "alphamissense_class"],
    }
    endpoints = ["future_condition_label_drift", "cross_environment_drift", "any_meaning_drift"]

    rows = []
    for endpoint in endpoints:
        for model_name, features in model_specs.items():
            rows.append(fit_model(eligible, endpoint, features, model_name))
    return pd.DataFrame(rows)


def build_qc(joined: pd.DataFrame, alpha_raw: pd.DataFrame) -> pd.DataFrame:
    n = len(joined)
    qc = []
    qc.append({"metric": "cab_rows", "value": n})
    qc.append({"metric": "alpha_raw_matches_before_resolution", "value": int(len(alpha_raw))})
    qc.append({"metric": "rows_joined_by_hg38_coordinate", "value": int(joined["alphamissense_joined"].sum())})
    qc.append({"metric": "rows_high_confidence_join_and_hgvs_p_agreement", "value": int(joined["alphamissense_analysis_eligible"].sum())})
    qc.append({"metric": "high_confidence_analysis_rate", "value": round(float(joined["alphamissense_analysis_eligible"].mean()), 4) if n else np.nan})
    qc.append({"metric": "rows_multi_match_highest_score_selected", "value": int(joined.get("alpha_multi_match", pd.Series(False, index=joined.index)).fillna(False).astype(bool).sum())})

    for status, count in joined["alphamissense_join_status"].value_counts(dropna=False).items():
        qc.append({"metric": f"join_status__{status}", "value": int(count)})

    # Gene bias: eligible rates.
    gene_bias = (
        joined.groupby("gene", dropna=False)
        .agg(
            n=("variation_id", "size"),
            high_confidence_join_n=("alphamissense_analysis_eligible", "sum"),
            condition_label_drift_rate=("future_condition_label_drift", "mean"),
            cross_environment_drift_rate=("cross_environment_drift", "mean"),
        )
        .reset_index()
        .sort_values(["high_confidence_join_n", "n"], ascending=[False, False])
    )
    gene_bias["high_confidence_join_rate"] = (gene_bias["high_confidence_join_n"] / gene_bias["n"]).round(4)
    gene_bias["condition_label_drift_rate"] = gene_bias["condition_label_drift_rate"].round(4)
    gene_bias["cross_environment_drift_rate"] = gene_bias["cross_environment_drift_rate"].round(4)
    gene_bias.to_csv(GENE_BIAS_OUT, index=False)

    return pd.DataFrame(qc)


def update_negative_control(model_results: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    # Replaces placeholder negative control with actual joined results where possible.
    if model_results.empty:
        return model_results

    rows = []
    for endpoint in ["future_condition_label_drift", "cross_environment_drift", "any_meaning_drift"]:
        sub = model_results[model_results["endpoint"] == endpoint].copy()
        for row in sub.to_dict("records"):
            interp = "AlphaMissense missense-only sensitivity analysis; not full CAB-universe control."
            if row["model"] == "AlphaMissense-only":
                interp = (
                    "Protein-level deleteriousness-only model in high-confidence missense subset; "
                    "compare against CAB/gene models to test whether molecular damage explains portability."
                )
            rows.append({
                "endpoint": endpoint,
                "model": row["model"],
                "AUROC": row.get("AUROC", np.nan),
                "AUPRC": row.get("AUPRC", np.nan),
                "Brier_score": row.get("Brier_score", np.nan),
                "N": row.get("N", np.nan),
                "positive_N": row.get("positive_N", np.nan),
                "alpha_source": "AlphaMissense_hg38_v3",
                "status": row.get("status", ""),
                "interpretation": interp,
            })
    return pd.DataFrame(rows)


def write_report(joined: pd.DataFrame, qc: pd.DataFrame, models: pd.DataFrame) -> None:
    eligible_n = int(joined["alphamissense_analysis_eligible"].sum())
    report = [
        "# CAB AlphaMissense Negative Explanatory Control Report",
        "",
        "Technical QC output; not manuscript prose.",
        "",
        "## Inputs",
        f"- `{ALPHAMISSENSE_HG38.relative_to(BASE)}`",
        f"- `{CAB_FRAMEWORK.relative_to(BASE)}`",
        f"- `{MAPPING_CANDIDATES.relative_to(BASE)}`",
        "",
        "## Outputs",
        f"- `{JOIN_OUT.relative_to(BASE)}`",
        f"- `{JOIN_QC_OUT.relative_to(BASE)}`",
        f"- `{MODEL_OUT.relative_to(BASE)}`",
        f"- `{GENE_BIAS_OUT.relative_to(BASE)}`",
        f"- `{NEG_CONTROL_OUT.relative_to(BASE)}`",
        "",
        "## QC summary",
        qc.to_string(index=False),
        "",
        "## Model comparison",
        models.to_string(index=False),
        "",
        "## Interpretation guardrails",
        f"- High-confidence AlphaMissense analysis subset: {eligible_n}/{len(joined)} CAB aligned assertions.",
        "- This is a missense-only sensitivity analysis, not a full CAB assertion-universe analysis.",
        "- AlphaMissense predictions are theoretical protein-level scores and are not clinical adjudications.",
        "- Claims of CAB versus AlphaMissense must be restricted to high-confidence missense joins.",
        "- If AlphaMissense-only is weak while CAB remains informative, this supports disease-model portability as distinct from molecular deleteriousness.",
        "- If AlphaMissense adds to CAB, interpret as complementary protein-level and assertion-portability layers.",
        "",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    cab = load_base()
    cab = prepare_endpoints(cab)

    keys = set(cab["am_join_key"].dropna().astype(str))
    keys = {k for k in keys if ":-1:" not in k and k.count(":") == 3}
    print(f"CAB rows: {len(cab):,}")
    print(f"Unique genomic join keys: {len(keys):,}")
    print(f"Reading AlphaMissense hg38 in chunks from {ALPHAMISSENSE_HG38} ...")

    alpha_subset = read_alpha_subset(keys)
    print(f"AlphaMissense raw coordinate matches: {len(alpha_subset):,}")

    alpha_resolved = resolve_alpha_matches(alpha_subset)
    joined = join_alpha(cab, alpha_resolved)
    joined = prepare_endpoints(joined)

    joined.to_csv(JOIN_OUT, index=False)

    qc = build_qc(joined, alpha_subset)
    qc.to_csv(JOIN_QC_OUT, index=False)

    models = run_models(joined)
    models.to_csv(MODEL_OUT, index=False)

    neg = update_negative_control(models, joined)
    neg.to_csv(NEG_CONTROL_OUT, index=False)

    write_report(joined, qc, models)

    print("CAB AlphaMissense hg38 join and negative control complete.")
    print(qc.to_string(index=False))
    print()
    print("Model comparison:")
    print(models.to_string(index=False))
    print(f"Wrote: {MODEL_OUT}")


if __name__ == "__main__":
    main()
