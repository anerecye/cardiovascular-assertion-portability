
#!/usr/bin/env python3
"""Materialize full three-domain CAB benchmark exports.

Run from repository root:

    python scripts/materialize_full_three_domain_benchmark_exports.py

Input:
    data/processed/cab_decision_challenge_tasks.csv

Outputs:
    benchmark/{domain}/baseline_assertions.csv
    benchmark/{domain}/followup_assertions.csv
    benchmark/{domain}/temporal_endpoints.csv
    benchmark/{domain}/expected_metrics.json
    reports/tables/cab_benchmark_index.csv
    reports/workflow_simulation/full_exports_materialization_report.md

Baseline routing inputs remain baseline-only. Future endpoint labels are written
only to temporal_endpoints.csv. Follow-up files are benchmark replay files; if
explicit follow-up labels are unavailable, synthetic follow-up labels are created
from endpoint flags only for CLI endpoint reconstruction.

CAB is not a diagnostic tool, does not reclassify variants, and does not replace
ACMG/AMP or expert curation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path.cwd()
SOURCE = ROOT / "data" / "processed" / "cab_decision_challenge_tasks.csv"

DOMAINS = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]

BASELINE_COLS = [
    "assertion_id",
    "domain",
    "variation_id",
    "gene",
    "input_condition_label",
    "classification",
    "review_status",
    "submitter_count",
    "baseline_environment",
    "cab_portability_regime",
    "cab_portability_score",
    "baseline_regime_primary",
    "baseline_architecture_family",
]

ENDPOINT_COLS = [
    "assertion_id",
    "domain",
    "future_condition_label_drift",
    "future_classification_change",
    "future_classification_severity_drift",
    "future_cross_environment_drift",
    "future_any_meaning_drift",
    "semantic_drift_without_reclassification",
    "review_status_change",
    "submitter_count_change",
]

COLUMN_ALIASES = {
    "assertion_id": ["assertion_id", "VariationID", "variation_id"],
    "variation_id": ["variation_id", "VariationID", "assertion_id"],
    "domain": ["domain"],
    "gene": ["gene", "GeneSymbol", "gene_symbol"],
    "input_condition_label": [
        "input_condition_label",
        "condition_label",
        "PhenotypeList",
        "phenotype_list",
        "condition_baseline",
        "normalized_condition_label",
    ],
    "classification": [
        "classification",
        "ClinicalSignificance",
        "classification_baseline",
        "clinical_significance_baseline",
    ],
    "review_status": [
        "review_status",
        "ReviewStatus",
        "review_status_baseline",
        "baseline_review_status",
    ],
    "submitter_count": [
        "submitter_count",
        "NumberSubmitters",
        "submitter_count_baseline",
        "baseline_submitter_count",
    ],
    "baseline_environment": [
        "baseline_environment",
        "environment_baseline",
        "condition_environment_baseline",
    ],
    "cab_portability_regime": [
        "cab_portability_regime",
        "baseline_regime_primary",
        "primary_regime",
        "baseline_architecture_family",
    ],
    "cab_portability_score": [
        "cab_portability_score",
        "baseline_portability_score",
        "CPI_baseline_only",
        "CPI",
    ],
    "baseline_regime_primary": [
        "baseline_regime_primary",
        "primary_regime",
        "cab_portability_regime",
    ],
    "baseline_architecture_family": [
        "baseline_architecture_family",
        "causal_architecture_category",
        "causal_architecture",
    ],
    "future_condition_label_drift": [
        "future_condition_label_drift",
        "condition_label_drift",
        "future_condition_drift",
        "condition_label_change",
    ],
    "future_classification_change": [
        "future_classification_change",
        "classification_change",
        "future_classification_severity_drift",
    ],
    "future_classification_severity_drift": [
        "future_classification_severity_drift",
        "classification_severity_drift",
        "future_classification_change",
    ],
    "future_cross_environment_drift": [
        "future_cross_environment_drift",
        "cross_environment_drift",
    ],
    "future_any_meaning_drift": [
        "future_any_meaning_drift",
        "any_meaning_drift",
        "future_condition_label_drift",
    ],
    "semantic_drift_without_reclassification": [
        "semantic_drift_without_reclassification",
        "future_semantic_drift_without_reclassification",
    ],
    "review_status_change": [
        "review_status_change",
        "future_review_status_change",
    ],
    "submitter_count_change": [
        "submitter_count_change",
        "future_submitter_count_change",
    ],
}


def ensure_dirs() -> None:
    for path in [
        ROOT / "benchmark",
        ROOT / "reports" / "tables",
        ROOT / "reports" / "workflow_simulation",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def first_existing(df: pd.DataFrame, aliases: List[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def pick(df: pd.DataFrame, target: str, default: Any = "") -> pd.Series:
    src = first_existing(df, COLUMN_ALIASES.get(target, [target]))
    if src is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[src]


def bool_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "t"}).fillna(False)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_snapshot(df: pd.DataFrame, candidates: List[str], fallback: str) -> str:
    for c in candidates:
        if c in df.columns:
            vals = [str(v) for v in df[c].dropna().unique() if str(v).strip()]
            if vals:
                return vals[0]
    return fallback


def make_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in BASELINE_COLS:
        out[col] = pick(df, col, "")

    out["assertion_id"] = out["assertion_id"].astype(str)
    out["domain"] = out["domain"].astype(str)
    out["variation_id"] = out["variation_id"].fillna(out["assertion_id"]).astype(str)
    out["gene"] = out["gene"].fillna("").astype(str)
    out["input_condition_label"] = out["input_condition_label"].fillna("").astype(str)
    out["classification"] = out["classification"].fillna("Pathogenic/Likely pathogenic").astype(str)
    out["review_status"] = out["review_status"].fillna("").astype(str)
    out["submitter_count"] = pd.to_numeric(out["submitter_count"], errors="coerce").fillna(0).astype(int)
    out["cab_portability_score"] = pd.to_numeric(out["cab_portability_score"], errors="coerce").fillna(60.0)
    out["cab_portability_regime"] = out["cab_portability_regime"].fillna("unresolved").astype(str)
    out["baseline_environment"] = out["baseline_environment"].fillna("").astype(str)
    return out


def make_endpoint_table(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["assertion_id"] = baseline["assertion_id"].astype(str)
    out["domain"] = baseline["domain"].astype(str)

    for col in ENDPOINT_COLS:
        if col in {"assertion_id", "domain"}:
            continue
        out[col] = bool_series(pick(df, col, False)).astype(int)

    if out["future_any_meaning_drift"].sum() == 0:
        out["future_any_meaning_drift"] = (
            (out["future_condition_label_drift"] == 1)
            | (out["future_classification_change"] == 1)
            | (out["future_cross_environment_drift"] == 1)
        ).astype(int)

    if out["semantic_drift_without_reclassification"].sum() == 0:
        out["semantic_drift_without_reclassification"] = (
            (out["future_condition_label_drift"] == 1)
            & (out["future_classification_change"] == 0)
        ).astype(int)

    return out


def make_followup_table(baseline: pd.DataFrame, endpoints: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    follow = baseline.copy()

    explicit_condition = first_existing(original, [
        "followup_condition_label",
        "condition_label_followup",
        "PhenotypeList_followup",
        "followup_phenotype_list",
    ])
    explicit_classification = first_existing(original, [
        "followup_classification",
        "classification_followup",
        "ClinicalSignificance_followup",
    ])

    if explicit_condition:
        follow["input_condition_label"] = original[explicit_condition].fillna(follow["input_condition_label"]).astype(str)
    else:
        drift_mask = endpoints["future_condition_label_drift"].astype(int) == 1
        follow.loc[drift_mask, "input_condition_label"] = (
            follow.loc[drift_mask, "input_condition_label"].astype(str)
            + " / follow-up semantic drift"
        )

    if explicit_classification:
        follow["classification"] = original[explicit_classification].fillna(follow["classification"]).astype(str)
    else:
        class_mask = (
            (endpoints["future_classification_change"].astype(int) == 1)
            | (endpoints["future_classification_severity_drift"].astype(int) == 1)
        )
        follow.loc[class_mask, "classification"] = "changed_at_followup"

    follow["benchmark_followup_source"] = (
        "explicit_followup_columns" if explicit_condition or explicit_classification
        else "synthetic_followup_for_endpoint_replay"
    )
    follow["benchmark_note"] = (
        "Follow-up replay file is used only for retrospective-prospective benchmark endpoint reconstruction; "
        "baseline routing remains baseline-only."
    )
    return follow


def write_domain_configs(domain_dir: Path, domain: str) -> None:
    config_text = f"""domain: {domain}
not_clinical_use: true
baseline_only_predictors:
  - gene
  - input_condition_label
  - classification
  - review_status
  - submitter_count
  - baseline_environment
  - cab_portability_regime
  - cab_portability_score
routing_modes:
  - ClinVar-label-only
  - CAB-Strict
  - CAB-Balanced
limitations:
  - CAB is not a diagnostic tool.
  - CAB does not reclassify variants.
  - CAB does not replace ACMG/AMP interpretation.
  - External expert adjudication remains pending.
"""
    for name in ["environment_mapping.yaml", "baseline_regime_rules.yaml"]:
        p = domain_dir / name
        if not p.exists():
            p.write_text(config_text, encoding="utf-8")


def expected_metrics(domain: str, endpoints: pd.DataFrame, baseline_snapshot: str, followup_snapshot: str) -> Dict[str, Any]:
    n = len(endpoints)

    def rate(col: str) -> float:
        if n == 0 or col not in endpoints:
            return 0.0
        return float(pd.to_numeric(endpoints[col], errors="coerce").fillna(0).mean())

    return {
        "domain": domain,
        "aligned_N": int(n),
        "baseline_snapshot": baseline_snapshot,
        "followup_snapshot": followup_snapshot,
        "condition_label_change_rate": rate("future_condition_label_drift"),
        "cross_environment_drift_rate": rate("future_cross_environment_drift"),
        "any_meaning_drift_rate": rate("future_any_meaning_drift"),
        "classification_change_rate": rate("future_classification_change"),
        "semantic_drift_without_reclassification_rate": rate("semantic_drift_without_reclassification"),
        "notes": (
            "Full export materialized from data/processed/cab_decision_challenge_tasks.csv. "
            "Baseline assertions are baseline-only. Temporal endpoints are stored separately."
        ),
    }


def main() -> None:
    ensure_dirs()

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    df = pd.read_csv(SOURCE, low_memory=False)
    if "domain" not in df.columns:
        raise ValueError("Source table must contain a domain column.")

    baseline_snapshot = infer_snapshot(df, ["baseline_snapshot", "baseline_date", "baseline_release"], "2023-01")
    followup_snapshot = infer_snapshot(df, ["followup_snapshot", "followup_date", "followup_release"], "2026-04")

    index_rows: List[Dict[str, Any]] = []
    report_lines = [
        "# Full Three-Domain Benchmark Export Materialization",
        "",
        f"Source: `{SOURCE.relative_to(ROOT)}`",
        "",
        "Baseline routing inputs are baseline-only. Future endpoint labels are written only to `temporal_endpoints.csv`.",
        "",
    ]

    for domain in DOMAINS:
        sub = df[df["domain"].astype(str).eq(domain)].copy()
        domain_dir = ROOT / "benchmark" / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        if sub.empty:
            report_lines.append(f"- `{domain}`: no rows found.")
            continue

        baseline = make_baseline_table(sub)
        endpoints = make_endpoint_table(sub, baseline)
        followup = make_followup_table(baseline, endpoints, sub)

        baseline.to_csv(domain_dir / "baseline_assertions.csv", index=False)
        followup.to_csv(domain_dir / "followup_assertions.csv", index=False)
        endpoints.to_csv(domain_dir / "temporal_endpoints.csv", index=False)

        metrics = expected_metrics(domain, endpoints, baseline_snapshot, followup_snapshot)
        write_json(domain_dir / "expected_metrics.json", metrics)
        write_domain_configs(domain_dir, domain)

        index_rows.append({
            "domain": domain,
            "aligned_N": metrics["aligned_N"],
            "baseline_snapshot": baseline_snapshot,
            "followup_snapshot": followup_snapshot,
            "condition_label_change_rate": metrics["condition_label_change_rate"],
            "cross_environment_drift_rate": metrics["cross_environment_drift_rate"],
            "any_meaning_drift_rate": metrics["any_meaning_drift_rate"],
            "classification_change_rate": metrics["classification_change_rate"],
            "notes": "Full three-domain export materialized; baseline-only routing preserved.",
        })

        report_lines.append(
            f"- `{domain}`: N={metrics['aligned_N']:,}; "
            f"condition drift={metrics['condition_label_change_rate']:.4f}; "
            f"cross-environment={metrics['cross_environment_drift_rate']:.4f}; "
            f"any meaning={metrics['any_meaning_drift_rate']:.4f}; "
            f"classification change={metrics['classification_change_rate']:.4f}."
        )

    index = pd.DataFrame(index_rows)
    index.to_csv(ROOT / "reports" / "tables" / "cab_benchmark_index.csv", index=False)

    report_lines.extend([
        "",
        "## Non-clinical limitation",
        "",
        "CAB is a research reference implementation for assertion portability and routing simulation. It is not a diagnostic tool, does not reclassify variants, and does not replace ACMG/AMP interpretation or expert curation.",
        "",
        "## CLI replay",
        "",
        "Run the benchmark command per domain:",
        "",
        "```bash",
        "python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia",
        "python -m cab_portability.cli benchmark --baseline benchmark/cardiomyopathy/baseline_assertions.csv --followup benchmark/cardiomyopathy/followup_assertions.csv --domain cardiomyopathy --output-dir reports/workflow_simulation/cardiomyopathy",
        "python -m cab_portability.cli benchmark --baseline benchmark/hereditary_cancer/baseline_assertions.csv --followup benchmark/hereditary_cancer/followup_assertions.csv --domain hereditary_cancer --output-dir reports/workflow_simulation/hereditary_cancer",
        "```",
    ])
    (ROOT / "reports" / "workflow_simulation" / "full_exports_materialization_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("Full three-domain benchmark exports materialized.")
    print(index.to_string(index=False))
    print()
    print("Wrote:")
    for domain in DOMAINS:
        print(f"  benchmark/{domain}/baseline_assertions.csv")
        print(f"  benchmark/{domain}/followup_assertions.csv")
        print(f"  benchmark/{domain}/temporal_endpoints.csv")
        print(f"  benchmark/{domain}/expected_metrics.json")
    print("  reports/tables/cab_benchmark_index.csv")
    print("  reports/workflow_simulation/full_exports_materialization_report.md")


if __name__ == "__main__":
    main()
