
#!/usr/bin/env python3
"""Sync full benchmark exports with final CAB operating-frontier decision columns.

Problem
-------
CLI routing has been patched to use final decision columns, but the benchmark
baseline_assertions.csv exports may not contain those columns. If missing, CLI
falls back to heuristics, which is not the final manuscript replay.

This script injects row-level final decision columns into:
- benchmark/inherited_arrhythmia/baseline_assertions.csv
- benchmark/cardiomyopathy/baseline_assertions.csv
- benchmark/hereditary_cancer/baseline_assertions.csv

Preferred source:
- data/processed/cab_routing_operating_modes_final.csv

Fallback sources:
- data/processed/cab_decision_challenge_tasks.csv

Added/updated columns:
- cab_strict_direct_use_allowed
- cab_balanced_direct_use_allowed
- direct_single_model_reuse_allowed
- contextual_repair_required
- disease_specific_expert_review_required
- population_or_penetrance_review_required
- final_decision_column_source

Outputs:
- reports/workflow_simulation/cli_final_decision_column_sync_report.md
- reports/tables/cli_final_decision_column_sync_qc.csv

CAB is research software only. Not diagnostic, not clinical deployment, not
variant reclassification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path.cwd()
DOMAINS = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]

MODES_FILE = ROOT / "data" / "processed" / "cab_routing_operating_modes_final.csv"
TASKS_FILE = ROOT / "data" / "processed" / "cab_decision_challenge_tasks.csv"

REPORT = ROOT / "reports" / "workflow_simulation" / "cli_final_decision_column_sync_report.md"
QC = ROOT / "reports" / "tables" / "cli_final_decision_column_sync_qc.csv"


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def boolish(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "t"}).fillna(False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def build_mode_decisions() -> pd.DataFrame:
    """Return assertion_id, cab_strict_direct_use_allowed, cab_balanced_direct_use_allowed."""
    if MODES_FILE.exists():
        modes = read_csv(MODES_FILE)
        needed = {"assertion_id", "cab_mode", "direct_use_allowed"}
        if needed.issubset(set(modes.columns)):
            m = modes[["assertion_id", "cab_mode", "direct_use_allowed"]].copy()
            m["assertion_id"] = norm_id(m["assertion_id"])
            m["direct_use_allowed"] = boolish(m["direct_use_allowed"])
            piv = (
                m[m["cab_mode"].isin(["CAB-Strict", "CAB-Balanced"])]
                .pivot_table(
                    index="assertion_id",
                    columns="cab_mode",
                    values="direct_use_allowed",
                    aggfunc="first",
                )
                .reset_index()
            )
            if "CAB-Strict" in piv.columns:
                piv["cab_strict_direct_use_allowed"] = piv["CAB-Strict"]
            if "CAB-Balanced" in piv.columns:
                piv["cab_balanced_direct_use_allowed"] = piv["CAB-Balanced"]
                piv["direct_single_model_reuse_allowed"] = piv["CAB-Balanced"]
            for c in ["CAB-Strict", "CAB-Balanced"]:
                if c in piv.columns:
                    piv = piv.drop(columns=[c])
            piv["final_decision_column_source"] = "cab_routing_operating_modes_final.csv"
            return piv

    # Fallback to tasks file.
    tasks = read_csv(TASKS_FILE)
    out = pd.DataFrame()
    out["assertion_id"] = norm_id(tasks["assertion_id"] if "assertion_id" in tasks.columns else tasks["VariationID"])
    if "direct_single_model_reuse_allowed" in tasks.columns:
        out["cab_balanced_direct_use_allowed"] = boolish(tasks["direct_single_model_reuse_allowed"])
        out["direct_single_model_reuse_allowed"] = out["cab_balanced_direct_use_allowed"]
    else:
        out["cab_balanced_direct_use_allowed"] = False
        out["direct_single_model_reuse_allowed"] = False

    # Strict fallback: if no row-level mode file, derive same way CLI would.
    high_risk = {
        "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
        "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC",
    }
    failure_tokens = [
        "collision", "underresolved", "nonspecific", "penetrance", "spectrum",
        "moderate", "nonportable", "low", "recessive", "biallelic", "overlap",
    ]
    gene = tasks["gene"].astype(str).str.upper() if "gene" in tasks.columns else pd.Series([""] * len(tasks))
    reg = tasks["baseline_regime_primary"].astype(str).str.lower() if "baseline_regime_primary" in tasks.columns else pd.Series([""] * len(tasks))
    arch = tasks["baseline_architecture_family"].astype(str).str.lower() if "baseline_architecture_family" in tasks.columns else pd.Series([""] * len(tasks))
    fail = reg.str.contains("|".join(failure_tokens), regex=True, na=False) | arch.str.contains("|".join(failure_tokens), regex=True, na=False)
    out["cab_strict_direct_use_allowed"] = ~(gene.isin(high_risk) | fail)
    out["final_decision_column_source"] = "cab_decision_challenge_tasks.csv_fallback"
    return out


def build_task_action_flags() -> pd.DataFrame:
    if not TASKS_FILE.exists():
        return pd.DataFrame(columns=[
            "assertion_id",
            "contextual_repair_required",
            "disease_specific_expert_review_required",
            "population_or_penetrance_review_required",
        ])

    tasks = read_csv(TASKS_FILE)
    id_col = "assertion_id" if "assertion_id" in tasks.columns else "VariationID"
    out = pd.DataFrame()
    out["assertion_id"] = norm_id(tasks[id_col])
    for c in [
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
    ]:
        if c in tasks.columns:
            out[c] = boolish(tasks[c])
        else:
            out[c] = False
    return out


def sync_domain(domain: str, decisions: pd.DataFrame, flags: pd.DataFrame) -> Dict[str, object]:
    path = ROOT / "benchmark" / domain / "baseline_assertions.csv"
    if not path.exists():
        return {
            "domain": domain,
            "status": "missing_baseline_assertions",
            "rows": 0,
            "matched_decision_rows": 0,
            "balanced_source_direct_column_rate": 0,
            "strict_source_direct_column_rate": 0,
        }

    base = read_csv(path)
    if "assertion_id" not in base.columns:
        raise ValueError(f"{path} missing assertion_id")

    base["assertion_id"] = norm_id(base["assertion_id"])

    # Drop old columns before merge to avoid _x/_y trash, the classic spreadsheet swamp.
    inject_cols = [
        "cab_strict_direct_use_allowed",
        "cab_balanced_direct_use_allowed",
        "direct_single_model_reuse_allowed",
        "final_decision_column_source",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
    ]
    base = base.drop(columns=[c for c in inject_cols if c in base.columns], errors="ignore")

    merged = base.merge(decisions, on="assertion_id", how="left")
    merged = merged.merge(flags, on="assertion_id", how="left")

    for c in [
        "cab_strict_direct_use_allowed",
        "cab_balanced_direct_use_allowed",
        "direct_single_model_reuse_allowed",
        "contextual_repair_required",
        "disease_specific_expert_review_required",
        "population_or_penetrance_review_required",
    ]:
        if c not in merged.columns:
            merged[c] = False
        merged[c] = merged[c].fillna(False).astype(bool)

    merged["final_decision_column_source"] = merged["final_decision_column_source"].fillna("missing_final_decision_match")

    matched = int((merged["final_decision_column_source"] != "missing_final_decision_match").sum())
    merged.to_csv(path, index=False)

    return {
        "domain": domain,
        "status": "synced",
        "rows": len(merged),
        "matched_decision_rows": matched,
        "matched_decision_rate": matched / len(merged) if len(merged) else 0,
        "balanced_direct_use_rate": float(merged["direct_single_model_reuse_allowed"].mean()) if len(merged) else 0,
        "strict_direct_use_rate": float(merged["cab_strict_direct_use_allowed"].mean()) if len(merged) else 0,
        "balanced_source_ready_for_cli": bool((merged["final_decision_column_source"] != "missing_final_decision_match").all()),
    }


def main() -> None:
    (ROOT / "reports" / "workflow_simulation").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)

    decisions = build_mode_decisions()
    flags = build_task_action_flags()

    rows: List[Dict[str, object]] = []
    for domain in DOMAINS:
        rows.append(sync_domain(domain, decisions, flags))

    qc = pd.DataFrame(rows)
    qc.to_csv(QC, index=False)

    lines = [
        "# CLI Final Decision Column Sync Report",
        "",
        "Benchmark baseline exports have been synced with final operating-frontier decision columns.",
        "",
        "## Decision columns injected",
        "",
        "- `cab_strict_direct_use_allowed`",
        "- `cab_balanced_direct_use_allowed`",
        "- `direct_single_model_reuse_allowed`",
        "- `contextual_repair_required`",
        "- `disease_specific_expert_review_required`",
        "- `population_or_penetrance_review_required`",
        "- `final_decision_column_source`",
        "",
        "## Source priority",
        "",
        "1. `data/processed/cab_routing_operating_modes_final.csv`",
        "2. fallback: `data/processed/cab_decision_challenge_tasks.csv`",
        "",
        "## QC",
        "",
        qc.to_string(index=False),
        "",
        "## Leakage boundary",
        "",
        "Injected columns are baseline-time routing decisions used by the final operating-frontier benchmark. Future endpoint labels remain only in `temporal_endpoints.csv`.",
        "",
        "## Limitation",
        "",
        "This is a research workflow simulation. CAB is not diagnostic, not clinical deployment, and not variant reclassification.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Synced benchmark baseline exports to final decision columns.")
    print(qc.to_string(index=False))
    print()
    print(f"Wrote: {QC.relative_to(ROOT)}")
    print(f"Wrote: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
