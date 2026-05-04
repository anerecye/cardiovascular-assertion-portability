#!/usr/bin/env python3
"""Sanitize final CAB operating-frontier outputs.

Final public/output layer must not contain old operating-mode names
(CAB-Core / CAB-Conservative) except in:
- reports/tables/cab_operating_mode_name_crosswalk.csv
- reports/tables/quarantined_operating_mode_wording.csv
- reports/deprecated/
"""

from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"
FIGURES = REPORTS / "figures"
DEPRECATED = REPORTS / "deprecated"

FINAL_TABLES_TO_SANITIZE = [
    TABLES / "routing_metrics_all_modes_all_endpoints.csv",
    TABLES / "routing_metrics_by_domain_all_modes.csv",
    TABLES / "routing_pareto_frontier_by_endpoint.csv",
    TABLES / "routing_operating_modes_bootstrap_ci.csv",
    DATA / "cab_routing_operating_modes_final.csv",
]

DEPRECATED_FILES = [
    TABLES / "cab_core_vs_conservative_bootstrap_ci.csv",
    TABLES / "cab_core_vs_conservative_by_domain.csv",
    TABLES / "cab_core_vs_conservative_routing_metrics.csv",
    TABLES / "domain_specific_cab_mode_recommendations.csv",
    TABLES / "routing_publication_safe_claims_operating_modes.csv",
    TABLES / "routing_pareto_frontier.csv",
    QC / "cab_operating_modes_definition.md",
    FIGURES / "cab_core_vs_conservative_tradeoff.svg",
    FIGURES / "routing_pareto_frontier.svg",
    FIGURES / "final_cab_operating_modes_figure.svg",
]

OUT_FINAL_DEF = QC / "cab_operating_modes_final_definition.md"
OUT_READY = REPORTS / "final_cab_readiness_report.md"
OUT_DEPRECATED_INDEX = TABLES / "deprecated_operating_mode_outputs_quarantine.csv"


def ensure_dirs():
    for p in [DATA, REPORTS, TABLES, QC, FIGURES, DEPRECATED]:
        p.mkdir(parents=True, exist_ok=True)


def sanitize_csv(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path, low_memory=False)
    changed = False

    if "old_mode_name" in df.columns:
        df = df.drop(columns=["old_mode_name"])
        changed = True

    for c in df.columns:
        if df[c].dtype == object:
            before = df[c].astype(str)
            after = before.str.replace("CAB-Core", "CAB-Strict", regex=False).str.replace("CAB-Conservative", "CAB-Balanced", regex=False)
            if not after.equals(before):
                df[c] = after
                changed = True

    if changed:
        df.to_csv(path, index=False)
    return changed


def write_final_definition():
    lines = [
        "# CAB Operating Modes Final Definition",
        "",
        "Technical definitions; not manuscript prose.",
        "",
        "## CAB-Strict",
        "- features: gene + baseline disease-model regime",
        "- behavior: high-stringency triage",
        "- goal: minimize false portability / unsupported deterministic reuse",
        "- limitation: high overrestriction and low direct-use allowance",
        "",
        "## CAB-Balanced",
        "- features: full CAB routing configuration",
        "- behavior: balanced safety-permissiveness routing",
        "- goal: retain large reduction in unsupported reuse while allowing more direct deterministic use",
        "- limitation: higher unsupported reuse than CAB-Strict but less overrestriction",
        "",
        "## ClinVar-label-only",
        "- P/LP treated as portable direct-use by default",
        "- behavior: maximal permissiveness",
        "- limitation: high unsupported deterministic reuse under drift endpoints",
        "",
        "## Traceability",
        "Historical operating-mode names are retained only in the crosswalk and quarantine tables.",
        "",
        "## Operating-frontier rule",
        "CAB is an operating-frontier framework, not a single universal classifier.",
        "",
        "## Non-negotiable reporting rules",
        "- Do not hide that CAB-Strict overrestricts.",
        "- Do not hide that CAB-Balanced allows more direct use but has higher unsupported reuse.",
        "- Do not present one mode as universally optimal.",
        "- Do not claim external decision validation.",
        "- Do not use historical operating-mode names in headline tables or reports.",
    ]
    OUT_FINAL_DEF.write_text("\n".join(lines), encoding="utf-8")


def sanitize_readiness_report():
    if not OUT_READY.exists():
        return
    s = OUT_READY.read_text(encoding="utf-8", errors="ignore")
    s = s.replace("CAB-Core", "CAB-Strict").replace("CAB-Conservative", "CAB-Balanced")
    note = "\n\n## Historical naming quarantine\nHistorical operating-mode names are retained only in `reports/tables/cab_operating_mode_name_crosswalk.csv` and `reports/tables/quarantined_operating_mode_wording.csv`.\n"
    if "Historical naming quarantine" not in s:
        s += note
    OUT_READY.write_text(s, encoding="utf-8")


def update_quarantine_wording():
    q = TABLES / "quarantined_operating_mode_wording.csv"
    rows = []
    if q.exists():
        try:
            rows = pd.read_csv(q).to_dict("records")
        except Exception:
            rows = []
    extra = [
        {
            "quarantined_wording": "CAB-Core",
            "reason": "historical operating-mode name; final headline outputs use CAB-Strict",
            "replace_with": "CAB-Strict",
            "allowed_context": "cab_operating_mode_name_crosswalk.csv only",
        },
        {
            "quarantined_wording": "CAB-Conservative",
            "reason": "historical operating-mode name; final headline outputs use CAB-Balanced",
            "replace_with": "CAB-Balanced",
            "allowed_context": "cab_operating_mode_name_crosswalk.csv only",
        },
    ]
    combined = rows + extra
    seen = set()
    clean = []
    for r in combined:
        key = (r.get("quarantined_wording"), r.get("replace_with"))
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
    pd.DataFrame(clean).to_csv(q, index=False)


def quarantine_old_files():
    rows = []
    for p in DEPRECATED_FILES:
        if not p.exists():
            rows.append({
                "original_path": str(p.relative_to(BASE)),
                "quarantined_path": "",
                "status": "not_found",
                "reason": "old operating-mode naming output; not part of final headline layer",
            })
            continue

        dest_dir = DEPRECATED / p.parent.relative_to(BASE)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(p), str(dest))

        rows.append({
            "original_path": str(p.relative_to(BASE)),
            "quarantined_path": str(dest.relative_to(BASE)),
            "status": "moved_to_deprecated",
            "reason": "old operating-mode naming output; final names are CAB-Strict and CAB-Balanced",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DEPRECATED_INDEX, index=False)
    return out


def main():
    ensure_dirs()
    changed = []
    for p in FINAL_TABLES_TO_SANITIZE:
        if sanitize_csv(p):
            changed.append(str(p.relative_to(BASE)))

    write_final_definition()
    sanitize_readiness_report()
    update_quarantine_wording()
    dep = quarantine_old_files()

    print("Final operating-frontier sanitation complete.")
    print()
    print("Sanitized final tables:")
    if changed:
        for x in changed:
            print(f"  - {x}")
    else:
        print("  - none needed")
    print()
    print("Deprecated outputs quarantine index:")
    print(dep.to_string(index=False))
    print()
    print("Allowed old-name locations only:")
    print("  - reports/tables/cab_operating_mode_name_crosswalk.csv")
    print("  - reports/tables/quarantined_operating_mode_wording.csv")
    print("  - reports/deprecated/")


if __name__ == "__main__":
    main()
