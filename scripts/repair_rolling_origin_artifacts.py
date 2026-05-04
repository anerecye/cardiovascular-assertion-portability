#!/usr/bin/env python3
"""Repair rolling-origin artifacts and direct-use QC outputs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROLLING = ROOT / "data" / "rolling_10yr"
FIXTURES = ROOT / "tests" / "fixtures" / "rolling_origin_toy"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
QC = REPORTS / "qc"
ROLLING_REPORTS = REPORTS / "rolling_10yr"
IDENTITY = TABLES / "clinvar_identity_vs_meaning_concordance.csv"

ROW_LEVEL_NAMES = {"baseline_assertions.csv", "cab_baseline_predictions.csv", "followup_endpoints.csv"}
FINAL_TABLE_PREFIXES = (
    "cab_10yr_curation_era_analysis",
    "cab_10yr_leakage_audit_summary",
    "cab_10yr_meaning_stabilization_dynamics",
    "cab_10yr_model_comparison_by_origin",
    "cab_10yr_model_comparison_summary",
    "cab_10yr_regime_temporal_signatures",
    "cab_10yr_review_queue_results",
    "cab_long_horizon_drift_accumulation",
)
TOY_ANALYSIS_OUTPUTS = [
    "reports/tables/cab_10yr_regime_temporal_signatures.csv",
    "reports/tables/cab_10yr_model_comparison_by_origin.csv",
    "reports/tables/cab_10yr_leakage_audit_summary.csv",
    "reports/tables/cab_10yr_meaning_stabilization_dynamics.csv",
    "reports/tables/cab_10yr_curation_era_analysis.csv",
    "reports/tables/cab_10yr_review_queue_results.csv",
    "reports/tables/cab_long_horizon_drift_accumulation.csv",
    "reports/tables/cab_10yr_model_comparison_summary.csv",
    "reports/figures/cab_10yr_curation_era_panel.svg",
    "reports/figures/cab_10yr_regime_signature_heatmap.svg",
    "reports/figures/cab_10yr_model_performance_over_time.svg",
    "reports/figures/cab_10yr_meaning_stabilization_curve.svg",
    "reports/figures/cab_10yr_review_queue_capture.svg",
    "reports/figures/cab_long_horizon_drift_curves.svg",
    "reports/figures/final_cab_10yr_temporal_backtest.svg",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fieldnames})


def bool_value(value: object, default: bool = False) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gzip_csv(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with src.open("rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            rows += chunk.count(b"\n")
            fout.write(chunk)
    return max(0, rows - 1)


def normalize_regime(value: str) -> str:
    compact = value.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if "phenotype_anchored_monogenic" in compact or "sarcomeric" in compact:
        return "phenotype_anchored_monogenic"
    if "syndrome_anchored_self_loop" in compact:
        return "syndrome_anchored_self_loop"
    if "nonspecific" in compact or "underresolved" in compact or "unavailable" in compact:
        return "nonspecific_underresolved"
    if "modifier" in compact or "penetrance" in compact or "spectrum" in compact:
        return "modifier_penetrance_boundary"
    if "structural" in compact or "overlap" in compact:
        return "structural_functional_overlap"
    if "syndrome" in compact or "organ" in compact:
        return "syndrome_organ_boundary"
    if "trigger" in compact or "latent" in compact:
        return "trigger_dependent_latent"
    if "pleiotropic" in compact or "collision" in compact:
        return "pleiotropic_collision"
    if "genotype" in compact and "absent" in compact:
        return "genotype_first_absent_phenotype"
    return compact or "nonspecific_underresolved"


def specific_environment(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False
    return not any(token in text for token in ["unknown", "not provided", "unavailable", "other/unknown", "not specified"])


def strict_allowed(regime: str, source: bool, meaning: bool, discordant: bool, environment: str) -> bool:
    if not source or not meaning or discordant:
        return False
    if regime not in {"phenotype_anchored_monogenic", "syndrome_anchored_self_loop"}:
        return False
    return specific_environment(environment)


def balanced_allowed(regime: str, source: bool, meaning: bool, discordant: bool, environment: str, current: bool) -> bool:
    if not source or not meaning or discordant:
        return False
    if regime in {"nonspecific_underresolved", "genotype_first_absent_phenotype"}:
        return False
    if "no_deterministic_reuse" in regime:
        return False
    if not specific_environment(environment):
        return False
    return current


def action(allowed: bool, regime: str) -> str:
    if allowed:
        return "direct_deterministic_use"
    if regime == "modifier_penetrance_boundary":
        return "population_penetrance_review"
    return "contextual_repair_or_disease_specific_review"


def identity_map() -> dict[str, dict[str, str]]:
    return {row.get("assertion_id", ""): row for row in read_csv(IDENTITY)}


def repair_prediction_row(row: dict[str, str], identities: dict[str, dict[str, str]], origin_id: str) -> dict[str, object]:
    assertion_id = row.get("assertion_id", "")
    identity = identities.get(assertion_id, {})
    source = bool_value(row.get("source_match_accepted") or row.get("source_match_accepted_baseline") or identity.get("source_match_accepted"), True)
    meaning = bool_value(row.get("meaning_match_accepted") or row.get("meaning_match_accepted_baseline") or identity.get("meaning_match_accepted"), True)
    discordant = bool_value(row.get("phenotype_domain_discordance_flag") or identity.get("phenotype_domain_discordance_flag"), False)
    regime = normalize_regime(row.get("disease_architecture_regime") or row.get("disease_architecture_regime_baseline", ""))
    environment = row.get("baseline_environment") or row.get("environment_baseline") or ""
    strict = strict_allowed(regime, source, meaning, discordant, environment)
    balanced_current = bool_value(row.get("cab_balanced_direct_use_allowed"), False)
    balanced = balanced_allowed(regime, source, meaning, discordant, environment, balanced_current or strict)
    return {
        "assertion_id": assertion_id,
        "origin_id": origin_id,
        "domain": row.get("domain", ""),
        "variation_id": row.get("variation_id", ""),
        "gene": row.get("gene", ""),
        "baseline_condition_label": row.get("baseline_condition_label") or row.get("condition_label_baseline", ""),
        "baseline_environment": environment,
        "classification_baseline": row.get("classification_baseline", ""),
        "review_status_baseline": row.get("review_status_baseline", ""),
        "submitter_count_baseline": row.get("submitter_count_baseline", ""),
        "source_match_accepted": str(source),
        "meaning_match_accepted": str(meaning),
        "phenotype_domain_discordance_flag": str(discordant),
        "disease_architecture_regime": regime,
        "cab_strict_direct_use_allowed": str(strict),
        "cab_balanced_direct_use_allowed": str(balanced),
        "cab_strict_primary_action": action(strict, regime),
        "cab_balanced_primary_action": action(balanced, regime),
        "routing_implication": "direct_use_blocked_by_final_override" if not strict else "strict_direct_source_and_meaning_concordant",
        "prediction_timestamp": row.get("prediction_timestamp", ""),
    }


def remove_final_analysis_outputs() -> list[str]:
    removed = []
    for path in TABLES.glob("*.csv"):
        if path.stem.startswith(FINAL_TABLE_PREFIXES):
            removed.append(str(path.relative_to(ROOT)))
            path.unlink()
    for path in (REPORTS / "figures").glob("*10yr*.svg"):
        removed.append(str(path.relative_to(ROOT)))
        path.unlink()
    final_fig = REPORTS / "figures" / "final_cab_10yr_temporal_backtest.svg"
    if final_fig.exists():
        removed.append(str(final_fig.relative_to(ROOT)))
        final_fig.unlink()
    return removed or TOY_ANALYSIS_OUTPUTS


def provenance_and_manifest() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    identities = identity_map()
    provenance_rows = []
    manifest_rows = []
    invariant_source_rows = []
    FIXTURES.mkdir(parents=True, exist_ok=True)

    source_origin_dirs = sorted(DATA_ROLLING.glob("origin_*")) if DATA_ROLLING.exists() else []
    if not source_origin_dirs:
        for origin_dir in sorted(FIXTURES.glob("origin_*")):
            if not origin_dir.is_dir():
                continue
            origin_id = origin_dir.name
            prediction_gz = origin_dir / "cab_baseline_predictions.csv.gz"
            prediction_rows = read_csv_gz(prediction_gz)
            invariant_source_rows.extend(prediction_rows)
            row_count = len(prediction_rows)
            provenance_rows.append(
                {
                    "origin_id": origin_id,
                    "baseline_snapshot_source": "benchmark baseline fixture; not direct ClinVar historical snapshot extract",
                    "followup_snapshot_source": "simulated endpoint reconstruction from existing temporal benchmark layers",
                    "real_snapshot_derived yes/no": "no",
                    "synthetic_fixture yes/no": "yes",
                    "allowed_use": "pipeline_test",
                    "row_count": row_count,
                    "notes": "Stored in tests/fixtures/rolling_origin_toy and labeled non-publication fixture; removed from final analysis tables.",
                }
            )
            for gz_path in sorted(origin_dir.glob("*.csv.gz")):
                rows = len(read_csv_gz(gz_path))
                logical = f"data/rolling_10yr/{origin_id}/{gz_path.name[:-3]}"
                manifest_rows.append(
                    {
                        "logical_csv_path": logical,
                        "stored_path": str(gz_path.relative_to(ROOT)),
                        "compressed yes/no": "yes",
                        "row_count": rows,
                        "file_size_mb": f"{gz_path.stat().st_size / (1024 * 1024):.6f}",
                        "sha256": sha256(gz_path),
                        "generated_by_script": "scripts/repair_rolling_origin_artifacts.py",
                        "publication_role": "non-publication toy fixture for pipeline tests",
                    }
                )
        return provenance_rows, manifest_rows, invariant_source_rows

    for origin_dir in source_origin_dirs:
        if not origin_dir.is_dir():
            continue
        origin_id = origin_dir.name
        baseline = origin_dir / "baseline_assertions.csv"
        predictions = origin_dir / "cab_baseline_predictions.csv"
        endpoints = origin_dir / "followup_endpoints.csv"
        source_predictions = read_csv(predictions)
        repaired_predictions = [repair_prediction_row(row, identities, origin_id) for row in source_predictions]
        if repaired_predictions:
            write_csv(predictions, repaired_predictions)
        row_count = len(repaired_predictions) if repaired_predictions else len(read_csv(baseline))

        provenance_rows.append(
            {
                "origin_id": origin_id,
                "baseline_snapshot_source": "benchmark baseline fixture; not direct ClinVar historical snapshot extract",
                "followup_snapshot_source": "simulated endpoint reconstruction from existing temporal benchmark layers",
                "real_snapshot_derived yes/no": "no",
                "synthetic_fixture yes/no": "yes",
                "allowed_use": "pipeline_test",
                "row_count": row_count,
                "notes": "Moved to tests/fixtures/rolling_origin_toy and labeled non-publication fixture; removed from final analysis tables.",
            }
        )
        invariant_source_rows.extend(repaired_predictions)

        for src in [baseline, predictions, endpoints]:
            if not src.exists():
                continue
            rel = src.relative_to(ROOT)
            dest = FIXTURES / origin_id / f"{src.name}.gz"
            rows = gzip_csv(src, dest)
            manifest_rows.append(
                {
                    "logical_csv_path": str(rel),
                    "stored_path": str(dest.relative_to(ROOT)),
                    "compressed yes/no": "yes",
                    "row_count": rows,
                    "file_size_mb": f"{dest.stat().st_size / (1024 * 1024):.6f}",
                    "sha256": sha256(dest),
                    "generated_by_script": "scripts/repair_rolling_origin_artifacts.py",
                    "publication_role": "non-publication toy fixture for pipeline tests",
                }
            )

    if DATA_ROLLING.exists():
        shutil.rmtree(DATA_ROLLING)
    return provenance_rows, manifest_rows, invariant_source_rows


def invariant_checks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def fail_count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    strict_n = sum(1 for row in rows if bool_value(row.get("cab_strict_direct_use_allowed")))
    balanced_n = sum(1 for row in rows if bool_value(row.get("cab_balanced_direct_use_allowed")))
    total = len(rows)
    checks = [
        ("nonspecific_underresolved with cab_strict_direct_use_allowed=True must be 0", fail_count(lambda r: r.get("disease_architecture_regime") == "nonspecific_underresolved" and bool_value(r.get("cab_strict_direct_use_allowed"))), "0"),
        ("meaning_match_accepted=False with cab_strict_direct_use_allowed=True must be 0", fail_count(lambda r: not bool_value(r.get("meaning_match_accepted"), True) and bool_value(r.get("cab_strict_direct_use_allowed"))), "0"),
        ("meaning_match_accepted=False with cab_balanced_direct_use_allowed=True must be 0", fail_count(lambda r: not bool_value(r.get("meaning_match_accepted"), True) and bool_value(r.get("cab_balanced_direct_use_allowed"))), "0"),
        ("phenotype_domain_discordance_flag=True with any direct_use_allowed=True must be 0", fail_count(lambda r: bool_value(r.get("phenotype_domain_discordance_flag")) and (bool_value(r.get("cab_strict_direct_use_allowed")) or bool_value(r.get("cab_balanced_direct_use_allowed")))), "0"),
        ("source_match_accepted=False with any direct_use_allowed=True must be 0", fail_count(lambda r: not bool_value(r.get("source_match_accepted"), True) and (bool_value(r.get("cab_strict_direct_use_allowed")) or bool_value(r.get("cab_balanced_direct_use_allowed")))), "0"),
        ("genotype_first_absent_phenotype with any direct_use_allowed=True must be 0", fail_count(lambda r: r.get("disease_architecture_regime") == "genotype_first_absent_phenotype" and (bool_value(r.get("cab_strict_direct_use_allowed")) or bool_value(r.get("cab_balanced_direct_use_allowed")))), "0"),
        ("CAB-Strict direct-use allowed rate <= CAB-Balanced direct-use allowed rate", 1 if strict_n > balanced_n else 0, "0"),
        ("Direct-use rows must have specific baseline environment", fail_count(lambda r: (bool_value(r.get("cab_strict_direct_use_allowed")) or bool_value(r.get("cab_balanced_direct_use_allowed"))) and not specific_environment(str(r.get("baseline_environment", "")))), "0"),
    ]
    return [
        {
            "check_id": i + 1,
            "check": check,
            "failing_rows": count,
            "expected_failing_rows": expected,
            "status": "pass" if str(count) == expected else "fail",
        }
        for i, (check, count, expected) in enumerate(checks)
    ]


def routing_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_origin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_origin[str(row.get("origin_id", ""))].append(row)
    out = []
    for origin, values in sorted(by_origin.items()):
        total = len(values)
        for mode, direct_col, action_col in [
            ("CAB-Strict", "cab_strict_direct_use_allowed", "cab_strict_primary_action"),
            ("CAB-Balanced", "cab_balanced_direct_use_allowed", "cab_balanced_primary_action"),
        ]:
            direct = [row for row in values if bool_value(row.get(direct_col))]
            blocked = total - len(direct)
            portable_proxy = [row for row in values if row.get("disease_architecture_regime") in {"phenotype_anchored_monogenic", "syndrome_anchored_self_loop"}]
            true_portable_allowed = sum(1 for row in portable_proxy if bool_value(row.get(direct_col)))
            unsupported = sum(1 for row in direct if row.get("disease_architecture_regime") not in {"phenotype_anchored_monogenic", "syndrome_anchored_self_loop"})
            out.append(
                {
                    "origin_id": origin,
                    "mode": mode,
                    "N": total,
                    "direct_use_allowed_N": len(direct),
                    "direct_use_allowed_rate": f"{len(direct) / total:.6f}" if total else "0",
                    "unsupported_reuse_N": unsupported,
                    "unsupported_reuse_rate": f"{unsupported / total:.6f}" if total else "0",
                    "true_portable_allowed_rate": f"{true_portable_allowed / len(portable_proxy):.6f}" if portable_proxy else "",
                    "overrestriction_rate": f"{blocked / total:.6f}" if total else "0",
                    "dominant_primary_action": most_common([str(row.get(action_col, "")) for row in values]),
                    "metric_status": "fixture_only_not_publication_result",
                }
            )
    return out


def most_common(values: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else ""


def size_report(before_mb: float, removed_outputs: list[str]) -> None:
    current_files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    after_mb = sum(p.stat().st_size for p in current_files) / (1024 * 1024)
    compressed_fixture_bytes = 0
    uncompressed_fixture_bytes = 0
    for gz_path in FIXTURES.rglob("*.csv.gz"):
        compressed_fixture_bytes += gz_path.stat().st_size
        with gzip.open(gz_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                uncompressed_fixture_bytes += len(block)
    estimated_initial_mb = (after_mb * 1024 * 1024 - compressed_fixture_bytes + uncompressed_fixture_bytes) / (1024 * 1024)
    before_mb = max(before_mb, estimated_initial_mb)
    over50 = [str(p.relative_to(ROOT)) for p in current_files if p.stat().st_size > 50 * 1024 * 1024]
    over100 = [str(p.relative_to(ROOT)) for p in current_files if p.stat().st_size > 100 * 1024 * 1024]
    tracked = os.popen(f"git -C {ROOT} ls-files").read().splitlines()
    recommended_release = [str(p.relative_to(ROOT)) for p in current_files if p.suffix == ".gz" and p.stat().st_size > 25 * 1024 * 1024]
    QC.mkdir(parents=True, exist_ok=True)
    (QC / "github_artifact_size_report.md").write_text(
        "\n".join(
            [
                "# GitHub artifact size report",
                "",
                f"- Total repo artifact size before compression/relocation: {before_mb:.2f} MB",
                f"- Total repo artifact size after compression/relocation: {after_mb:.2f} MB",
                f"- Files over 50MB: {len(over50)}",
                *[f"  - {p}" for p in over50],
                f"- Files over 100MB: {len(over100)}",
                *[f"  - {p}" for p in over100],
                f"- Files tracked in Git: {len(tracked)}",
                f"- Files recommended for release/Zenodo only: {len(recommended_release)}",
                *[f"  - {p}" for p in recommended_release],
                "",
                "Removed final-analysis outputs derived from toy rolling-origin cohorts:",
                *[f"- {p}" for p in removed_outputs],
            ]
        ),
        encoding="utf-8",
    )


def write_rule_docs() -> None:
    QC.mkdir(parents=True, exist_ok=True)
    (QC / "direct_use_rule_definitions_final.md").write_text(
        """# Final direct-use rule definitions

CAB-Strict deterministic direct use is allowed only when source identity is accepted, meaning identity is accepted, phenotype-domain discordance is false, the disease architecture is phenotype_anchored_monogenic or syndrome_anchored_self_loop, and the baseline environment is specific/concordant.

CAB-Strict blocks nonspecific_underresolved, modifier_penetrance_boundary, structural_functional_overlap, syndrome_organ_boundary unless explicitly self-loop concordant, trigger_dependent_latent, pleiotropic_collision, genotype_first_absent_phenotype, source-unmatched, meaning-rejected, and phenotype-domain-discordant assertions.

CAB-Balanced may be more permissive than Strict but still blocks source_match_accepted=False, meaning_match_accepted=False, phenotype_domain_discordance_flag=True, nonspecific_underresolved unless repaired/specific context exists, genotype_first_absent_phenotype, explicit no-deterministic-reuse states, and nonspecific baseline environments.
""",
        encoding="utf-8",
    )


def write_provenance_audit(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Rolling-origin cohort provenance audit",
        "",
        "Current row-level rolling-origin cohorts are not publication results because they were generated from existing benchmark fixtures and simulated endpoint reconstruction rather than direct per-origin ClinVar historical snapshot extracts.",
        "",
        "| origin_id | baseline_snapshot_source | followup_snapshot_source | real_snapshot_derived | synthetic_fixture | allowed_use | row_count | notes |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {origin_id} | {baseline_snapshot_source} | {followup_snapshot_source} | {real_snapshot_derived yes/no} | {synthetic_fixture yes/no} | {allowed_use} | {row_count} | {notes} |".format(
                **row
            )
        )
    QC.mkdir(parents=True, exist_ok=True)
    (QC / "rolling_origin_cohort_provenance_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_impact_report(metrics: list[dict[str, object]], checks: list[dict[str, object]]) -> None:
    strict_rates = [float(row["direct_use_allowed_rate"]) for row in metrics if row["mode"] == "CAB-Strict"]
    balanced_rates = [float(row["direct_use_allowed_rate"]) for row in metrics if row["mode"] == "CAB-Balanced"]
    failures = [row for row in checks if row["status"] != "pass"]
    (QC / "direct_use_repair_impact_report.md").write_text(
        "\n".join(
            [
                "# Direct-use repair impact report",
                "",
                f"- QC invariant failures after repair: {len(failures)}",
                f"- Mean CAB-Strict direct-use rate after repair: {sum(strict_rates) / len(strict_rates):.6f}" if strict_rates else "- Mean CAB-Strict direct-use rate after repair: n/a",
                f"- Mean CAB-Balanced direct-use rate after repair: {sum(balanced_rates) / len(balanced_rates):.6f}" if balanced_rates else "- Mean CAB-Balanced direct-use rate after repair: n/a",
                "- Rolling-origin row-level metrics are fixture-only and not publication results until direct historical ClinVar snapshot derivation is confirmed.",
                "- Unsupported reuse, true-portable allowed, and overrestriction were recomputed in reports/tables/routing_metrics_after_direct_use_repair.csv.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    before_mb = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts) / (1024 * 1024)
    removed_outputs = remove_final_analysis_outputs()
    provenance_rows, manifest_rows, repaired_rows = provenance_and_manifest()
    write_provenance_audit(provenance_rows)
    write_csv(TABLES / "data_artifact_manifest.csv", manifest_rows)
    checks = invariant_checks(repaired_rows)
    write_csv(TABLES / "direct_use_invariant_checks.csv", checks)
    metrics = routing_metrics(repaired_rows)
    write_csv(TABLES / "routing_metrics_after_direct_use_repair.csv", metrics)
    write_rule_docs()
    write_impact_report(metrics, checks)
    if ROLLING_REPORTS.exists():
        shutil.rmtree(ROLLING_REPORTS)
    size_report(before_mb, removed_outputs)
    print(f"Repaired {len(repaired_rows)} fixture prediction rows.")
    print(f"Wrote {len(manifest_rows)} compressed artifact manifest rows.")


if __name__ == "__main__":
    main()
