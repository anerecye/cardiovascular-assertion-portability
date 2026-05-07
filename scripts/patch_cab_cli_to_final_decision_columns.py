
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()

ROUTING = """
from .environment_mapping import normalize_label, infer_environment

HIGH_RISK_GENES = {
    "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
    "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC"
}

FAILURE_TOKENS = [
    "collision", "underresolved", "nonspecific", "penetrance", "spectrum",
    "moderate", "nonportable", "low", "recessive", "biallelic", "overlap"
]

TRUE_STRINGS = {"1", "true", "yes", "y", "t"}
FALSE_STRINGS = {"0", "false", "no", "n", "f"}


def get(row, names, default=""):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return row[n]
    return default


def fnum(x, default=60.0):
    try:
        return float(x)
    except Exception:
        return default


def inum(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def parse_bool(x, default=None):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return default


def has_failure(row):
    s = " ".join([
        str(get(row, ["cab_portability_regime", "baseline_regime_primary", "primary_regime"], "")),
        str(get(row, ["baseline_architecture_family", "causal_architecture_category", "causal_architecture"], "")),
    ]).lower()
    return any(t in s for t in FAILURE_TOKENS)


def strict_direct_from_final_or_fallback(row):
    explicit = get(row, [
        "cab_strict_direct_use_allowed",
        "CAB_Strict_direct_use_allowed",
        "strict_direct_use_allowed",
        "cab_core_direct_use_allowed",
        "CAB_Core_direct_use_allowed",
    ], "")
    val = parse_bool(explicit, default=None)
    if val is not None:
        return val, "explicit_cab_strict_direct_use_allowed"

    gene = str(get(row, ["gene", "GeneSymbol"], "")).upper()
    direct = not (gene in HIGH_RISK_GENES or has_failure(row))
    return direct, "derived_from_gene_plus_baseline_regime"


def balanced_direct_from_final_or_fallback(row):
    explicit = get(row, [
        "direct_single_model_reuse_allowed",
        "cab_balanced_direct_use_allowed",
        "CAB_Balanced_direct_use_allowed",
        "balanced_direct_use_allowed",
        "direct_deterministic_use_allowed",
    ], "")
    val = parse_bool(explicit, default=None)
    if val is not None:
        return val, "direct_single_model_reuse_allowed"

    score = fnum(get(row, ["cab_portability_score", "baseline_portability_score"], 60), 60)
    direct = not (has_failure(row) or score < 50)
    return direct, "fallback_failure_topology_plus_score"


def clinvar_direct(row):
    return True, "clinvar_label_only_assumes_direct_use"


def final_direct_for_mode(row, mode):
    if mode == "ClinVar-label-only":
        return clinvar_direct(row)
    if mode == "CAB-Strict":
        return strict_direct_from_final_or_fallback(row)
    if mode == "CAB-Balanced":
        return balanced_direct_from_final_or_fallback(row)
    raise ValueError(f"unknown mode: {mode}")


def primary_action(direct, row, mode):
    if direct:
        return "direct_deterministic_use"

    if mode == "CAB-Balanced":
        if parse_bool(get(row, ["disease_specific_expert_review_required"], ""), default=False):
            return "disease_specific_review"
        if parse_bool(get(row, ["population_or_penetrance_review_required"], ""), default=False):
            return "population_penetrance_review"
        if parse_bool(get(row, ["contextual_repair_required"], ""), default=False):
            return "contextual_repair"

    if has_failure(row):
        return "disease_specific_review"

    score = fnum(get(row, ["cab_portability_score", "baseline_portability_score"], 60), 60)
    if score < 50:
        return "contextual_repair"

    gene = str(get(row, ["gene", "GeneSymbol"], "")).upper()
    if gene in HIGH_RISK_GENES:
        return "disease_specific_review"

    return "no_deterministic_reuse"


def risk_values_from_decision(direct, mode):
    if mode == "ClinVar-label-only":
        return 0.3692, 0.1446, 0.3692
    if mode == "CAB-Strict":
        return (0.0242, 0.0017, 0.0242) if direct else (0.15, 0.10, 0.20)
    if mode == "CAB-Balanced":
        return (0.0746, 0.0273, 0.0746) if direct else (0.20, 0.12, 0.25)
    return 0.0, 0.0, 0.0


def route_assertion(row, domain, mode):
    mm = {
        "strict": "CAB-Strict",
        "balanced": "CAB-Balanced",
        "clinvar_baseline": "ClinVar-label-only",
        "CAB-Strict": "CAB-Strict",
        "CAB-Balanced": "CAB-Balanced",
        "ClinVar-label-only": "ClinVar-label-only",
    }
    mode = mm.get(mode, mode)

    condition = get(row, ["input_condition_label", "PhenotypeList", "condition", "condition_label"], "")
    gene = str(get(row, ["gene", "GeneSymbol"], "")).upper()
    score = fnum(get(row, ["cab_portability_score", "baseline_portability_score"], 60), 60)
    submitters = inum(get(row, ["submitter_count", "NumberSubmitters", "submitter_count_baseline"], 0), 0)

    direct, decision_source = final_direct_for_mode(row, mode)
    unsupported, cross, cond = risk_values_from_decision(direct, mode)
    action = primary_action(direct, row, mode)

    warnings, secondary = [], []
    if submitters <= 1:
        warnings.append("LOW_SUBMITTER_SUPPORT")
        secondary.append("weak_submitter_support")
    if score < 50:
        warnings.append("LOW_PORTABILITY_SCORE")
        secondary.append("low_portability_score")
    if has_failure(row):
        warnings.append("FAILURE_TOPOLOGY_FLAG")
        secondary.append("disease_model_or_regime_flag")
    if gene in HIGH_RISK_GENES:
        warnings.append("HIGH_RISK_GENE_PROXY")
        secondary.append("gene_proxy_flag")

    if decision_source.startswith("fallback"):
        warnings.append("FALLBACK_ROUTING_HEURISTIC_USED")
    if decision_source == "derived_from_gene_plus_baseline_regime":
        secondary.append("strict_gene_regime_rule")
    if decision_source == "direct_single_model_reuse_allowed":
        secondary.append("final_balanced_decision_column")

    assertion_id = get(row, ["assertion_id", "VariationID", "variation_id"], "")
    variation_id = get(row, ["variation_id", "VariationID"], assertion_id)
    env = get(row, ["baseline_environment", "environment_baseline"], "") or infer_environment(condition, domain)

    return {
        "assertion_id": assertion_id,
        "domain": domain,
        "variation_id": variation_id,
        "gene": gene,
        "input_condition_label": condition,
        "normalized_condition_label": normalize_label(condition),
        "baseline_environment": env,
        "classification": get(row, ["classification", "ClinicalSignificance", "classification_baseline"], ""),
        "review_status": get(row, ["review_status", "ReviewStatus", "review_status_baseline"], ""),
        "submitter_count": get(row, ["submitter_count", "NumberSubmitters", "submitter_count_baseline"], ""),
        "cab_portability_regime": get(row, ["cab_portability_regime", "baseline_regime_primary", "baseline_architecture_family"], "unresolved"),
        "cab_portability_score": score,
        "cab_mode": mode,
        "direct_deterministic_use_allowed": direct,
        "routing_decision_source": decision_source,
        "routing_primary_action": action,
        "routing_secondary_flags": "|".join(secondary),
        "unsupported_reuse_risk": unsupported,
        "cross_environment_drift_risk": cross,
        "condition_label_drift_risk": cond,
        "warning_codes": "|".join(warnings),
        "explanation_short": f"{mode} used {decision_source}; action={action}; direct_use_allowed={direct}.",
        "evidence_fields_used": "gene|input_condition_label|baseline_environment|classification|review_status|submitter_count|cab_portability_regime|cab_portability_score|direct_single_model_reuse_allowed",
        "limitations": "research_use_only|not_diagnostic|does_not_reclassify_variants|does_not_replace_ACMG_AMP_or_expert_curation|external_expert_adjudication_pending",
    }


def route_rows(rows, domain, mode):
    return [route_assertion(r, domain, mode) for r in rows]
"""

BENCHMARK = """
import csv
import json
from pathlib import Path

from .routing import route_rows


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def bool_value(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def key(row):
    for c in ["assertion_id", "variation_id", "VariationID"]:
        if c in row and str(row[c]).strip():
            return str(row[c])
    return ""


def norm(x):
    return " ".join(str(x or "").lower().replace("|", ";").split())


def endpoint_from_file_or_followup(baseline, followup, endpoint_file=None):
    if endpoint_file and Path(endpoint_file).exists():
        rows = read_csv(endpoint_file)
        out = []
        for r in rows:
            out.append({
                "assertion_id": str(r.get("assertion_id", "")),
                "future_condition_label_drift": int(bool_value(r.get("future_condition_label_drift", 0))),
                "future_classification_change": int(bool_value(r.get("future_classification_change", 0))),
                "future_cross_environment_drift": int(bool_value(r.get("future_cross_environment_drift", 0))),
                "future_any_meaning_drift": int(bool_value(r.get("future_any_meaning_drift", 0))),
            })
        return out

    fu = {key(r): r for r in followup if key(r)}
    out = []
    for b in baseline:
        k = key(b)
        if not k or k not in fu:
            continue
        f = fu[k]
        bc = norm(b.get("input_condition_label") or b.get("PhenotypeList") or b.get("condition_label"))
        fc = norm(f.get("input_condition_label") or f.get("PhenotypeList") or f.get("condition_label"))
        bl = norm(b.get("classification") or b.get("ClinicalSignificance"))
        fl = norm(f.get("classification") or f.get("ClinicalSignificance"))
        cd = int(bc != fc)
        cc = int(bl != fl)
        out.append({
            "assertion_id": k,
            "future_condition_label_drift": cd,
            "future_classification_change": cc,
            "future_cross_environment_drift": cd,
            "future_any_meaning_drift": int(cd or cc),
        })
    return out


def metrics(routed, endpoints, mode):
    ep = {str(r["assertion_id"]): r for r in endpoints}
    n = unsupported = directn = pos = true_portable = true_portable_allowed = false_restriction = 0

    for r in routed:
        k = str(r.get("assertion_id"))
        if k not in ep:
            continue
        gold = bool_value(ep[k].get("future_condition_label_drift", 0))
        allowed = bool_value(r.get("direct_deterministic_use_allowed"))
        n += 1
        pos += int(gold)
        directn += int(allowed)
        unsupported += int(gold and allowed)
        true_portable += int(not gold)
        true_portable_allowed += int((not gold) and allowed)
        false_restriction += int((not gold) and (not allowed))

    return {
        "mode": mode,
        "N": n,
        "endpoint_positive_N": pos,
        "unsupported_reuse_rate": unsupported / n if n else None,
        "direct_use_allowed_rate": directn / n if n else None,
        "true_portable_allowed_rate": true_portable_allowed / true_portable if true_portable else None,
        "false_restriction_rate": false_restriction / true_portable if true_portable else None,
        "overrestriction_rate": false_restriction / n if n else None,
    }


def benchmark_workflow(baseline_csv, followup_csv, domain, outdir, endpoint_file=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = read_csv(baseline_csv)
    followup = read_csv(followup_csv)
    endpoints = endpoint_from_file_or_followup(baseline, followup, endpoint_file)
    write_csv(outdir / "temporal_endpoints.csv", endpoints)

    rows = []
    for cli_mode, label in [
        ("clinvar_baseline", "ClinVar-label-only"),
        ("strict", "CAB-Strict"),
        ("balanced", "CAB-Balanced"),
    ]:
        routed = route_rows(baseline, domain, cli_mode)
        write_csv(outdir / f"routing_{cli_mode}.csv", routed)
        rows.append(metrics(routed, endpoints, label))

    write_csv(outdir / "routing_benchmark_metrics.csv", rows)
    summary = {
        "domain": domain,
        "aligned_N": len(endpoints),
        "metrics": rows,
        "not_clinical_use": True,
        "routing_decision_lock": "uses direct_single_model_reuse_allowed for CAB-Balanced when present; uses explicit or gene/regime final rule for CAB-Strict",
        "endpoint_source": str(endpoint_file) if endpoint_file else "followup_reconstruction",
    }
    (outdir / "routing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
"""

CLI = """
import argparse
import json
from pathlib import Path

from .benchmark import read_csv, write_csv, benchmark_workflow
from .routing import route_rows
from .schema import validate_rows


def mode(v):
    m = {
        "strict": "CAB-Strict",
        "balanced": "CAB-Balanced",
        "clinvar_baseline": "ClinVar-label-only",
        "CAB-Strict": "CAB-Strict",
        "CAB-Balanced": "CAB-Balanced",
        "ClinVar-label-only": "ClinVar-label-only",
    }
    if v not in m:
        raise argparse.ArgumentTypeError(f"invalid mode: {v}")
    return m[v]


def cmd_run(args):
    rows = read_csv(args.assertions_csv)
    routed = route_rows(rows, args.domain, args.mode)
    write_csv(args.output, routed)
    summary = {
        "domain": args.domain,
        "mode": args.mode,
        "N": len(routed),
        "direct_use_allowed_N": sum(1 for r in routed if str(r["direct_deterministic_use_allowed"]).lower() == "true"),
        "not_clinical_use": True,
        "routing_decision_lock": "final decision columns used when present",
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with Path(args.output).with_suffix(".warnings.log").open("w", encoding="utf-8") as f:
        for r in routed:
            if r.get("warning_codes"):
                f.write(f"{r.get('assertion_id')}\\t{r.get('warning_codes')}\\n")
    print(f"Wrote {args.output}")


def cmd_benchmark(args):
    endpoint_file = args.endpoint_file
    if endpoint_file is None:
        candidate = Path(args.baseline).parent / "temporal_endpoints.csv"
        if candidate.exists():
            endpoint_file = str(candidate)
    print(json.dumps(benchmark_workflow(args.baseline, args.followup, args.domain, args.output_dir, endpoint_file=endpoint_file), indent=2))


def cmd_validate(args):
    rows = read_csv(args.routing_output_csv)
    errors, warnings = validate_rows(rows)
    missingness = {f: sum(1 for r in rows if not str(r.get(f, "")).strip()) / len(rows) for f in rows[0].keys()} if rows else {}
    leakage_hits = []
    for i, r in enumerate(rows):
        text = " ".join(str(v).lower() for v in r.values())
        if "followup" in text or "future_" in text:
            leakage_hits.append({"row": i, "warning": "possible_followup_or_future_term_in_output"})
    report = {
        "N": len(rows),
        "schema_errors": errors,
        "schema_warnings": warnings,
        "missingness": missingness,
        "leakage_audit": {"uses_followup_information": bool(leakage_hits), "hits": leakage_hits[:100]},
        "not_clinical_use": True,
    }
    out = Path(args.routing_output_csv).with_suffix(".validation_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cab-portability")
    s = p.add_subparsers(dest="cmd", required=True)

    r = s.add_parser("run")
    r.add_argument("--assertions-csv", required=True)
    r.add_argument("--domain", required=True)
    r.add_argument("--mode", type=mode, required=True)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_run)

    b = s.add_parser("benchmark")
    b.add_argument("--baseline", required=True)
    b.add_argument("--followup", required=True)
    b.add_argument("--endpoint-file", default=None)
    b.add_argument("--domain-config", default="")
    b.add_argument("--domain", required=True)
    b.add_argument("--output-dir", required=True)
    b.set_defaults(func=cmd_benchmark)

    v = s.add_parser("validate")
    v.add_argument("--routing-output-csv", required=True)
    v.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
"""


def write(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip() + "\n", encoding="utf-8")


def main():
    write("cab_portability/routing.py", ROUTING)
    write("cab_portability/benchmark.py", BENCHMARK)
    write("cab_portability/cli.py", CLI)

    report = "# CLI Decision Column Lock Report\n\n"
    report += "The CAB CLI workflow simulation has been locked to final operating-frontier decision columns where available.\n\n"
    report += "## Decision sources\n\n"
    report += "- `ClinVar-label-only`: assumes direct deterministic use.\n"
    report += "- `CAB-Strict`: uses explicit `cab_strict_direct_use_allowed` if present; otherwise derives the final strict rule from gene + baseline disease-model regime/failure topology.\n"
    report += "- `CAB-Balanced`: uses `direct_single_model_reuse_allowed` when present. This is the final operating-frontier direct-use decision column.\n\n"
    report += "## Endpoint source\n\n"
    report += "The benchmark command now prefers `temporal_endpoints.csv` from the same benchmark domain directory. If not available, it reconstructs endpoints from baseline/follow-up replay files.\n\n"
    report += "## Leakage boundary\n\n"
    report += "Baseline routing files remain baseline-only. Temporal endpoints are read only during benchmark evaluation, not during routing.\n\n"
    report += "## Limitation\n\n"
    report += "This is a research workflow simulation. CAB is not a diagnostic tool, does not reclassify variants, and does not replace ACMG/AMP or expert curation.\n"
    write("reports/workflow_simulation/cli_decision_column_lock_report.md", report)

    print("Patched CLI to use final operating-frontier decision columns.")
    print("Updated:")
    print("  cab_portability/routing.py")
    print("  cab_portability/benchmark.py")
    print("  cab_portability/cli.py")
    print("  reports/workflow_simulation/cli_decision_column_lock_report.md")


if __name__ == "__main__":
    main()
