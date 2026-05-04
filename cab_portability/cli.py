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
                f.write(f"{r.get('assertion_id')}\t{r.get('warning_codes')}\n")
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

