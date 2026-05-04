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

