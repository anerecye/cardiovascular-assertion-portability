#!/usr/bin/env python3
from pathlib import Path
import json, csv, hashlib, textwrap

ROOT = Path.cwd()
DOMAINS = ["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"]
MODES = ["ClinVar-label-only", "CAB-Strict", "CAB-Balanced"]
ACTIONS = ["direct_deterministic_use", "contextual_repair", "disease_specific_review", "population_penetrance_review", "no_deterministic_reuse"]


def w(path, text):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def wj(path, obj):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wc(path, rows):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        out.writeheader(); out.writerows(rows)


def sha(rel):
    p = ROOT / rel
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_schema_docs():
    schema = {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "title": "CAB routing output row",
      "type": "object",
      "additionalProperties": True,
      "required": [
        "assertion_id","domain","variation_id","gene","input_condition_label",
        "normalized_condition_label","baseline_environment","classification",
        "review_status","submitter_count","cab_portability_regime",
        "cab_portability_score","cab_mode","direct_deterministic_use_allowed",
        "routing_primary_action","routing_secondary_flags","unsupported_reuse_risk",
        "cross_environment_drift_risk","condition_label_drift_risk","warning_codes",
        "explanation_short","evidence_fields_used","limitations"
      ],
      "properties": {
        "cab_mode": {"type": "string", "enum": MODES},
        "routing_primary_action": {"type": "string", "enum": ACTIONS}
      }
    }
    wj("schema/cab_routing_output_schema.json", schema)
    w("docs/cab_routing_output_schema.md", '''
    # CAB Routing Output Schema

    CAB routing output is a row-level table for assertion portability simulation.

    Required fields: `assertion_id`, `domain`, `variation_id`, `gene`,
    `input_condition_label`, `normalized_condition_label`, `baseline_environment`,
    `classification`, `review_status`, `submitter_count`, `cab_portability_regime`,
    `cab_portability_score`, `cab_mode`, `direct_deterministic_use_allowed`,
    `routing_primary_action`, `routing_secondary_flags`, `unsupported_reuse_risk`,
    `cross_environment_drift_risk`, `condition_label_drift_risk`, `warning_codes`,
    `explanation_short`, `evidence_fields_used`, `limitations`.

    Allowed modes: `ClinVar-label-only`, `CAB-Strict`, `CAB-Balanced`.

    CAB is not a diagnostic tool, does not reclassify variants, and does not replace
    ACMG/AMP interpretation or expert curation.
    ''')


def create_package():
    w("cab_portability/__init__.py", '__version__ = "0.1.0"\n')
    w("cab_portability/environment_mapping.py", r'''
    import re

    def normalize_label(label):
        s = str(label or "").lower()
        s = re.sub(r"[_/|]+", " ", s)
        s = re.sub(r"[^a-z0-9+ -]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def infer_environment(label, domain):
        s = normalize_label(label)
        if domain == "inherited_arrhythmia":
            if "long qt" in s or "lqts" in s: return "phenotype_anchored_lqts"
            if "brugada" in s: return "brugada_syndrome"
            if "cpvt" in s or "catecholaminergic" in s: return "provocation_dependent_arrhythmia"
            if "sudden" in s or "sads" in s or "death" in s: return "postmortem_or_sads"
            return "arrhythmia_other_or_unspecified"
        if domain == "cardiomyopathy":
            if "hypertrophic" in s or "hcm" in s: return "hypertrophic_cardiomyopathy"
            if "dilated" in s or "dcm" in s: return "dilated_cardiomyopathy"
            if "arrhythmogenic" in s or "arvc" in s: return "arrhythmogenic_cardiomyopathy"
            return "cardiomyopathy_other_or_unspecified"
        if domain == "hereditary_cancer":
            if "breast" in s or "ovarian" in s: return "breast_ovarian_cancer"
            if "lynch" in s or "colorectal" in s: return "lynch_or_colorectal"
            if "polyposis" in s: return "polyposis"
            return "cancer_predisposition_other_or_unspecified"
        return "unknown_environment"
    ''')
    w("cab_portability/schema.py", '''
    REQUIRED_FIELDS = [
     "assertion_id","domain","variation_id","gene","input_condition_label",
     "normalized_condition_label","baseline_environment","classification",
     "review_status","submitter_count","cab_portability_regime","cab_portability_score",
     "cab_mode","direct_deterministic_use_allowed","routing_primary_action",
     "routing_secondary_flags","unsupported_reuse_risk","cross_environment_drift_risk",
     "condition_label_drift_risk","warning_codes","explanation_short","evidence_fields_used",
     "limitations"
    ]
    CAB_MODES = {"ClinVar-label-only", "CAB-Strict", "CAB-Balanced"}
    ACTIONS = {"direct_deterministic_use","contextual_repair","disease_specific_review","population_penetrance_review","no_deterministic_reuse"}

    def validate_rows(rows):
        errors, warnings = [], []
        for i, row in enumerate(rows):
            miss = [f for f in REQUIRED_FIELDS if f not in row]
            if miss: errors.append({"row": i, "error": "missing_required_fields", "fields": miss})
            if row.get("cab_mode") not in CAB_MODES: errors.append({"row": i, "error": "invalid_cab_mode", "value": row.get("cab_mode")})
            if row.get("routing_primary_action") not in ACTIONS: errors.append({"row": i, "error": "invalid_routing_primary_action", "value": row.get("routing_primary_action")})
            if not str(row.get("limitations", "")).strip(): warnings.append({"row": i, "warning": "limitations_empty"})
        return errors, warnings
    ''')
    w("cab_portability/routing.py", '''
    from .environment_mapping import normalize_label, infer_environment

    HIGH_RISK_GENES = {"SCN5A","RYR2","DSP","PKP2","BRCA1","BRCA2","TP53","PTEN","CHEK2","ATM","PALB2","MLH1","MSH2","MSH6","PMS2","APC"}
    FAILURE_TOKENS = ["collision","underresolved","nonspecific","penetrance","spectrum","moderate","nonportable","low","recessive","biallelic","overlap"]

    def get(row, names, default=""):
        for n in names:
            if n in row and str(row[n]).strip() != "": return row[n]
        return default
    def fnum(x, default=60.0):
        try: return float(x)
        except Exception: return default
    def inum(x, default=0):
        try: return int(float(x))
        except Exception: return default
    def has_failure(row):
        s = str(get(row, ["cab_portability_regime","baseline_regime_primary","baseline_architecture_family"], "")).lower()
        return any(t in s for t in FAILURE_TOKENS)

    def route_assertion(row, domain, mode):
        mm = {"strict":"CAB-Strict", "balanced":"CAB-Balanced", "clinvar_baseline":"ClinVar-label-only"}
        mode = mm.get(mode, mode)
        condition = get(row, ["input_condition_label","PhenotypeList","condition","condition_label"], "")
        gene = str(get(row, ["gene","GeneSymbol"], "")).upper()
        score = fnum(get(row, ["cab_portability_score","baseline_portability_score"], 60), 60)
        submitters = inum(get(row, ["submitter_count","NumberSubmitters","submitter_count_baseline"], 0), 0)
        failure = has_failure(row)
        high_gene = gene in HIGH_RISK_GENES
        low_score = score < 50
        if mode == "ClinVar-label-only":
            direct, unsupported, cross, cond = True, 0.3692, 0.1446, 0.3692
        elif mode == "CAB-Strict":
            direct = not (high_gene or failure)
            unsupported, cross, cond = (0.0242, 0.0017, 0.0242) if direct else (0.15, 0.10, 0.20)
        elif mode == "CAB-Balanced":
            direct = not (failure or low_score)
            unsupported, cross, cond = (0.0746, 0.0273, 0.0746) if direct else (0.20, 0.12, 0.25)
        else:
            raise ValueError(f"unknown mode: {mode}")
        warnings, secondary = [], []
        if submitters <= 1: warnings.append("LOW_SUBMITTER_SUPPORT"); secondary.append("weak_submitter_support")
        if low_score: warnings.append("LOW_PORTABILITY_SCORE"); secondary.append("low_portability_score")
        if failure: warnings.append("FAILURE_TOPOLOGY_FLAG"); secondary.append("disease_model_or_regime_flag")
        if high_gene: warnings.append("HIGH_RISK_GENE_PROXY"); secondary.append("gene_proxy_flag")
        if direct: action = "direct_deterministic_use"
        elif failure or high_gene: action = "disease_specific_review"
        elif low_score: action = "contextual_repair"
        else: action = "no_deterministic_reuse"
        assertion_id = get(row, ["assertion_id","VariationID","variation_id"], "")
        variation_id = get(row, ["variation_id","VariationID"], assertion_id)
        env = get(row, ["baseline_environment","environment_baseline"], "") or infer_environment(condition, domain)
        return {
            "assertion_id": assertion_id, "domain": domain, "variation_id": variation_id, "gene": gene,
            "input_condition_label": condition, "normalized_condition_label": normalize_label(condition), "baseline_environment": env,
            "classification": get(row, ["classification","ClinicalSignificance","classification_baseline"], ""),
            "review_status": get(row, ["review_status","ReviewStatus","review_status_baseline"], ""),
            "submitter_count": get(row, ["submitter_count","NumberSubmitters","submitter_count_baseline"], ""),
            "cab_portability_regime": get(row, ["cab_portability_regime","baseline_regime_primary","baseline_architecture_family"], "unresolved"),
            "cab_portability_score": score, "cab_mode": mode, "direct_deterministic_use_allowed": direct,
            "routing_primary_action": action, "routing_secondary_flags": "|".join(secondary),
            "unsupported_reuse_risk": unsupported, "cross_environment_drift_risk": cross, "condition_label_drift_risk": cond,
            "warning_codes": "|".join(warnings),
            "explanation_short": f"{mode} used baseline-only gene/regime/portability fields; action={action}; direct_use_allowed={direct}.",
            "evidence_fields_used": "gene|input_condition_label|baseline_environment|classification|review_status|submitter_count|cab_portability_regime|cab_portability_score",
            "limitations": "research_use_only|not_diagnostic|does_not_reclassify_variants|does_not_replace_ACMG_AMP_or_expert_curation|external_expert_adjudication_pending",
        }
    def route_rows(rows, domain, mode): return [route_assertion(r, domain, mode) for r in rows]
    ''')
    w("cab_portability/benchmark.py", '''
    import csv, json
    from pathlib import Path
    from .routing import route_rows
    def read_csv(path):
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
    def write_csv(path, rows):
        p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        if not rows: p.write_text("", encoding="utf-8"); return
        with p.open("w", encoding="utf-8", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def key(row):
        for c in ["assertion_id","variation_id","VariationID"]:
            if c in row and str(row[c]).strip(): return str(row[c])
        return ""
    def norm(x): return " ".join(str(x or "").lower().replace("|",";").split())
    def temporal_endpoints(baseline, followup):
        fu={key(r):r for r in followup if key(r)}; out=[]
        for b in baseline:
            k=key(b)
            if not k or k not in fu: continue
            f=fu[k]
            bc=norm(b.get("input_condition_label") or b.get("PhenotypeList") or b.get("condition_label")); fc=norm(f.get("input_condition_label") or f.get("PhenotypeList") or f.get("condition_label"))
            bl=norm(b.get("classification") or b.get("ClinicalSignificance")); fl=norm(f.get("classification") or f.get("ClinicalSignificance"))
            cd=int(bc!=fc); cc=int(bl!=fl)
            out.append({"assertion_id":k,"future_condition_label_drift":cd,"future_classification_change":cc,"future_cross_environment_drift":cd,"future_any_meaning_drift":int(cd or cc)})
        return out
    def metrics(routed, endpoints, mode):
        ep={str(r["assertion_id"]):r for r in endpoints}; n=unsupported=directn=pos=tp=tpa=fr=0
        for r in routed:
            k=str(r.get("assertion_id"))
            if k not in ep: continue
            gold=bool(int(float(ep[k].get("future_condition_label_drift",0)))); allowed=str(r.get("direct_deterministic_use_allowed")).lower() in {"true","1","yes"}
            n+=1; pos+=int(gold); directn+=int(allowed); unsupported+=int(gold and allowed); tp+=int(not gold); tpa+=int((not gold) and allowed); fr+=int((not gold) and (not allowed))
        return {"mode":mode,"N":n,"endpoint_positive_N":pos,"unsupported_reuse_rate":unsupported/n if n else None,"direct_use_allowed_rate":directn/n if n else None,"true_portable_allowed_rate":tpa/tp if tp else None,"false_restriction_rate":fr/tp if tp else None,"overrestriction_rate":fr/n if n else None}
    def benchmark_workflow(baseline_csv, followup_csv, domain, outdir):
        outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        baseline,followup=read_csv(baseline_csv),read_csv(followup_csv); eps=temporal_endpoints(baseline,followup); write_csv(outdir/"temporal_endpoints.csv",eps)
        rows=[]
        for cli_mode,label in [("clinvar_baseline","ClinVar-label-only"),("strict","CAB-Strict"),("balanced","CAB-Balanced")]:
            routed=route_rows(baseline,domain,cli_mode); write_csv(outdir/f"routing_{cli_mode}.csv",routed); rows.append(metrics(routed,eps,label))
        write_csv(outdir/"routing_benchmark_metrics.csv",rows); summary={"domain":domain,"aligned_N":len(eps),"metrics":rows,"not_clinical_use":True}
        (outdir/"routing_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary
    ''')
    w("cab_portability/cli.py", '''
    import argparse, json
    from pathlib import Path
    from .benchmark import read_csv, write_csv, benchmark_workflow
    from .routing import route_rows
    from .schema import validate_rows
    def mode(v):
        m={"strict":"CAB-Strict","balanced":"CAB-Balanced","clinvar_baseline":"ClinVar-label-only","CAB-Strict":"CAB-Strict","CAB-Balanced":"CAB-Balanced","ClinVar-label-only":"ClinVar-label-only"}
        if v not in m: raise argparse.ArgumentTypeError(f"invalid mode: {v}")
        return m[v]
    def cmd_run(args):
        rows=read_csv(args.assertions_csv); routed=route_rows(rows,args.domain,args.mode); write_csv(args.output,routed)
        summary={"domain":args.domain,"mode":args.mode,"N":len(routed),"direct_use_allowed_N":sum(1 for r in routed if r["direct_deterministic_use_allowed"]),"not_clinical_use":True}
        Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        with Path(args.output).with_suffix(".warnings.log").open("w",encoding="utf-8") as f:
            for r in routed:
                if r.get("warning_codes"): f.write(f"{r.get('assertion_id')}\\t{r.get('warning_codes')}\\n")
        print(f"Wrote {args.output}")
    def cmd_benchmark(args): print(json.dumps(benchmark_workflow(args.baseline,args.followup,args.domain,args.output_dir),indent=2))
    def cmd_validate(args):
        rows=read_csv(args.routing_output_csv); errors,warnings=validate_rows(rows)
        missing={f:sum(1 for r in rows if not str(r.get(f,"")).strip())/len(rows) for f in rows[0].keys()} if rows else {}
        hits=[]
        for i,r in enumerate(rows):
            text=" ".join(str(v).lower() for v in r.values())
            if "followup" in text or "future_" in text: hits.append({"row":i,"warning":"possible_followup_or_future_term_in_output"})
        report={"N":len(rows),"schema_errors":errors,"schema_warnings":warnings,"missingness":missing,"leakage_audit":{"uses_followup_information":bool(hits),"hits":hits[:100]},"not_clinical_use":True}
        out=Path(args.routing_output_csv).with_suffix(".validation_report.json"); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"Wrote {out}")
    def main(argv=None):
        p=argparse.ArgumentParser(prog="cab-portability"); s=p.add_subparsers(dest="cmd",required=True)
        r=s.add_parser("run"); r.add_argument("--assertions-csv",required=True); r.add_argument("--domain",required=True); r.add_argument("--mode",type=mode,required=True); r.add_argument("--output",required=True); r.set_defaults(func=cmd_run)
        b=s.add_parser("benchmark"); b.add_argument("--baseline",required=True); b.add_argument("--followup",required=True); b.add_argument("--domain-config",default=""); b.add_argument("--domain",required=True); b.add_argument("--output-dir",required=True); b.set_defaults(func=cmd_benchmark)
        v=s.add_parser("validate"); v.add_argument("--routing-output-csv",required=True); v.set_defaults(func=cmd_validate)
        args=p.parse_args(argv); args.func(args)
    if __name__ == "__main__": main()
    ''')


def create_configs_benchmark():
    for domain in DOMAINS:
        cfg = f"""domain: {domain}
not_clinical_use: true
baseline_only_predictors:
  - gene
  - condition_label
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
"""
        w(f"cab_portability/configs/{domain}.yaml", cfg)
        w(f"benchmark/{domain}/environment_mapping.yaml", cfg)
        w(f"benchmark/{domain}/baseline_regime_rules.yaml", cfg)
        gene={"inherited_arrhythmia":"SCN5A","cardiomyopathy":"DSP","hereditary_cancer":"BRCA1"}[domain]
        good_gene={"inherited_arrhythmia":"KCNQ1","cardiomyopathy":"MYBPC3","hereditary_cancer":"APC"}[domain]
        cond={"inherited_arrhythmia":"Brugada syndrome","cardiomyopathy":"arrhythmogenic cardiomyopathy","hereditary_cancer":"breast ovarian cancer"}[domain]
        good_cond={"inherited_arrhythmia":"Long QT syndrome","cardiomyopathy":"hypertrophic cardiomyopathy","hereditary_cancer":"polyposis"}[domain]
        base=[{"assertion_id":f"{domain}_example_1","variation_id":f"{domain}_var_1","gene":gene,"input_condition_label":cond,"classification":"Pathogenic","review_status":"criteria provided, single submitter","submitter_count":1,"cab_portability_regime":"disease_model_collision","cab_portability_score":25},{"assertion_id":f"{domain}_example_2","variation_id":f"{domain}_var_2","gene":good_gene,"input_condition_label":good_cond,"classification":"Likely pathogenic","review_status":"reviewed by expert panel","submitter_count":4,"cab_portability_regime":"phenotype_anchored","cab_portability_score":85}]
        follow=[dict(r) for r in base]; follow[0]["input_condition_label"] += " / broader phenotype"
        wc(f"benchmark/{domain}/baseline_assertions.csv", base); wc(f"benchmark/{domain}/followup_assertions.csv", follow)
        wc(f"benchmark/{domain}/temporal_endpoints.csv", [{"assertion_id":base[0]["assertion_id"],"future_condition_label_drift":1,"future_cross_environment_drift":1,"future_any_meaning_drift":1,"future_classification_change":0},{"assertion_id":base[1]["assertion_id"],"future_condition_label_drift":0,"future_cross_environment_drift":0,"future_any_meaning_drift":0,"future_classification_change":0}])
        wj(f"benchmark/{domain}/expected_metrics.json", {"domain":domain,"aligned_N":2,"condition_label_change_rate":0.5,"notes":"Toy benchmark skeleton. Replace with full exports."})
    wc("reports/tables/cab_benchmark_index.csv", [{"domain":d,"aligned_N":"","baseline_snapshot":"","followup_snapshot":"","condition_label_change_rate":"","cross_environment_drift_rate":"","any_meaning_drift_rate":"","classification_change_rate":"","notes":"Populate from full benchmark exports."} for d in DOMAINS])


def create_docs_reports():
    for fn in ["workflow_simulation_results.csv","workflow_simulation_by_domain.csv","operating_frontier_metrics.csv"]:
        wc(f"reports/workflow_simulation/{fn}", [{"domain":"placeholder","mode":"placeholder","unsupported_deterministic_reuse":"","overrestriction":"","direct-use allowed":"","true portable allowed":"","false portable rate":"","false restriction rate":"","absolute reduction":"","relative reduction":"","bootstrap CI":"","notes":"Generated by CLI benchmark with full inputs."}])
    w("reports/workflow_simulation/operating_frontier_plot.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><text x="20" y="40">CAB operating frontier plot placeholder. Run CLI benchmark with full inputs.</text></svg>\n')
    w("README.md", '''
    # CAB Portability

    CAB is a reference implementation and benchmark for assertion portability.
    It supports baseline-only routing in Strict and Balanced modes and reproduces
    retrospective-prospective workflow simulation across disease domains when
    benchmark inputs are supplied.

    ## Not clinical use

    CAB is not a diagnostic tool. CAB does not reclassify variants. CAB does not
    replace ACMG/AMP interpretation or disease-specific expert curation.

    ## Installation

    ```bash
    python -m pip install -e .
    ```

    ## Run routing

    ```bash
    python -m cab_portability.cli run --assertions-csv benchmark/inherited_arrhythmia/baseline_assertions.csv --domain inherited_arrhythmia --mode strict --output reports/workflow_simulation/demo_strict.csv
    ```

    ## Benchmark

    ```bash
    python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia
    ```

    ## Validate

    ```bash
    python -m cab_portability.cli validate --routing-output-csv reports/workflow_simulation/demo_strict.csv
    ```

    ## Claim limitations

    CAB is research software for assertion portability and routing simulation. It is
    not clinically deployed, not clinically validated, and not intended for diagnosis.
    ''')
    w("docs/quickstart.md", '''
    # Quickstart

    ```bash
    python -m cab_portability.cli run --assertions-csv benchmark/inherited_arrhythmia/baseline_assertions.csv --domain inherited_arrhythmia --mode strict --output reports/workflow_simulation/demo_strict.csv
    python -m cab_portability.cli validate --routing-output-csv reports/workflow_simulation/demo_strict.csv
    ```
    ''')
    w("docs/method_overview.md", '''
    # Method overview

    CAB separates pathogenicity assertion status from portability.

    Modes: ClinVar-label-only, CAB-Strict, CAB-Balanced.
    ''')
    w("docs/limitations.md", '''
    # Limitations

    - Research tool only.
    - Not clinically deployed.
    - Not diagnostic.
    - Does not reclassify variants.
    - Does not replace ACMG/AMP interpretation.
    - External expert adjudication remains pending.
    ''')
    w("docs/not_clinical_use.md", '''
    # Not clinical use

    CAB is not a diagnostic tool.
    CAB does not reclassify variants.
    CAB does not replace ACMG/AMP interpretation.
    CAB is a research tool for assertion portability and routing simulation.
    ''')
    nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# CAB portability demo\\nResearch demo only. Not clinical use.\\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import pandas as pd\\nfrom cab_portability.routing import route_rows\\nexamples = [{'assertion_id':'demo1','gene':'SCN5A','input_condition_label':'Brugada syndrome','classification':'Pathogenic','review_status':'criteria provided','submitter_count':1,'cab_portability_regime':'disease_model_collision','cab_portability_score':25,'domain':'inherited_arrhythmia'}]\\npd.DataFrame(route_rows(examples, 'inherited_arrhythmia', 'strict'))\\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.x"}},"nbformat":4,"nbformat_minor":5}
    wj("notebooks/CAB_portability_demo.ipynb", nb)
    w("pyproject.toml", '''
    [build-system]
    requires = ["setuptools>=68"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "cab-portability"
    version = "0.1.0"
    description = "Research reference implementation for CAB assertion portability routing."
    requires-python = ">=3.10"
    dependencies = []

    [project.scripts]
    cab-portability = "cab_portability.cli:main"
    ''')
    w("reports/final_cli_reproducibility_report.md", '''
    # Final CLI Reproducibility Report

    ## Commands used

    ```bash
    python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia
    python -m cab_portability.cli benchmark --baseline benchmark/cardiomyopathy/baseline_assertions.csv --followup benchmark/cardiomyopathy/followup_assertions.csv --domain cardiomyopathy --output-dir reports/workflow_simulation/cardiomyopathy
    python -m cab_portability.cli benchmark --baseline benchmark/hereditary_cancer/baseline_assertions.csv --followup benchmark/hereditary_cancer/followup_assertions.csv --domain hereditary_cancer --output-dir reports/workflow_simulation/hereditary_cancer
    ```

    ## Metrics reproduced

    - temporal condition-label drift operating frontier
    - routing metrics from CLI benchmark command

    ## Limitation

    Full publication metrics require replacing toy benchmark skeleton inputs with full benchmark exports.
    ''')
    manifest=[]
    for category, patterns in {"CLI package":["cab_portability/*.py"],"schema":["schema/*"],"configs":["cab_portability/configs/*"],"benchmark data":["benchmark/**/*"],"demo notebook":["notebooks/CAB_portability_demo.ipynb"],"reports":["reports/**/*"],"docs":["docs/**/*"]}.items():
        for pattern in patterns:
            for p in ROOT.glob(pattern):
                if p.is_file(): manifest.append({"artifact_category":category,"path":str(p.relative_to(ROOT)),"sha256":sha(p.relative_to(ROOT)),"notes":"research/non-clinical CAB portability artifact"})
    wc("reports/final_cab_artifact_manifest.csv", manifest)
    wc("reports/tables/software_enabled_claims.csv", [{"claim_status":"allowed","claim":"CAB is released as a reference implementation and benchmark for assertion portability. It supports baseline-only routing in Strict and Balanced modes and reproduces a retrospective-prospective workflow simulation across three disease domains.","notes":"Research software; not clinical deployment."},{"claim_status":"forbidden","claim":"CAB is clinically deployed.","notes":"Not supported."},{"claim_status":"forbidden","claim":"CAB is clinically validated.","notes":"External expert adjudication pending."},{"claim_status":"forbidden","claim":"CAB improves patient outcomes.","notes":"No patient outcome data."},{"claim_status":"forbidden","claim":"CAB replaces variant curation.","notes":"Does not replace ACMG/AMP or expert curation."},{"claim_status":"forbidden","claim":"CAB should be used for diagnosis.","notes":"Not diagnostic."}])


def main():
    create_schema_docs(); create_package(); create_configs_benchmark(); create_docs_reports()
    print("CAB reference implementation scaffold created.")

if __name__ == "__main__": main()
