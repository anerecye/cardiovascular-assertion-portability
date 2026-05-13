#!/usr/bin/env python3
"""Build a true-blind CAB expert-endpoint adjudication run package.

This package turns the existing 400-case CAB casebook into an executable
expert-verification endpoint run. It does not invent adjudicator labels. It
creates domain reviewer assignments, verdict templates, endpoint keys, SADS/CPVT
priority addenda, and a validation analysis script contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "reports" / "adjudication"
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

CORE_CASEBOOK = ADJ / "cab_expert_adjudication_casebook_blinded.csv"
ANSWER_KEY = ADJ / "cab_expert_adjudication_casebook_answer_key.csv"
SADS_CASES = ADJ / "cab_sads_adjudication_pathway_cases.csv"

ENDPOINT_COLS = [
    "followup_condition_label",
    "condition_label_drift",
    "cross_environment_drift",
    "semantic_drift_without_reclassification",
    "conservative_composite_non_portability",
    "identity_vs_meaning_discordance",
    "expected_expert_decision_category",
    "endpoint_status_hidden_shown_flag",
]

CAB_PREDICTION_COLS = [
    "sample_bucket",
    "CAB_regime",
    "CAB_routing_action",
    "reason_code",
    "evidence_fields_shown_to_adjudicator",
]

REVIEWERS = {
    "hereditary_cancer": ["HC_R1", "HC_R2", "HC_R3"],
    "cardiomyopathy": ["CM_R1", "CM_R2", "CM_R3"],
    "inherited_arrhythmia": ["ARR_R1", "ARR_R2", "ARR_R3", "ARR_R4", "ARR_R5"],
}

ARRHYTHMIA_HIGH_VALUE_GENES = {
    "ANK2",
    "CACNA1C",
    "CACNA2D1",
    "CALM1",
    "CALM2",
    "CALM3",
    "CASQ2",
    "HCN4",
    "KCNH2",
    "KCNJ2",
    "KCNQ1",
    "RYR2",
    "SCN5A",
    "TRDN",
}

CPVT_SADS_GENES = {"RYR2", "CASQ2", "CALM1", "CALM2", "CALM3", "TRDN", "SCN5A", "KCNQ1", "KCNH2"}


def ensure_dirs() -> None:
    ADJ.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)


def yes_no_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def target_context(row: pd.Series) -> str:
    action = str(row.get("CAB_routing_action", "")).lower()
    bucket = str(row.get("sample_bucket", "")).lower()
    regime = str(row.get("CAB_regime", "")).lower()
    if "population" in action or "penetrance" in action or "population" in bucket:
        return "genotype-first, population, or penetrance-sensitive reuse context"
    if "disease" in action or "disease-specific" in bucket:
        return "disease-specific interpretive context named by the assertion"
    if "contextual" in action or "repair" in bucket:
        return "reuse after condition-label or disease-model context repair"
    if "identity" in bucket:
        return "same-variant identity with potentially discordant disease meaning"
    if "trigger" in regime:
        return "trigger-dependent or context-dependent arrhythmia liability context"
    if "direct" in bucket:
        return "direct reuse in the stated baseline disease context"
    return "assertion portability into the stated target context"


def primary_question(row: pd.Series) -> str:
    return (
        "Can this P/LP assertion be reused in the specified target context without "
        "additional disease-specific, population/penetrance, genotype-first, or "
        "contextual reinterpretation?"
    )


def high_value_flags(row: pd.Series) -> tuple[bool, bool]:
    gene = str(row.get("gene", "")).upper()
    domain = str(row.get("domain", ""))
    high_value = domain == "inherited_arrhythmia" and gene in ARRHYTHMIA_HIGH_VALUE_GENES
    sads_cpvt = domain == "inherited_arrhythmia" and gene in CPVT_SADS_GENES
    return high_value, sads_cpvt


def build_core_packets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(CORE_CASEBOOK)
    key = pd.read_csv(ANSWER_KEY)
    core["target_context"] = core.apply(target_context, axis=1)
    core["primary_expert_endpoint_question"] = core.apply(primary_question, axis=1)
    flags = core.apply(high_value_flags, axis=1, result_type="expand")
    flags.columns = ["arrhythmia_high_value_gene", "sads_cpvt_priority"]
    core = pd.concat([core, flags], axis=1)

    blind_drop = [col for col in ENDPOINT_COLS + CAB_PREDICTION_COLS if col in core.columns]
    blind = core.drop(columns=blind_drop)
    blind["expert_endpoint_status"] = "pending_real_expert_review"
    blind["reviewer_blinding"] = "true_blind_no_followup_temporal_endpoint_or_CAB_prediction_columns"

    key_cols = [
        "blinded_case_id",
        "assertion_id",
        "domain",
        "gene",
        "variant_identifier",
        "sample_bucket",
        "CAB_regime",
        "CAB_routing_action",
        "reason_code",
    ] + [col for col in ENDPOINT_COLS if col in key.columns]
    endpoint_key = key[key_cols].copy()
    endpoint_key["key_scope"] = "analyst_only_not_for_reviewers"

    assignment_rows: list[dict[str, object]] = []
    for _, row in blind.iterrows():
        reviewers = REVIEWERS.get(str(row["domain"]), [])
        for reviewer_id in reviewers:
            assignment_rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_domain": row["domain"],
                    "blinded_case_id": row["blinded_case_id"],
                    "assignment_type": "core_400_casebook",
                    "case_domain": row["domain"],
                    "gene": row["gene"],
                    "variant_identifier": row["variant_identifier"],
                    "baseline_condition_label": row["baseline_condition_label"],
                    "target_context": row["target_context"],
                    "primary_expert_endpoint_question": row["primary_expert_endpoint_question"],
                    "arrhythmia_high_value_gene": row["arrhythmia_high_value_gene"],
                    "sads_cpvt_priority": row["sads_cpvt_priority"],
                    "portable_without_additional_interpretation": "",
                    "confidence_1_to_5": "",
                    "requires_disease_specific_review": "",
                    "requires_population_or_penetrance_context": "",
                    "requires_genotype_first_or_trigger_context": "",
                    "not_adjudicable": "",
                    "not_adjudicable_reason": "",
                    "short_rationale": "",
                    "review_minutes": "",
                }
            )
    assignments = pd.DataFrame(assignment_rows)

    consensus = blind[
        [
            "blinded_case_id",
            "domain",
            "gene",
            "variant_identifier",
            "target_context",
            "arrhythmia_high_value_gene",
            "sads_cpvt_priority",
        ]
    ].copy()
    consensus["expert_reviewer_N_expected"] = consensus["domain"].map(lambda d: len(REVIEWERS.get(str(d), [])))
    consensus["expert_reviewer_N_completed"] = ""
    consensus["portable_yes_N"] = ""
    consensus["portable_no_N"] = ""
    consensus["consensus_portable_without_additional_interpretation"] = ""
    consensus["consensus_rule"] = "majority_yes_no; tie_or_low_confidence_goes_to_panel_resolution"
    consensus["consensus_short_rationale"] = ""

    return blind, endpoint_key, assignments, consensus


def build_sads_priority_packet() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SADS_CASES.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sads = pd.read_csv(SADS_CASES).copy()
    sads = sads.head(50)
    sads["blinded_case_id"] = [f"SADS_PRIORITY_{i:03d}" for i in range(1, len(sads) + 1)]
    sads["target_context"] = "SADS, molecular autopsy, family-risk, genotype-first, or trigger-dependent reuse context"
    sads["primary_expert_endpoint_question"] = sads.get(
        "adjudication_question",
        "Can this assertion be reused in a SADS/genotype-first context without additional interpretation?",
    )
    sads["sads_cpvt_priority"] = True
    blind_keep = [
        "blinded_case_id",
        "assertion_id",
        "ClinVar VariationID",
        "gene",
        "condition label",
        "classification",
        "review status",
        "submitter count",
        "baseline environment",
        "domain",
        "target_context",
        "primary_expert_endpoint_question",
        "sads_cpvt_priority",
        "underpowered_flag",
        "individual_risk_prediction_claim",
    ]
    blind_keep = [col for col in blind_keep if col in sads.columns]
    blind = sads[blind_keep].copy()
    blind["expert_endpoint_status"] = "pending_real_expert_review"
    blind["reviewer_blinding"] = "true_blind_prospective_sads_addendum_no_CAB_prediction_or_followup_endpoint"

    key_keep = [
        "blinded_case_id",
        "assertion_id",
        "ClinVar VariationID",
        "gene",
        "condition label",
        "domain",
        "predicted_routing_action",
        "predicted_disease_architecture_regime",
        "predicted_PRF_needed",
        "predicted_review_priority_score",
        "predicted_condition_label_drift_risk",
        "predicted_cross_environment_drift_risk",
        "predicted_any_meaning_drift_risk",
    ]
    key_keep = [col for col in key_keep if col in sads.columns]
    key = sads[key_keep].copy()
    key["key_scope"] = "analyst_only_not_for_reviewers"

    assignment_rows: list[dict[str, object]] = []
    for _, row in blind.iterrows():
        for reviewer_id in REVIEWERS["inherited_arrhythmia"]:
            assignment_rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_domain": "inherited_arrhythmia",
                    "blinded_case_id": row["blinded_case_id"],
                    "assignment_type": "SADS_CPVT_priority_addendum",
                    "case_domain": row["domain"],
                    "gene": row["gene"],
                    "variant_identifier": row.get("ClinVar VariationID", ""),
                    "baseline_condition_label": row.get("condition label", ""),
                    "target_context": row["target_context"],
                    "primary_expert_endpoint_question": row["primary_expert_endpoint_question"],
                    "arrhythmia_high_value_gene": True,
                    "sads_cpvt_priority": True,
                    "portable_without_additional_interpretation": "",
                    "confidence_1_to_5": "",
                    "requires_disease_specific_review": "",
                    "requires_population_or_penetrance_context": "",
                    "requires_genotype_first_or_trigger_context": "",
                    "not_adjudicable": "",
                    "not_adjudicable_reason": "",
                    "short_rationale": "",
                    "review_minutes": "",
                }
            )
    assignments = pd.DataFrame(assignment_rows)
    return blind, key, assignments


def build_analysis_contract() -> pd.DataFrame:
    rows = [
        {
            "analysis_id": "endpoint_validation_primary",
            "question": "Does CAB routing predict expert non-portability better than ClinVar temporal drift?",
            "input_required": "completed cab_expert_endpoint_verdict_template.csv",
            "metric": "AUROC/AUPRC/Brier/precision@top10 for expert consensus endpoint",
            "claim_enabled_if_supported": "CAB endpoint validity is supported by expert portability verdicts",
        },
        {
            "analysis_id": "clinvar_drift_proxy_audit",
            "question": "Does ClinVar drift under- or over-estimate expert meaning non-portability?",
            "input_required": "expert consensus plus endpoint analyst key",
            "metric": "agreement, false-positive/false-negative drift proxy table",
            "claim_enabled_if_supported": "ClinVar drift is retained as proxy, not sole truth endpoint",
        },
        {
            "analysis_id": "regime_calibration",
            "question": "Do structural-functional, trigger-dependent, and PRF-needed regimes actually require reinterpretation?",
            "input_required": "expert consensus by CAB_regime",
            "metric": "expert non-portability rate by regime with exact CI",
            "claim_enabled_if_supported": "routing-rule enrichments are calibrated against independent expert endpoint",
        },
        {
            "analysis_id": "sads_cpvt_validation",
            "question": "Do SADS/CPVT priority cases require PRF, trigger, or genotype-first context?",
            "input_required": "completed SADS priority verdicts",
            "metric": "expert non-portability rate and reason-code distribution in SADS/CPVT addendum",
            "claim_enabled_if_supported": "SADS becomes an adjudicated high-value use case, not a future-work promise",
        },
        {
            "analysis_id": "inter_rater_reliability",
            "question": "Are expert endpoint labels stable enough for manuscript use?",
            "input_required": "three-to-five domain specialist verdicts per case",
            "metric": "percent agreement, pairwise kappa, majority strength, adjudication escalation rate",
            "claim_enabled_if_supported": "expert endpoint is reproducible enough for primary validation",
        },
    ]
    return pd.DataFrame(rows)


def write_protocol(core: pd.DataFrame, assignments: pd.DataFrame, sads: pd.DataFrame, sads_assignments: pd.DataFrame) -> None:
    domain_counts = core["domain"].value_counts().to_dict()
    assignment_counts = assignments["reviewer_id"].value_counts().sort_index().to_dict()
    sads_n = len(sads)
    lines = [
        "# CAB Expert Endpoint Validation Protocol",
        "",
        "This run replaces ClinVar temporal drift as the primary validation endpoint once real expert verdicts are collected.",
        "ClinVar drift remains an analyst-only proxy and must not be shown to reviewers.",
        "",
        "## Primary Expert Endpoint",
        "",
        "Binary verdict: can this P/LP assertion be reused in the specified target context without additional interpretation?",
        "",
        "Allowed values: yes, no, not_adjudicable.",
        "",
        "## Blinding",
        "",
        "- Reviewer packets exclude follow-up condition labels and all temporal endpoint columns.",
        "- Reviewer packets also exclude CAB regime, CAB routing action, sample bucket, and model reason codes.",
        "- Analyst endpoint and CAB prediction keys are written separately and are not reviewer inputs.",
        "",
        "## Reviewer Design",
        "",
        f"- Core casebook cases: {len(core)}.",
        f"- Domain counts: {domain_counts}.",
        f"- Reviewer assignments: {len(assignments)} rows.",
        f"- Assignment counts by reviewer: {assignment_counts}.",
        f"- SADS/CPVT priority addendum cases: {sads_n}.",
        f"- SADS/CPVT priority reviewer assignments: {len(sads_assignments)} rows.",
        "",
        "## Consensus Rule",
        "",
        "Use simple majority among completed yes/no verdicts. Ties, not-adjudicable majorities, or median confidence <3 go to panel resolution.",
        "",
        "## Manuscript Upgrade Logic",
        "",
        "1. Validate endpoint: compare CAB routing/scores against expert non-portability consensus and against ClinVar drift.",
        "2. Calibrate regimes: estimate expert non-portability rates for structural-functional overlap, trigger-dependent latent, PRF-needed, and syndrome-organ-boundary regimes.",
        "3. Convert SADS from future-work language to adjudicated evidence if the SADS/CPVT addendum has at least 40 completed expert-consensus cases.",
        "",
        "## Claim Boundary",
        "",
        "Do not report expert-endpoint performance until real specialist verdicts are entered. The current artifact is an executable run packet, not completed external validation.",
        "",
    ]
    (QC / "cab_expert_endpoint_validation_protocol.md").write_text("\n".join(lines), encoding="utf-8")


def update_indexes() -> None:
    table_index = TABLES / "final" / "TABLE_INDEX.md"
    if table_index.exists():
        existing = table_index.read_text(encoding="utf-8")
    else:
        existing = "# CAB Table Index\n\n"
    marker = "\n## Expert Endpoint Adjudication Run Tables\n"
    block = marker + """
| Table | Role | Source |
|---|---|---|
| cab_expert_endpoint_run_packet_blinded.csv | true-blind reviewer case packet | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_verdict_template.csv | reviewer assignment and verdict-entry template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_consensus_template.csv | consensus endpoint aggregation template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_temporal_endpoint_key.csv | analyst-only ClinVar drift endpoint key | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_priority_cases.csv | SADS/CPVT high-value adjudication addendum | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_verdict_template.csv | SADS/CPVT five-reviewer verdict template | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_sads_cpvt_expert_endpoint_prediction_key.csv | analyst-only SADS/CPVT CAB prediction key | scripts/build_cab_expert_endpoint_adjudication_run.py |
| cab_expert_endpoint_validation_analysis_contract.csv | endpoint-validation analysis contract | scripts/build_cab_expert_endpoint_adjudication_run.py |
"""
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + block
    else:
        existing = existing.rstrip() + "\n" + block
    table_index.write_text(existing, encoding="utf-8")


def upsert_rows(path: Path, key_col: str, rows: list[dict[str, object]]) -> None:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=list(rows[0].keys()))
    for row in rows:
        key = row[key_col]
        if key_col in df.columns and (df[key_col].astype(str) == str(key)).any():
            for col, value in row.items():
                df.loc[df[key_col].astype(str) == str(key), col] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def update_claim_and_boundary_tables() -> None:
    upsert_rows(
        TABLES / "cab_positive_claims_supported_by_new_analyses.csv",
        "claim_id",
        [
            {
                "claim_id": "adjudication_ready",
                "positive_claim": "CAB produces true-blind expert-adjudication-ready portability questions, including a five-reviewer SADS/CPVT priority path.",
                "supporting_new_analysis": "cab_expert_endpoint_run_packet_blinded; cab_expert_endpoint_verdict_template; cab_sads_cpvt_expert_endpoint_priority_cases; cab_expert_endpoint_validation_protocol",
                "key_metric": "400 core cases + 50 SADS/CPVT priority cases",
                "main_text_use": "validation-readiness evidence",
            }
        ],
    )

    upsert_rows(
        TABLES / "cab_claim_boundaries_quarantined.csv",
        "boundary_id",
        [
            {
                "boundary_id": "expert_endpoint_pending_boundary",
                "quarantined_claim": "Expert endpoint validates CAB performance",
                "where_kept": "reports/qc/cab_expert_endpoint_validation_protocol.md",
                "main_text_rule": "report only adjudication-readiness until real specialist verdicts are entered and analyzed",
            },
            {
                "boundary_id": "arrhythmia_small_n_boundary",
                "quarantined_claim": "Inherited-arrhythmia/SADS results have the same evidentiary weight as hereditary-cancer big-N results",
                "where_kept": "reports/qc/cab_arrhythmia_small_n_expert_endpoint_boundary.md",
                "main_text_rule": "main text treats arrhythmia as high-value underpowered domain with explicit expert adjudication path",
            },
        ],
    )

    upsert_rows(
        TABLES / "cab_reviewer_evidence_map.csv",
        "reviewer_issue",
        [
            {
                "reviewer_issue": "endpoint validity / expert truth",
                "upgrade_response": "true-blind expert endpoint adjudication run packet created to replace ClinVar drift as primary validation endpoint once real verdicts are returned",
                "evidence_artifacts": "cab_expert_endpoint_run_packet_blinded; cab_expert_endpoint_verdict_template; cab_expert_endpoint_validation_protocol",
            },
            {
                "reviewer_issue": "arrhythmia small-N",
                "upgrade_response": "inherited-arrhythmia signal is treated as high-value but underpowered; SADS/CPVT receives a five-reviewer priority addendum rather than big-N equivalence claims",
                "evidence_artifacts": "cab_arrhythmia_small_n_expert_endpoint_boundary; cab_sads_cpvt_expert_endpoint_priority_cases",
            },
        ],
    )

    boundary = """# Arrhythmia Small-N Expert Endpoint Boundary

Inherited arrhythmia is the conceptually highest-value stress test for CAB because it contains SADS, CPVT, trigger-dependent, genotype-first, and penetrance-sensitive portability questions.

It is also underpowered in the current historical benchmark:

- Historical benchmark N: 942 inherited-arrhythmia assertions versus 20,865 hereditary-cancer assertions.
- Baseline environment coverage: inherited-arrhythmia baseline environment is currently underresolved/unknown in the materialized benchmark.
- Cross-environment drift events in inherited arrhythmia: 0, so cross-environment AUROC is not estimable inside this domain.
- Trigger-dependent latent regime count: 45 assertions.
- Core true-blind expert packet inherited-arrhythmia cases: 66.
- SADS/CPVT priority addendum cases: 50, with 5 arrhythmia-domain reviewers per case.

Manuscript rule:

Do not claim inherited arrhythmia has the same evidentiary weight as hereditary cancer. Treat it as a high-value, underpowered, adjudication-priority domain. A positive expert endpoint result in the SADS/CPVT addendum can upgrade the section from future validation path to adjudicated high-value use case, but only after real specialist verdicts are returned and analyzed.
"""
    (QC / "cab_arrhythmia_small_n_expert_endpoint_boundary.md").write_text(boundary, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    blind, endpoint_key, assignments, consensus = build_core_packets()
    sads, sads_key, sads_assignments = build_sads_priority_packet()
    contract = build_analysis_contract()

    blind.to_csv(ADJ / "cab_expert_endpoint_run_packet_blinded.csv", index=False)
    endpoint_key.to_csv(ADJ / "cab_expert_endpoint_temporal_endpoint_key.csv", index=False)
    assignments.to_csv(ADJ / "cab_expert_endpoint_verdict_template.csv", index=False)
    consensus.to_csv(ADJ / "cab_expert_endpoint_consensus_template.csv", index=False)
    sads.to_csv(ADJ / "cab_sads_cpvt_expert_endpoint_priority_cases.csv", index=False)
    sads_key.to_csv(ADJ / "cab_sads_cpvt_expert_endpoint_prediction_key.csv", index=False)
    sads_assignments.to_csv(ADJ / "cab_sads_cpvt_expert_endpoint_verdict_template.csv", index=False)
    contract.to_csv(TABLES / "cab_expert_endpoint_validation_analysis_contract.csv", index=False)
    write_protocol(blind, assignments, sads, sads_assignments)
    update_indexes()
    update_claim_and_boundary_tables()
    print("Wrote CAB expert endpoint adjudication run package")


if __name__ == "__main__":
    main()
