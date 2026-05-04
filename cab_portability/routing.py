from .environment_mapping import normalize_label, infer_environment

HIGH_RISK_GENES = {
    "SCN5A", "RYR2", "DSP", "PKP2", "BRCA1", "BRCA2", "TP53", "PTEN",
    "CHEK2", "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC"
}

FAILURE_TOKENS = [
    "collision", "underresolved", "nonspecific", "penetrance", "spectrum",
    "moderate", "nonportable", "low", "recessive", "biallelic", "overlap"
]

STRICT_ALLOWED_REGIME_TOKENS = [
    "phenotype_anchored_monogenic",
    "phenotype-anchored monogenic",
    "phenotype anchored monogenic",
    "syndrome_anchored_self_loop",
    "syndrome-anchored self-loop",
]

STRICT_BLOCKING_REGIME_TOKENS = [
    "nonspecific_underresolved",
    "nonspecific/underresolved",
    "underresolved",
    "modifier_penetrance_boundary",
    "modifier/penetrance boundary",
    "penetrance",
    "structural_functional_overlap",
    "structural-functional overlap",
    "structural",
    "syndrome_organ_boundary",
    "syndrome-organ boundary",
    "trigger_dependent_latent",
    "trigger-dependent latent",
    "pleiotropic_collision",
    "pleiotropic collision",
    "genotype_first_absent_phenotype",
    "genotype-first absent phenotype",
]

BALANCED_BLOCKING_REGIME_TOKENS = [
    "nonspecific_underresolved",
    "nonspecific/underresolved",
    "underresolved",
    "genotype_first_absent_phenotype",
    "genotype-first absent phenotype",
    "no_deterministic_reuse",
    "explicit no-deterministic-reuse",
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


def architecture_text(row):
    return " ".join([
        str(get(row, ["disease_architecture_regime", "disease_architecture_regime_baseline"], "")),
        str(get(row, ["cab_portability_regime", "baseline_regime_primary", "primary_regime"], "")),
        str(get(row, ["baseline_architecture_family", "causal_architecture_category", "causal_architecture"], "")),
        str(get(row, ["routing_implication"], "")),
    ]).lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def has_any_token(text, tokens):
    compact = text.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    for token in tokens:
        if token.lower().replace("-", "_").replace("/", "_").replace(" ", "_") in compact:
            return True
    return False


def specific_environment(row):
    env = str(get(row, ["baseline_environment", "environment_baseline"], "")).strip().lower()
    condition = str(get(row, ["input_condition_label", "condition_label_baseline", "PhenotypeList", "condition", "condition_label"], "")).strip().lower()
    text = f"{env} {condition}"
    if not env:
        return False
    return not any(token in text for token in ["unknown", "not provided", "unavailable", "other/unknown", "not specified"])


def identity_concordance_block(row):
    source = parse_bool(get(row, ["source_match_accepted", "source_match_accepted_baseline", "external_clinvar_match"], "true"), default=True)
    meaning = parse_bool(get(row, ["meaning_match_accepted", "meaning_match_accepted_baseline"], "true"), default=True)
    discordant = parse_bool(get(row, ["phenotype_domain_discordance_flag"], "false"), default=False)
    if not source:
        return True, "source_match_rejected"
    if not meaning:
        return True, "meaning_match_rejected"
    if discordant:
        return True, "phenotype_domain_discordant"
    return False, ""


def final_direct_use_override(row, mode, candidate_direct, decision_source):
    if mode == "ClinVar-label-only":
        return candidate_direct, decision_source

    blocked, reason = identity_concordance_block(row)
    if blocked:
        return False, reason

    text = architecture_text(row)
    if mode == "CAB-Strict":
        if has_any_token(text, STRICT_BLOCKING_REGIME_TOKENS):
            return False, "strict_blocking_disease_architecture_regime"
        if not has_any_token(text, STRICT_ALLOWED_REGIME_TOKENS):
            return False, "strict_requires_anchored_self_loop_regime"
        if not specific_environment(row):
            return False, "strict_requires_specific_concordant_environment"
        return candidate_direct, decision_source

    if mode == "CAB-Balanced":
        if has_any_token(text, BALANCED_BLOCKING_REGIME_TOKENS):
            repaired = parse_bool(get(row, ["contextual_repair_completed", "specific_context_repaired"], ""), default=False)
            if not repaired:
                return False, "balanced_blocking_unrepaired_regime"
        return candidate_direct, decision_source

    return candidate_direct, decision_source


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
        direct, source = strict_direct_from_final_or_fallback(row)
        return final_direct_use_override(row, mode, direct, source)
    if mode == "CAB-Balanced":
        direct, source = balanced_direct_from_final_or_fallback(row)
        return final_direct_use_override(row, mode, direct, source)
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
        return "contextual_repair_or_disease_specific_review"

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
