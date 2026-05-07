#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"

def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")

def phase1_ladder() -> None:
    rows = [
        {
            "resource": "eMERGE SCN5A/KCNH2 JAMA 2016",
            "resource_type": "published genotype-EHR cohort study",
            "disease/domain relevance": "Inherited arrhythmia; SCN5A/KCNH2; long-QT/arrhythmia phenotype transfer",
            "data accessibility": "published article/summary; individual-level raw data not open",
            "unit of evidence": "cohort-level genotype-EHR phenotype association and laboratory classification discordance",
            "what it can validate": "external genotype-first evidence that potentially pathogenic variant status can be discordant with EHR/ECG phenotype realization in an unselected cohort",
            "what it cannot validate": "CAB output correctness, SADS prediction, patient outcomes, clinical deployment",
            "CAB/PRF use": "external validation proxy for pathogenicity-versus-phenotype-realization separation",
            "claim_strength": "external_genotype_phenotype_proxy",
            "citation": "Van Driest et al. JAMA 2016 / ACC summary: https://www.acc.org/latest-in-cardiology/journal-scans/2016/01/12/12/21/association-of-arrhythmia-related-genetic-variants",
        },
        {
            "resource": "eMERGE-III inherited arrhythmia sequencing study",
            "resource_type": "published/preprint genotype-EHR sequencing cohort with return-of-results and functional follow-up",
            "disease/domain relevance": "Inherited arrhythmia; 10 arrhythmia genes; P/LP, VUS, EHR phenotypes, RoR",
            "data accessibility": "article/supplements; individual-level raw participant data not fully open",
            "unit of evidence": "cohort-level carrier counts, EHR phenotype burden, diagnoses after RoR, VUS reclassification",
            "what it can validate": "external support for genotype-first phenotype-concordance uncertainty and reclassification as a real-world issue",
            "what it cannot validate": "CAB predictions, SADS outcomes, or automated routing correctness",
            "CAB/PRF use": "external proxy for penetrance gap, phenotype concordance uncertainty, and return-of-result context transfer",
            "claim_strength": "external_genotype_phenotype_proxy",
            "citation": "Glazer et al. eMERGE-III: https://www.medrxiv.org/content/10.1101/2021.03.30.21254549.full",
        },
        {
            "resource": "ClinGen Evidence Repository / VCEP / CSpec",
            "resource_type": "expert-curated pathogenicity classification infrastructure",
            "disease/domain relevance": "Cross-domain hereditary disease; relevant depending on VCEP scope",
            "data accessibility": "public Evidence Repository / ClinVar tags / APIs where available",
            "unit of evidence": "expert-curated variant pathogenicity classification with evidence/provenance",
            "what it can validate": "classification comparator and expert-curation benchmark for variant pathogenicity assertions",
            "what it cannot validate": "CAB portability, context transfer, penetrance realization, patient outcomes, FDA recognition for CAB",
            "CAB/PRF use": "expert-curation comparator; CAB runs after classification to evaluate reuse/routing portability",
            "claim_strength": "expert_curation_comparator",
            "citation": "ClinGen FDA recognition: https://www.clinicalgenome.org/about/fda-recognition/ ; VCEP protocol: https://www.clinicalgenome.org/docs/clingen-variant-curation-expert-panel-vcep-protocol/",
        },
        {
            "resource": "Personal Genome Project / PGP",
            "resource_type": "open-consent public genome, health, and trait resource",
            "disease/domain relevance": "Cross-domain proof-of-concept; depends on variants/traits present in public profiles",
            "data accessibility": "open public participant profiles and data; highly identifiable and participant-controlled",
            "unit of evidence": "individual-level open genotype/trait record, when public data are available",
            "what it can validate": "technical executability of CAB/PRF on open genotype-trait records",
            "what it cannot validate": "penetrance estimates, SADS risk, clinical outcome prediction, population validity",
            "CAB/PRF use": "open individual proof-of-concept demonstration only",
            "claim_strength": "open_individual_demo",
            "citation": "Harvard PGP data page: https://pgp.med.harvard.edu/data ; open consent article: https://pmc.ncbi.nlm.nih.gov/articles/PMC3978420/",
        },
        {
            "resource": "DiscovEHR / Geisinger genotype-first penetrance studies",
            "resource_type": "health-system genotype-first EHR cohort studies",
            "disease/domain relevance": "Cross-domain penetrance and phenotype-realization rationale; not CAB-specific",
            "data accessibility": "published cohort-level results; individual-level raw data generally not open",
            "unit of evidence": "genotype-first carrier ascertainment with EHR phenotypes and variable penetrance",
            "what it can validate": "external rationale that P/LP or predicted functional variation does not equal uniform phenotype realization",
            "what it cannot validate": "CAB routing correctness, CAB clinical impact, individual prognosis",
            "CAB/PRF use": "PRF external penetrance rationale; P/LP classification != penetrance != event risk",
            "claim_strength": "penetrance_rationale",
            "citation": "Dewey et al. Science/DiscovEHR: https://pubmed.ncbi.nlm.nih.gov/28008009/ ; COL4A5 genotype-first: https://pubmed.ncbi.nlm.nih.gov/39625784/",
        },
        {
            "resource": "LOVD",
            "resource_type": "public/gene-centered variant database network",
            "disease/domain relevance": "Gene-specific variant/phenotype assertion comparator if CAB genes have accessible LOVD instances",
            "data accessibility": "public web instances; accessibility and licensing vary by database/installation",
            "unit of evidence": "gene/variant-level public entries with phenotype/disease labels where available",
            "what it can validate": "feasibility of independent label comparison against ClinVar/CAB environments",
            "what it cannot validate": "clinical correctness, comprehensive coverage, CAB validation, patient outcomes",
            "CAB/PRF use": "external assertion comparator feasibility and limited manual sample comparison",
            "claim_strength": "feasibility_only",
            "citation": "LOVD home: https://www.lovd.nl/ ; LOVD 3: https://www.lovd.nl/3.0/",
        },
        {
            "resource": "GPCards",
            "resource_type": "genotype-phenotype correlation database",
            "disease/domain relevance": "Cross-domain genotype-phenotype evidence and patient-level phenotype mapping where accessible",
            "data accessibility": "published open-access database article; web accessibility should be checked at run time",
            "unit of evidence": "variant/gene/phenotype correlations from published studies",
            "what it can validate": "feasibility of phenotype-label comparison and phenotype correlation context",
            "what it cannot validate": "CAB correctness, clinical outcomes, prospective deployment",
            "CAB/PRF use": "optional comparator if accessible; do not depend on it as validation",
            "claim_strength": "feasibility_only",
            "citation": "GPCards: https://www.sciencedirect.com/science/article/pii/S2001037021000830 ; Database Commons: https://ngdc.cncb.ac.cn/databasecommons/database/id/7755",
        },
        {
            "resource": "PhysioNet ECG/rhythm datasets",
            "resource_type": "open physiological waveform datasets",
            "disease/domain relevance": "Arrhythmia phenotype layer; ECG/rhythm signals and labels",
            "data accessibility": "open or credentialed access depending on dataset",
            "unit of evidence": "ECG/rhythm labels and waveform records, usually not linked to germline variant assertions",
            "what it can validate": "phenotype-model layer feasibility and external rhythm-label vocabulary alignment",
            "what it cannot validate": "CAB genotype-to-portability claims without linked genotype/assertion data",
            "CAB/PRF use": "phenotype-side comparator only; no genotype-linked CAB validation",
            "claim_strength": "feasibility_only",
            "citation": "PhysioNet ECG arrhythmia dataset: https://physionet.org/content/ecg-arrhythmia/ ; access policies: https://www.physionet.org/physiobank/",
        },
    ]
    write_csv(TABLES / "cab_external_validation_proxy_ladder.csv", rows)

def phase2_emerge() -> None:
    rows = [
        {
            "study": "eMERGE SCN5A/KCNH2 JAMA 2016",
            "cohort_size": "2022",
            "genes": "SCN5A;KCNH2",
            "variant_carrier_counts": "223 individuals with 122 pathogenic/potentially pathogenic variants reported in summary",
            "phenotype_association_metrics": "ICD-9 arrhythmia code: 11/63 variant carriers vs 264/1959 noncarriers; ECG subset QTc not significantly different in carriers vs noncarriers in ACC summary",
            "reclassification_details": "Low laboratory concordance; only 4 variants designated by all three labs in ACC summary",
            "CAB_PRF_concept": "variant pathogenicity label / rare variant status != phenotype realization",
            "proxy_mapping": "P/LP/rare variant status; phenotype concordance uncertainty; genotype-first EHR ascertainment; portability/penetrance gap",
            "allowed_interpretation": "eMERGE supports the need to separate variant carriership from phenotype realization and provides external genotype-EHR evidence relevant to CAB/PRF.",
            "forbidden_interpretation": "CAB is validated by eMERGE; CAB predicts SADS outcomes in eMERGE.",
            "claim_strength": "external_genotype_phenotype_proxy",
            "citation": "https://www.acc.org/latest-in-cardiology/journal-scans/2016/01/12/12/21/association-of-arrhythmia-related-genetic-variants",
        },
        {
            "study": "eMERGE-III inherited arrhythmia sequencing study",
            "cohort_size": "21846",
            "genes": "ANK2;CACNA1C;KCNE1;KCNE2;KCNH2;KCNJ2;KCNQ1;LMNA;RYR2;SCN5A",
            "variant_carrier_counts": "123 individuals with P/LP variants; 1838 with ultra-rare VUS in article text",
            "phenotype_association_metrics": "P/LP carriers had higher EHR arrhythmia phenotype burden; 51 returned results; 18/51 inherited arrhythmia diagnoses; 11/18 diagnoses after return-of-results",
            "reclassification_details": "Functional study of 50 VUS reclassified 11 variants: 3 likely benign, 8 P/LP",
            "CAB_PRF_concept": "genotype-first ascertainment, phenotype concordance, return-of-results context, reclassification",
            "proxy_mapping": "supports PRF/CAB separation among P/LP status, phenotype realization, and context-dependent interpretation",
            "allowed_interpretation": "eMERGE-III provides external genotype-EHR support for phenotype-concordance uncertainty and reclassification as real-world issues.",
            "forbidden_interpretation": "CAB is validated by eMERGE-III; CAB predicts SADS or clinical outcomes.",
            "claim_strength": "external_genotype_phenotype_proxy",
            "citation": "https://www.medrxiv.org/content/10.1101/2021.03.30.21254549.full",
        },
    ]
    write_csv(TABLES / "emerge_arrhythmia_proxy_summary.csv", rows)
    write_md(QC / "emerge_proxy_interpretation.md", """
# eMERGE Arrhythmia Proxy Interpretation

## Allowed interpretation

eMERGE supports the need to separate variant carriership from phenotype realization and provides external genotype-EHR evidence relevant to CAB/PRF.

The 2016 SCN5A/KCNH2 eMERGE/JAMA analysis is useful as a negative/uncertain genotype-first phenotype-concordance proxy. The eMERGE-III arrhythmia sequencing study is useful as a positive genotype-EHR proxy showing phenotype burden, return-of-results diagnoses, and VUS reclassification.

## CAB/PRF mapping

- P/LP or rare variant status is not equivalent to phenotype realization.
- Genotype-first ascertainment creates a portability/penetrance gap.
- Reclassification and phenotype concordance are external reasons to separate pathogenicity, portability, and phenotype realization.

## Forbidden interpretation

- Do not write that CAB is validated by eMERGE.
- Do not write that CAB predicts SADS outcomes in eMERGE.
- Do not write that eMERGE provides patient outcome validation for CAB.
""")

def phase3_clingen() -> None:
    rows = [
        {
            "dimension": "primary object",
            "ClinGen_role": "expert variant pathogenicity classification",
            "CAB_PRF_role": "assertion portability and context-transfer routing after classification",
            "complementarity": "ClinGen answers whether a variant assertion is expert-curated; CAB asks whether an assertion can be deterministically reused in a new disease-model context.",
            "allowed_claim": "CAB complements ClinGen by evaluating portability after classification.",
            "forbidden_claim": "CAB replaces ClinGen; ClinGen validates CAB.",
            "citation": "https://www.clinicalgenome.org/about/fda-recognition/",
        },
        {
            "dimension": "ACMG/AMP specifications",
            "ClinGen_role": "VCEPs develop disease/gene-specific ACMG/AMP specifications and submit expert-reviewed classifications",
            "CAB_PRF_role": "uses disease-model context to route assertions before reuse; does not modify ACMG/AMP criteria",
            "complementarity": "CAB can flag context-transfer risk when a curated assertion is reused outside the curation scope.",
            "allowed_claim": "CAB is downstream of pathogenicity classification and upstream of reuse/routing.",
            "forbidden_claim": "CAB is an ACMG/AMP replacement or FDA-recognized variant database.",
            "citation": "https://www.clinicalgenome.org/docs/clingen-variant-curation-expert-panel-vcep-protocol/",
        },
        {
            "dimension": "Evidence Repository",
            "ClinGen_role": "public finalized classifications with supporting evidence/provenance; FDA-recognized scope for qualifying expert-curated human variant data",
            "CAB_PRF_role": "comparator for classification provenance; not a portability endpoint",
            "complementarity": "ClinGen/ERepo can stratify whether source pathogenicity assertions are expert-curated before CAB portability analysis.",
            "allowed_claim": "ClinGen is an expert-curation comparator for CAB inputs.",
            "forbidden_claim": "ClinGen Evidence Repository validates CAB output correctness.",
            "citation": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13069835/",
        },
    ]
    write_csv(TABLES / "cab_vs_clingen_comparator.csv", rows)
    write_md(QC / "cab_clingen_positioning.md", """
# CAB vs ClinGen Positioning

CAB evaluates assertion portability after pathogenicity classification. ClinGen provides expert variant curation infrastructure, disease/gene-specific ACMG/AMP specifications through VCEPs, curated pathogenicity classifications, and public evidence/provenance.

## Safe positioning

CAB complements ClinGen by evaluating portability after classification.

## Forbidden positioning

- CAB replaces ClinGen.
- CAB has FDA recognition.
- ClinGen validates CAB.
- CAB is an expert variant classification database.
""")

def phase4_pgp() -> None:
    rows = [
        {
            "demo_id": "PGP_source_identification_Harvard",
            "record_source": "Harvard Personal Genome Project public data/profiles",
            "open_consent_status": "open consent; non-anonymous public genome/health/trait data",
            "genotype_input_status": "public genome/genetic data source identified; no individual genome bundled in repository",
            "trait_input_status": "public trait/survey/profile source identified; no participant-level trait table bundled",
            "variant_candidate": "not selected in scaffold",
            "public_PLP_assertion_match": "not executed",
            "CAB_portability_output": "pipeline-ready; not run on named participant in this scaffold",
            "PRF_readiness_output": "pipeline-ready; phenotype concordance categories: concordant/absent/insufficient/unknown",
            "phenotype_record_class": "not evaluated",
            "privacy_note": "Do not reidentify beyond public PGP terms; do not bundle participant-level data unless explicitly needed and permitted.",
            "allowed_claim": "CAB/PRF is executable on open genotype-trait records.",
            "forbidden_claim": "PGP validates penetrance; PGP validates SADS risk; CAB predicts individual clinical outcome.",
            "claim_strength": "open_individual_demo",
            "citation": "https://pgp.med.harvard.edu/data",
        },
        {
            "demo_id": "PGP_source_identification_PGP_UK",
            "record_source": "Personal Genome Project UK",
            "open_consent_status": "open access/open research resource",
            "genotype_input_status": "PGP-UK open multi-omics/genome resource identified",
            "trait_input_status": "health/trait data resource identified at dataset level",
            "variant_candidate": "not selected in scaffold",
            "public_PLP_assertion_match": "not executed",
            "CAB_portability_output": "pipeline-ready; not run on named participant in this scaffold",
            "PRF_readiness_output": "pipeline-ready; phenotype concordance categories: concordant/absent/insufficient/unknown",
            "phenotype_record_class": "not evaluated",
            "privacy_note": "Use only public/open-consent data and minimize participant-level disclosure in repository artifacts.",
            "allowed_claim": "CAB/PRF proof-of-concept can be run on open individual genotype-trait records when a public profile is selected.",
            "forbidden_claim": "PGP validates penetrance or individual event risk.",
            "claim_strength": "open_individual_demo",
            "citation": "https://www.nature.com/articles/s41597-019-0205-4",
        },
    ]
    write_csv(TABLES / "pgp_cab_prf_demo_results.csv", rows)
    write_md(QC / "pgp_proof_of_concept_limitations.md", """
# PGP Proof-of-Concept Limitations

This repository does not bundle participant-level PGP records. It identifies public/open-consent sources and defines the executable proof-of-concept pathway.

## POC pipeline

1. Select an explicitly public PGP profile with available genotype and health/trait data.
2. Extract candidate variants in CAB/PRF genes.
3. Match candidate variant to public P/LP assertion where present.
4. Run CAB portability mode.
5. Run PRF readiness check.
6. Classify phenotype record as `concordant`, `absent`, `insufficient`, or `unknown`.
7. Report only minimum necessary details.

## Allowed claim

CAB/PRF is executable on open genotype-trait records.

## Forbidden claims

- PGP validates penetrance.
- PGP validates SADS risk.
- CAB predicts individual clinical outcome.
- PGP provides clinical validation for CAB.
""")

def phase5_penetrance() -> None:
    rows = [
        {
            "resource": "DiscovEHR Science 2016",
            "cohort_or_population": "50,726 whole-exome sequences from DiscovEHR study",
            "genotype_first_feature": "functional variants ascertained in a health-system-linked sequencing cohort",
            "phenotype_realization_observation": "health-system genotype-first analyses show clinical impact/phenotypic associations vary by gene/variant and context",
            "PRF_principle_supported": "P/LP or functional variant status does not equal penetrance or event risk",
            "CAB_PRF_use": "external penetrance rationale; not a CAB validation dataset",
            "claim_strength": "penetrance_rationale",
            "citation": "https://pubmed.ncbi.nlm.nih.gov/28008009/",
        },
        {
            "resource": "Geisinger MyCode/DiscovEHR COL4A5 genotype-first study",
            "cohort_or_population": "170,856 unselected health-system patients in PubMed/source summaries",
            "genotype_first_feature": "COL4A5 P/LP or protein-truncating variant carriers identified by genotype",
            "phenotype_realization_observation": "penetrance and severity varied by genotype/sex; many carriers lacked known Alport/thin basement membrane diagnosis in source summary",
            "PRF_principle_supported": "pathogenic variant label is not identical to realized phenotype, diagnosis, or event risk",
            "CAB_PRF_use": "PRF rationale for separating pathogenicity, penetrance, and clinical-event readiness",
            "claim_strength": "penetrance_rationale",
            "citation": "https://pubmed.ncbi.nlm.nih.gov/39625784/",
        },
        {
            "resource": "Genome-first myeloid malignancy predisposition studies",
            "cohort_or_population": "population-based cohorts including DiscovEHR and UK Biobank",
            "genotype_first_feature": "pathogenic germline variants in predisposition genes ascertained independent of phenotype",
            "phenotype_realization_observation": "elevated risk exists at cohort level, but penetrance/event realization is not universal",
            "PRF_principle_supported": "classification and risk must be separated from deterministic phenotype prediction",
            "CAB_PRF_use": "cross-domain penetrance rationale; not CAB validation",
            "claim_strength": "penetrance_rationale",
            "citation": "https://www.nature.com/articles/s41375-024-02436-y",
        },
    ]
    write_csv(TABLES / "genotype_first_penetrance_proxy_summary.csv", rows)
    write_md(QC / "prf_external_penetrance_rationale.md", """
# PRF External Penetrance Rationale

Published genotype-first studies support the PRF principle that pathogenicity classification, penetrance, and event risk are separate layers.

## Safe claim

Genotype-first studies support the rationale for PRF by showing that variant carriers in unselected health-system or population cohorts can have variable phenotype realization, ascertainment, and diagnosis status.

## Forbidden claim

These studies do not validate CAB or PRF output correctness, do not provide CAB clinical deployment evidence, and do not prove individual event prediction.
""")

def phase6_lovd() -> None:
    rows = [
        {
            "gene_or_resource": "LOVD network",
            "availability_check": "public LOVD installations exist; gene-specific availability must be checked per gene",
            "sample_status": "feasibility only; no bulk extraction bundled",
            "possible_comparison": "compare LOVD disease/phenotype labels against ClinVar/CAB environments for selected variants",
            "licensing_or_use_constraint": "respect per-installation terms, robots/access limits, and copyright/database rights",
            "CAB_use": "external assertion comparator feasibility",
            "limitation": "coverage is heterogeneous; entries are not a clinical validation endpoint",
            "claim_strength": "feasibility_only",
            "citation": "https://www.lovd.nl/3.0/",
        },
        {
            "gene_or_resource": "BRCA1.lovd.nl example pattern",
            "availability_check": "LOVD examples include gene-specific subdomains such as BRCA1.lovd.nl in LOVD documentation",
            "sample_status": "manual sampling feasible if access terms allow",
            "possible_comparison": "hereditary cancer label comparison for BRCA1/BRCA2-like genes",
            "licensing_or_use_constraint": "do not scrape aggressively; cite database and preserve provenance",
            "CAB_use": "external label comparator for hereditary cancer domain",
            "limitation": "not comprehensive; not a gold standard for CAB routing",
            "claim_strength": "feasibility_only",
            "citation": "https://www.lovd.nl/",
        },
        {
            "gene_or_resource": "GPCards",
            "availability_check": "published as genotype-phenotype correlation database; runtime accessibility should be verified before use",
            "sample_status": "optional comparator; no extraction bundled",
            "possible_comparison": "map variant/gene phenotype features to CAB/PRF phenotype realization layer",
            "licensing_or_use_constraint": "cite source; verify database terms before automated extraction",
            "CAB_use": "phenotype-correlation comparator feasibility",
            "limitation": "does not validate CAB output; literature-derived and heterogeneous",
            "claim_strength": "feasibility_only",
            "citation": "https://www.sciencedirect.com/science/article/pii/S2001037021000830",
        },
    ]
    write_csv(TABLES / "lovd_feasibility_and_sample_comparison.csv", rows)
    write_md(QC / "lovd_limitations.md", """
# LOVD / GPCards Comparator Limitations

LOVD and GPCards are feasibility comparators, not CAB validation sources.

## Allowed use

- Assess whether independent public variant/phenotype labels exist for CAB genes.
- Manually sample entries if licensing and access terms allow.
- Compare disease/phenotype labels against ClinVar/CAB environments.

## Forbidden use

- Do not claim LOVD validates CAB.
- Do not claim LOVD is comprehensive across CAB domains.
- Do not violate database terms, copyright, or access limits.
""")

def phase7_claims() -> None:
    rows = [
        {
            "claim_label": "external_genotype_phenotype_proxy",
            "allowed_claim": "eMERGE supports the need to separate variant carriership from phenotype realization and provides external genotype-EHR evidence relevant to CAB/PRF.",
            "evidence_source": "eMERGE SCN5A/KCNH2 JAMA 2016; eMERGE-III inherited arrhythmia sequencing study",
            "claim_strength": "external validation proxy; not CAB validation",
            "required_caveat": "Does not validate CAB outputs, SADS prediction, or patient outcomes.",
        },
        {
            "claim_label": "expert_curation_comparator",
            "allowed_claim": "CAB complements ClinGen by evaluating portability after classification.",
            "evidence_source": "ClinGen Evidence Repository / VCEP / CSpec / FDA-recognized variant data scope",
            "claim_strength": "external comparator",
            "required_caveat": "CAB does not replace ClinGen and does not have FDA recognition.",
        },
        {
            "claim_label": "open_individual_demo",
            "allowed_claim": "CAB/PRF is executable on open genotype-trait records.",
            "evidence_source": "PGP public/open-consent data sources",
            "claim_strength": "proof-of-concept demonstration",
            "required_caveat": "Does not validate penetrance, SADS risk, or individual clinical outcomes.",
        },
        {
            "claim_label": "penetrance_rationale",
            "allowed_claim": "Genotype-first penetrance studies support the PRF distinction between pathogenicity classification, penetrance, and event risk.",
            "evidence_source": "DiscovEHR/Geisinger genotype-first studies and related population cohorts",
            "claim_strength": "external rationale",
            "required_caveat": "Not a direct CAB validation test.",
        },
        {
            "claim_label": "feasibility_only",
            "allowed_claim": "LOVD, GPCards, and PhysioNet provide feasible external comparators for variant/phenotype/rhythm-label layers.",
            "evidence_source": "LOVD/GPCards/PhysioNet public resources",
            "claim_strength": "feasibility only",
            "required_caveat": "Not definitive validation; do not overclaim coverage or correctness.",
        },
        {
            "claim_label": "not_clinical_validation",
            "allowed_claim": "No fully open, no-application dataset provides definitive patient-outcome validation for CAB. However, eMERGE, ClinGen, PGP, and genotype-first penetrance studies provide an immediate external evidence stack supporting CAB's distinction between pathogenicity, portability, and phenotype realization.",
            "evidence_source": "External proxy stack",
            "claim_strength": "final cautious conclusion",
            "required_caveat": "No patient-outcome validation, no prospective deployment, no clinical validation claim.",
        },
    ]
    write_csv(TABLES / "cab_external_evidence_claims.csv", rows)
    write_md(QC / "cab_external_validation_proxy_summary.md", """
# CAB External Validation-Proxy Stack Summary

## Purpose

This external stack supports CAB/PRF concepts using public or published resources without claiming clinical validation.

## Final allowed conclusion

No fully open, no-application dataset provides definitive patient-outcome validation for CAB. However, eMERGE, ClinGen, PGP, and genotype-first penetrance studies provide an immediate external evidence stack supporting CAB's distinction between pathogenicity, portability, and phenotype realization.

## Non-negotiable caveats

- Do not claim patient outcome validation.
- Do not claim prospective clinical deployment.
- Do not claim CAB is validated by eMERGE, ClinGen, PGP, LOVD, GPCards, PhysioNet, or DiscovEHR unless CAB outputs are directly tested.
- Use `external validation proxy`, `external comparator`, or `proof-of-concept demonstration`.
""")

def main() -> None:
    ensure_dirs()
    phase1_ladder()
    phase2_emerge()
    phase3_clingen()
    phase4_pgp()
    phase5_penetrance()
    phase6_lovd()
    phase7_claims()
    outputs = [
        "reports/tables/cab_external_validation_proxy_ladder.csv",
        "reports/tables/emerge_arrhythmia_proxy_summary.csv",
        "reports/qc/emerge_proxy_interpretation.md",
        "reports/tables/cab_vs_clingen_comparator.csv",
        "reports/qc/cab_clingen_positioning.md",
        "reports/tables/pgp_cab_prf_demo_results.csv",
        "reports/qc/pgp_proof_of_concept_limitations.md",
        "reports/tables/genotype_first_penetrance_proxy_summary.csv",
        "reports/qc/prf_external_penetrance_rationale.md",
        "reports/tables/lovd_feasibility_and_sample_comparison.csv",
        "reports/qc/lovd_limitations.md",
        "reports/tables/cab_external_evidence_claims.csv",
        "reports/qc/cab_external_validation_proxy_summary.md",
    ]
    print("CAB external validation-proxy stack complete.")
    for out in outputs:
        print("  - " + out)

if __name__ == "__main__":
    main()
