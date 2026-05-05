
#!/usr/bin/env python3
"""Build CAB SADS/Postmortem Portability Framing Layer.

This is a framing/QC/evidence-positioning layer, not a clinical validation layer.

Non-negotiables:
- Do not claim CAB proves cause of death.
- Do not claim CAB predicts sudden death.
- Do not claim CAB predicts individual family-member risk.
- Do not claim CAB validates SADS genotype-first risk.
- Do not claim CAB replaces ClinGen/VCEP.
- Do not call this clinical validation.
- Treat SADS as a stress-test for portability theory, not as a new validated outcome domain.
"""

from pathlib import Path
import csv
import html
import pandas as pd

ROOT = Path.cwd()
TABLES = ROOT / "reports" / "tables"
QC = ROOT / "reports" / "qc"
FIGS = ROOT / "reports" / "figures"

OUT_USE = TABLES / "sads_cab_portability_use_cases.csv"
OUT_MEMO = QC / "sads_postmortem_portability_positioning.md"
OUT_CLAIMS = TABLES / "sads_publication_safe_claims.csv"
OUT_SUPPORT = TABLES / "sads_external_support_sources.csv"
OUT_FIG = FIGS / "sads_cab_portability_context_map.svg"
EVIDENCE_LADDER = TABLES / "cab_evidence_ladder_final.csv"


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def build_use_cases():
    rows = [
        {
            "use_case": "postmortem_assertion_portability",
            "source_context": "P/LP inherited-arrhythmia or channelopathy assertion.",
            "target_context": "autopsy-negative sudden death / SADS causal explanation.",
            "portability_question": "Can inherited-arrhythmia pathogenic meaning travel into cause-of-death relevance'",
            "disease_architecture_regime": "trigger_dependent_latent / structural_functional_overlap / nonspecific_underresolved",
            "CAB_test": "test disease-model concordance, trigger/phenotype-context dependence, and unsupported deterministic reuse",
            "CAB_routing_action": "disease-specific review or contextual repair",
            "PRF_required": "conditional",
            "what_CAB_can_claim": "CAB can audit whether an inherited-arrhythmia P/LP assertion is portable as a cause-of-death explanation.",
            "what_CAB_cannot_claim": "CAB proves cause of death; CAB predicts SADS outcome.",
            "external_support": "Brugada/SADS literature rationale; ClinGen channelopathy VCEP comparator",
            "claim_strength": "conceptual_high_value_use_case",
        },
        {
            "use_case": "family_screening_after_sudden_death",
            "source_context": "variant identified in deceased proband.",
            "target_context": "living relatives with genotype-first or phenotype-negative context.",
            "portability_question": "Does postmortem causal relevance travel into predictive family-risk relevance'",
            "disease_architecture_regime": "genotype_first_absent_phenotype / modifier_penetrance_boundary / trigger_dependent_latent",
            "CAB_test": "separate cause-of-death relevance from predictive family-risk relevance",
            "CAB_routing_action": "PRF-needed; population/penetrance review; phenotype ascertainment review; no deterministic reuse if phenotype absent",
            "PRF_required": "yes",
            "what_CAB_can_claim": "CAB separates cause-of-death relevance from predictive family-risk relevance.",
            "what_CAB_cannot_claim": "CAB predicts individual risk in relatives; CAB diagnoses relatives.",
            "external_support": "genotype-first / penetrance rationale; SADS family-screening portability logic",
            "claim_strength": "conceptual_high_value_use_case",
        },
        {
            "use_case": "disease_specific_channelopathy_interpretation",
            "source_context": "generic P/LP or ClinVar assertion.",
            "target_context": "SCN5A/channelopathy-specific disease model, including Brugada/LQTS/channelopathy contexts.",
            "portability_question": "Does generic pathogenicity travel into a specific arrhythmia disease model'",
            "disease_architecture_regime": "structural_functional_overlap / trigger_dependent_latent / phenotype_anchored_monogenic",
            "CAB_test": "audit disease-specific portability after classification",
            "CAB_routing_action": "disease-specific review",
            "PRF_required": "conditional",
            "what_CAB_can_claim": "CAB complements disease-specific curation by auditing portability after classification.",
            "what_CAB_cannot_claim": "CAB replaces ClinGen/VCEP; CAB has ClinGen validation.",
            "external_support": "ClinGen Sodium/Calcium Channel Arrhythmia VCEP; Potassium Channel Arrhythmia VCEP",
            "claim_strength": "external_comparator_supported",
        },
        {
            "use_case": "brugada_sads_architecture_boundary",
            "source_context": "SCN5A / Brugada / inherited-arrhythmia assertion.",
            "target_context": "autopsy-negative sudden death explanation or SADS family-risk context.",
            "portability_question": "Is the variant meaning monogenic, trigger-dependent, modifier-sensitive, or phenotype-context-limited'",
            "disease_architecture_regime": "trigger_dependent_latent / structural_functional_overlap / modifier_penetrance_boundary",
            "CAB_test": "identify whether deterministic reuse is unsafe without phenotype, trigger, and disease-specific context",
            "CAB_routing_action": "trigger/phenotype-context review; disease-specific review; PRF-needed for family-risk use",
            "PRF_required": "conditional_to_yes",
            "what_CAB_can_claim": "Brugada/SADS is a high-value CAB boundary case because pathogenic meaning is constrained by phenotype, trigger, and disease-model context.",
            "what_CAB_cannot_claim": "CAB validates Brugada risk prediction; CAB predicts sudden death.",
            "external_support": "Brugada/SADS rationale; SCN5A disease-spectrum curation comparator",
            "claim_strength": "external_rationale_supported",
        },
        {
            "use_case": "genotype_first_phenotype_negative_relative",
            "source_context": "familial P/LP assertion after sudden death.",
            "target_context": "living genotype-positive, phenotype-negative relative.",
            "portability_question": "Can disease meaning travel deterministically into a phenotype-negative genotype-first context'",
            "disease_architecture_regime": "genotype_first_absent_phenotype / modifier_penetrance_boundary",
            "CAB_test": "detect unsupported deterministic disease transfer and route to PRF",
            "CAB_routing_action": "PRF-needed; no deterministic reuse; phenotype ascertainment review",
            "PRF_required": "yes",
            "what_CAB_can_claim": "CAB identifies genotype-first phenotype-negative relatives as PRF-needed rather than deterministic disease-positive.",
            "what_CAB_cannot_claim": "CAB predicts individual penetrance, sudden death, or clinical outcome.",
            "external_support": "PRF conceptual layer; genotype-first portability rationale",
            "claim_strength": "conceptual_PRF_supported",
        },
    ]
    write_csv(OUT_USE, rows)


def build_memo():
    text = """# SADS/Postmortem Portability Positioning for CAB/PRF

## Core statement

SADS provides a high-value portability setting because pathogenic meaning must move across non-equivalent contexts: inherited-arrhythmia classification, postmortem causal interpretation, family screening, genotype-first living relatives, trigger/provocation dependence, and disease-specific channelopathy curation.

SADS is not simply “variant pathogenicity in a dead proband.” It is a pathogenic-meaning portability problem across non-equivalent contexts:

1. inherited-arrhythmia P/LP assertion
2. postmortem causal interpretation
3. family-risk relevance
4. genotype-first / phenotype-negative living relatives
5. trigger/provocation dependence
6. disease-specific channelopathy curation

## CAB value proposition

CAB’s SADS value proposition is not outcome prediction; it is assertion-meaning triage across postmortem, family-risk, genotype-first, trigger-dependent, and disease-specific curation contexts.

CAB does not reclassify variants and does not predict sudden death. CAB audits whether a P/LP assertion can be reused as:

- deterministic disease meaning
- cause-of-death relevance
- family-risk relevance
- PRF-style conditional liability
- or whether it requires contextual repair / disease-specific review

## Required caveat

This framework does not infer cause of death, diagnose relatives, or validate SADS risk prediction. It defines portability checks for whether an existing P/LP assertion can be reused across postmortem and family-risk contexts without unsupported deterministic meaning transfer.

## Mapping to CAB disease-architecture regimes

| CAB regime | SADS/postmortem implication |
|---|---|
| phenotype_anchored_monogenic | direct deterministic use only inside concordant self-loop |
| trigger_dependent_latent | SADS/Brugada-like context; requires trigger/phenotype-context review |
| structural_functional_overlap | SCN5A/channel genes can sit across Brugada/LQTS/DCM-like boundaries; disease-specific review |
| genotype_first_absent_phenotype | living relatives after proband death; PRF-needed, no deterministic reuse |
| modifier_penetrance_boundary | family-risk framing; conditional liability, not deterministic disease |
| nonspecific_underresolved | autopsy-negative sudden death labels; contextual repair |

## Practical interpretation

CAB can say: “this P/LP assertion is source-valid but its disease meaning may not be portable into a postmortem causal explanation or phenotype-negative family-risk setting without additional context.”

CAB cannot say: “this variant caused death,” “this relative will develop disease,” or “this family has validated SADS risk.”

## Publication-safe phrasing

SADS/postmortem genomics is a high-value CAB/PRF stress-test because it forces separation between variant pathogenicity, postmortem causal relevance, family-risk relevance, trigger-dependent liability, and disease-specific curation. CAB provides a portability audit and assertion-meaning triage layer for these transfers.
"""
    OUT_MEMO.write_text(text, encoding="utf-8")


def build_claims():
    allowed = [
        ("CAB can frame SADS as an assertion-portability problem rather than a variant-classification-only problem.", "CAB/PRF portability theory; SADS use-case table; external comparator/rationale support", "does not validate outcome prediction or cause-of-death inference"),
        ("CAB can distinguish postmortem cause-of-death relevance from predictive family-risk relevance.", "postmortem/family-risk context separation in CAB use-case framing", "does not predict individual family-member risk"),
        ("CAB can route genotype-first / phenotype-negative relatives to PRF-needed rather than deterministic disease reuse.", "PRF framing and genotype-first absent phenotype regime", "does not estimate penetrance or diagnose relatives"),
        ("CAB complements disease-specific channelopathy curation by auditing portability after classification.", "ClinGen channelopathy VCEP external comparator support", "does not replace ClinGen/VCEP or have ClinGen validation"),
        ("Brugada/SADS is a high-value portability boundary because pathogenic meaning may depend on phenotype, trigger/provocation, penetrance, and disease model.", "Brugada/SADS rationale and SCN5A/channelopathy disease-model specificity", "does not validate Brugada risk prediction"),
    ]
    rows = [
        {
            "claim_type": "allowed",
            "claim_text": claim,
            "allowed_or_forbidden": "allowed",
            "evidence_basis": basis,
            "caveat": caveat,
        }
        for claim, basis, caveat in allowed
    ]
    forbidden = [
        "CAB proves cause of death.",
        "CAB predicts sudden death.",
        "CAB predicts individual family-member risk.",
        "CAB validates SADS genotype-first risk.",
        "CAB replaces ClinGen/VCEP or expert curation.",
        "CAB is clinically validated for postmortem interpretation.",
        "CAB improves patient outcomes or family outcomes.",
    ]
    for txt in forbidden:
        rows.append({
            "claim_type": "forbidden",
            "claim_text": txt,
            "allowed_or_forbidden": "forbidden",
            "evidence_basis": "outside CAB/PRF scope",
            "caveat": "must not be used",
        })
    write_csv(OUT_CLAIMS, rows)


def build_support_sources():
    rows = [
        {
            "source_name": "ClinGen Sodium and Calcium Channel Arrhythmia VCEP",
            "source_type": "expert curation comparator",
            "relevance_to_CAB": "Shows disease/gene-specific variant interpretation exists for arrhythmia/channelopathy contexts.",
            "supports": "generic P/LP is insufficient for disease-specific meaning transfer",
            "does_not_support": "ClinGen/VCEP validates CAB",
            "citation_or_url": "https://www.clinicalgenome.org/affiliation/50160/",
            "claim_strength": "external_comparator_supported",
        },
        {
            "source_name": "ClinGen SCN5A gene-disease validity / arrhythmia spectrum resources",
            "source_type": "expert disease-specific curation resource",
            "relevance_to_CAB": "Supports disease-model specificity and spectrum interpretation around SCN5A.",
            "supports": "SCN5A assertions may require disease-context-specific portability review",
            "does_not_support": "CAB replaces ClinGen or has ClinGen validation",
            "citation_or_url": "https://www.clinicalgenome.org/affiliation/50160/",
            "claim_strength": "external_comparator_supported",
        },
        {
            "source_name": "ClinGen Potassium Channel Arrhythmia VCEP",
            "source_type": "expert curation comparator",
            "relevance_to_CAB": "Shows gene-specific channelopathy curation for KCNQ1/KCNH2/KCNE1/KCNE2/KCNJ2 contexts.",
            "supports": "channelopathy P/LP assertions require disease/gene-specific curation and portability review",
            "does_not_support": "CAB replaces VCEP or has external expert validation",
            "citation_or_url": "https://www.clinicalgenome.org/affiliation/50108/",
            "claim_strength": "external_comparator_supported",
        },
        {
            "source_name": "Brugada/SADS review literature",
            "source_type": "published literature rationale",
            "relevance_to_CAB": "Brugada/SADS illustrates autopsy-negative sudden death and incomplete rare-variant explanation.",
            "supports": "SADS is a trigger/phenotype/context-dependent portability boundary",
            "does_not_support": "CAB predicts SADS",
            "citation_or_url": "literature_review_required_for_manuscript_reference_manager",
            "claim_strength": "external_rationale_supported",
        },
        {
            "source_name": "SCN5A/Brugada variant yield literature",
            "source_type": "published literature rationale",
            "relevance_to_CAB": "Shows many Brugada cases lack rare SCN5A variants, making simple monogenic deterministic explanation insufficient.",
            "supports": "rare variant presence alone is not equivalent to cause-of-death explanation",
            "does_not_support": "CAB predicts Brugada outcome",
            "citation_or_url": "literature_review_required_for_manuscript_reference_manager",
            "claim_strength": "external_rationale_supported",
        },
    ]
    write_csv(OUT_SUPPORT, rows)


def esc(x):
    return html.escape(str(x))


def build_figure():
    width, height = 1500, 850
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="42" font-family="Arial" font-size="26" font-weight="700">SADS/Postmortem CAB Portability Context Map</text>',
        '<text x="40" y="70" font-family="Arial" font-size="14">Portability audit, not cause-of-death inference or risk prediction.</text>',
    ]

    parts.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#111"/></marker></defs>')

    x, y = 50, 120
    parts.append(f'<rect x="{x}" y="{y}" width="430" height="560" rx="18" fill="#fafafa" stroke="#111"/>')
    parts.append(f'<text x="{x+20}" y="{y+35}" font-family="Arial" font-size="18" font-weight="700">A. Context transfer</text>')
    boxes = [
        "Inherited-arrhythmia P/LP assertion",
        "Postmortem causal interpretation",
        "Family-risk interpretation",
        "Genotype-first phenotype-negative relative",
    ]
    yy = y + 80
    for i, b in enumerate(boxes):
        parts.append(f'<rect x="{x+35}" y="{yy}" width="350" height="58" rx="12" fill="#fff" stroke="#111"/>')
        parts.append(f'<text x="{x+55}" y="{yy+35}" font-family="Arial" font-size="13">{esc(b)}</text>')
        if i < len(boxes) - 1:
            parts.append(f'<line x1="{x+210}" y1="{yy+58}" x2="{x+210}" y2="{yy+88}" stroke="#111" marker-end="url(#arrow)"/>')
        yy += 100

    x2, y2 = 530, 120
    parts.append(f'<rect x="{x2}" y="{y2}" width="430" height="560" rx="18" fill="#fafafa" stroke="#111"/>')
    parts.append(f'<text x="{x2+20}" y="{y2+35}" font-family="Arial" font-size="18" font-weight="700">B. CAB routing</text>')
    routes = [
        ("direct self-loop", "only concordant phenotype loop"),
        ("trigger-context review", "Brugada/SADS-like contexts"),
        ("disease-specific review", "channelopathy/VCEP-style review"),
        ("PRF-needed", "family-risk / phenotype-negative"),
        ("no deterministic reuse", "unsupported transfer"),
    ]
    yy = y2 + 80
    for r, note in routes:
        parts.append(f'<rect x="{x2+35}" y="{yy}" width="350" height="58" rx="12" fill="#fff" stroke="#111"/>')
        parts.append(f'<text x="{x2+55}" y="{yy+25}" font-family="Arial" font-size="13" font-weight="700">{esc(r)}</text>')
        parts.append(f'<text x="{x2+55}" y="{yy+43}" font-family="Arial" font-size="11">{esc(note)}</text>')
        yy += 82

    x3, y3 = 1010, 120
    parts.append(f'<rect x="{x3}" y="{y3}" width="430" height="560" rx="18" fill="#fafafa" stroke="#111"/>')
    parts.append(f'<text x="{x3+20}" y="{y3+35}" font-family="Arial" font-size="18" font-weight="700">C. Forbidden overclaim boundary</text>')
    claims = [
        "variant classification ≠ cause of death",
        "cause-of-death relevance ≠ individual family risk",
        "P/LP assertion ≠ deterministic disease in phenotype-negative relative",
        "CAB audit ≠ clinical validation",
        "CAB complement ≠ VCEP replacement",
    ]
    yy = y3 + 80
    for c in claims:
        parts.append(f'<rect x="{x3+35}" y="{yy}" width="350" height="62" rx="12" fill="#fff" stroke="#111"/>')
        parts.append(f'<text x="{x3+55}" y="{yy+37}" font-family="Arial" font-size="12">{esc(c)}</text>')
        yy += 82

    parts.append('<text x="40" y="805" font-family="Arial" font-size="12">SADS is treated as a portability stress-test for assertion meaning, not as a validated outcome domain.</text>')
    parts.append('</svg>')
    OUT_FIG.write_text("\n".join(parts), encoding="utf-8")


def update_evidence_ladder():
    if not EVIDENCE_LADDER.exists():
        return
    row = {
        "evidence_layer": "SADS/postmortem portability use-case framing",
        "dataset": "conceptual + external comparator/rationale sources",
        "N": "not applicable",
        "result": "SADS provides a high-value assertion-portability stress-test across postmortem, family-risk, genotype-first, trigger-dependent, and disease-specific curation contexts.",
        "claim_strength": "conceptual_external_rationale_supported",
        "limitation": "No cause-of-death inference, no patient-risk prediction, no clinical validation.",
        "what_upgrades_it_further": "expert adjudication of postmortem/family screening cases; longitudinal SADS-specific assertion follow-up; curated genotype-first family datasets.",
    }
    df = pd.read_csv(EVIDENCE_LADDER, dtype=str, keep_default_na=False)
    if "evidence_layer" not in df.columns:
        return
    for k in row:
        if k not in df.columns:
            df[k] = ""
    mask = df["evidence_layer"].eq(row["evidence_layer"])
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EVIDENCE_LADDER, index=False)


def main():
    ensure_dirs()
    build_use_cases()
    build_memo()
    build_claims()
    build_support_sources()
    build_figure()
    update_evidence_ladder()
    print("CAB SADS/Postmortem Portability Framing Layer complete.")
    for p in [OUT_USE, OUT_MEMO, OUT_CLAIMS, OUT_SUPPORT, OUT_FIG]:
        print(f"  - {p.relative_to(ROOT)}")
    if EVIDENCE_LADDER.exists():
        print(f"  - updated {EVIDENCE_LADDER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
