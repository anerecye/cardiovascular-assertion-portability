from pathlib import Path

p = Path("scripts/resolve_arrhythmia_arr_ids_to_clinvar.py")
text = p.read_text(encoding="utf-8")

old = '''    if "gene" in resolved.columns and "arr_clinvar_GeneSymbol" in resolved.columns:
        resolved["arr_gene_concordant"] = (
            resolved["gene"].astype(str).str.upper().str.strip()
            == resolved["arr_clinvar_GeneSymbol"].astype(str).str.upper().str.strip()
        )
    else:
        resolved["arr_gene_concordant"] = False
'''

new = '''    def gene_token_concordant(local_gene, clinvar_gene_field):
        local = str(local_gene or "").upper().strip()
        tokens = {
            t.strip().upper()
            for t in str(clinvar_gene_field or "").replace(",", ";").split(";")
            if t.strip()
        }
        return local in tokens

    if "gene" in resolved.columns and "arr_clinvar_GeneSymbol" in resolved.columns:
        resolved["arr_gene_concordant"] = resolved.apply(
            lambda r: gene_token_concordant(r.get("gene", ""), r.get("arr_clinvar_GeneSymbol", "")),
            axis=1,
        )
    else:
        resolved["arr_gene_concordant"] = False
'''

if old not in text:
    raise SystemExit("Target arr_gene_concordant block not found. Nothing patched.")

text = text.replace(old, new)

marker = '''    resolved.loc[
        resolved["arr_resolution_accepted"] & local_gene_missing,
        "arr_resolution_note",
    ] = "ARR_suffix_resolved_to_ClinVar_VariationID_local_gene_missing_or_unavailable"
'''

insert = marker + '''

    def arr_phenotype_domain_note(phenotype):
        ptxt = str(phenotype or "").lower()
        arrhythmia_terms = [
            "long qt", "lqts", "arrhythmia", "brugada", "cpvt",
            "catecholaminergic", "sudden", "cardiac", "tachycardia",
            "fibrillation", "conduction"
        ]
        if any(t in ptxt for t in arrhythmia_terms):
            return "ClinVar phenotype label appears arrhythmia-relevant"
        return "ClinVar phenotype label is not inherited-arrhythmia specific; retain as source match with phenotype-domain discordance flag"

    if "arr_clinvar_PhenotypeList" in resolved.columns:
        resolved["arr_phenotype_domain_note"] = resolved["arr_clinvar_PhenotypeList"].map(arr_phenotype_domain_note)
    else:
        resolved["arr_phenotype_domain_note"] = ""
'''

if marker in text and "arr_phenotype_domain_note" not in text:
    text = text.replace(marker, insert)

old_cols = '''        "candidate_clinvar_variation_id",
        "arr_resolution_note",
    ] + CLINVAR_COLS
'''

new_cols = '''        "candidate_clinvar_variation_id",
        "arr_resolution_note",
        "arr_phenotype_domain_note",
    ] + CLINVAR_COLS
'''

if old_cols in text:
    text = text.replace(old_cols, new_cols)

old_update = '''        final.loc[mask, "arr_resolution_note"] = str(row.get("arr_resolution_note", ""))
'''

new_update = '''        final.loc[mask, "arr_resolution_note"] = str(row.get("arr_resolution_note", ""))
        final.loc[mask, "arr_phenotype_domain_note"] = str(row.get("arr_phenotype_domain_note", ""))
'''

if old_update in text and 'final.loc[mask, "arr_phenotype_domain_note"]' not in text:
    text = text.replace(old_update, new_update)

p.write_text(text, encoding="utf-8")
print("Patched ARR gene concordance to tokenized matching.")
