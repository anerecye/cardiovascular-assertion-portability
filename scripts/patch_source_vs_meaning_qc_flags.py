from pathlib import Path

p = Path("scripts/resolve_arrhythmia_arr_ids_to_clinvar.py")
text = p.read_text(encoding="utf-8")

# 1) Add phenotype-domain implication helpers after arr_phenotype_domain_note block.
marker = '''    if "arr_clinvar_PhenotypeList" in resolved.columns:
        resolved["arr_phenotype_domain_note"] = resolved["arr_clinvar_PhenotypeList"].map(arr_phenotype_domain_note)
    else:
        resolved["arr_phenotype_domain_note"] = ""
'''

insert = marker + '''

    def phenotype_domain_concordant_for_arrhythmia(phenotype):
        ptxt = str(phenotype or "").lower()
        arrhythmia_terms = [
            "long qt", "lqts", "arrhythmia", "brugada", "cpvt",
            "catecholaminergic", "sudden", "cardiac", "tachycardia",
            "fibrillation", "conduction", "short qt", "qt syndrome"
        ]
        return any(t in ptxt for t in arrhythmia_terms)

    if "arr_clinvar_PhenotypeList" in resolved.columns:
        resolved["phenotype_domain_concordant"] = resolved["arr_clinvar_PhenotypeList"].map(phenotype_domain_concordant_for_arrhythmia)
    else:
        resolved["phenotype_domain_concordant"] = False

    resolved["phenotype_domain_discordance_flag"] = (
        resolved["arr_resolution_accepted"] & ~resolved["phenotype_domain_concordant"]
    )
    resolved["source_match_accepted"] = resolved["arr_resolution_accepted"]
    resolved["meaning_match_accepted"] = (
        resolved["arr_resolution_accepted"] & resolved["phenotype_domain_concordant"]
    )
    resolved["routing_implication"] = "direct_source_match_only"
    resolved.loc[
        resolved["phenotype_domain_discordance_flag"],
        "routing_implication",
    ] = "contextual_repair_or_disease_specific_review"
'''

if marker in text and "phenotype_domain_discordance_flag" not in text:
    text = text.replace(marker, insert)

# 2) Ensure final output columns exist as object/string columns.
old_cols = '''        "arr_phenotype_domain_note",
    ] + CLINVAR_COLS
'''

new_cols = '''        "arr_phenotype_domain_note",
        "phenotype_domain_concordant",
        "phenotype_domain_discordance_flag",
        "source_match_accepted",
        "meaning_match_accepted",
        "routing_implication",
    ] + CLINVAR_COLS
'''

if old_cols in text:
    text = text.replace(old_cols, new_cols)

# 3) Update final accepted rows with string-safe QC flags.
old_update = '''        final.loc[mask, "arr_phenotype_domain_note"] = str(row.get("arr_phenotype_domain_note", ""))
'''

new_update = '''        final.loc[mask, "arr_phenotype_domain_note"] = str(row.get("arr_phenotype_domain_note", ""))
        final.loc[mask, "phenotype_domain_concordant"] = "True" if bool(row.get("phenotype_domain_concordant", False)) else "False"
        final.loc[mask, "phenotype_domain_discordance_flag"] = "True" if bool(row.get("phenotype_domain_discordance_flag", False)) else "False"
        final.loc[mask, "source_match_accepted"] = "True" if bool(row.get("source_match_accepted", False)) else "False"
        final.loc[mask, "meaning_match_accepted"] = "True" if bool(row.get("meaning_match_accepted", False)) else "False"
        final.loc[mask, "routing_implication"] = str(row.get("routing_implication", ""))
'''

if old_update in text and 'final.loc[mask, "phenotype_domain_discordance_flag"]' not in text:
    text = text.replace(old_update, new_update)

# 4) Add final-level defaults after ARR updates but before final.to_csv.
old_before_csv = '''    final.to_csv(OUT_FINAL, index=False)
'''

new_before_csv = '''    # Final-level QC defaults for all rows.
    if "source_match_accepted" not in final.columns:
        final["source_match_accepted"] = final["external_clinvar_match"].map(lambda x: "True" if str(x).lower() in {"true", "1", "yes"} else "False")
    else:
        blank_source = final["source_match_accepted"].astype(str).str.strip().isin({"", "nan", "None"})
        final.loc[blank_source, "source_match_accepted"] = final.loc[blank_source, "external_clinvar_match"].map(lambda x: "True" if str(x).lower() in {"true", "1", "yes"} else "False")

    if "phenotype_domain_concordant" not in final.columns:
        final["phenotype_domain_concordant"] = "True"
    final["phenotype_domain_concordant"] = final["phenotype_domain_concordant"].replace({"": "True", "nan": "True"})

    if "phenotype_domain_discordance_flag" not in final.columns:
        final["phenotype_domain_discordance_flag"] = "False"
    final["phenotype_domain_discordance_flag"] = final["phenotype_domain_discordance_flag"].replace({"": "False", "nan": "False"})

    if "meaning_match_accepted" not in final.columns:
        final["meaning_match_accepted"] = final.apply(
            lambda r: "False" if str(r.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"} else str(r.get("source_match_accepted", "False")),
            axis=1,
        )
    else:
        blank_meaning = final["meaning_match_accepted"].astype(str).str.strip().isin({"", "nan", "None"})
        final.loc[blank_meaning, "meaning_match_accepted"] = final.loc[blank_meaning].apply(
            lambda r: "False" if str(r.get("phenotype_domain_discordance_flag", "")).lower() in {"true", "1", "yes"} else str(r.get("source_match_accepted", "False")),
            axis=1,
        )

    if "routing_implication" not in final.columns:
        final["routing_implication"] = ""
    final.loc[
        final["phenotype_domain_discordance_flag"].astype(str).str.lower().isin({"true", "1", "yes"}),
        "routing_implication",
    ] = "contextual_repair_or_disease_specific_review"

    final.to_csv(OUT_FINAL, index=False)
'''

if old_before_csv in text and "Final-level QC defaults for all rows" not in text:
    text = text.replace(old_before_csv, new_before_csv)

# 5) Add discordance count to audit if possible.
old_audit = '''        "arr_resolution_accepted": arr_accepted,
        "matched_after": matched_after,
'''

new_audit = '''        "arr_resolution_accepted": arr_accepted,
        "phenotype_domain_discordance_flagged": int(final.get("phenotype_domain_discordance_flag", pd.Series(["False"] * len(final))).astype(str).str.lower().isin({"true", "1", "yes"}).sum()),
        "meaning_match_rejected": int(final.get("meaning_match_accepted", pd.Series(["True"] * len(final))).astype(str).str.lower().isin({"false", "0", "no"}).sum()),
        "matched_after": matched_after,
'''

if old_audit in text and "phenotype_domain_discordance_flagged" not in text:
    text = text.replace(old_audit, new_audit)

p.write_text(text, encoding="utf-8")
print("Patched ARR resolver with source-vs-meaning QC flags.")
