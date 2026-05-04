#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.cwd()
p = ROOT / "scripts" / "resolve_arrhythmia_arr_ids_to_clinvar.py"

if not p.exists():
    raise FileNotFoundError(p)

text = p.read_text(encoding="utf-8")

replacements = {
    'final.loc[mask, "external_clinvar_match"] = True': 'final.loc[mask, "external_clinvar_match"] = "True"',
    'final.loc[mask, "arr_resolution_accepted"] = bool(row.get("arr_resolution_accepted", False))': 'final.loc[mask, "arr_resolution_accepted"] = "True" if bool(row.get("arr_resolution_accepted", False)) else "False"',
    'final.loc[mask, "arr_gene_concordant"] = bool(row.get("arr_gene_concordant", False))': 'final.loc[mask, "arr_gene_concordant"] = "True" if bool(row.get("arr_gene_concordant", False)) else "False"',
    'final[c] = False': 'final[c] = "False"',
}

changed = 0
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        changed += 1

old_block = (
    '    bool_update_cols = [\n'
    '        "external_clinvar_match",\n'
    '        "arr_resolution_accepted",\n'
    '        "arr_gene_concordant",\n'
    '    ]\n'
    '    for c in bool_update_cols:\n'
    '        if c not in final.columns:\n'
    '            final[c] = "False"\n'
)

new_block = (
    '    bool_update_cols = [\n'
    '        "external_clinvar_match",\n'
    '        "arr_resolution_accepted",\n'
    '        "arr_gene_concordant",\n'
    '    ]\n'
    '    for c in bool_update_cols:\n'
    '        if c not in final.columns:\n'
    '            final[c] = "False"\n'
    '        final[c] = final[c].astype("object")\n'
)

if old_block in text:
    text = text.replace(old_block, new_block)
    changed += 1

p.write_text(text, encoding="utf-8")

print(f"Patched {p}")
print(f"Replacement groups applied: {changed}")
print(r"Now rerun: python scripts\resolve_arrhythmia_arr_ids_to_clinvar.py")
