
#!/usr/bin/env python3
"""Patch CAB prospective registry builder to enforce post-cutoff baseline cohort.

Problem fixed:
The first registry builder excluded original benchmark VariationIDs, but for
non-benchmark rows it marked all remaining P/LP domain-matched ClinVar rows as
"new_after_original_cutoff" without requiring LastEvaluated > original cutoff.

This patch:
- backs up scripts/build_cab_prospective_portability_prediction_registry.py
- requires LastEvaluated > --original-cutoff for all included rows by default
- labels non-benchmark rows with LastEvaluated > cutoff as new_or_updated_after_original_cutoff
- labels benchmark rows with LastEvaluated > cutoff as updated_after_original_cutoff only when --include-update-cohort is used
"""

from pathlib import Path
import shutil

ROOT = Path.cwd()
target = ROOT / "scripts" / "build_cab_prospective_portability_prediction_registry.py"

if not target.exists():
    raise FileNotFoundError(target)

text = target.read_text(encoding="utf-8")
backup = target.with_suffix(".py.bak_postcutoff")
shutil.copy2(target, backup)

old = '''        is_existing = chunk["VariationID"].isin(existing_ids)
        if include_update:
            last_eval = chunk.get("LastEvaluated", pd.Series([""] * len(chunk), index=chunk.index)).astype(str)
            updated = is_existing & last_eval.map(lambda x: date_after(x, original_cutoff))
            new = ~is_existing
            chunk["new_or_updated_status"] = "new_after_original_cutoff"
            chunk.loc[updated, "new_or_updated_status"] = "updated_after_original_cutoff"
            chunk = chunk[new | updated].copy()
        else:
            chunk = chunk[~is_existing].copy()
            chunk["new_or_updated_status"] = "new_after_original_cutoff"
'''

new = '''        is_existing = chunk["VariationID"].isin(existing_ids)
        last_eval = chunk.get("LastEvaluated", pd.Series([""] * len(chunk), index=chunk.index)).astype(str)
        post_cutoff = last_eval.map(lambda x: date_after(x, original_cutoff))

        # Prospective-style baseline rule:
        # include only assertions with a baseline public evaluation/update after the original cutoff.
        # Non-benchmark rows without post-cutoff LastEvaluated are not "new after cutoff"; they are older
        # public assertions merely absent from the training benchmark and must not enter the sealed registry.
        if include_update:
            updated_existing = is_existing & post_cutoff
            new_or_updated_nonbenchmark = (~is_existing) & post_cutoff
            chunk["new_or_updated_status"] = ""
            chunk.loc[updated_existing, "new_or_updated_status"] = "updated_after_original_cutoff"
            chunk.loc[new_or_updated_nonbenchmark, "new_or_updated_status"] = "new_or_updated_after_original_cutoff"
            chunk = chunk[updated_existing | new_or_updated_nonbenchmark].copy()
        else:
            chunk = chunk[(~is_existing) & post_cutoff].copy()
            chunk["new_or_updated_status"] = "new_or_updated_after_original_cutoff"
'''

if old not in text:
    raise RuntimeError("Expected filter block not found. Patch not applied; inspect script manually.")

text = text.replace(old, new)

old_note = "- cohort N: {len(cohort):,}\n"
new_note = "- cohort N: {len(cohort):,}\n- post-cutoff inclusion rule: all included rows require ClinVar LastEvaluated > original cutoff unless future script version explicitly documents a different source-date field\n"
if new_note not in text:
    text = text.replace(old_note, new_note)

target.write_text(text, encoding="utf-8")

print(f"Patched {target}")
print(f"Backup written to {backup}")
print("Now rerun:")
print('python scripts\\build_cab_prospective_portability_prediction_registry.py `')
print('  --baseline-date 2026-05-05 `')
print('  --original-cutoff 2026-04-01')
