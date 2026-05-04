
#!/usr/bin/env python3
# Patch CAB prospective registry builder to use ClinVar snapshot-diff inclusion.

from pathlib import Path
import re
import shutil

ROOT = Path.cwd()
target = ROOT / "scripts" / "build_cab_prospective_portability_prediction_registry.py"

if not target.exists():
    raise FileNotFoundError(target)

text = target.read_text(encoding="utf-8")
backup = target.with_suffix(".py.bak_snapshotdiff")
shutil.copy2(target, backup)

needle = '''    p.add_argument("--raw-clinvar", default="",
                   help="Optional path to already downloaded current variant_summary.txt.gz.")
'''
if needle not in text:
    needle = '''    p.add_argument("--raw-clinvar", default="",
                   help="Optional path to already downloaded variant_summary.txt.gz.")
'''
replacement = '''    p.add_argument("--raw-clinvar", default="",
                   help="Optional path to already downloaded current variant_summary.txt.gz.")
    p.add_argument("--prior-clinvar", default="",
                   help="Required for locked registry: prior ClinVar variant_summary.txt.gz path or URL for snapshot-diff inclusion.")
    p.add_argument("--allow-broad-dry-run", action="store_true",
                   help="Allow broad non-benchmark dry run without prior snapshot diff. Output is NOT a locked registry.")
'''
if needle not in text:
    raise RuntimeError("Could not find argparse raw-clinvar block.")
text = text.replace(needle, replacement, 1)

needle = "def sha256_file(path: Path) -> str:\n"
insert = r'''
def fetch_path_or_url(path_or_url: str, label: str, baseline_date: str) -> Path:
    # Return local path for a path or URL.
    if not path_or_url:
        raise ValueError(f"{label} is required.")
    if re.match(r"^https?://", path_or_url):
        suffix = ".txt.gz"
        out = RAW_PROSPECTIVE / f"{label}_{baseline_date}{suffix}"
        if out.exists() and out.stat().st_size > 0:
            return out
        print(f"[download] {path_or_url} -> {out}")
        with urllib.request.urlopen(path_or_url, timeout=120) as r, open(out, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return out
    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def canonical_variant_key(df: pd.DataFrame) -> pd.Series:
    assembly = df["Assembly"].astype(str) if "Assembly" in df.columns else pd.Series([""] * len(df), index=df.index)
    return (
        df["VariationID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        + "||" + df.get("GeneSymbol", "").astype(str).str.strip()
        + "||" + df.get("PhenotypeList", "").astype(str).str.strip()
        + "||" + assembly.str.strip()
    )


def snapshot_signature(df: pd.DataFrame) -> pd.Series:
    fields = ["ClinicalSignificance", "PhenotypeList", "ReviewStatus", "NumberSubmitters", "GeneSymbol"]
    vals = []
    for f in fields:
        if f in df.columns:
            vals.append(df[f].fillna("").astype(str).str.strip())
        else:
            vals.append(pd.Series([""] * len(df), index=df.index, dtype=str))
    sig = vals[0]
    for v in vals[1:]:
        sig = sig + "||" + v
    return sig


def load_prior_snapshot_index(prior_path: Path, max_rows: int = 0) -> dict[str, str]:
    # Load prior snapshot key -> material signature for P/LP-ish rows in target domains.
    idx: dict[str, str] = {}
    total = 0
    for chunk in pd.read_csv(prior_path, sep="\t", compression="gzip", chunksize=250_000, low_memory=False, dtype=str):
        total += len(chunk)
        if max_rows and total > max_rows:
            chunk = chunk.iloc[: max(0, len(chunk) - (total - max_rows))]

        if "Assembly" in chunk.columns:
            chunk = chunk[chunk["Assembly"].astype(str).eq("GRCh38")].copy()
        if chunk.empty or "ClinicalSignificance" not in chunk.columns or "VariationID" not in chunk.columns:
            if max_rows and total >= max_rows:
                break
            continue

        chunk = chunk[chunk["ClinicalSignificance"].map(is_p_lp)].copy()
        if chunk.empty:
            if max_rows and total >= max_rows:
                break
            continue

        chunk["VariationID"] = chunk["VariationID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        chunk["GeneSymbol"] = chunk.get("GeneSymbol", "").astype(str)
        chunk["PhenotypeList"] = chunk.get("PhenotypeList", "").astype(str)
        chunk["domain"] = chunk.apply(lambda r: classify_domain(r.get("GeneSymbol", ""), r.get("PhenotypeList", "")), axis=1)
        chunk = chunk[chunk["domain"].isin(["inherited_arrhythmia", "cardiomyopathy", "hereditary_cancer"])].copy()
        if chunk.empty:
            if max_rows and total >= max_rows:
                break
            continue

        keys = canonical_variant_key(chunk)
        sigs = snapshot_signature(chunk)
        for k, s in zip(keys, sigs):
            idx[str(k)] = str(s)

        if max_rows and total >= max_rows:
            break

    return idx


'''
if needle not in text:
    raise RuntimeError("Could not find sha256_file block.")
text = text.replace(needle, insert + "\n" + needle, 1)

old_sig = "def load_clinvar_candidates(path: Path, existing_ids: set[str], original_cutoff: str, include_update: bool, max_rows: int) -> pd.DataFrame:\n"
new_sig = "def load_clinvar_candidates(path: Path, existing_ids: set[str], original_cutoff: str, include_update: bool, max_rows: int, prior_index: dict[str, str] | None = None, allow_broad_dry_run: bool = False) -> pd.DataFrame:\n"
if old_sig not in text:
    raise RuntimeError("Could not find load_clinvar_candidates signature.")
text = text.replace(old_sig, new_sig, 1)

pattern = re.compile(
    r'''        is_existing = chunk\["VariationID"\]\.isin\(existing_ids\)\n.*?        if not chunk\.empty:\n''',
    re.DOTALL
)
new_block = '''        is_existing = chunk["VariationID"].isin(existing_ids)

        if prior_index is not None:
            keys = canonical_variant_key(chunk)
            sigs = snapshot_signature(chunk)
            prior_sigs = keys.map(prior_index)
            new_key = prior_sigs.isna()
            materially_updated = (~prior_sigs.isna()) & (prior_sigs.astype(str) != sigs.astype(str))

            chunk["new_or_updated_status"] = ""
            chunk.loc[new_key, "new_or_updated_status"] = "new_in_current_snapshot"
            chunk.loc[materially_updated, "new_or_updated_status"] = "materially_updated_since_prior_snapshot"

            if include_update:
                include_mask = new_key | materially_updated
            else:
                include_mask = (new_key | materially_updated) & (~is_existing)

            chunk = chunk[include_mask].copy()
        else:
            if not allow_broad_dry_run:
                raise RuntimeError(
                    "Locked prospective registry requires --prior-clinvar for snapshot-diff inclusion. "
                    "Use --allow-broad-dry-run only for non-locked exploratory dry runs."
                )
            chunk = chunk[~is_existing].copy()
            chunk["new_or_updated_status"] = "DRYRUN_not_locked_absent_from_training_benchmark_only"

        if not chunk.empty:
'''
matches = list(pattern.finditer(text))
if not matches:
    raise RuntimeError("Could not locate filtering block in load_clinvar_candidates.")
m = matches[0]
text = text[:m.start()] + new_block + text[m.end():]

old_main = '''    raw_path = download_clinvar(args.clinvar_url, args.baseline_date, args.raw_clinvar)
    existing_ids = read_existing_benchmark_ids()
    print(f"[benchmark exclusion] existing VariationIDs loaded: {len(existing_ids):,}")

    candidates = load_clinvar_candidates(
        raw_path,
        existing_ids,
        original_cutoff=args.original_cutoff,
        include_update=args.include_update_cohort,
        max_rows=args.max_rows,
    )
'''
new_main = '''    raw_path = download_clinvar(args.clinvar_url, args.baseline_date, args.raw_clinvar)
    existing_ids = read_existing_benchmark_ids()
    print(f"[benchmark exclusion] existing VariationIDs loaded: {len(existing_ids):,}")

    prior_index = None
    prior_path = None
    if args.prior_clinvar:
        prior_path = fetch_path_or_url(args.prior_clinvar, "prior_clinvar_variant_summary", args.baseline_date)
        print(f"[prior snapshot] loading snapshot-diff index from {prior_path}")
        prior_index = load_prior_snapshot_index(prior_path, max_rows=args.max_rows)
        print(f"[prior snapshot] indexed keys: {len(prior_index):,}")
    elif not args.allow_broad_dry_run:
        raise RuntimeError(
            "For a locked prospective registry, provide --prior-clinvar and use snapshot-diff inclusion. "
            "ClinVar LastEvaluated is not a valid source-date filter."
        )

    candidates = load_clinvar_candidates(
        raw_path,
        existing_ids,
        original_cutoff=args.original_cutoff,
        include_update=args.include_update_cohort,
        max_rows=args.max_rows,
        prior_index=prior_index,
        allow_broad_dry_run=args.allow_broad_dry_run,
    )
'''
if old_main not in text:
    raise RuntimeError("Could not find main candidate loading block.")
text = text.replace(old_main, new_main, 1)

old_report = '''- downloaded raw file: `{raw_path.relative_to(ROOT)}`
- raw file SHA256: `{raw_sha}`
'''
new_report = '''- downloaded current raw file: `{raw_path.relative_to(ROOT)}`
- current raw file SHA256: `{raw_sha}`
- inclusion rule: snapshot-diff against prior ClinVar release when `--prior-clinvar` is supplied; otherwise broad dry run is not lockable
'''
if old_report in text:
    text = text.replace(old_report, new_report, 1)

text = text.replace(
    "- post-cutoff inclusion rule: all included rows require ClinVar LastEvaluated > original cutoff unless future script version explicitly documents a different source-date field\\n",
    ""
)

target.write_text(text, encoding="utf-8")

print(f"Patched {target}")
print(f"Backup written to {backup}")
print()
print("Now use snapshot-diff mode. Example:")
print('python scripts\\build_cab_prospective_portability_prediction_registry.py `')
print('  --baseline-date 2026-05-05 `')
print('  --original-cutoff 2026-04-01 `')
print('  --prior-clinvar data\\prospective\\raw\\clinvar_variant_summary_prior_2026-04.txt.gz')
