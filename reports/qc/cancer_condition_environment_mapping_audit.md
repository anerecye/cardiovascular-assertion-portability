# Cancer Condition Environment Mapping Audit

Technical QC output; not manuscript prose.

- aligned cancer assertions: 20865
- baseline environment other/unknown count: 3
- follow-up environment other/unknown count: 3

## Reproducibility
- Environment mapping is implemented in `src/run_cancer_predisposition_replication_FIXED.py` via the `cancer_environment()` function.
- Synonym normalization includes Li-Fraumeni/LFS, Lynch/mismatch repair/Muir-Torre/CMMRD, PTEN/Cowden, FAP/polyposis/MUTYH-associated, breast/ovarian/HBOC, gastric, pancreatic, moderate-risk, pan-cancer/nonspecific labels.
- Failed/ambiguous mappings are preserved as `other/unknown`, not silently dropped after temporal alignment.
- Baseline portability regimes use baseline labels/environments only.

## Baseline environment distribution
environment_baseline
hereditary breast/ovarian cancer                          7343
pan-cancer / nonspecific cancer predisposition            5002
colorectal cancer / polyposis                             3190
Lynch syndrome / mismatch repair cancer predisposition    1701
breast cancer predisposition                              1461
syndromic cancer predisposition                           1129
PTEN hamartoma tumor syndrome / Cowden                     420
gastric cancer predisposition                              280
Li-Fraumeni syndrome                                       272
ovarian cancer predisposition                               59
pancreatic cancer predisposition                             4
other/unknown                                                3
moderate-risk cancer susceptibility                          1

## Leakage check
- no follow-up condition label used in baseline regime: yes
- no cross_environment_drift used in predictor: yes
- no condition_label_change used in predictor: yes
- no follow-up review status or submitter count used: yes
