# Domain-calibrated CAB-Balanced v2

This is a separate calibration layer. It does not alter the base CAB routing rules.

## Main comparison

| policy                         |   direct_use_rate |   unsupported_reuse_E3_rate |   overrestriction_rate_E3E4_portable |   true_portable_allowance_rate |   E4_false_direct_use_rate_all |
|:-------------------------------|------------------:|----------------------------:|-------------------------------------:|-------------------------------:|-------------------------------:|
| CAB-Balanced-global            |          0.273115 |                   0.0274275 |                             0.604303 |                       0.289047 |                      0.0274275 |
| CAB-Balanced-domain-calibrated |          0.546155 |                   0.034724  |                             0.338559 |                       0.60169  |                      0.034724  |
| CAB-SADS-high-stringency       |          0.540243 |                   0.0346118 |                             0.344359 |                       0.594867 |                      0.0346118 |

## BRCA1/2-like stable stratum

| policy                         |   direct_use_rate |   overrestriction_rate_E3E4_portable |   E4_false_direct_use_rate_all |
|:-------------------------------|------------------:|-------------------------------------:|-------------------------------:|
| CAB-Balanced-global            |          0        |                            1         |                              0 |
| CAB-Balanced-domain-calibrated |          0.906374 |                            0.0936263 |                              0 |
| CAB-SADS-high-stringency       |          0.906374 |                            0.0936263 |                              0 |

## SADS-sensitive stratum

| policy                         |   direct_use_rate |   SADS_false_direct_use_rate_among_SADS_direct |   E4_false_direct_use_rate_all |
|:-------------------------------|------------------:|-----------------------------------------------:|-------------------------------:|
| CAB-Balanced-global            |          0.18656  |                                     0.00755668 |                     0.00140977 |
| CAB-Balanced-domain-calibrated |          0.18656  |                                     0.00755668 |                     0.00140977 |
| CAB-SADS-high-stringency       |          0.112312 |                                     0          |                     0          |

## Repair simulation

| repair_stratum                          |   initial_review_or_repair_N |   after_repair_direct_use_N |   after_repair_still_review_N |   repair_rescue_rate |   E3_false_direct_after_repair_N |   E3_false_direct_after_repair_rate |   E4_false_direct_after_repair_N |   E4_false_direct_after_repair_rate |   stable_rescued_N |
|:----------------------------------------|-----------------------------:|----------------------------:|------------------------------:|---------------------:|---------------------------------:|------------------------------------:|---------------------------------:|------------------------------------:|-------------------:|
| all_overrestricted_hereditary_cancer    |                        14567 |                        8958 |                          5609 |             0.614952 |                              628 |                           0.0701049 |                              628 |                           0.0701049 |               8330 |
| BRCA1/2_like                            |                         8127 |                        7331 |                           796 |             0.902055 |                              102 |                           0.0139135 |                              102 |                           0.0139135 |               7229 |
| MMR_Lynch_like                          |                         3607 |                        1627 |                          1980 |             0.451067 |                              526 |                           0.323294  |                              526 |                           0.323294  |               1101 |
| stable_overrestricted_hereditary_cancer |                        11899 |                        8330 |                          3569 |             0.700059 |                                0 |                           0         |                                0 |                           0         |               8330 |
| stable_BRCA1/2_like                     |                         7841 |                        7229 |                           612 |             0.921949 |                                0 |                           0         |                                0 |                           0         |               7229 |
| stable_MMR_Lynch_like                   |                         2439 |                        1101 |                          1338 |             0.451415 |                                0 |                           0         |                                0 |                           0         |               1101 |
