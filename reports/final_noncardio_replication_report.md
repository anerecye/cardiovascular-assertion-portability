# Final Non-Cardiovascular Replication Report

Analysis report; not manuscript prose.

## Inputs
- data/processed/clinvar_snapshot_baseline_202301.csv
- data/processed/clinvar_snapshot_followup_202604.csv

## 1. Does hereditary cancer show meaning drift despite classification stability?
                               endpoint  numerator  denominator   rate  ci95_low  ci95_high endpoint_role
                  classification_change          0        20865 0.0000    0.0000     0.0000     secondary
                 condition_label_change       7602        20865 0.3643    0.3578     0.3709       primary
                cross_environment_drift       3378        20865 0.1619    0.1569     0.1669       primary
         within_environment_label_drift       4224        20865 0.2024    0.1970     0.2079     secondary
                       self_loop_stable      17487        20865 0.8381    0.8331     0.8431     secondary
                   review_status_change       2952        20865 0.1415    0.1368     0.1462     secondary
                 submitter_count_change       9246        20865 0.4431    0.4364     0.4499     secondary
                      any_meaning_drift       7970        20865 0.3820    0.3754     0.3886     secondary
semantic_drift_without_reclassification       7602        20865 0.3643    0.3578     0.3709     secondary

## 2. Are drift patterns structured by baseline portability regimes?
                                               test                            exposure                 outcome  a_exposed_outcome  b_exposed_no_outcome  c_unexposed_outcome  d_unexposed_no_outcome  odds_ratio       p_value status   FDR_p_value
syndrome_organ_collision_enriched_cross_environment             baseline_collision_flag cross_environment_drift                581                  1401                 2797                   16086    2.385029  3.281223e-54    fit  8.749927e-54
          tumor_spectrum_enriched_cross_environment        baseline_tumor_spectrum_flag cross_environment_drift                906                  8362                 2472                    9125    0.399947  1.000000e+00    fit  1.000000e+00
      moderate_risk_enriched_condition_label_change         baseline_moderate_risk_flag  condition_label_change               1078                  1920                 6524                   11343    0.976184  7.277542e-01    fit  9.703389e-01
               syndrome_anchored_enriched_self_loop     baseline_syndrome_anchored_flag        self_loop_stable               3021                    28                14466                    3350   24.985557 3.079843e-205    fit 1.231937e-204
        nonspecific_enriched_condition_label_change           baseline_nonspecific_flag  condition_label_change                867                  1595                 6735                   11668    0.941710  9.133808e-01    fit  1.000000e+00
    broad_ambiguous_enriched_condition_label_change baseline_broad_ambiguous_label_flag  condition_label_change               2064                  2932                 5538                   10331    1.313213  1.763197e-16    fit  3.526394e-16
         low_portability_enriched_cross_environment      low_baseline_portability_score cross_environment_drift               1960                  3045                 1418                   14442    6.555712  0.000000e+00    fit  0.000000e+00
    low_portability_enriched_condition_label_change      low_baseline_portability_score  condition_label_change               2066                  2939                 5536                   10324    1.310939  2.609724e-16    fit  4.175559e-16

## 3. Does regime-only or gene+regime improve over gene-only?
- condition_label_change: gene-only AUROC=0.6353; regime-only AUROC=0.6467; gene+regime AUROC=0.6951.
- cross_environment_drift: gene-only AUROC=0.7907; regime-only AUROC=0.7636; gene+regime AUROC=0.8732.

## 4. Does the portability principle replicate outside cardiovascular genetics?
- Supported only if hereditary cancer shows nonzero meaning drift plus baseline-only regime stratification in the tables above.

## 5. Is this enough for general assertion portability theory?
- No. If hereditary cancer supports replication, this is three-domain evidence, not all-disease generalization.

## Blocked claims
- no future-label leakage
- no all-disease claim
- no clinical actionability claim
- no variant reclassification claim
- no expert validation claim

## Three-domain summary
                          domain  aligned_N  condition_label_change_rate  cross_environment_drift_rate  classification_change_rate  any_meaning_drift_rate  self_loop_stable_rate  gene_only_AUROC  regime_only_AUROC  gene_plus_regime_AUROC                                                                                                                                                                       primary_unstable_grammar                                            primary_stable_grammar                                                                                                                        external_constraint_available                          claim_strength
            inherited_arrhythmia      942.0                       0.3875                        0.1550                      0.0998                  0.4501                 0.8450           0.7659             0.7655                  0.8063 CAB architecture decomposes gene-level drift; low portability and disease-model collision enrich cross-environment transitions; counterfactual routing reduces unsupported deterministic reuse                            domain-specific self-loop architecture AlphaMissense missense-only comparator: CAB stronger than AlphaMissense-only for condition drift; protein deleteriousness not sufficient explanation             previously_supported_domain
                  cardiomyopathy     4918.0                       0.3865                        0.0986                      0.0000                  0.4036                 0.9014           0.6556             0.7024                  0.7277                                                                            baseline-only cardiomyopathy regimes and low baseline portability stratify future condition/cross-environment drift                            domain-specific self-loop architecture              CMP VCEP scope: 1135/4918 (0.2308) gene-level scope candidate only; CMP CSpec scope: 1135/4918 (0.2308) gene-level scope candidate only             previously_supported_domain
hereditary_cancer_predisposition    20865.0                       0.3643                        0.1619                      0.0000                  0.3820                 0.8381           0.6353             0.6467                  0.6951                                                                                             syndrome/organ/tumor-spectrum/moderate-risk grammar; low portability OR=6.555712270803949, FDR=0.0 syndrome-anchored or organ-specific self-loop states if supported                                                                                                               not joined; no expert validation claim noncardiovascular_replication_candidate