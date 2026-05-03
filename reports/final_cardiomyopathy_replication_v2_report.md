# Final Cardiomyopathy Replication v2 Report

Analysis report, not manuscript prose.

## 1. What survived after removing leakage?
- v1 cross-environment AUROC=0.9742 is blocked; v2 recomputes all predictors baseline-only.
- Baseline-only regime, score, and enrichment outputs are the only eligible cardiomyopathy v2 predictors.

## 2. Did cardiomyopathy replicate broad meaning drift?
                               endpoint  numerator  denominator   rate  ci95_low  ci95_high endpoint_role
                  classification_change          0         4918 0.0000    0.0000     0.0000     secondary
                 condition_label_change       1901         4918 0.3865    0.3729     0.4001       primary
                cross_environment_drift        485         4918 0.0986    0.0903     0.1070       primary
         within_environment_label_drift       1416         4918 0.2879    0.2753     0.3006     secondary
                       self_loop_stable       4433         4918 0.9014    0.8930     0.9097     secondary
                   review_status_change        890         4918 0.1810    0.1702     0.1917     secondary
                 submitter_count_change       1743         4918 0.3544    0.3410     0.3678     secondary
                      any_meaning_drift       1985         4918 0.4036    0.3899     0.4173       primary
semantic_drift_without_reclassification       1901         4918 0.3865    0.3729     0.4001     secondary

## 3. Did baseline-only regimes predict condition-label drift?
- gene-only AUROC=0.6556; baseline-regime-only AUROC=0.7024; gene+baseline-regime AUROC=0.7277.

## 4. Did baseline-only regimes predict cross-environment drift?
- baseline-regime-only cross-environment AUROC=0.7713; see enrichment tests below.

## 5. Did gene+CAB-like improve over gene-only?
- See model comparison and LR-style tests.
                      endpoint                                      model    N  positive_N  AUROC  AUROC_CI95_low  AUROC_CI95_high  AUPRC  Brier_score  log_loss  calibration_slope  cross_validated_AUROC  AIC_approx  BIC_approx status
        condition_label_change                                    M0_null 4918        1901 0.5000          0.5000           0.5000 0.3865       0.2500    0.6931             0.0000                 0.5002   6821.7957   6834.7970    fit
        condition_label_change                               M1_gene_only 4918        1901 0.6556          0.6392           0.6690 0.5610       0.2200    0.6316             1.0285                 0.6443   6216.7324   6229.7337    fit
        condition_label_change                    M2_baseline_regime_only 4918        1901 0.7024          0.6861           0.7157 0.6240       0.2096    0.6105             1.0002                 0.7000   6012.6918   6038.6944    fit
        condition_label_change          M3_baseline_ClinVar_metadata_only 4918        1901 0.6120          0.5982           0.6250 0.4917       0.2363    0.6659             0.9962                 0.6086   6558.2193   6584.2219    fit
        condition_label_change               M4_gene_plus_baseline_regime 4918        1901 0.7277          0.7123           0.7411 0.6751       0.2034    0.5908             1.0334                 0.7180   5820.7897   5853.2929    fit
        condition_label_change           M5_baseline_regime_plus_metadata 4918        1901 0.7307          0.7153           0.7444 0.6468       0.2056    0.6042             0.9853                 0.7252   5957.3942   6002.8988    fit
        condition_label_change M6_gene_plus_baseline_regime_plus_metadata 4918        1901 0.7422          0.7277           0.7552 0.6889       0.1985    0.5807             1.0271                 0.7331   5727.9742   5779.9795    fit
       cross_environment_drift                                    M0_null 4918         485 0.5000          0.5000           0.5000 0.0986       0.2500    0.6931             0.0000                 0.5000   6821.7957   6834.7970    fit
       cross_environment_drift                               M1_gene_only 4918         485 0.5743          0.5501           0.5958 0.1157       0.2446    0.6789             1.0442                 0.5575   6681.5353   6694.5366    fit
       cross_environment_drift                    M2_baseline_regime_only 4918         485 0.7713          0.7555           0.7876 0.2177       0.1866    0.5364             0.9974                 0.7706   5283.9764   5309.9790    fit
       cross_environment_drift          M3_baseline_ClinVar_metadata_only 4918         485 0.5339          0.5095           0.5571 0.1050       0.2484    0.6892             0.9233                 0.5265   6787.2375   6813.2402    fit
       cross_environment_drift               M4_gene_plus_baseline_regime 4918         485 0.8339          0.8180           0.8514 0.2996       0.1677    0.4998             1.0130                 0.8213   4925.9111   4958.4144    fit
       cross_environment_drift           M5_baseline_regime_plus_metadata 4918         485 0.8255          0.8114           0.8426 0.3100       0.1772    0.5161             1.0161                 0.8181   5090.1000   5135.6046    fit
       cross_environment_drift M6_gene_plus_baseline_regime_plus_metadata 4918         485 0.8447          0.8306           0.8617 0.3301       0.1631    0.4866             1.0310                 0.8276   4802.1016   4854.1069    fit
             any_meaning_drift                                    M0_null 4918        1985 0.5000          0.5000           0.5000 0.4036       0.2500    0.6931            -0.0000                 0.4998   6821.7957   6834.7970    fit
             any_meaning_drift                               M1_gene_only 4918        1985 0.6492          0.6322           0.6625 0.5710       0.2223    0.6362             1.0268                 0.6423   6261.3920   6274.3933    fit
             any_meaning_drift                    M2_baseline_regime_only 4918        1985 0.6951          0.6791           0.7094 0.6296       0.2128    0.6164             1.0018                 0.6867   6071.1105   6097.1131    fit
             any_meaning_drift          M3_baseline_ClinVar_metadata_only 4918        1985 0.5993          0.5846           0.6112 0.5002       0.2385    0.6703             0.9922                 0.5949   6601.0920   6627.0947    fit
             any_meaning_drift               M4_gene_plus_baseline_regime 4918        1985 0.7156          0.6987           0.7305 0.6726       0.2073    0.5994             1.0278                 0.7096   5905.2144   5937.7177    fit
             any_meaning_drift           M5_baseline_regime_plus_metadata 4918        1985 0.7229          0.7073           0.7371 0.6523       0.2090    0.6108             0.9882                 0.7181   6022.0958   6067.6004    fit
             any_meaning_drift M6_gene_plus_baseline_regime_plus_metadata 4918        1985 0.7341          0.7185           0.7494 0.6890       0.2022    0.5893             1.0233                 0.7272   5812.0548   5864.0600    fit
within_environment_label_drift                                    M0_null 4918        1416 0.5000          0.5000           0.5000 0.2879       0.2500    0.6931             0.0000                 0.5003   6821.7957   6834.7970    fit
within_environment_label_drift                               M1_gene_only 4918        1416 0.6893          0.6711           0.7062 0.5013       0.2086    0.6116             1.0210                 0.6741   6019.6721   6032.6734    fit
within_environment_label_drift                    M2_baseline_regime_only 4918        1416 0.8035          0.7896           0.8163 0.6386       0.1716    0.5291             0.9937                 0.8004   5212.6810   5238.6836    fit
within_environment_label_drift          M3_baseline_ClinVar_metadata_only 4918        1416 0.6445          0.6303           0.6589 0.4249       0.2278    0.6494             1.0009                 0.6407   6395.2081   6421.2107    fit
within_environment_label_drift               M4_gene_plus_baseline_regime 4918        1416 0.8134          0.7991           0.8252 0.6887       0.1680    0.5107             1.0390                 0.8074   5032.8829   5065.3862    fit
within_environment_label_drift           M5_baseline_regime_plus_metadata 4918        1416 0.8046          0.7905           0.8167 0.6465       0.1713    0.5287             0.9924                 0.8016   5214.6906   5260.1952    fit
within_environment_label_drift M6_gene_plus_baseline_regime_plus_metadata 4918        1416 0.8155          0.8012           0.8281 0.6945       0.1664    0.5072             1.0373                 0.8075   5005.2182   5057.2234    fit
              self_loop_stable                                    M0_null 4918        4433 0.5000          0.5000           0.5000 0.9014       0.2500    0.6931             0.0000                 0.5000   6821.7957   6834.7970    fit
              self_loop_stable                               M1_gene_only 4918        4433 0.5743          0.5501           0.5958 0.9234       0.2446    0.6789             1.0442                 0.5575   6681.5352   6694.5365    fit
              self_loop_stable                    M2_baseline_regime_only 4918        4433 0.7713          0.7555           0.7876 0.9663       0.1866    0.5364             0.9974                 0.7706   5283.9764   5309.9790    fit
              self_loop_stable          M3_baseline_ClinVar_metadata_only 4918        4433 0.5339          0.5095           0.5571 0.9118       0.2484    0.6892             0.9233                 0.5265   6787.2375   6813.2402    fit
              self_loop_stable               M4_gene_plus_baseline_regime 4918        4433 0.8339          0.8180           0.8514 0.9747       0.1677    0.4998             1.0130                 0.8213   4925.9110   4958.4143    fit
              self_loop_stable           M5_baseline_regime_plus_metadata 4918        4433 0.8255          0.8114           0.8426 0.9731       0.1772    0.5161             1.0161                 0.8181   5090.1000   5135.6046    fit
              self_loop_stable M6_gene_plus_baseline_regime_plus_metadata 4918        4433 0.8447          0.8306           0.8617 0.9767       0.1631    0.4866             1.0310                 0.8276   4802.1012   4854.1064    fit

## 6. What is the safe ClinGen/VCEP/CSpec claim?
                     resource                            coverage_level  covered_assertions  total_assertions  coverage_rate                                      allowed_statement                                                                            forbidden_statement
ClinGen Gene-Disease Validity gene/gene-condition if local file present                   0              4918         0.0000              unavailable unless local join file exists VCEP validates assertions / variant-level ClinGen validation / ClinGen confirmed pathogenicity
               CMP VCEP scope           gene-level scope candidate only                1135              4918         0.2308    gene-level scope only; not variant-level validation VCEP validates assertions / variant-level ClinGen validation / ClinGen confirmed pathogenicity
              CMP CSpec scope           gene-level scope candidate only                1135              4918         0.2308    gene-level scope only; not variant-level validation VCEP validates assertions / variant-level ClinGen validation / ClinGen confirmed pathogenicity
  ClinGen Evidence Repository                             variant-level                   0              4918         0.0000 no variant-level Evidence Repository validation joined VCEP validates assertions / variant-level ClinGen validation / ClinGen confirmed pathogenicity

## 7. Does this support external domain replication or only descriptive replication?
                                                                                                       claim_text    N numerator denominator percent                                                                     model_or_statistic            CI                      p_or_FDR                                                                                                                                   source_file                                                 script                      claim_strength
               Cardiomyopathy P/LP assertions showed condition-label meaning drift despite stable classification. 4918      1901        4918   38.65                                      endpoint_count; baseline_regime_only_AUROC=0.7024 0.6861-0.7157                                             reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv; reports/tables/cardiomyopathy_model_comparison_baseline_only.csv src/run_cardiomyopathy_replication_v2_baseline_only.py         external_domain_replication
                                             Cardiomyopathy P/LP assertions showed broad assertion meaning drift. 4918      1985        4918   40.36                                      endpoint_count; baseline_regime_only_AUROC=0.6951 0.6791-0.7094                                             reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv; reports/tables/cardiomyopathy_model_comparison_baseline_only.csv src/run_cardiomyopathy_replication_v2_baseline_only.py         external_domain_replication
                                           Cardiomyopathy P/LP assertions showed cross-environment meaning drift. 4918       485        4918    9.86                                      endpoint_count; baseline_regime_only_AUROC=0.7713 0.7555-0.7876                                             reports/tables/cardiomyopathy_temporal_endpoint_counts_v2.csv; reports/tables/cardiomyopathy_model_comparison_baseline_only.csv src/run_cardiomyopathy_replication_v2_baseline_only.py             descriptive_replication
          Baseline-only CAB-like regimes stratified condition-label drift if gene+regime improves over gene-only. 4918                  4918            gene_only_AUROC=0.6556; baseline_regime_AUROC=0.7024; gene_plus_regime_AUROC=0.7277                                                                                                                          reports/tables/cardiomyopathy_model_comparison_baseline_only.csv src/run_cardiomyopathy_replication_v2_baseline_only.py    baseline_only_predictive_support
               Baseline-only CAB-like regimes stratified cross-environment drift if leakage-clean signal remains. 4918       485        4918    9.86 baseline_regime_only_AUROC=0.7713; baseline_collision_enrichment_OR=0.1870364302523011               p=0.9999999999999957; FDR=1.0 reports/tables/cardiomyopathy_model_comparison_baseline_only.csv; reports/tables/cardiomyopathy_transition_enrichment_tests_baseline_only.csv src/run_cardiomyopathy_replication_v2_baseline_only.py    baseline_only_predictive_support
  CMP VCEP/CSpec coverage is gene-level only; no variant-level ClinGen Evidence Repository validation was joined. 4918      1135        4918   23.08                                                                  gene-level scope only                                                                                                                            reports/tables/cardiomyopathy_clingen_overlay_status_clean.csv src/run_cardiomyopathy_replication_v2_baseline_only.py gene_level_external_constraint_only
Previous cardiomyopathy cross-environment AUROC=0.9742 is blocked because v1 regimes used follow-up environments. 4918                                                                                                 blocked prior metric                                                                                                                                    reports/tables/cardiomyopathy_regime_leakage_audit.csv src/run_cardiomyopathy_replication_v2_baseline_only.py                  blocked_by_leakage

## 8. What remains blocked?
- Prior v1 cross-environment AUROC=0.9742.
- Variant-level ClinGen validation.
- Full general assertion portability theory if baseline-only signals are weak.
- Any claim using follow-up labels/environments as predictors.

## Baseline-only enrichment tests
                                                          test                                exposure                 outcome  a_exposed_outcome  b_exposed_no_outcome  c_unexposed_outcome  d_unexposed_no_outcome  odds_ratio       p_value status   FDR_p_value
                 baseline_collision_enriched_cross_environment                 baseline_collision_flag cross_environment_drift                 13                   569                  472                    3864    0.187036  1.000000e+00    fit  1.000000e+00
        baseline_underresolved_enriched_condition_label_change             baseline_underresolved_flag  condition_label_change                598                  1260                 1303                    1757    0.639968  1.000000e+00    fit  1.000000e+00
          baseline_nonspecific_enriched_condition_label_change               baseline_nonspecific_flag  condition_label_change                593                  1241                 1308                    1776    0.648811  1.000000e+00    fit  1.000000e+00
                        baseline_sarcomeric_enriched_self_loop                baseline_sarcomeric_flag        self_loop_stable               1055                   101                 3378                     384    1.187415  7.800868e-02    fit  3.510391e-01
     baseline_structural_electrical_enriched_cross_environment     baseline_structural_electrical_flag cross_environment_drift                 29                  1826                  456                    2607    0.090797  1.000000e+00    fit  1.000000e+00
baseline_structural_electrical_enriched_condition_label_change     baseline_structural_electrical_flag  condition_label_change                678                  1177                 1223                    1840    0.866652  9.916041e-01    fit  1.000000e+00
           low_baseline_portability_enriched_cross_environment          low_baseline_portability_score cross_environment_drift                414                  1572                   71                    2861   10.612246 4.767844e-103    fit 4.291060e-102
      low_baseline_portability_enriched_condition_label_change          low_baseline_portability_score  condition_label_change                748                  1238                 1153                    1779    0.932240  8.856628e-01    fit  1.000000e+00
broad_ambiguous_baseline_label_enriched_condition_label_change baseline_broad_ambiguous_condition_flag  condition_label_change                391                   662                 1510                    2355    0.921155  8.810704e-01    fit  1.000000e+00

## Baseline portability score performance
               endpoint                              model    N  positive_N  AUROC  AUROC_CI95_low  AUROC_CI95_high  AUPRC  Brier_score  log_loss  calibration_slope  cross_validated_AUROC  AIC_approx  BIC_approx status
 condition_label_change baseline_nonportability_score_only 4918        1901 0.5610          0.5435           0.5738 0.4314       0.2472    0.6875             0.9740                 0.5603   6766.5123   6779.5136    fit
cross_environment_drift baseline_nonportability_score_only 4918         485 0.7665          0.7467           0.7893 0.2403       0.2044    0.5894             1.1228                 0.7643   5801.1393   5814.1406    fit
      any_meaning_drift baseline_nonportability_score_only 4918        1985 0.5508          0.5332           0.5656 0.4415       0.2480    0.6892             0.9709                 0.5507   6782.6782   6795.6796    fit

## Arrhythmia vs cardiomyopathy v2 comparison
                         domain  aligned_N                        temporal_alignment_rate  classification_change_rate  condition_label_change_rate  cross_environment_drift_rate  within_environment_label_drift_rate  self_loop_stable_rate  any_meaning_drift_rate  baseline_regime_only_AUROC_condition_label_change  gene_only_AUROC_condition_label_change  gene_plus_regime_AUROC_condition_label_change  metadata_only_AUROC_condition_label_change  baseline_regime_only_AUROC_cross_environment  gene_only_AUROC_cross_environment  gene_plus_regime_AUROC_cross_environment  cross_environment_enrichment_OR_baseline_collision  cross_environment_enrichment_OR_baseline_structural                        ClinGen_VCEP_CSpec_status
cardiomyopathy_v2_baseline_only       4918 not_recomputed_from_total_target_gene_universe                         0.0                       0.3865                        0.0986                               0.2879                 0.9014                  0.4036                                             0.7024                                  0.6556                                         0.7277                                       0.612                                        0.7713                             0.5743                                    0.8339                                            0.187036                                             0.090797 gene-level scope only; VCEP coverage rate=0.2308