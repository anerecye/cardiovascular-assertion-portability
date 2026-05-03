# Final CAB Readiness Report

Technical integration update; not manuscript prose.

## CAB validation equivalent
CAB's current validation equivalent is counterfactual routing intervention plus temporal/cross-domain validation.

## Gold-standard correction
The routing benchmark reports two separate internal gold standards: temporal condition-label drift and conservative composite routing. Claims must specify which gold standard is used.

## Operational intervention outcome
The routing benchmark provides an operational intervention outcome: reduced unsupported deterministic reuse in an internal counterfactual benchmark. It is not a clinical outcome.

## Remaining missing piece
External expert adjudication or adoption by a curation body remains pending.

## Non-negotiable limitations
- expert adjudication pending
- VCEP/CSpec variant-level validation blocked unless variant-level data are joined
- quarantined claims remain visible
- no clinical outcome improvement claim
- no expert-validated routing claim
- no clinically actionable decision-system claim beyond routing

## Publication-safe routing claims
                                                                                                                                                                                                                                                                                                           claim_text     N                numerator_denominator                     percent                                                                                                                        CI                                                statistic                                                                                                                                                        source_table                                               source_script                    claim_strength
          Across 26,725 temporally aligned P/LP assertions in three domains, CAB reduced unsupported deterministic reuse from 36.92% under a ClinVar-label-only baseline to 7.46% using the temporal condition-label drift gold standard, corresponding to a 29.47% absolute reduction and 79.80% relative reduction. 26725  baseline 9868/26725; CAB 1993/26725  baseline 36.92%; CAB 7.46%   baseline 36.33% to 37.49%; CAB 7.15% to 7.76%; absolute reduction 28.94% to 30.03%; relative reduction 79.04% to 80.60% absolute_reduction=0.294668; relative_reduction=0.798034 reports/tables/routing_metric_summary_by_domain.csv; reports/tables/routing_error_reduction_by_gold_standard.csv; reports/tables/routing_benchmark_bootstrap_ci.csv src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py       temporal_endpoint_supported
Across 26,725 temporally aligned P/LP assertions in three domains, CAB reduced unsupported deterministic reuse from 85.69% under a ClinVar-label-only baseline to 13.00% using the conservative composite internal routing gold standard, corresponding to a 72.69% absolute reduction and 84.83% relative reduction. 26725 baseline 22900/26725; CAB 3474/26725 baseline 85.69%; CAB 13.00% baseline 85.27% to 86.09%; CAB 12.56% to 13.39%; absolute reduction 72.14% to 73.22%; relative reduction 84.37% to 85.33% absolute_reduction=0.726885; relative_reduction=0.848297 reports/tables/routing_metric_summary_by_domain.csv; reports/tables/routing_error_reduction_by_gold_standard.csv; reports/tables/routing_benchmark_bootstrap_ci.csv src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py internal_counterfactual_benchmark
                                                                                                                                                                                         CAB routing provides operational routing support, not clinical outcome improvement or expert-validated decision correctness. 26725                       not applicable              not applicable                                                                                                            not applicable                                         scope limitation                                                                                                                      reports/qc/cab_routing_benchmark_definition.md src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py           external_expert_pending
                                                                                                                                                    Claims of expert-validated routing, patient outcome improvement, or clinically actionable decision support are blocked until external expert adjudication exists. 26725                       not applicable              not applicable                                                                                                            not applicable                                       blocked claim rule                                                                                                                  reports/tables/routing_publication_safe_claims.csv src/run_cab_routing_benchmark_publication_audit_FIXED_v2.py           external_expert_pending

## Bootstrap uncertainty
          gold_standard                          metric  estimate  ci95_low  ci95_high  bootstrap_replicates stratification
temporal_condition_gold baseline_unsupported_reuse_rate  0.369321  0.363330   0.374858                  1000  within_domain
temporal_condition_gold      cab_unsupported_reuse_rate  0.074579  0.071506   0.077643                  1000  within_domain
temporal_condition_gold              absolute_reduction  0.294743  0.289428   0.300282                  1000  within_domain
temporal_condition_gold              relative_reduction  0.798067  0.790351   0.805999                  1000  within_domain
 composite_routing_gold baseline_unsupported_reuse_rate  0.856891  0.852722   0.860882                  1000  within_domain
 composite_routing_gold      cab_unsupported_reuse_rate  0.129978  0.125613   0.133920                  1000  within_domain
 composite_routing_gold              absolute_reduction  0.726913  0.721422   0.732238                  1000  within_domain
 composite_routing_gold              relative_reduction  0.848315  0.843687   0.853288                  1000  within_domain