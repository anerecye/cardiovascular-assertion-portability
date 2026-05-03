# CAB Decision Challenge Report

Technical benchmark report; not manuscript prose.

## Systems
- Baseline system: ClinVar-label-only reuse; P/LP is treated as portable unless raw label conflict is detected.
- CAB system: baseline portability regime, portability score, disease-model environment, gene/regime architecture, and population/penetrance flags where available.

## Evaluation endpoints
- future condition-label drift
- future cross-environment drift
- self-loop stability
- internal routing gold standard from decision layer
- expert adjudication if available in future

## Metrics
         system     N  unsupported_deterministic_reuse_rate  false_portable_rate  repair_recall  high_drift_risk_recall  cross_environment_drift_capture  self_loop_direct_reuse_rate  net_reduction_in_unsupported_reuse_vs_baseline
baseline_system 26725                              0.369242             0.369242       0.000000                0.000000                          0.00000                     1.000000                                        0.000000
            cab 26725                              0.074574             0.074574       0.870186                0.776938                          0.80818                     0.314765                                        0.294668

## Domain breakdown
              domain          system     N  unsupported_deterministic_reuse_rate  repair_recall  cross_environment_drift_capture
      cardiomyopathy baseline_system  4918                              0.386539       0.000000                         0.000000
      cardiomyopathy             cab  4918                              0.039244       0.898474                         0.948454
   hereditary_cancer baseline_system 20865                              0.364342       0.000000                         0.000000
   hereditary_cancer             cab 20865                              0.085310       0.859511                         0.788040
inherited_arrhythmia baseline_system   942                              0.387473       0.000000                         0.000000
inherited_arrhythmia             cab   942                              0.021231       0.945205                         0.000000

## Error reduction
              domain     N  baseline_unsupported_deterministic_reuse_rate  cab_unsupported_deterministic_reuse_rate  absolute_reduction  relative_reduction
                 all 26725                                       0.369242                                  0.074574            0.294668            0.798034
      cardiomyopathy  4918                                       0.386539                                  0.039244            0.347296            0.898474
   hereditary_cancer 20865                                       0.364342                                  0.085310            0.279032            0.765851
inherited_arrhythmia   942                                       0.387473                                  0.021231            0.366242            0.945205

## Claim scope
CAB converts static P/LP labels into routed portability decisions and reduces unsupported deterministic reuse compared with label-only interpretation. This is routing actionability, not clinical actionability beyond routing.