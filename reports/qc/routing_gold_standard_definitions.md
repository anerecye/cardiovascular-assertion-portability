# Routing Gold Standard Definitions

Technical definitions; not manuscript prose.

## 1. temporal_condition_label_drift_gold_standard

Definition: unsupported deterministic reuse is defined by future condition-label drift between baseline and follow-up snapshots.

Interpretation: this is the primary temporal counterfactual routing benchmark because the endpoint is external to the baseline routing decision.

Claim strength: temporal_counterfactual_benchmark.

Allowed wording: CAB reduced unsupported deterministic reuse against a future condition-label drift endpoint.

Prohibited wording: CAB reduced clinical errors; CAB improved patient outcomes; CAB produced expert-validated decisions.

## 2. conservative_composite_routing_gold_standard

Definition: unsupported deterministic reuse is defined by a broader internal routing standard including temporal drift endpoints plus baseline low-portability, failure/regime topology, and decision-layer restriction.

Interpretation: this is an internal operational stress test. It may include CAB-derived rule logic and therefore is not independent external validation.

Claim strength: internal_operational_benchmark.

Allowed wording: under a conservative composite internal routing gold standard, CAB reduced unsupported deterministic reuse.

Prohibited wording: composite benchmark is independent validation; composite benchmark is external expert validation; CAB should be used clinically without expert review.

## Shared limitations
- internal counterfactual routing benchmark
- no clinical outcome validation
- no expert adjudication yet
- no claim of deployed clinical decision support
- over-restriction and direct-use allowance must be reported