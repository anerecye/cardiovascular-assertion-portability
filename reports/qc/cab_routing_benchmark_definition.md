# CAB Routing Benchmark Definition

Technical benchmark definition; not manuscript prose.

## Key correction
This benchmark reports two separate gold standards and does not mix them.

## Temporal condition-label gold standard
Unsupported deterministic reuse is defined by future condition-label drift. This reproduces the original headline counterfactual routing benchmark.

## Conservative composite routing gold standard
Unsupported deterministic reuse is defined by any of: future condition-label drift, future cross-environment drift, low baseline portability, failure/regime topology, or decision-layer restriction.

## ClinVar-label-only baseline
P/LP is treated as directly portable unless raw label conflict is detected. In this materialized task table, all P/LP assertions are direct-use allowed by baseline.

## CAB routing
CAB uses baseline portability regime, portability score, disease-model environment, gene/regime architecture, and population/penetrance or expert-review flags where available.

## False-portable assertion
An assertion allowed for direct deterministic reuse despite the selected internal gold standard marking it non-portable.

## Internal vs external gold standard
Both current gold standards are internal and rule-adjudicated. Neither is an external expert-adjudicated clinical truth set.

## Limitations
- counterfactual routing benchmark only
- no clinical outcome improvement claim
- no expert-validated decision correctness claim
- no clinical actionability beyond routing
- external expert adjudication remains pending