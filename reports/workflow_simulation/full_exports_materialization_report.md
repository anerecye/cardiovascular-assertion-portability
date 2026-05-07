# Full Three-Domain Benchmark Export Materialization

Source: `data\processed\cab_decision_challenge_tasks.csv`

Baseline routing inputs are baseline-only. Future endpoint labels are written only to `temporal_endpoints.csv`.

- `inherited_arrhythmia`: N=942; condition drift=0.3875; cross-environment=0.0000; any meaning=0.3875; classification change=0.0000.
- `cardiomyopathy`: N=4,918; condition drift=0.3865; cross-environment=0.0986; any meaning=0.4036; classification change=0.0000.
- `hereditary_cancer`: N=20,865; condition drift=0.3643; cross-environment=0.1619; any meaning=0.3820; classification change=0.0000.

## Non-clinical limitation

CAB is a research reference implementation for assertion portability and routing simulation. It is not a diagnostic tool, does not reclassify variants, and does not replace ACMG/AMP interpretation or expert curation.

## CLI replay

Run the benchmark command per domain:

```bash
python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia
python -m cab_portability.cli benchmark --baseline benchmark/cardiomyopathy/baseline_assertions.csv --followup benchmark/cardiomyopathy/followup_assertions.csv --domain cardiomyopathy --output-dir reports/workflow_simulation/cardiomyopathy
python -m cab_portability.cli benchmark --baseline benchmark/hereditary_cancer/baseline_assertions.csv --followup benchmark/hereditary_cancer/followup_assertions.csv --domain hereditary_cancer --output-dir reports/workflow_simulation/hereditary_cancer
```
