# Quickstart

```bash
python -m cab_portability.cli run --assertions-csv benchmark/inherited_arrhythmia/baseline_assertions.csv --domain inherited_arrhythmia --mode strict --output reports/workflow_simulation/demo_strict.csv
python -m cab_portability.cli validate --routing-output-csv reports/workflow_simulation/demo_strict.csv
```
