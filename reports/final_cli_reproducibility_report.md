# Final CLI Reproducibility Report

## Commands used

```bash
python -m cab_portability.cli benchmark --baseline benchmark/inherited_arrhythmia/baseline_assertions.csv --followup benchmark/inherited_arrhythmia/followup_assertions.csv --domain inherited_arrhythmia --output-dir reports/workflow_simulation/inherited_arrhythmia
python -m cab_portability.cli benchmark --baseline benchmark/cardiomyopathy/baseline_assertions.csv --followup benchmark/cardiomyopathy/followup_assertions.csv --domain cardiomyopathy --output-dir reports/workflow_simulation/cardiomyopathy
python -m cab_portability.cli benchmark --baseline benchmark/hereditary_cancer/baseline_assertions.csv --followup benchmark/hereditary_cancer/followup_assertions.csv --domain hereditary_cancer --output-dir reports/workflow_simulation/hereditary_cancer
```

## Metrics reproduced

- temporal condition-label drift operating frontier
- routing metrics from CLI benchmark command

## Limitation

Full publication metrics require replacing toy benchmark skeleton inputs with full benchmark exports.
