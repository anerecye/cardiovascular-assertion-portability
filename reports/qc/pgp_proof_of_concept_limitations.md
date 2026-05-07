# PGP Proof-of-Concept Limitations

This repository does not bundle participant-level PGP records. It identifies public/open-consent sources and defines the executable proof-of-concept pathway.

## POC pipeline

1. Select an explicitly public PGP profile with available genotype and health/trait data.
2. Extract candidate variants in CAB/PRF genes.
3. Match candidate variant to public P/LP assertion where present.
4. Run CAB portability mode.
5. Run PRF readiness check.
6. Classify phenotype record as `concordant`, `absent`, `insufficient`, or `unknown`.
7. Report only minimum necessary details.

## Allowed claim

CAB/PRF is executable on open genotype-trait records.

## Forbidden claims

- PGP validates penetrance.
- PGP validates SADS risk.
- CAB predicts individual clinical outcome.
- PGP provides clinical validation for CAB.
