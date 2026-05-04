# CAB Operating Modes Final Definition

Technical definitions; not manuscript prose.

## CAB-Strict
- features: gene + baseline disease-model regime
- behavior: high-stringency triage
- goal: minimize false portability / unsupported deterministic reuse
- limitation: high overrestriction and low direct-use allowance

## CAB-Balanced
- features: full CAB routing configuration
- behavior: balanced safety-permissiveness routing
- goal: retain large reduction in unsupported reuse while allowing more direct deterministic use
- limitation: higher unsupported reuse than CAB-Strict but less overrestriction

## ClinVar-label-only
- P/LP treated as portable direct-use by default
- behavior: maximal permissiveness
- limitation: high unsupported deterministic reuse under drift endpoints

## Traceability
Historical operating-mode names are retained only in the crosswalk and quarantine tables.

## Operating-frontier rule
CAB is an operating-frontier framework, not a single universal classifier.

## Non-negotiable reporting rules
- Do not hide that CAB-Strict overrestricts.
- Do not hide that CAB-Balanced allows more direct use but has higher unsupported reuse.
- Do not present one mode as universally optimal.
- Do not claim external decision validation.
- Do not use historical operating-mode names in headline tables or reports.