# CAB Operating Modes Definition

Technical definitions; not manuscript prose.

## CAB-Core
- features: gene + baseline disease-model regime
- purpose: temporal portability prediction and minimal routing
- benchmark: temporal condition-label drift gold standard
- expected behavior: reduce unsupported deterministic reuse while allowing more true portable assertions
- reporting rule: use as the primary temporal portability operating mode if it outperforms CAB-Conservative under temporal gold.

## CAB-Conservative
- features: full CAB routing flags, portability score, failure topology, repair/review flags
- purpose: safety-first triage and contextual repair
- benchmark: conservative composite routing gold standard
- expected behavior: stronger restriction, higher repair/review routing, possible overrestriction
- reporting rule: use as conservative operational stress-test / safety-first triage mode, not as the best temporal portability model if CAB-Core performs better.

## Required distinction
CAB-Core and CAB-Conservative are distinct operating modes and must not be collapsed into one claim.

## Non-negotiable reporting rules
- Do not present full CAB/CAB-Conservative as best for temporal condition-label drift if CAB-Core is better.
- Do not hide overrestriction.
- Do not call CAB-Conservative clinically validated.
- Do not treat composite routing gold as independent external validation.
- Report unsupported reuse together with false restriction, direct-use allowance, and true-portable allowance.