# CAB Expert Adjudication Protocol

Purpose: create blinded, expert-adjudication-ready portability questions from CAB routing outputs.

Casebook sizes:
- Blinded cases: 400
- Answer-key rows: 400
- Sampling buckets: 8

Blinding:
- Follow-up condition labels and endpoint statuses are hidden in the blinded file.
- The answer key restores follow-up labels and endpoint statuses for scoring.

Adjudication task:
For each case, decide whether the source-valid assertion can be reused as deterministic disease meaning in the target context, or whether it requires contextual repair, disease-specific review, population/penetrance review, PRF framing, or no deterministic reuse.

SADS path:
Prospective SADS stratum cases are exported separately as an explicit high-value adjudication path. The task is assertion-portability adjudication across postmortem, family-risk, genotype-first, and disease-specific curation contexts.
