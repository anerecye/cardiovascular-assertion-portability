# Assertion Portability Formal Definition

Technical definition; not manuscript prose.

## Variant pathogenicity classification
A discrete clinical-significance label assigned to a variant, such as pathogenic or likely pathogenic.

## Variant assertion
A variant-level public claim linking a variant, gene, classification, condition label, review metadata, and submitter context at a snapshot date.

## Assertion portability
The extent to which a public variant assertion can be reused across downstream inference environments without losing or changing its disease-model interpretation.

## Disease-model environment
A domain-specific normalized clinical inference environment derived from condition labels, such as LQTS, cardiomyopathy, hereditary cancer syndrome, organ-specific cancer predisposition, or other mapped environments.

## Cross-environment drift
A temporal change where an assertion's normalized disease-model environment differs between baseline and follow-up snapshots.

## Condition-label drift
A temporal change in the assertion condition label after normalization, regardless of whether the broader disease-model environment changes.

## Self-loop stability
A temporal state where the assertion remains in the same disease-model environment between baseline and follow-up.

## Contextual repair
A routing state where reuse is not rejected, but requires added context such as disease model, phenotype environment, penetrance, population frequency, or expert disease-specific review.

## Unsupported deterministic reuse
Reuse of a public P/LP assertion as directly portable without contextual repair when baseline portability or future drift endpoints indicate the assertion should be routed or restricted.

## Formal statement
A P/LP classification is not equivalent to portable disease-model meaning. Assertion portability is the extent to which a public variant assertion can be reused across downstream inference environments without losing or changing its disease-model interpretation.