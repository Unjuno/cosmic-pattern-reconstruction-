# Independent matched-galaxy redshift-slice segregation confirmation

## Motivation

The preregistered matched DR11-extended x DESI DR1 redshift-view gate found `actual full-projection correlation < survey/program-stratified z-shuffle correlation` in both the primary and independent replication sets. That gate correctly fails its original positive-correlation hypothesis, but the observed sign has a second interpretation: shuffling redshift labels makes every z bin a random thinning of the full angular field, which tends to make the bins resemble the full projection. Real redshift-localized structures can instead make fixed-z slices **more spatially distinct from one another**.

This interpretation was recognized after seeing the earlier matched-tracer result. It is therefore not tested on those same fields as a confirmatory claim. The decision below uses a **new third independent sky set**.

## Fixed confirmation set

- Candidate centers: deterministic shuffled 10-degree sky grid, seed `20260912`.
- Each confirmation center must be >=6 degrees from every original expanded48 center, every center in the previous independent redshift replication, and every other accepted confirmation center.
- Field acceptance uses only the same DR11/DESI coverage counts as the previous replication; no correlation outcome enters selection.
- Target: 24 coverage-qualified confirmation centers.

## Matched tracer and null

Use the same one-to-one 1.5-arcsec match between high-quality DESI DR1 primary galaxies and DR11 extended `REX/EXP/DEV/SER` sources as the matched-tracer redshift gate. The observed Data Lab `type/ref_cat` content mismatch is audited field by field before using the morphology-like values.

Redshift bins remain fixed at `[0.05, 0.40, 0.80, 1.20, 1.80]`. A bin needs >=12 matched galaxies, and a field needs >=60 matched galaxies and at least two usable bins.

For every usable z bin, construct the same 32x32 local-bandpass angular map. The **primary observed statistic** is the count-weighted mean pairwise correlation between different z-bin local maps.

Generate 300 null realizations by permuting observed redshifts separately within the exact `(survey, program)` groups while leaving every angular position fixed. The field segregation effect is:

`mean(null pairwise z-slice correlation) - observed pairwise z-slice correlation`.

Positive values mean real fixed-z slices are more spatially distinct than random label partitions of the same angular catalog.

## Preregistered decision

This is a new confirmation test after the directional hypothesis was recognized.

**PASS_REDSHIFT_SLICE_SPATIAL_SEGREGATION** requires:

- at least 12 accepted confirmation fields;
- median segregation effect >= 0.02;
- one-sided sign-test p < 0.01;
- one-sided Wilcoxon p < 0.01;
- global lower-tail permutation p < 0.01.

**FAIL** is reported if the median effect <=0, either paired p>=0.10, or the global p>=0.10. Otherwise the result is UNCERTAIN.

## Guardrail

A PASS would show that observed redshift labels partition the matched galaxy angular field into more spatially distinct structures than survey/program-stratified random labels. This would be a redshift-structure finding, but it would **not** by itself prove that the original all-source DR11 hole-completion/locality signal is cosmological, nor establish matter-density reconstruction, gravity, or higher-order cosmic grammar.
