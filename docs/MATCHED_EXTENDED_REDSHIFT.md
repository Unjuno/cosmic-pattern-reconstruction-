# DR11 x DESI DR1 matched-extended redshift localization

## Question

Does the robust two-dimensional DR11 angular locality correspond to redshift-localized structure when the imaging and spectroscopic views are restricted to the **same broad observed galaxy population**?

The earlier all-source DR11 imaging vs DESI DR1 redshift-view gate did not replicate on independent sky and disappeared under survey/program-stratified redshift permutation. This experiment addresses the remaining tracer-mismatch counterhypothesis.

## Fixed design

- Primary centers: all 48 fixed `expanded48` REAL_DR11 centers.
- Independent replication centers: the same deterministic shuffled sky-grid and coverage-only selection protocol used by `redshift_view_replication.py`; 24 centers, each >=6 deg from the original set and each other.
- Imaging catalog: `ls_dr11.tractor_s`, `brick_primary=1`, fixed 0.5 deg square.
- Spectroscopic catalog: `desi_dr1.zpix`, `ZWARN=0`, `ZCAT_PRIMARY=TRUE`, `SPECTYPE='GALAXY'`, fixed z range [0.05, 1.80).
- Redshift bins fixed in advance: [0.05, 0.40, 0.80, 1.20, 1.80].

## Matched tracer

DR11 extended sources are the documented Tractor morphology classes `REX/EXP/DEV/SER`. Because live DR11 Data Lab probes on 2026-09-11 showed morphology-like values in `ref_cat` and REF_CAT-like values in `type`, the run performs a strict field-by-field audit before using `ref_cat` as the effective morphology field. If the observed mapping changes or unexpected values appear, the run stops rather than inferring a new encoding.

High-quality DESI galaxies are one-to-one nearest-neighbour matched to DR11 extended sources within **1.5 arcsec** in the tangent-plane metric. Duplicate imaging matches retain only the closest DESI object. The full imaging comparison map contains only matched DR11 positions.

## Statistic and null

For each accepted field:

1. construct the fixed 32x32 local-bandpass map of all matched DR11 galaxy positions;
2. construct DESI angular count maps in the four fixed redshift bins;
3. compute the count-weighted mean local correlation across bins with at least 12 matched galaxies;
4. generate 300 null realizations by permuting the observed redshifts separately within each exact `(survey, program)` group while leaving every angular position fixed.

A field requires >=60 matched objects and at least two usable redshift bins.

The primary field effect is `actual local correlation - mean stratified-null local correlation`.

## Preregistered decision

At least 20 accepted fields combined and at least 8 independent replication fields are required.

**PASS_MATCHED_TRACER_REDSHIFT_LOCALIZATION** requires all of:

- primary-set median effect > 0;
- independent-replication median effect > 0;
- combined median effect >= 0.01;
- combined one-sided sign-test p < 0.01;
- combined one-sided Wilcoxon p < 0.01;
- set-level global permutation p < 0.01 for both primary and replication sets.

**FAIL_MATCHED_TRACER_REDSHIFT_LOCALIZATION** is reported if the replication median <= 0, the combined median <= 0, either combined paired p >= 0.10, or either set-level global permutation p >= 0.10. Otherwise the result is UNCERTAIN.

The thresholds are fixed before observing this experiment's results.

## Interpretation boundary

A PASS would establish redshift-localized clustering in an observed, matched broad galaxy tracer under the tested survey/program-stratified null. It would not by itself establish matter density reconstruction, a gravity measurement, or higher-order cosmic grammar. A FAIL would close the current redshift-localization roadmap gate as not established with DESI DR1 at this sampling density.
