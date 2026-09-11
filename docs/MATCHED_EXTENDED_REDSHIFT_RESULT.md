# Matched DR11 extended x DESI DR1 redshift result

Status: **FAIL_MATCHED_TRACER_REDSHIFT_LOCALIZATION**.

## Result

The preregistered matched-tracer gate completed successfully on 37 evaluable fields: 18 from the fixed primary set and 19 from the independent-sky replication set.

DR11 extended sources were one-to-one matched within 1.5 arcsec to high-quality DESI DR1 primary galaxy spectra. Median DESI match fractions were **0.864** in the primary set and **0.873** in the independent replication, so the negative result is not driven by a very low match rate.

The primary statistic is the excess local correlation of fixed redshift slices over a null that permutes observed redshifts only within the same field and exact `(survey, program)` group, leaving every angular position fixed.

### Primary set

- accepted fields: **18**
- actual local correlation median: **0.54998**
- stratified-null local correlation median: **0.55749**
- actual-minus-null median: **-0.00768**
- positive fields: **4/18**
- one-sided sign p: **0.9962**
- one-sided Wilcoxon p: **0.9972**
- set-level global permutation p: **0.9967**

### Independent replication

- accepted fields: **19**
- actual local correlation median: **0.55813**
- stratified-null local correlation median: **0.56223**
- actual-minus-null median: **-0.00659**
- positive fields: **3/19**
- one-sided sign p: **0.9996**
- one-sided Wilcoxon p: **0.9983**
- set-level global permutation p: **0.9967**

### Combined

- accepted fields: **37**
- actual-minus-null median: **-0.00707**
- positive fields: **7/37**
- one-sided sign p: **0.999979**
- one-sided Wilcoxon p: **0.999976**

The broad-map comparison is also negative (combined median difference **-0.02508**).

## Interpretation

The matched angular catalogs themselves are strongly correlated with their redshift-slice maps: the actual local correlations are around 0.55. However, the survey/program-stratified redshift-shuffle null is at least as strong. Therefore the experiment finds **no additional evidence that the robust DR11 2D locality is localized in the fixed DESI DR1 redshift slices**.

This closes the current redshift-view roadmap gate negatively. The earlier all-source initial positive was already not replicated; restricting both views to a one-to-one matched broad galaxy tracer does not recover it.

This does not show that the observed 2D angular locality is non-cosmological. Projection over redshift, DESI DR1 sparsity, radial selection, bias evolution, and the fixed binning can all suppress a 3D correspondence. The narrower conclusion is that **redshift localization is not established by the current DESI DR1 tests**.

Source workflow run: `34585038214`; artifact id: `10193348202`; artifact digest: `sha256:955a46b1c3072e05ca084c24b598a4f737d0132fa48f77fc529f83317b3794fb`.
