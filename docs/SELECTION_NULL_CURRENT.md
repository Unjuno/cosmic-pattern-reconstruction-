# REAL DR11 pixel-level selection-null status

## Current accepted evidence

The survey-selection stress tests use observed DESI Legacy Imaging Surveys DR11 source positions and official DR11 coadd observing-condition products. No simulated cosmological field is used as a science target.

The strongest completed replication is the 36-field multiband whole-field leave-one-field-out experiment. Thirty-six fields from the fixed first-48 candidate order passed coadd availability and coverage requirements. The selection model receives g/r/i/z depth, NEXP, PSF-size, and mask/primary information, including information in the artificially hidden region. The observed predictor receives only source-density information outside the hidden region.

### Motif results

- Void: observed median AUC 0.750; selection g/r/i/z median AUC 0.569. Observed minus selection was positive in 16/23 evaluable fields; one-sided sign p=0.0466, Wilcoxon p=0.0127.
- Overdense: observed median AUC 0.735; selection g/r/i/z median AUC 0.571. Observed minus selection was positive in 19/28 evaluable fields; one-sided sign p=0.0436, Wilcoxon p=0.0109.
- Peak: observed median AUC 0.510 and selection median AUC 0.500. This stress test does not support peak predictability; treat peak as FAIL/UNCERTAIN rather than carrying forward the earlier 48-field position-only result.

### Continuous residual locality

A count model using the g/r/i/z selection maps has median held-out cell R2 = 0.00594. After subtracting its predicted component, the median Spearman correlation between visible-ring residual and hidden residual is 0.368, versus 0.0347 for a matched-shift control. Residual minus shift is positive in 27/36 fields (one-sided sign p=0.00197; Wilcoxon p=0.000180).

This does not prove that every remaining correlation is cosmological; unmodeled selection, foregrounds, deblending and tracer-population effects remain possible. It does show that the known pixel-level g/r/i/z coadd selection maps tested here do not account for most of the local void/overdensity continuity signal.

### Directionality QC

In a separate 12-field post-hoc QC, several official depth/NEXP/PSF maps show a significant x-y anisotropy, while source-count x-y anisotropy is near zero. This mismatch is evidence against a simple explanation in which the observed source locality is copied directly from those survey maps, but it is not a preregistered primary test.

## Negative and discarded results

- The original 48-brick brute-force LOFO HistGradientBoosting run was cancelled at the 70-minute workflow limit and produced no accepted science artifact.
- Peak prediction does not survive the 36-field pixel-level selection-null stress.
- Selection-map count prediction has very low held-out cell R2, so residualization should be interpreted as a nuisance stress test, not as a complete selection-function reconstruction.
