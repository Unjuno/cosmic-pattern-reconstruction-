# DR11 point-random selection-residual experiment

## Scope

This is an independent selection-function replication using the official **point-level DR11 random catalogs** rather than the direct coadd maps used in the accepted 36-field selection-null experiment.

The science target remains observed REAL_DR11 source positions. No simulated cosmological field is introduced.

## Fixed field set

Use exactly the 36 fields from the accepted direct-coadd selection residual locality test so that the new result can be compared to the existing stress test without field selection after seeing the new result.

## Point-random acquisition

- 20 official `randoms-1-[0..19].fits` files.
- First 2% of rows from every file.
- The official DR11 documentation states that row order within each file is random, so prefix sampling retains randomness.
- Strict HTTP Range requests only; each response must be HTTP 206 with an exact Content-Range.
- No full remote FITS file is written to disk.

The 2% x 20 sample corresponds nominally to 1,000 random locations per square degree before footprint effects, with mean spacing of order a few arcminutes. This is sufficient for an arcminute-scale smoothed selection-surface stress test, but it is not an exact reconstruction of every sub-arcminute mask edge.

## Selection surface

For every field, point-random metadata are interpolated onto the same 64x64 grid as the source-count locality analysis using inverse-distance weighting of the four nearest random points. Inputs include:

- MASKBITS cleanliness / primary / bright-star indicators;
- NOBS g/r/i/z;
- PSFDEPTH g/r/i/z;
- GALDEPTH g/r/i/z;
- PSFSIZE g/r/i/z;
- EBV;
- PHOTSYS;
- fourth-nearest-random distance as a local sampling/footprint proxy.

Cells whose fourth-nearest point is farther than 0.08 deg are invalid. A field must retain at least 85% valid cells to enter the model.

## Cross-validation

A Poisson HistGradientBoosting nuisance model is trained with whole-field grouped 6-fold cross-validation. The target is source count divided by that field's mean count, matching the normalization logic of the accepted direct-coadd selection model. The held-out field is residualized using only the point-random-derived selection expectation.

## H / T / D / C / U

**H**: after official point-random selection residualization, local visible-to-hidden source-density coupling remains above the within-field matched-shift control.

**T**: exact 36-field direct-coadd comparison set; first 2% of each of 20 randomized official point-random files; k=4 IDW selection surfaces; 6-fold whole-field CV.

**D**: PASS if at least 30 fields are valid, the median residual real-minus-shift advantage is positive, and both one-sided sign and Wilcoxon p-values are below 0.05. FAIL if at least 30 fields are valid and the median is non-positive or either p-value is at least 0.10. Otherwise UNCERTAIN.

**C**: sampled point-level survey selection explains the local source-density continuity.

**U**: finite point-random sampling, kNN smoothing at roughly arcminute scales, unmodeled deblending/tracer effects, and random-catalog construction. PASS does not establish a cosmological origin.

## Historical comparator only

The accepted direct-coadd 36-field result had median held-out selection-model R2 about 0.00594 and residual local-visible rho 0.368 versus 0.0347 matched-shift, with 27/36 positive fields. These values are context only and are not part of this experiment's decision rule.
