# DR11 point-random bright/faint cross-locality experiment

## Question

Does the strong local source-density continuity that survives official DR11 point-random selection residualization also appear **between two disjoint brightness populations of extended sources**?

The purpose is to test a remaining counterhypothesis: that the locality is mainly a magnitude-population-specific selection, morphology, deblending, or surface-brightness effect rather than a spatial field shared by independent source populations.

No simulated cosmological field is used.

## Fixed fields and geometry

- Use exactly the 36 fields locked by the accepted direct-coadd / point-random selection program.
- Use the same 0.5 deg sky-centered RA/Dec field geometry and 64x64 grid as the point-random selection-residual experiment.
- A field is evaluable only if the point-random k=4 selection surface has at least 85% valid cells.
- At least 30 fields are required for a scientific decision.

## Source populations

For each field, query `ls_dr11.tractor_s` with `brick_primary=1` for `ra, dec, type, flux_r, mw_transmission_r` inside the fixed field.

Extended sources are `TYPE in {REX, EXP, DEV, SER}` with finite positive `flux_r` and `mw_transmission_r`. Define dereddened r flux as `flux_r / mw_transmission_r`. Sort deterministically within each field and split the even-sized usable sample exactly in half:

- **faint**: lower dereddened-r half;
- **bright**: upper dereddened-r half.

The split uses catalog properties only to define tracers. Downstream locality statistics use only the resulting source positions/count maps.

A deterministic random permutation of the exact same extended set produces equal-count disjoint `random_a` and `random_b` halves. This is the required positive control for shared-field recoverability at the same sampling density.

Each field must contain at least 1,000 sources in each half.

## Selection model

Use the first 4% of rows from every one of the 20 official randomized `randoms-1-[0..19].fits` files. This matches the dense point-random stress test.

Interpolate official point-random metadata to the 64x64 source grid with k=4 inverse-distance weighting. Inputs are the same point-random nuisance features used by the accepted point-random selection test: MASKBITS-derived indicators, g/r/i/z NOBS, PSFDEPTH, GALDEPTH, PSFSIZE, EBV, PHOTSYS, and fourth-nearest-random distance.

Fit separate Poisson HistGradientBoosting nuisance models for bright, faint, random_a, and random_b counts with 6-fold cross-validation grouped by whole field. The held-out field is never used to fit its selection model.

Residual maps are `(count - expected) / sqrt(expected + 1)` followed by the same robust normalization/fill convention as the accepted point-random selection experiment.

## Locality statistic

For each held-out residual map, use the same 3.75 arcmin patch / 1.875 arcmin hidden-center geometry as `locality_validate.py`.

For a source tracer A and target tracer B:

- real statistic = Spearman correlation of A visible same-patch context with B hidden-center density across the 64 patches;
- matched-shift null = cyclic spatial shift of A visible context before correlation with the unchanged B hidden target;
- field advantage = real rho - matched-shift rho.

Evaluate bright self, faint self, bright->faint, faint->bright, random_a->random_b, and random_b->random_a. The **primary bright/faint statistic** is the per-field mean of the two cross directions. The positive-control statistic is the mean of the two random-half directions.

## Preregistered H / T / D / C / U

**H**: disjoint bright and faint extended-source populations share local angular continuity after official point-random selection residualization.

**T**: exact locked 36 fields; 4% x 20 official point-random files; exact within-field bright/faint halves; deterministic equal-count random halves; 6-fold whole-field selection CV; one symmetric paired statistic per field.

**D**:

1. Require at least 30 valid fields.
2. Require the random-half positive control to have median advantage >= 0.10 and one-sided sign and Wilcoxon p < 0.01.
3. Conditional on 1-2, **PASS_SHARED_BRIGHT_FAINT_LOCALITY** iff the bright/faint median cross advantage is >= 0.05 and one-sided sign and Wilcoxon p < 0.01.
4. Conditional on 1-2, **FAIL_SHARED_BRIGHT_FAINT_LOCALITY** if the bright/faint median is <= 0 or either p >= 0.10.
5. Otherwise report UNCERTAIN.

The effect floors are fixed before inspecting the new result; nominal significance alone is insufficient for PASS.

**C**: magnitude-population-specific effects plus the sampled official point-random selection model are sufficient to explain the observed local continuity.

**U**: source-property measurement error, flux-dependent incompleteness not captured by point-random covariates, deblending, surface-brightness selection, redshift-distribution differences, Galactic foregrounds, finite point-random sampling, and the fact that angular tracer populations need not sample identical radial structure.

## Interpretation boundary

A PASS would establish shared **observed angular residual locality** across two disjoint brightness populations under the tested nuisance model. It would not establish a cosmological origin, matter density, gravity, or higher-order spatial information. A failure could reflect real tracer/redshift separation as well as a population-specific systematic, so it would not by itself falsify the previously established all-source locality.

## Reproducibility outputs

The workflow records exact Data Lab SQL, retrieval timestamps, bounded-catalog SHA-256 hashes, point-random HTTP Range/SHA provenance, field QC, nuisance-model metrics, field-level locality metrics, and cached selection/residual maps for follow-up tests.
