# DR11 point-random residual Gaussianized exact-power test

## Purpose

The raw point-random residual exact-power test was preregistered and returned `UNCERTAIN_HIGHER_ORDER_LOCALITY`: the observed-minus-surrogate median was close to, but below, the 0.05 effect floor. This follow-up asks a different and cleaner question: after removing the residual field's one-point marginal by rank-Gaussianization, does the spatial locality statistic still require Fourier phase information beyond the exact full 2-D power spectrum?

This follow-up does not relax or replace the earlier decision. The earlier result remains uncertain.

## Fixed data and residualization

- Exact same 36 REAL_DR11 fields.
- First 4% of every one of the 20 randomly ordered official DR11 `randoms-1-[0..19].fits` files.
- Same dense4 point-random selection features, k=4 IDW surface, fourth-neighbour 0.08 deg validity cutoff, Poisson HistGradientBoosting nuisance model, 6-fold whole-field CV, and residual-map construction.
- Strict HTTP 206 Range acquisition; retries may repeat identical byte ranges only.

The reconstructed 36 residual maps will also be saved as a provenance-locked Actions artifact (`residual_maps.npz`) so later statistical tests do not need to rescan the remote point-random files. Saving these maps is infrastructure only and does not affect the science statistic.

## Primary representation

Each held-out residual map is rank-Gaussianized cell-by-cell using the repository's existing `bispectrum_phase_validate.gaussianize` implementation. The locality statistic is then recomputed on that transformed map. Generate 32 independent exact-power real-field phase surrogates preserving the Gaussianized map's full 2-D Fourier amplitude spectrum to numerical precision.

For each field, primary paired difference = observed Gaussianized residual locality rho minus mean exact-power surrogate locality rho.

## Secondary raw-phase replication

Using the same saved residual maps, generate a second independent set of 32 raw-residual phase surrogates with seeds disjoint from the first raw phase experiment. This is Monte-Carlo stability QC only. It does not overwrite or pool with the original raw-phase primary decision.

## H / T / D / C / U

**H**: after point-random selection residualization and one-point rank-Gaussianization, observed local visible-to-hidden coupling exceeds what is reproduced by an exact full-2D power-spectrum surrogate.

**T**: 36 fixed fields; 4% official point-random selection residualization; rank-Gaussianized residual representation; 32 exact-power phase surrogates per field; one paired field-level primary statistic.

**D**: `PASS_HIGHER_ORDER_GAUSSIANIZED_LOCALITY` only if at least 30 fields are valid, median observed-minus-surrogate rho is at least 0.05, and both one-sided sign-test and Wilcoxon p-values are below 0.01. `FAIL_HIGHER_ORDER_GAUSSIANIZED_LOCALITY` if at least 30 fields are valid and the median is non-positive or either p-value is at least 0.10. Otherwise `UNCERTAIN_HIGHER_ORDER_GAUSSIANIZED_LOCALITY`. Any surrogate maximum Fourier-amplitude relative error above 1e-10 invalidates the run rather than producing a science FAIL.

**C**: after marginal control, exact two-point/power information adequately reproduces the current locality statistic.

**U**: rank-Gaussianization changes patch-mean ordering and therefore defines a transformed statistic; exact-power surrogates are not full discrete point-process models; residual selection, deblending, foreground, and tracer-population effects remain possible.

A PASS would strengthen evidence for higher-order spatial phase information. FAIL/UNCERTAIN means higher-order 'cosmic grammar' remains unestablished.
