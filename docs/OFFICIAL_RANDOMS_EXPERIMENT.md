# Official DR11 random-catalog experiment

## Scope

This experiment tests whether the previously observed REAL_DR11 local ring-to-hidden-center continuity can be explained by the **official DR11 resolved brick footprint and primary-coverage fractions** used to interpret the official random catalogs.

It is deliberately narrower than a point-level random-catalog selection-function test. The science target remains the provenance-fixed REAL_DR11 source positions; no simulated cosmological field is introduced.

## Official inputs

Two official DR11 products are joined by `BRICKID`:

- `randoms/survey-bricks-dr11-randoms-5.1.0.fits`: resolved `PHOTSYS` plus brick geometry / random-catalog area metadata.
- `survey-bricks.fits.gz`: `DRVERSION` and primary-coverage fractions.

Resolved South coverage is defined before examining the science result as:

- `PHOTSYS=S`, `DRVERSION=11` -> `FALLPRIMGE1DR11`;
- `PHOTSYS=S`, `DRVERSION=9` -> `FALLPRIMGE1DR9S`;
- otherwise coverage `0`.

Input hashes for workflow run `33995675602`:

- random brick summary SHA-256: `4ac81e7f0d3ab5eca98913868677e54c1cfaca9d11084a30b4ee5180cec25d51`
- survey-bricks SHA-256: `4cb1388b661760a9ed785f6027e72bf4d677f68508d304e035522572f53e04fb`

Both tables contain 662,174 bricks and join one-to-one for this test.

## Fixed hypothesis and decision rule

**H**: after official brick-coverage adjustment, ring-to-hidden-center locality remains above the within-field matched-shift control.

**T**: 48 provenance-fixed REAL_DR11 fields, tangent-plane grid, one paired field-level statistic; minimum valid field count `n_min=36`.

**D**: `PASS` iff the median adjusted real-minus-shift advantage is at least `0.10` and both one-sided sign-test and Wilcoxon p-values are below `0.01`. `FAIL` if the median is non-positive or either p-value is at least `0.05`; otherwise `UNCERTAIN`.

**C**: the official resolved brick footprint / primary-coverage structure explains the observed local continuity.

**U**: sub-brick masks, depth, PSF, deblending, Galactic foregrounds, tracer-population effects, and the coarseness of the brick summary remain unresolved.

## Result

Status: **PASS** for the brick-level hypothesis test.

| quantity | result |
|---|---:|
| requested fields | 48 |
| evaluable fields | 44 |
| fields with positive adjusted advantage | 41 / 44 |
| median adjusted real rho | 0.414949 |
| median adjusted matched-shift rho | -0.071531 |
| median adjusted advantage | 0.488622 |
| mean adjusted advantage | 0.452935 |
| one-sided sign p | 8.09166e-10 |
| one-sided Wilcoxon p | 1.87583e-12 |
| median coverage-count rho | 0.003033 |
| raw median real rho | 0.417194 |

Four fixed fields (`f14_ra264_m05`, `f16_ra288_m25`, `f21_ra084_p05`, `f38_ra084_m05`) have official resolved coverage `0` throughout the test square and therefore contribute no valid cells under the pre-defined coverage floor `0.05`; they are excluded mechanically, not according to their science outcome.

The brick-level correction changes the median real locality only from `0.417194` to `0.414949`. Thus the official **coarse brick footprint / primary-coverage variation** tested here does not account for the local continuity signal.

## Interpretation boundary

This result does **not** establish that the residual signal is cosmological, traces matter density, measures gravity, or requires higher-order structure beyond two-point statistics.

The official point-random files remain necessary for the stronger selection-function test because they carry sub-brick observing-condition and mask sampling. The official files are twenty randomly shuffled catalogs at 2,500 points per square degree; the probed first files are about 27.2 GB each. Because rows are randomly shuffled rather than spatially indexed, a bounded RA/Dec test requires a one-time full scan or a separately generated provenance-locked spatial index rather than HTTP range extraction.

## Reproducibility

Successful science workflow: GitHub Actions run `33995675602`.

Artifact: `real-dr11-official-random-brick-33995675602`, artifact id `9977997766`, digest `sha256:6cb0fdb879d49dbfbe4b995dd535fd5c9c9fdb7d2ce3f3abfe88ce28ed5e8f5f`.

Machine-readable outputs are committed under `results/real_dr11/official_random_brick48/` and the availability probe under `results/real_dr11/official_randoms_probe/`.
