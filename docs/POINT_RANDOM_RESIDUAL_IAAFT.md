# DR11 point-random residual IAAFT locality test

## Purpose

Resolve the remaining ambiguity in the raw exact-power phase test. Exact-power phase surrogates preserve Fourier amplitude exactly but do not preserve the original residual map's one-point marginal. Rank-Gaussianization controls the marginal by changing the representation. IAAFT provides the complementary control: preserve the original residual values exactly while approximately matching the Fourier power and scrambling phase.

This experiment uses only the provenance-locked dense4 residual-map cache produced by workflow run `34144835351`; it does not rescan the remote DR11 random files.

## Fixed input

- `residual_maps.npz` from the successful dense4 point-random Gaussianized-phase workflow artifact.
- 36 held-out residual maps, each 64x64, built from the same fixed REAL_DR11 fields and 4% official point-random selection residualization.
- Artifact manifest must report status `REAL_DR11_POINT_RANDOM_DENSE4_RESIDUAL_MAP_CACHE`, shape `[36,64,64]`, and selection fraction `0.04`.

## Surrogate construction

For each field generate 32 independent IAAFT surrogates with 150 iterations each.

Each IAAFT iteration alternates:
1. replace Fourier amplitude by the target residual map amplitude while retaining the current surrogate phase;
2. rank-remap the result onto the exact sorted values of the original residual map.

Thus the final surrogate preserves the one-point sample exactly and approximates the target power while scrambling phase.

## Quality gate

The run is invalid rather than a science FAIL unless all of the following ensemble-level conditions hold:

- maximum sorted one-point absolute difference <= `1e-12`;
- median radial log-power correlation >= `0.995`;
- mean radial power relative error <= `0.05`.

Full 2-D Fourier-amplitude relative error is also reported descriptively but is not the preregistered quality gate because final rank remapping necessarily perturbs exact mode amplitudes.

## H / T / D / C / U

**H**: the point-random-selection-residual local visible-to-hidden coupling exceeds what is reproduced when both the one-point marginal and approximately the two-point/power structure are preserved by IAAFT.

**T**: 36 fixed cached dense4 residual maps; 32 IAAFT surrogates per field; 150 iterations; same local-visible/hidden Spearman statistic; one paired field-level primary comparison using observed rho minus mean surrogate rho.

**D**: `PASS_HIGHER_ORDER_IAAFT_LOCALITY` only if the quality gate passes, at least 30 fields are valid, median observed-minus-IAAFT rho is at least `0.05`, and both one-sided sign-test and Wilcoxon p-values are below `0.01`. `FAIL_HIGHER_ORDER_IAAFT_LOCALITY` if the quality gate passes, at least 30 fields are valid, and median is non-positive or either p-value is at least `0.10`. Otherwise `UNCERTAIN_HIGHER_ORDER_IAAFT_LOCALITY`.

**C**: one-point marginal plus two-point/power structure is sufficient, at the tested IAAFT fidelity, to reproduce the current residual locality statistic.

**U**: IAAFT power matching is approximate rather than exact; the residual maps are nuisance-model products rather than a complete physical field; unmodeled observing/tracer effects remain; no cosmological inference follows directly.

This test does not overwrite the earlier exact-power results. Agreement across exact-power, Gaussianized exact-power, and IAAFT controls is the relevant robustness criterion.
