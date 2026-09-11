# DR11 point-random residual exact-power phase-surrogate test

## Purpose

Test the strongest remaining interpretation boundary after the official point-random selection stress tests: whether the local visible-to-hidden coupling that survives point-random selection residualization contains information beyond the field's full 2-D two-point/power structure.

This is not a cosmology test. It is a statistical sufficiency test for the current locality statistic.

## Fixed data and residualization

- Exact same 36 REAL_DR11 fields as the accepted direct-coadd and point-random selection tests.
- First 4% of rows from each of the 20 randomly ordered official DR11 `randoms-1-[0..19].fits` files.
- Same point-random features, k=4 IDW selection surface, 0.08 deg fourth-neighbour validity cutoff, Poisson HistGradientBoosting nuisance model, 6-fold whole-field CV, and residual-map construction as the dense4 selection test.
- Strict HTTP 206 Range acquisition; transport retries may repeat identical byte ranges only.

## Exact-power surrogate

For each held-out residual map, generate 32 independent real-valued phase surrogates. Each surrogate preserves the observed residual map's exact full 2-D Fourier amplitude spectrum (to numerical precision) while replacing Fourier phases with phases from an independent real Gaussian field. The map mean is restored exactly.

The locality statistic is the same-patch visible-ring mean versus hidden-center mean Spearman correlation used in the accepted selection tests. For each field, compare the observed residual locality rho with the mean rho across its 32 exact-power phase surrogates.

Power-fidelity diagnostics are recorded for every surrogate. The maximum relative Fourier-amplitude error must remain below 1e-10; otherwise the experiment is invalid rather than a science FAIL.

## H / T / D / C / U

**H**: after official point-random selection residualization, the observed local visible-to-hidden coupling exceeds what is reproduced by an exact full-2D power-spectrum surrogate.

**T**: 36 fixed fields; 4% official point-random selection residualization; 32 exact-power phase surrogates per field; one paired field-level primary statistic.

**D**: `PASS_HIGHER_ORDER_LOCALITY` only if (a) at least 30 fields are valid, (b) median observed-minus-surrogate rho is at least 0.05, and (c) both one-sided sign-test and Wilcoxon p-values are below 0.01. `FAIL_HIGHER_ORDER_LOCALITY` if at least 30 fields are valid and the median is non-positive or either p-value is at least 0.10. Otherwise `UNCERTAIN_HIGHER_ORDER_LOCALITY`. A surrogate power-fidelity failure makes the run invalid, not a science FAIL.

**C**: the surviving locality is adequately reproduced, for this statistic, by exact two-point/power information after the tested survey-selection residualization.

**U**: Fourier surrogates do not reproduce every property of the original discrete point process; residual selection/deblending/tracer effects remain possible; failure to exceed the surrogate does not prove the universe is Gaussian or fully two-point determined.

A PASS would be evidence for higher-order information in this particular selection-stressed locality statistic. A FAIL/UNCERTAIN result keeps the current conservative position: robust locality exists, but higher-order cosmic grammar is not established.
