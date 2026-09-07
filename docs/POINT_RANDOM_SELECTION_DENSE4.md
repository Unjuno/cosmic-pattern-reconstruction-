# DR11 point-random higher-density stress test

## Purpose

Test whether the point-random selection-residual PASS depends on the relatively sparse 2% prefix surface. Increase the deterministic point-random density to the first 4% of every official random file while keeping the science pipeline unchanged.

## Fixed design

- Exact same 36 REAL_DR11 fields.
- Same source maps, 64x64 grid, point-random features, k=4 inverse-distance interpolation, 0.08 deg validity cutoff, Poisson HistGradientBoosting nuisance model, 6-fold whole-field CV, residual map construction, local-visible statistic, and matched-shift null.
- Use first 4% of rows from each of the 20 randomly ordered official `randoms-1-[0..19].fits` files.
- Strict HTTP 206 byte ranges; retries may repeat identical requests only.
- No tuning of k, model parameters, field set, or decision thresholds after seeing the result.

## H / T / D / C / U

**H**: after residualization with the denser 4% official point-random selection surface, local visible-to-hidden coupling remains above matched shift.

**T**: same 36-field point-random pipeline, but 4% rather than 2% of each official random file.

**D**: identical primary rule: PASS if at least 30 fields are valid, the median residual real-minus-shift advantage is positive, and both one-sided sign and Wilcoxon p-values are below 0.05. FAIL if at least 30 fields are valid and the median is non-positive or either p-value is at least 0.10. Otherwise UNCERTAIN.

**C**: the previous PASS is an artifact of an under-sampled, overly smoothed point-random selection surface.

**U**: 4% still does not equal the complete random catalog; deblending/tracer effects and random-catalog construction remain outside the claim.

A PASS supports stability as point-random sampling becomes denser. It does not establish cosmological origin or higher-order structure.
