# DR11 point-random disjoint replication

## Purpose

Replicate the completed 0%-2% official point-random selection-residual test with a non-overlapping row sample. This isolates sensitivity to finite random-catalog sampling and the derived k=4 IDW selection surface.

## Fixed design

- Exact same 36 REAL_DR11 fields as the completed 0%-2% test.
- Exact same source-count maps, 64x64 grid, selection features, k=4 IDW interpolation, 0.08 deg validity limit, Poisson HistGradientBoosting nuisance model, 6-fold whole-field cross-validation, residual normalization, local-visible statistic, and matched-shift null.
- Sample exactly rows corresponding to nominal fractions 0.02 <= row/total < 0.04 from each of the 20 randomly ordered official `randoms-1-[0..19].fits` files.
- No overlap with the 0%-2% point-random sample.
- Strict HTTP 206 Range requests; transport retries may repeat the same byte range but may not change sampled rows or science parameters.

## H / T / D / C / U

**H**: after selection residualization using the disjoint 2%-4% point-random sample, local visible-to-hidden coupling remains above matched shift.

**T**: same 36 fields and same analysis pipeline; disjoint 2%-4% rows from every official point-random file.

**D**: identical to the first point-random test: PASS if at least 30 fields are valid, median residual real-minus-shift advantage > 0, and both one-sided sign and Wilcoxon p-values < 0.05. FAIL if at least 30 fields are valid and median <= 0 or either p-value >= 0.10. Otherwise UNCERTAIN.

**C**: the first 0%-2% PASS was an accident of finite random sampling / kNN surface realization rather than a stable selection-function result.

**U**: both disjoint samples remain smoothed finite-density reconstructions rather than the complete point-random catalog; deblending/tracer effects and random-catalog construction remain outside the claim.

A second PASS supports stability to finite point-random realization. It still does not establish a cosmological origin or higher-order cosmic structure.
