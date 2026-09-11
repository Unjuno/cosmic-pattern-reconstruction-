# REAL_DR11 covariance-capacity audit

Status before execution: **POSTHOC_MECHANISM_TEST_PREREGISTERED_ON_EXPERIMENT_BRANCH**.

## Scope

The accepted blind result shows that the isotropic two-point covariance reconstructs the hidden 1.875-arcmin center better than a train-mean baseline, while the unrestricted 64x64 covariance does not. This audit asks whether that contrast is specific evidence for radial/isotropic structure or is mostly finite-sample covariance regularization.

This is **not** a new blind confirmation: it reuses the previously inspected 18-train / 9-validation / 9-test field split. Results may select the next experiment, but cannot upgrade the scientific claim by themselves.

## H — falsifiable hypothesis

**H-capacity:** isotropic covariance contains predictive structure beyond generic high-dimensional covariance shrinkage. The observable is per-test-field hidden-cell MSE on the frozen nine-field test set.

## T — minimum test

Use exactly the existing 36 selection-qualified DR11 bricks, the frozen 18/9/9 split, the same official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY residualization, the same 3.75-arcmin patch and 1.875-arcmin hidden center, and validation-only diagonal regularization.

Compare five covariance estimators: unrestricted empirical, translation-invariant directional stationary, radial isotropic, Ledoit-Wolf shrinkage, and OAS shrinkage. No model may use test-field labels or test MSE for tuning.

Expected sample count from the accepted run is 576 train patches, 284 validation patches and 283 test patches; the run must abort if the acquisition/split invariants change materially.

## D — decision rule

Diagnostic support for isotropy beyond generic shrinkage requires isotropic MSE to be lower than **both** OAS and Ledoit-Wolf in at least 8 of 9 test fields, with one-sided sign-test and Wilcoxon p-values below 0.05 for both comparisons.

Directional-stationary versus radial-isotropic structure is classified separately with the same 8-of-9 plus dual-p-value criterion. If neither direction passes, the comparison is **UNCERTAIN / no strong difference**.

All labels remain explicitly post-hoc because the test set was already observed in the predecessor experiment.

## C — ways the hypothesis can fail

1. OAS or Ledoit-Wolf matches or beats isotropic covariance: generic regularization is sufficient to explain the earlier advantage.
2. Directional stationary covariance beats isotropic covariance: translation invariance matters, but radial symmetry is over-constraining.
3. All structured covariance models are similar: the current nine test fields are underpowered for mechanistic discrimination.
4. Results vary strongly by field: residual survey-selection heterogeneity or field clustering remains a plausible driver.

## U — uncertainty and constraints

Primary uncertainty sources are only 18 independent training fields, only nine test fields, within-field patch dependence, selection-model misspecification, and reuse of a previously inspected test set. Patch count must not be treated as the number of independent astronomical replicates; inference is field-paired.

## Required next confirmation

If this audit identifies a preferred covariance class, confirm it on a newly locked, spatially independent set of selection-qualified DR11 fields before making a stronger structural claim. The confirmatory field list, acquisition rules, model set and thresholds must be frozen before fetching target-density outcomes.
