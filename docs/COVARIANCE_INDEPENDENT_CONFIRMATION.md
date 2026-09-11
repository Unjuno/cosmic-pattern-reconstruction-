# REAL_DR11 independent covariance confirmation

Status before execution: **PREREGISTERED_CONFIRMATORY_TEST**.

## Motivation

The accepted 36-field blind reconstruction showed that radial/isotropic two-point covariance predicts a hidden 1.875-arcmin center better than a train-mean baseline. A subsequent explicitly post-hoc capacity audit showed that both radial/isotropic and translation-invariant directional covariance beat generic Ledoit-Wolf/OAS shrinkage, while radial and directional stationary covariance were not strongly distinguishable.

This experiment is the required fresh-field confirmation. The old 36 selection-qualified fields are now training data only. No new-field density outcome may be used for model selection, regularization selection, field ordering, or stopping.

## Frozen field acquisition

Candidate sky centers use the exact historical deterministic grid and shuffle (`seed=20260824`) from `fetch_dr11_expanded.py`, but begin at candidate index 54, after all 48 historical expanded fields had already been fixed. Candidate order is therefore determined before this test.

A candidate is eligible only when all of the following outcome-independent acquisition/coverage gates pass:

1. its center is at least 6 degrees from every historical expanded48 center and every already accepted confirmation center;
2. it resolves to a unique official DR11-South Tractor brick;
3. the official Tractor catalog and official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY products are retrievable;
4. the existing selection-feature coverage gate yields at least eight complete patches.

No minimum target-density, reconstruction-score, covariance-score, or effect-size gate is allowed. Candidates are scanned in the fixed order until exactly 20 fields are accepted; the run aborts if 20 cannot be obtained from candidate indices 54 through 173.

## Frozen training and models

Training uses all 36 historical selection-qualified bricks. Their selection residuals are generated out-of-fold to avoid nuisance-model self-fit. The final nuisance model applied to the 20 new fields is fitted only on the 36 historical fields.

Covariance parameters are estimated only from historical residual patches. Hyperparameters are frozen from the completed capacity audit and are not retuned on the new fields:

- translation-invariant directional stationary covariance: diagonal regularization 0.1;
- radial/isotropic covariance: diagonal regularization 0.01;
- OAS covariance: diagonal regularization 1.0;
- Ledoit-Wolf covariance: diagonal regularization 0.3.

The train-mean predictor is retained as the minimal baseline. Patch geometry is unchanged: 3.75 arcmin square with the central 1.875 arcmin region hidden.

## H — falsifiable hypothesis

**H-confirm:** the predictive advantage is not merely generic covariance shrinkage. A translation-invariant covariance learned on historical fields will reduce hidden-cell mean-squared error on fresh fields relative to both the historical train-mean predictor and OAS shrinkage.

## T — minimum test

Use exactly 20 fresh accepted fields, field-level paired inference, the frozen acquisition order/gates, the frozen historical training set, frozen model classes and frozen regularization values. Patches within one field are not treated as independent astronomical replicates.

## D — decision rule

**PASS_STRUCTURED_STATIONARY_CONFIRMATION** requires both comparisons below to hold on the 20 new fields:

- stationary MSE is lower than train-mean MSE in at least 16 of 20 fields, with one-sided sign-test p < 0.01 and one-sided Wilcoxon p < 0.01;
- stationary MSE is lower than OAS MSE in at least 16 of 20 fields, with one-sided sign-test p < 0.01 and one-sided Wilcoxon p < 0.01.

**FAIL_STRUCTURED_STATIONARY_CONFIRMATION** is declared if the stationary median MSE advantage over either train-mean or OAS is non-positive. All other outcomes are **UNCERTAIN_STRUCTURED_STATIONARY_CONFIRMATION**.

Radial/isotropic versus directional stationary covariance is secondary. This experiment does not claim radial isotropy unless a separate predeclared directional comparison supports it.

## C — ways the hypothesis can fail

1. OAS matches or beats stationary covariance on fresh fields: generic shrinkage remains sufficient.
2. Stationary covariance fails to beat the train mean: the earlier predictive gain does not generalize spatially.
3. Isotropic and stationary results diverge strongly: the directionality question must be isolated before physical interpretation.
4. Effects cluster by field geometry or observing conditions: remaining survey-systematic structure is a plausible explanation.

## U — uncertainty and guardrails

Main uncertainties are field-to-field heterogeneity, residual selection-model misspecification, within-field patch dependence, official-product acquisition failures, deblending/morphology systematics and survey-coordinate anisotropy. Inference is therefore paired at the field level.

A PASS establishes only that a low-dimensional translation-invariant two-point covariance carries reproducible predictive information in the observed DR11 angular point field after the tested nuisance controls. It does **not** establish cosmological origin, matter-density reconstruction, gravity, three-dimensional structure, or a discrete cosmic grammar.
