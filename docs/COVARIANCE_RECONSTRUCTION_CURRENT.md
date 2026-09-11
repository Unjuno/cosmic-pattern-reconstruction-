# REAL_DR11 selection-residual covariance reconstruction

Status: **PASS_ISOTROPIC_COVARIANCE_RECONSTRUCTION**.

## Question

Can the locality that survives official pixel-level selection residualization be used to reconstruct a hidden central region using only an empirical two-point covariance model?

## Blind design

- Exact 36 selection-qualified REAL_DR11 fields.
- Fixed split before test evaluation: 18 train / 9 validation / 9 blind test fields.
- Train residual maps are themselves cross-fitted; validation/test selection expectations are predicted from training fields only.
- 3.75 arcmin patches with a 1.875 arcmin hidden center.
- Two Gaussian conditional reconstruction models: unrestricted empirical covariance and an isotropized covariance.
- Diagonal regularization tuned only on validation MSE.
- 90% interval scale calibrated only on validation data.
- Primary PASS: isotropic MSE below the train-mean baseline in >=7/9 blind fields and one-sided sign and Wilcoxon p<0.05.

## Result

The isotropic covariance model passes the preregistered blind criterion:

- isotropic beats mean MSE in **8/9** test fields;
- median `mean MSE - isotropic MSE` = **+0.02106**;
- one-sided sign p = **0.01953**;
- one-sided Wilcoxon p = **0.00586**;
- test median correlation = **0.12445** versus **0.01047** for the mean baseline;
- hidden-region mean correlation = **0.40886**;
- calibrated nominal-90% coverage median = **0.87153**.

The unrestricted full covariance does not improve on the mean baseline, and is worse than the isotropic covariance in **9/9** blind fields. The full-minus-isotropic MSE difference has median **+0.03588**, with one-sided sign and Wilcoxon p = **0.001953**.

## Interpretation

This is positive evidence that a compact, approximately isotropic two-point/covariance description contains reproducible predictive information about the hidden selection-residual field. It is consistent with the exact-power and IAAFT stress tests, which did not establish a robust higher-order phase contribution.

It is not evidence that the observed field is Gaussian, purely cosmological, or fully determined by the two-point function. The current evidence supports a simpler statement: for the tested local reconstruction task, isotropized covariance is useful and richer unrestricted covariance does not add blind-test value.

Source workflow run: `33998716930`; artifact id: `9978928445`; artifact digest: `sha256:6a76efe9756d74bd88a609b2a6691c7599e2825f74bf38ccf0fcc2717503f03e`.
