# REAL_DR11 point-random selection and higher-order status

## Selection-function gate

Three independent official point-random tests establish that the strong local angular continuity is not explained by the tested sampled DR11 observing-condition/mask nuisance model.

- 0-2% prefix: residual real-minus-matched-shift median **0.49146**, 35/36 positive, sign p=5.38e-10, Wilcoxon p=2.91e-11.
- disjoint 2-4% prefix: median **0.51358**, 36/36 positive, both p=1.46e-11. Field-level effects agree with the first realization at Spearman rho=0.942 and Pearson r=0.971.
- dense 0-4% prefix: median **0.48793**, 36/36 positive, both p=1.46e-11. The median kNN4 interpolation radius shrinks from about 2.24 arcmin at 2% to 1.56 arcmin at 4% with essentially unchanged residual locality.

The point-random selection models have low held-out predictive R2, so these tests do **not** establish that the full selection function is known or removed. The defensible statement is narrower: the sampled official point-random nuisance structure and finite-random interpolation coarseness do not explain most of the locality.

## Higher-order / phase gate

The residual field was then compared with increasingly strict controls.

### Exact full-2D power, raw residual representation

- observed residual rho median: 0.50151
- exact-power surrogate rho median: 0.44981
- observed-minus-surrogate median: **0.04663**
- 25/36 positive
- sign p=0.01441; Wilcoxon p=0.01195
- Fourier-amplitude relative error <=7.07e-16

The preregistered effect floor was 0.05 with both p<0.01, so the result is **UNCERTAIN_HIGHER_ORDER_LOCALITY**.

### Rank-Gaussianized exact-power control

- median observed-minus-surrogate: **0.04626**
- 23/36 positive
- sign p=0.06625; Wilcoxon p=0.01540
- amplitude error <=7.92e-16

Primary decision: **UNCERTAIN_HIGHER_ORDER_GAUSSIANIZED_LOCALITY**. A secondary independent raw-phase ensemble was stronger (median 0.05176; 26/36; p=0.00567/0.00959), but it is not pooled into the preregistered primary result.

### IAAFT marginal+power control

IAAFT preserves the observed one-point sample exactly and approximates the power spectrum extremely closely:

- one-point max absolute difference: 0
- median radial log-power correlation: 0.99999896
- mean radial-power relative error: 0.0013679
- median full-amplitude L1 relative error: 0.001811

Against this control:

- median observed-minus-IAAFT locality: **0.02503**
- 24/36 positive
- sign p=0.03262; Wilcoxon p=0.04272

Primary decision: **UNCERTAIN_HIGHER_ORDER_IAAFT_LOCALITY**.

## Current interpretation

A small phase-dependent excess appears repeatedly in the same fields, so it is not well described as pure surrogate-seed noise. However, its magnitude falls by roughly half when the observed one-point marginal is preserved, and no preregistered marginal-controlled test establishes a robust higher-order contribution.

Therefore the accepted status is:

1. **robust local angular continuity: established for the observed DR11 source field under the tested selection controls;**
2. **two-point/power structure: sufficient to explain most current predictive performance;**
3. **higher-order cosmic grammar: not established.**

No result here establishes cosmological origin, matter density, gravity, or a natural discrete taxonomy.
