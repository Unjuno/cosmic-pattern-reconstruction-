# Current REAL_DR11 observational results

Status: **REAL_DR11**. The model input is RA/Dec only. Artificial holes hide locations that were genuinely observed in DR11; no simulated cosmological target is used.

## Data

The baseline sample contains 359,920 primary DR11-South Tractor sources in twelve independent 0.5° × 0.5° fields. Every field file is provenance-locked by the exact query, row count, and SHA-256 hashes.

## 12-field leave-one-field-out baseline

Compared with a within-field fine-cell permutation null:

| metric | real median | null median | real>null fields | one-sided exact sign-flip p |
|---|---:|---:|---:|---:|
| center-hole Gaussian correlation | 0.1210 | 0.0021 | 11/12 | 0.000488 |
| void AUC | 0.7006 | 0.5002 | 12/12 | 0.000244 |
| overdense AUC | 0.7214 | 0.4914 | 12/12 | 0.000244 |
| peak AUC | 0.6418 | 0.4915 | 9/12 | 0.0571 |
| center-hole Gaussian MSE | 1.0526 | 1.0851 | 8/12 | 0.0676 |

Void/overdense boundary predictability is robust. Peak prediction and MSE improvement are not established under this first null.

## Stronger matched-shift null and angular scale

The matched-shift control preserves observed patches and their internal clustering but pairs a visible boundary with the hidden center of a different observed patch.

- **1.875 arcmin patch width:** Gaussian hidden-pixel correlation 0.1661 versus -0.0133 null; 12/12 fields, p=0.000244.
- **3.75 arcmin:** correlation 0.1210 versus -0.0061; 12/12, p=0.000244. Void/overdense AUCs are 0.7125/0.7180.
- **7.5 arcmin:** correlation falls to 0.0235, while void/overdense AUCs rise to 0.8364/0.8289. Peak AUC is not robust at this scale (p=0.375).

Interpretation: fine spatial phase/location information is most recoverable on the smallest tested angular scale, while coarse environment classification becomes easier on larger patches. These are different inference tasks.

## Boundary-feature ablation: important correction

For the 3.75 arcmin patches, a single feature — the mean density in the ring immediately outside the hole — performs as well as or better than the full eight-feature boundary descriptor.

| motif | ring-mean median AUC | full-8 median AUC | full-8 better fields | one-sided p for full-8 gain |
|---|---:|---:|---:|---:|
| void | 0.7023 | 0.7006 | 3/12 | 0.972 |
| overdense | 0.7389 | 0.7214 | 3/12 | 0.952 |
| peak | 0.6133 | 0.6418 | 7/12 | 0.674 |

Therefore the current real-data evidence does **not** establish a rich multi-feature “cosmic grammar” for void/overdense prediction. The dominant signal is consistent with ordinary local density continuity.

## Directional continuation

A vector inferred from left/right/top/bottom boundary differences was compared with the gradient inside the hidden center.

- median field mean cosine, real: **0.1866**
- matched-shift null: **-0.0552**
- real>null: 11/12 fields
- exact one-sided p: **0.000488**

Directional continuation exists statistically, but it is far weaker than the earlier simulation proof-of-concept and should be treated as a secondary component.

## Artificial mask geometry

At 25% hidden area, Gaussian hidden-pixel correlations are nearly identical across center, corner, stripe, and randomly distributed masks: approximately 0.117–0.122. All four exceed matched-shift controls at p<=0.000244 for correlation.

The large mask-geometry effect seen in synthetic development experiments is **not reproduced** in this DR11 pilot.

## Tracer selection and thinning

Selections use DR11 observing-condition columns only to choose retained observed sources; the pattern model still receives RA/Dec only.

- all sources: void/overdense AUC 0.701/0.721; Gaussian correlation 0.121.
- MASKBITS=0: 0.787/0.677; correlation 0.385.
- all griz NOBS>0 and MASKBITS=0: 0.856/0.760; correlation 0.469.
- random 50% thinning: 0.647/0.698; correlation 0.088. Peak prediction becomes non-significant (p=0.118).
- clean catalogs equalized to at most 11,000 sources/field: 0.741/0.743; correlation 0.246. Peak remains borderline (p=0.052).

Cleaning changes the tracer population and survey selection, so its stronger predictability must **not** be interpreted as more physical without an independent selection-function random catalog.

## Observing-condition residualization

A cross-field nuisance model using MASKBITS cleanliness, NOBS, PSFDEPTH, PSFSIZE, and MW transmission was fit only for QC and then subtracted from the fine-cell count field.

The held-out nuisance-model median cell-level R² is **-0.0337**. Therefore this particular source-sampled QC model does not generalize well enough to be considered a successful selection-function model.

Nevertheless, after residualization, void/overdense AUCs remain approximately 0.702/0.717 and strongly exceed a residual matched-shift null. This is supportive evidence that the signal is not trivially removed by these covariates, but it is **not proof that survey selection has been eliminated**.

## Official random catalogs

Data Lab schema discovery found no random-catalog table under `ls_dr11`; the official DR11 randoms are file-based products. The next selection-null stage must therefore use the public random FITS files or an equivalent provenance-verified file-based extraction.

## Current interpretation

**Fact:** neighboring observed DR11 density predicts whether a held-out local region is underdense or overdense across twelve independent sky fields, and this survives a matched-patch null and 50% random thinning.

**Fact:** exact hidden-pixel phase/location recovery is weak but significantly above matched-shift null on ~2–4 arcmin patches.

**Fact:** the void/overdense result is mostly explained by the local boundary mean; richer boundary geometry has not yet added significant information.

**Not established:** that the signal traces matter density rather than residual survey selection; that it measures gravity; that a natural discrete pattern taxonomy exists; or that peak locations are robustly predictable under sparse sampling.
