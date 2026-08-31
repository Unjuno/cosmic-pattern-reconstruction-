# REAL_DR11 morphology tracer-split result

Status: **accepted real-observation stress test**.

Source: official DESI Legacy Imaging Surveys DR11 South Tractor FITS. Tractor morphology is used only to define subsets; downstream locality predictors use positions only.

## Design

- Fixed first-48 candidate order from the provenance-frozen expanded DR11 sample.
- Availability gate only; target 36 accepted fields, minimum 1,000 objects in each morphology group.
- PSF-like: `TYPE=PSF`.
- Extended-like: `TYPE in {REX, EXP, DEV, SER}`.
- PSF and extended samples are downsampled to equal counts within each brick before the primary comparison.
- Whole-field leave-one-out evaluation.
- Matched-shift spatial null.
- Cross-tracer tests: PSF boundary -> extended hidden region and extended boundary -> PSF hidden region.

## Accepted result

For the equal-count extended sample:

- void median AUC = **0.7055**, matched-shift median = **0.5042**; Wilcoxon one-sided p = **0.000381**.
- overdense median AUC = **0.6783**, matched-shift median = **0.5131**; Wilcoxon p = **1.34e-5**.
- peak median AUC = **0.5939**, matched-shift median = **0.5190**; Wilcoxon p = **0.00183**.

Cross-tracer PSF boundary -> extended hidden target is much weaker:

- void median AUC = **0.5816**.
- overdense median AUC = **0.5415**.
- peak median AUC = **0.5020**.

Paired extended-self minus PSF->extended differences:

- void median delta = **+0.0822**; sign p = **0.0106**, Wilcoxon p = **0.00134**.
- overdense median delta = **+0.1522**; sign p = **0.0207**, Wilcoxon p = **0.00243**.
- peak median delta = **+0.1004**; sign p = **0.00591**, Wilcoxon p = **0.000241**.

The equal-count PSF sample has weaker locality: median AUC 0.6152 (void), 0.6119 (overdense), 0.5563 (peak). Extended->PSF cross prediction is also weak/moderate.

## Interpretation

The accepted DR11 angular-locality signal is not well explained as a simple mixture of a strongly clustered PSF-like foreground population with an extended population. Extended sources retain their own significant local density continuity after equalizing tracer counts, while PSF boundary density is a substantially poorer predictor of extended hidden density.

This does **not** prove a cosmological origin. Morphology-dependent completeness, deblending, surface-brightness selection, and residual imaging systematics remain counterhypotheses. The next gate is a within-extended bright/faint cross-population test, with random half/half splits as a positive control.

## Invalid / repaired runs

1. An earlier fixed 48-brick run stopped at brick `2641m050` because the hard equal-count requirement was 1,500 while only 1,188 extended objects were available. No scientific result from that run was used.
2. The first availability-gated replication rejected all fields because FITS numeric columns were big-endian and pandas on the runner rejected the buffers. The error was `Big-endian buffer not supported on little-endian compiler`. No scientific result was used.
3. The accepted run converts FITS numeric columns to native endian before DataFrame construction. GitHub Actions run: `33393601720`.
