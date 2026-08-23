# REAL_DR11 Run 002

Objectives:

1. test angular scale dependence at 1.875, 3.75, and 7.5 arcmin patch widths;
2. replace simple cell-shuffle-only controls with a matched-shift null that preserves observed patches and field-level selection structure while breaking local boundary/interior correspondence;
3. retrieve observing-condition QC covariates (MASKBITS, NOBS, PSFDEPTH, PSFSIZE, MW_TRANSMISSION) without using them as cosmological model inputs;
4. cross-fit a nuisance count model by whole sky field, subtract its predicted observing-condition component, and re-test boundary motif predictability;
5. discover whether a DR11 random-catalog table is queryable through Data Lab TAP metadata.

All science targets remain real observed DR11 counts before artificial masking. No synthetic cosmological field is generated.
