# DR11 point-random deterministic-prefix pilot

## Scope

This is the first point-level official-random acquisition experiment after the brick-level selection-function test. It makes **no cosmological claim**.

The official DR11 random-catalog documentation states that the order of points inside each `randoms-1-X.fits` file is random, so reading only the first N rows preserves randomness. The pilot therefore reads the same deterministic fraction from all 20 official files using bounded HTTP Range requests and filters those rows to the 48 provenance-fixed REAL_DR11 0.5-degree RA/Dec boxes.

## Fixed pilot

- 20 official `randoms-1-[0..19].fits` files.
- First `0.2%` of binary-table rows from each file.
- No full FITS file is written to disk.
- Required point-level columns: RA, DEC, MASKBITS, NOBS G/R/Z, PSFDEPTH G/R/Z, EBV.
- The FITS binary-table row layout is parsed from the official header; Range responses must be HTTP 206 with an exact `Content-Range`, otherwise the pilot aborts rather than risk a full-file download.

## H / T / D / C / U

**H**: deterministic prefixes from all official point-random files recover usable point-level survey metadata across nearly all 48 fixed REAL_DR11 fields.

**T**: first 0.2% of rows from each of 20 randomly ordered official random files; 48 fixed 0.5-degree RA/Dec boxes.

**D**: PASS iff at least 46 fields contain at least one sampled point, at least 40 contain at least five, the median is at least 8 points/field, and the median observed-to-nominal-expected count ratio lies in [0.45, 1.55]. FAIL if fewer than 40 fields contain points, the median is below 4, or the median ratio lies outside [0.20, 2.50]. Otherwise UNCERTAIN.

**C**: deterministic prefix sampling is too sparse or operationally biased for a bounded point-level selection-function pilot.

**U**: finite prefix-sampling noise, footprint/PHOTSYS edges, small-angle box-area approximation, and HTTP server behavior. This pilot does not test whether the previously observed locality is cosmological.

## Next gate

If this acquisition pilot passes, the next experiment will use the recovered point-level MASKBITS/NOBS/depth metadata to construct a field-wise selection surface and repeat the ring-to-hidden-center matched-shift test after point-level selection correction.
