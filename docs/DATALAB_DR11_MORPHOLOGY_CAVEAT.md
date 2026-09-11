# DR11 Data Lab morphology-column caveat

## Observed issue

Live probes of the NOIRLab Astro Data Lab DR11 tables on 2026-09-11 found a column-content mismatch relative to the table schema descriptions used by this repository.

For both `ls_dr11.tractor_s` and `ls_dr11.tractor` in a bounded field near RA=156 deg, Dec=+5 deg:

- the exposed schema describes `type` as the Tractor morphological model (`PSF`, `REX`, `DEV`, `EXP`, `SER`, `DUP`);
- the exposed schema describes `ref_cat` as the external reference-catalog code;
- the live query results instead had `type` dominated by `G3`/blank-like reference-catalog values, while `ref_cat` contained `PSF`, `REX`, `EXP`, `DEV`, and `SER` values.

The same pattern appeared in both the South-only and joint Tractor Data Lab tables, so it was not treated as a one-view transient. A direct row-by-row `LS_ID_DR11` comparison against official Tractor FITS was attempted but was operationally too slow in CI and did not complete; therefore this note records an **observed Data Lab content/schema mismatch**, not a definitive diagnosis of the upstream service.

## Repository policy

1. **Preferred morphology source:** official DR11 Tractor FITS `TYPE` whenever practical.
2. Existing accepted morphology/tracer experiments such as `tracer_split_validate.py` and `magnitude_split_validate.py` already follow this pattern: Data Lab may identify a brick, but morphology values are read from the official Tractor FITS.
3. If Data Lab must be used for bounded sky-field morphology acquisition, query **both** `type` and `ref_cat` and perform an explicit value-domain audit before interpreting either column.
4. A temporary audited workaround may use the column containing only the known morphology vocabulary, but the run must abort if the live mapping changes or unexpected values appear. The raw columns and audit result must be preserved in provenance.
5. Do not silently hard-code `type -> morphology` from TAP schema metadata alone until the live-content behavior is independently verified or corrected upstream.

## Scientific impact

The repository's primary RA/Dec-only locality, point-random selection, covariance-reconstruction, and redshift tests do not depend on Data Lab `type` for their core all-source maps. The accepted PSF/extended morphology split is based on official Tractor FITS and is therefore not invalidated by this caveat.

The 36-field bright/faint point-random follow-up uses an explicit field-by-field live-content audit and aborts if the observed mapping no longer matches the documented workaround; its scientific thresholds and field set are unchanged by the acquisition correction.
