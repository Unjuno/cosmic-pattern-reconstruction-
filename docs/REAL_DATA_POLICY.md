# Real-data policy

This repository is intentionally observational-data-only.

## Accepted evidence

A result can be presented as a DR11 observational result only if the corresponding result directory contains a `provenance.json` with:

- `dataset = DESI Legacy Imaging Surveys DR11`
- database table or official file URL
- exact SQL or file path
- exact RA/Dec bounds
- retrieval timestamp in UTC
- input row count
- SHA-256 of each stored input
- tracer-selection rule
- software commit SHA when available

## Forbidden substitutions

The following may be useful for method development elsewhere, but are not accepted as observational evidence in this repository:

- Particle-Mesh simulations
- Zel'dovich toys
- HOD-like synthetic tracers
- synthetic completeness maps
- injected foreground catalogs used as primary data
- random Gaussian fields

No numerical result from such data should be copied into `results/real_dr11/`.

## Artificial masking is permitted

A real catalog may be artificially masked after acquisition for a held-out reconstruction test. In that case the hidden values must be retained only for evaluation and must never enter the reconstruction model. This is analogous to cross-validation on observational data and does not create synthetic target structure.

## Interpretation hierarchy

1. **Observed fact**: a statistic measured directly on the stated DR11 tracer sample.
2. **Statistical inference**: held-out predictability or reproducible structure across independent sky regions.
3. **Physical hypothesis**: interpretation in terms of matter density, cosmic web, gravity, or cosmology.

Levels 2 and 3 must never be silently promoted to level 1.

## Tracer selections

The pilot publishes two position catalogs derived from the same DR11 queries:

- `all_primary`: all `brick_primary=1` sources returned by the query;
- `extended_clean`: `brick_primary=1`, `maskbits=0`, and morphology not equal to `PSF`.

Morphology and mask information are used only to define the tracer sample. The pattern models themselves use sky positions/counts only.

## Region splitting

Train/validation/test splits are performed by complete sky field. Patches from a held-out field must not appear in training.

## Third-party data

The repository code is Apache-2.0. DESI Legacy Surveys / NOIRLab data remain under their upstream terms and are not relicensed here.
