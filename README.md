# Cosmic Pattern Reconstruction

**Real-observation-only research repository.**

This repository studies whether multiscale spatial patterns can be learned and conditionally reconstructed from sparse astronomical sky positions, using public observational data rather than synthetic cosmological mocks.

## Current data source

Primary source: **DESI Legacy Imaging Surveys DR11** via NOIRLab Astro Data Lab and the official Legacy Survey public release.

DR11 contains roughly 3.93 billion Tractor catalog rows across the combined North/South database. This repository does **not** mirror the full release. It retrieves bounded sky regions, records the exact SQL/query provenance and SHA-256 hashes, and publishes derived experiment outputs.

## Hard rule: no mock results

Synthetic PM/HOD experiments used during method development are intentionally excluded from this repository. A result may be labelled `REAL_DR11` only when its input provenance records:

- DR11 table/release
- exact sky bounds and query
- retrieval timestamp
- row count
- SHA-256 of the retrieved input file
- tracer-selection rule
- train/validation/test region split

Artificial holes are allowed only as **evaluation masks applied to real observed catalogs**: the hidden points existed in DR11 before masking, so reconstruction accuracy is measured against real observations rather than simulated truth.

## First real-data experiment

The initial workflow downloads independent 0.5° × 0.5° DR11 fields and evaluates:

1. multiscale local-density pattern representation;
2. a 2-D pattern surface (PCA map) with region-held-out evaluation;
3. artificial central/random/corner holes in real observed density patches;
4. conditional reconstruction with Gaussian, PCA, nearest-neighbour and mean baselines;
5. boundary-ring → hidden-motif prediction (void / overdense / peak probabilities);
6. comparison of all primary sources versus a morphology-selected extended-source tracer sample.

The workflow is deliberately region-split: complete sky fields, not neighbouring patches, are held out to reduce spatial leakage.

## Repository layout

- `src/fetch_dr11.py` — anonymous public DR11 retrieval and provenance hashing
- `src/analyze_dr11.py` — real point-pattern manifold and hole-reconstruction experiments
- `data/real/dr11/pilot/` — bounded real DR11 inputs produced by CI
- `results/real_dr11/` — numerical results, plots and machine-readable summaries
- `docs/REAL_DATA_POLICY.md` — provenance and interpretation rules
- `.github/workflows/real-dr11-pilot.yml` — executes the full real-data pipeline on GitHub Actions

## Scientific interpretation

The reconstruction target is the **observed DR11 point field under the stated tracer selection**, not the unobserved matter density and not gravity itself. A successful hole-completion test means that neighbouring observed structure predicts held-out observed structure. Claims about matter density, gravity, or cosmological parameters require separate selection-function and theory tests.

## Data licensing and attribution

Code in this repository is Apache-2.0. Third-party survey data are not relicensed by this repository. Users must follow the Legacy Surveys / NOIRLab data-use and citation requirements.

Official references:

- https://www.legacysurvey.org/dr11/
- https://www.legacysurvey.org/dr11/files/
- https://datalab.noirlab.edu/data/legacy-surveys

## Status

The repository begins with the real-data acquisition and analysis pipeline only. Results are committed automatically after a successful GitHub Actions run; until that run succeeds, no numerical observational claim is made.
