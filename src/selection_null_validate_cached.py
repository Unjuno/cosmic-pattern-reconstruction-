#!/usr/bin/env python3
"""Run the REAL_DR11 pixel selection-null test using already-provenanced RA/Dec catalogs.

This wrapper keeps the scientific/statistical implementation in
selection_null_validate_fast.py unchanged.  It replaces only the expensive
per-brick Data Lab source query with the REAL_DR11 expanded48 catalogs already
stored in the repository and verified by SHA-256 provenance.  The official
DR11 coadd depth/mask/NEXP/PSF maps are still fetched from NERSC.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import selection_null_validate_fast as core
from analyze_dr11 import verify_and_load

PROV_PATH = Path("data/real/dr11/expanded48/provenance.json")
PROV = json.loads(PROV_PATH.read_text())
if PROV.get("status") != "REAL_DR11":
    raise RuntimeError("REAL_DR11 expanded48 provenance required")
REGIONS = PROV.get("regions", [])

# Record which fixed field selected each brick.  choose_brick_from_center is
# still the original tiny Data Lab cone query; it does not rank by density.
BRICK_TO_META: dict[str, dict] = {}
_ORIG_CHOOSE = core.choose_brick_from_center


def _nearest_meta(ra: float, dec: float) -> dict:
    best = None
    best_d2 = np.inf
    for r in REGIONS:
        dra = ((float(r["center_ra_deg"]) - ra + 180.0) % 360.0) - 180.0
        d2 = (dra * np.cos(np.deg2rad(dec))) ** 2 + (float(r["center_dec_deg"]) - dec) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = r
    if best is None or best_d2 > 1e-8:
        raise RuntimeError(f"fixed center not found in REAL_DR11 provenance: {ra},{dec}")
    return best


def choose_brick_from_center(ra: float, dec: float):
    brick, sql, near = _ORIG_CHOOSE(ra, dec)
    meta = _nearest_meta(ra, dec)
    if brick in BRICK_TO_META and BRICK_TO_META[brick]["name"] != meta["name"]:
        raise RuntimeError(f"brick mapped to multiple fixed fields: {brick}")
    BRICK_TO_META[brick] = meta
    return brick, sql, near


def source_catalog(brick: str):
    meta = BRICK_TO_META.get(brick)
    if meta is None:
        raise RuntimeError(f"no provenance mapping for brick {brick}")
    # verify_and_load checks stored gzip SHA, canonical CSV SHA, row count and
    # enforces exactly ['ra','dec'] input columns.
    df = verify_and_load(meta)
    provenance_ref = f"CACHED_REAL_DR11_EXPANDED48:{meta['file']}"
    return df, provenance_ref


core.choose_brick_from_center = choose_brick_from_center
core.source_catalog = source_catalog

if __name__ == "__main__":
    core.main()
