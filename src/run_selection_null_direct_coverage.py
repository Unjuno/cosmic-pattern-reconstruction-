#!/usr/bin/env python3
"""Coverage-gated REAL_DR11 selection null using official Tractor FITS sources.

The existing selection_null_validate_coverage.py logic is preserved exactly.
Only its cached-source adapter is replaced at import time with the official
DR11 south Tractor file for each selected brick. This removes the expensive
Data Lab materialization/full-brick source query while preserving the
pre-registered field order and observing-coverage-only gate.
"""
from __future__ import annotations

import io
import sys
import time
import types

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

import selection_null_validate_fast as core

TRACTOR_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor"


def source_catalog_direct(brick: str):
    url = f"{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits"
    last = None
    for i in range(4):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            b = r.content
            if not b.startswith(b"SIMPLE"):
                raise RuntimeError(f"not FITS: {b[:60]!r}")
            with fits.open(io.BytesIO(b), memmap=False) as H:
                h = next(x for x in H if getattr(x.data, "dtype", None) is not None and x.data.dtype.names)
                tab = h.data
                names = {n.lower(): n for n in tab.dtype.names}
                if "ra" not in names or "dec" not in names:
                    raise RuntimeError(f"RA/DEC missing: {tab.dtype.names}")
                keep = np.ones(len(tab), dtype=bool)
                if "brick_primary" in names:
                    keep &= np.asarray(tab[names["brick_primary"]]).astype(bool)
                df = pd.DataFrame({
                    "ra": np.asarray(tab[names["ra"]], dtype=float)[keep],
                    "dec": np.asarray(tab[names["dec"]], dtype=float)[keep],
                })
            if len(df) == 0:
                raise RuntimeError("zero BRICK_PRIMARY rows")
            return df, f"OFFICIAL_TRACTOR_FITS:{url}:sha256={core.sha256(b)}:bytes={len(b)}"
        except Exception as e:
            last = e
            if i + 1 < 4:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"official Tractor download failed for {brick}: {last}")


core.source_catalog = source_catalog_direct

# selection_null_validate_coverage imports `selection_null_validate_cached` only
# to obtain its `core` object. Supply a minimal in-memory adapter so its
# statistical/coverage code remains unchanged.
adapter = types.ModuleType("selection_null_validate_cached")
adapter.core = core
sys.modules["selection_null_validate_cached"] = adapter

import selection_null_validate_coverage as coverage

if __name__ == "__main__":
    coverage.main()
