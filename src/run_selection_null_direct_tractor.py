#!/usr/bin/env python3
"""Run the 12-brick REAL_DR11 pixel selection-null test using official Tractor FITS.

This wrapper replaces the slow Data Lab full-brick source query with the
corresponding official DR11 south/tractor file. Brick selection still uses the
small center cone query from selection_null_validate_fast.py.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
from astropy.io import fits

import selection_null_validate_fast as base

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
                tabhdu = next(h for h in H if getattr(h.data, "dtype", None) is not None and h.data.dtype.names)
                tab = tabhdu.data
                names = {n.lower(): n for n in tab.dtype.names}
                if "ra" not in names or "dec" not in names:
                    raise RuntimeError(f"RA/DEC absent in {url}: {tab.dtype.names}")
                keep = np.ones(len(tab), dtype=bool)
                if "brick_primary" in names:
                    keep &= np.asarray(tab[names["brick_primary"]]).astype(bool)
                df = pd.DataFrame({
                    "ra": np.asarray(tab[names["ra"]], dtype=float)[keep],
                    "dec": np.asarray(tab[names["dec"]], dtype=float)[keep],
                })
            if len(df) == 0:
                raise RuntimeError(f"zero BRICK_PRIMARY sources in {url}")
            return df, f"OFFICIAL_TRACTOR_FITS {url} sha256={base.sha256(b)} bytes={len(b)}"
        except Exception as e:
            last = e
            if i + 1 < 4:
                import time
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"official tractor download failed for {brick}: {last}")


base.source_catalog = source_catalog_direct

if __name__ == "__main__":
    base.main()
