#!/usr/bin/env python3
"""Endian-safe entry point for the REAL_DR11 tracer-split experiment.

This wrapper changes only FITS-array byte order handling. The scientific
analysis in tracer_split_validate.py is otherwise unchanged.
"""
from __future__ import annotations
import hashlib, io
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
import tracer_split_validate as base


def native_array(a):
    a = np.asarray(a)
    if a.dtype.byteorder not in ('=', '|'):
        return a.astype(a.dtype.newbyteorder('='), copy=True)
    return a.copy()


def get_tractor_native(brick):
    url = f'{base.TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits'
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    b = r.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'not FITS {url}: {b[:40]!r}')
    with fits.open(io.BytesIO(b), memmap=False) as H:
        t = H[1].data
        cols = {c.lower(): c for c in t.names}
        need = ['brick_primary', 'type', 'ra', 'dec', 'bx', 'by', 'ref_cat']
        miss = [c for c in need if c not in cols]
        if miss:
            raise RuntimeError(f'missing columns {miss}')
        d = pd.DataFrame({c: native_array(t[cols[c]]) for c in need})
    for c in ['type', 'ref_cat']:
        d[c] = d[c].map(
            lambda x: x.decode().strip() if isinstance(x, (bytes, bytearray)) else str(x).strip()
        )
    d = d[d.brick_primary.astype(bool)].copy()
    return d, {
        'url': url,
        'sha256': hashlib.sha256(b).hexdigest(),
        'bytes': len(b),
        'rows_primary': int(len(d)),
        'type_counts': {str(k): int(v) for k, v in d.type.value_counts().items()},
        'endian_fix': 'numeric FITS arrays converted to native byte order before pandas',
    }


base.get_tractor = get_tractor_native

if __name__ == '__main__':
    base.main()
