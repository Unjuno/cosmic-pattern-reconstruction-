#!/usr/bin/env python3
"""Direct-file entry point for REAL_DR11 pixel selection-null validation.

This wrapper replaces only the large Data Lab per-brick source query with the
official DR11 Tractor FITS file for that same brick. Brick selection still uses
a tiny fixed-center cone query and does not rank by source density.
"""
from __future__ import annotations
import hashlib, io
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
import selection_null_validate_fast as base

TRACTOR_BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor'

def native_array(a):
    a=np.asarray(a)
    if a.dtype.byteorder not in ('=','|'):
        return a.astype(a.dtype.newbyteorder('='),copy=True)
    return a.copy()

def source_catalog_direct(brick):
    url=f'{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits'
    r=requests.get(url,timeout=180); r.raise_for_status(); b=r.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'not FITS {url}: {b[:40]!r}')
    with fits.open(io.BytesIO(b),memmap=False) as H:
        t=H[1].data; cols={c.lower():c for c in t.names}
        need=['brick_primary','ra','dec']
        miss=[c for c in need if c not in cols]
        if miss: raise RuntimeError(f'missing Tractor columns {miss}')
        primary=native_array(t[cols['brick_primary']]).astype(bool)
        ra=native_array(t[cols['ra']]).astype(float)
        dec=native_array(t[cols['dec']]).astype(float)
    d=pd.DataFrame({'ra':ra[primary],'dec':dec[primary]})
    prov=(f"DIRECT_FITS {url} sha256={hashlib.sha256(b).hexdigest()} "
          f"rows_primary={len(d)}")
    return d,prov

base.source_catalog=source_catalog_direct

if __name__=='__main__':
    base.main()
