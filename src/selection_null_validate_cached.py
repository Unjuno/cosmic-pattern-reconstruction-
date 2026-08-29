#!/usr/bin/env python3
"""Run the REAL_DR11 pixel selection-null test from hash-verified cached RA/Dec.

Brick identity is resolved by the existing tiny Data Lab center-cone query only.
The expensive full-brick source query is replaced by the exact provenance-
verified expanded48 RA/Dec file that already covers the selected central brick.
Official DR11 coadd depth/mask/NEXP/PSF maps are still fetched from NERSC.
No mock or simulated catalog is used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import selection_null_validate_fast as core
from analyze_dr11 import verify_and_load

PROV_PATH=Path('data/real/dr11/expanded48/provenance.json')
PROV=json.loads(PROV_PATH.read_text())
if PROV.get('status')!='REAL_DR11':
    raise RuntimeError('REAL_DR11 expanded48 provenance required')
REGIONS=PROV.get('regions',[])
BRICK_TO_META:dict[str,dict]={}
ORIG_CHOOSE=core.choose_brick_from_center


def _nearest_meta(ra:float,dec:float)->dict:
    best=None; best_d2=np.inf
    for r in REGIONS:
        dra=((float(r['center_ra_deg'])-ra+180.0)%360.0)-180.0
        d2=(dra*np.cos(np.deg2rad(dec)))**2+(float(r['center_dec_deg'])-dec)**2
        if d2<best_d2:
            best_d2=d2; best=r
    if best is None or best_d2>1e-8:
        raise RuntimeError(f'fixed center not found in REAL_DR11 provenance: {ra},{dec}')
    return best


def choose_brick_from_center(ra:float,dec:float):
    brick,query,near=ORIG_CHOOSE(ra,dec)
    meta=_nearest_meta(ra,dec)
    old=BRICK_TO_META.get(brick)
    if old is not None and old['name']!=meta['name']:
        raise RuntimeError(f'duplicate brick mapping: {brick}')
    BRICK_TO_META[brick]=meta
    return brick,query,near


def source_catalog(brick:str):
    meta=BRICK_TO_META.get(brick)
    if meta is None:
        raise RuntimeError(f'no cached provenance mapping for brick {brick}')
    df=verify_and_load(meta)
    return df,f"CACHED_REAL_DR11_EXPANDED48:{meta['file']}"

core.choose_brick_from_center=choose_brick_from_center
core.source_catalog=source_catalog

if __name__=='__main__':
    core.main()
