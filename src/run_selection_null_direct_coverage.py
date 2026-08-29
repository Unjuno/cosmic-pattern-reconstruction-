#!/usr/bin/env python3
"""Coverage-gated REAL_DR11 selection null using cached provenance-verified sources.

Brick identity is resolved by the original small center-cone query. Source
positions are then read from the already committed expanded48 REAL_DR11 field
that contains that central brick, with SHA-256 verification via analyze_dr11.
Only official DR11 coadd selection products are newly downloaded. No mock or
simulated catalog is used.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
import numpy as np

import selection_null_validate_fast as core
from analyze_dr11 import verify_and_load

PROV_PATH=Path('data/real/dr11/expanded48/provenance.json')
PROV=json.loads(PROV_PATH.read_text())
if PROV.get('status')!='REAL_DR11':
    raise RuntimeError('REAL_DR11 expanded48 provenance required')
REGIONS=PROV.get('regions',[])
ORIG_CHOOSE=core.choose_brick_from_center
BRICK_TO_META:dict[str,dict]={}


def nearest_meta(ra:float,dec:float)->dict:
    best=None; best_d2=np.inf
    for r in REGIONS:
        dra=((float(r['center_ra_deg'])-ra+180.0)%360.0)-180.0
        d2=(dra*np.cos(np.deg2rad(dec)))**2+(float(r['center_dec_deg'])-dec)**2
        if d2<best_d2:
            best_d2=d2; best=r
    if best is None or best_d2>1e-8:
        raise RuntimeError(f'fixed center missing from REAL_DR11 provenance: {ra},{dec}')
    return best


def choose_cached(ra:float,dec:float):
    brick,query,near=ORIG_CHOOSE(ra,dec)
    meta=nearest_meta(ra,dec)
    old=BRICK_TO_META.get(brick)
    if old is not None and old['name']!=meta['name']:
        raise RuntimeError(f'duplicate central brick {brick}')
    BRICK_TO_META[brick]=meta
    return brick,query,near


def source_catalog_cached(brick:str):
    meta=BRICK_TO_META.get(brick)
    if meta is None:
        raise RuntimeError(f'no cached REAL_DR11 field registered for {brick}')
    df=verify_and_load(meta)
    return df,f"CACHED_REAL_DR11_EXPANDED48:{meta['file']}:sha256={meta['canonical_csv_sha256']}"

core.choose_brick_from_center=choose_cached
core.source_catalog=source_catalog_cached

# selection_null_validate_coverage imports this adapter only for `core`.
adapter=types.ModuleType('selection_null_validate_cached')
adapter.core=core
sys.modules['selection_null_validate_cached']=adapter

import selection_null_validate_coverage as coverage

if __name__=='__main__':
    coverage.main()
