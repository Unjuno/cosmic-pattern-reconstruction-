#!/usr/bin/env python3
"""Run the REAL_DR11 pixel selection-null test from hash-verified cached RA/Dec.

The scientific/statistical implementation remains in
``selection_null_validate_fast.py``.  This wrapper removes all Data Lab calls
from the selection-analysis stage: fixed center -> brick mappings are read
from ``selection_bricks24.json`` created during SHA-verified materialization,
and source catalogs are the exact provenance-verified expanded48 RA/Dec files.
Official DR11 coadd depth/mask/NEXP/PSF maps are fetched from NERSC.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import selection_null_validate_fast as core
from analyze_dr11 import verify_and_load

PROV_PATH=Path('data/real/dr11/expanded48/provenance.json')
MAP_PATH=Path('data/real/dr11/expanded48/selection_bricks24.json')
PROV=json.loads(PROV_PATH.read_text()); MAP=json.loads(MAP_PATH.read_text())
if PROV.get('status')!='REAL_DR11': raise RuntimeError('REAL_DR11 expanded48 provenance required')
if MAP.get('status')!='REAL_DR11_BRICK_MAPPING' or int(MAP.get('n_fields',0))<12:
    raise RuntimeError('REAL_DR11 brick mapping with at least 12 fixed fields required')
REGIONS=PROV.get('regions',[]); MAP_REGIONS=MAP.get('regions',[]); MAP_BY_NAME={r['name']:r for r in MAP_REGIONS}; MAPPED_NAMES=set(MAP_BY_NAME)
BRICK_TO_META:dict[str,dict]={}


def _nearest_meta(ra:float,dec:float)->dict:
    best=None; best_d2=np.inf
    for r in REGIONS:
        if r['name'] not in MAPPED_NAMES: continue
        dra=((float(r['center_ra_deg'])-ra+180.0)%360.0)-180.0
        d2=(dra*np.cos(np.deg2rad(dec)))**2+(float(r['center_dec_deg'])-dec)**2
        if d2<best_d2:best_d2=d2;best=r
    if best is None or best_d2>1e-8:
        raise RuntimeError(f'fixed center not found in materialized REAL_DR11 provenance: {ra},{dec}')
    return best


def choose_brick_from_center(ra:float,dec:float):
    meta=_nearest_meta(ra,dec); m=MAP_BY_NAME.get(meta['name'])
    if m is None: raise RuntimeError(f"no materialized brick mapping for {meta['name']}")
    if abs(float(m['center_ra_deg'])-ra)>1e-9 or abs(float(m['center_dec_deg'])-dec)>1e-9:
        raise RuntimeError(f"brick sidecar center mismatch for {meta['name']}")
    brick=str(m['brick']).strip()
    if brick in BRICK_TO_META and BRICK_TO_META[brick]['name']!=meta['name']:
        raise RuntimeError(f'duplicate brick mapping: {brick}')
    BRICK_TO_META[brick]=meta
    return brick,f"MATERIALIZED_BRICK_MAPPING:{MAP_PATH}:{meta['name']}",float(m['nearest_primary_source_deg'])


def source_catalog(brick:str):
    meta=BRICK_TO_META.get(brick)
    if meta is None: raise RuntimeError(f'no provenance mapping for brick {brick}')
    df=verify_and_load(meta)
    return df,f"CACHED_REAL_DR11_EXPANDED48:{meta['file']}"

core.choose_brick_from_center=choose_brick_from_center
core.source_catalog=source_catalog

if __name__=='__main__': core.main()
