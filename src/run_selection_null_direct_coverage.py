#!/usr/bin/env python3
"""Coverage-gated REAL_DR11 selection null with indexed fixed-cone sources.

Brick identity is resolved from a tiny center cone. Source positions are then
queried with the same Q3C-indexed 0.4-degree fixed-center query that was used
successfully for expanded48. The selected brick WCS performs the final spatial
cut. This avoids the slow unindexed `brickname=...` full-table query while
remaining a direct official DR11 observation query. No mock data are used.
"""
from __future__ import annotations
import sys,types
import selection_null_validate_fast as core

ORIG_CHOOSE=core.choose_brick_from_center
BRICK_CENTER:dict[str,tuple[float,float,str]]={}


def choose_indexed(ra:float,dec:float):
    brick,q,near=ORIG_CHOOSE(ra,dec)
    old=BRICK_CENTER.get(brick)
    if old is not None and (abs(old[0]-ra)>1e-9 or abs(old[1]-dec)>1e-9):
        raise RuntimeError(f'duplicate fixed-center brick {brick}')
    BRICK_CENTER[brick]=(ra,dec,q)
    return brick,q,near


def source_catalog_indexed(brick:str):
    if brick not in BRICK_CENTER:
        raise RuntimeError(f'no fixed center registered for {brick}')
    ra,dec,_=BRICK_CENTER[brick]
    sql=(f"SELECT ra,dec FROM {core.TABLE} WHERE brick_primary=1 "
         f"AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},0.40000000)")
    d=core.query_df(sql)
    if list(d.columns)!=['ra','dec']:
        raise RuntimeError(f'bad source columns: {list(d.columns)}')
    if len(d)==0:
        raise RuntimeError(f'zero REAL_DR11 rows around {ra},{dec}')
    return d,sql

core.choose_brick_from_center=choose_indexed
core.source_catalog=source_catalog_indexed

adapter=types.ModuleType('selection_null_validate_cached')
adapter.core=core
sys.modules['selection_null_validate_cached']=adapter

import selection_null_validate_coverage as coverage

if __name__=='__main__':
    coverage.main()
