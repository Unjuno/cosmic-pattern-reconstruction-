#!/usr/bin/env python3
"""Coverage-aware REAL_DR11 pixel selection-null entry point.

For each fixed sky center, candidate BRICK_PRIMARY bricks are ordered by angular
proximity. The first candidate with enough valid depth/NEXP/primary support is
used. Source density is never used in brick selection.
"""
from __future__ import annotations
import numpy as np
import selection_null_validate_direct as direct

base = direct.base


def valid_patch_count(brick):
    urls=base.product_urls(brick)
    depth,_,_=base.read_image(urls['depth_r'],'DEPTH_R')
    mask,_,_=base.read_image(urls['maskbits'],'MASKBITS')
    nexp,_,_=base.read_image(urls['nexp_r'])
    if not (depth.shape==mask.shape==nexp.shape):
        return 0
    sm=base.sample_cells(np.asarray(mask,np.int64))
    sd=base.sample_cells(np.asarray(depth,float))
    sn=base.sample_cells(np.asarray(nexp,float))
    primary=((sm & 1)==0).mean(axis=(2,3))
    depth_cov=(np.isfinite(sd)&(sd>0)).mean(axis=(2,3))
    nexp_cov=(np.isfinite(sn)&(sn>0)).mean(axis=(2,3))
    valid=(primary>=0.9375)&(depth_cov>=0.9375)&(nexp_cov>=0.9375)
    n=0
    for y in range(0,base.GRID-base.PATCH+1,base.STRIDE):
        for x in range(0,base.GRID-base.PATCH+1,base.STRIDE):
            v=valid[y:y+base.PATCH,x:x+base.PATCH]
            if v.mean()>=0.98 and np.all(v[base.HIDDEN]) and np.all(v[base.RING]):
                n+=1
    return n


def choose_brick_coverage(ra,dec):
    seen=set(); candidates=[]; last_sql=None
    for radius in [0.06,0.12,0.25,0.40]:
        sql=(f"SELECT brickname,ra,dec FROM {base.TABLE} WHERE brick_primary=1 "
             f"AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{radius:.8f})")
        last_sql=sql; d=base.query_df(sql)
        if len(d)==0:
            continue
        d=d.copy(); d['_d2']=base.tangent_sep2(d.ra,d.dec,ra,dec)
        for brick,g in d.groupby('brickname'):
            b=str(brick).strip()
            if b in seen: continue
            seen.add(b); candidates.append((float(g['_d2'].min()),b))
        candidates.sort()
        # Try the closest candidates first; acceptance uses only official selection-map coverage.
        for d2,b in candidates:
            n=valid_patch_count(b)
            print(f'[selection-null12] candidate {b}: distance={np.sqrt(d2):.4f} deg valid_patches={n}',flush=True)
            if n>=8:
                return b, last_sql + f" /* chosen by distance+selection coverage; valid_patches={n} */", float(np.sqrt(d2))
    raise RuntimeError(f'no nearby brick with >=8 valid survey patches at {ra},{dec}; candidates={len(candidates)}')


base.choose_brick_from_center=choose_brick_coverage

if __name__=='__main__':
    base.main()
