#!/usr/bin/env python3
"""Materialize the first 12 provenance-fixed REAL_DR11 expanded fields.

The repository already records the accepted centers, original q3c queries,
row counts and SHA-256 hashes.  This script re-runs only the first 12 indexed
queries, adding ``brickname`` solely to record the fixed center -> coadd-brick
mapping.  The science RA/Dec CSV is reconstructed exactly as before and is
written only if row count, canonical CSV SHA and deterministic gzip SHA all
match the recorded provenance.  No field selection is repeated and no mock
fallback exists.
"""
from __future__ import annotations
import gzip, hashlib, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from dl import queryClient as qc

PROV=Path('data/real/dr11/expanded48/provenance.json')
SIDECAR=Path('data/real/dr11/expanded48/selection_bricks12.json')
N=12

def sha(b): return hashlib.sha256(b).hexdigest()

def query(sql,attempts=5):
    last=None
    for i in range(attempts):
        try:
            out=qc.query(sql=sql,fmt='csv',async_=False)
            if isinstance(out,bytes): out=out.decode()
            if not isinstance(out,str): raise RuntimeError(type(out))
            return out
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(3*(i+1))
    raise RuntimeError(f'query failed: {last}')

def tangent_sep2(ra,dec,ra0,dec0):
    dra=((np.asarray(ra,float)-ra0+180.0)%360.0)-180.0
    return (dra*np.cos(np.deg2rad(dec0)))**2+(np.asarray(dec,float)-dec0)**2

def main():
    p=json.loads(PROV.read_text())
    if p.get('status')!='REAL_DR11' or len(p.get('regions',[]))<N:
        raise RuntimeError('REAL_DR11 expanded48 provenance required')
    mappings=[]
    for i,r in enumerate(p['regions'][:N]):
        print(f"[materialize12] {i+1}/{N} {r['name']}",flush=True)
        original_sql=str(r['query'])
        if 'SELECT ra,dec' not in original_sql:
            raise RuntimeError(f"unexpected recorded query for {r['name']}")
        sql=original_sql.replace('SELECT ra,dec','SELECT ra,dec,brickname',1)
        d=pd.read_csv(io.StringIO(query(sql))); d.columns=[str(c).lower() for c in d.columns]
        if list(d.columns)!=['ra','dec','brickname']:
            raise RuntimeError(f"bad columns {list(d.columns)}")
        ra0=float(r['center_ra_deg']); dec0=float(r['center_dec_deg']); half=float(r.get('box_width_deg',.5))/2
        dra=((d.ra.astype(float)-ra0+180)%360)-180
        keep=dra.abs().le(half)&d.dec.astype(float).ge(dec0-half)&d.dec.astype(float).lt(dec0+half)
        d=d.loc[keep,['ra','dec','brickname']].reset_index(drop=True)
        if len(d)==0: raise RuntimeError(f"no rows after fixed box crop for {r['name']}")
        # Brick mapping is determined by the nearest already-BRICK_PRIMARY source
        # to the pre-registered field center.  No density ranking is involved.
        j=int(np.argmin(tangent_sep2(d.ra,d.dec,ra0,dec0)))
        brick=str(d.iloc[j].brickname).strip()
        near=float(np.sqrt(tangent_sep2([d.iloc[j].ra],[d.iloc[j].dec],ra0,dec0)[0]))
        # Rebuild the exact original science file from RA/Dec only.
        rd=d[['ra','dec']].sort_values(['ra','dec'],kind='mergesort').reset_index(drop=True)
        raw=rd.to_csv(index=False,lineterminator='\n').encode(); gz=gzip.compress(raw,compresslevel=9,mtime=0)
        if len(rd)!=int(r['rows']): raise RuntimeError(f"row mismatch {r['name']}: {len(rd)} != {r['rows']}")
        if sha(raw)!=r['canonical_csv_sha256']: raise RuntimeError(f"canonical SHA mismatch {r['name']}")
        if sha(gz)!=r['stored_gzip_sha256']: raise RuntimeError(f"gzip SHA mismatch {r['name']}")
        path=Path(r['file']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(gz)
        mappings.append({'name':r['name'],'center_ra_deg':ra0,'center_dec_deg':dec0,
                         'brick':brick,'nearest_primary_source_deg':near,
                         'mapping_query':sql,'science_file':str(path),
                         'canonical_csv_sha256':r['canonical_csv_sha256'],
                         'stored_gzip_sha256':r['stored_gzip_sha256']})
    side={'status':'REAL_DR11_BRICK_MAPPING','method':'nearest BRICK_PRIMARY source within the provenance-fixed q3c/box sample; no density ranking',
          'n_fields':N,'regions':mappings}
    SIDECAR.write_text(json.dumps(side,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'REAL_DR11','materialized_fields':N,'hash_verified':True,'brick_mapping_sidecar':str(SIDECAR)},indent=2))

if __name__=='__main__': main()
