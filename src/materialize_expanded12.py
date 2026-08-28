#!/usr/bin/env python3
"""Materialize the first 12 provenance-fixed REAL_DR11 expanded fields.

The repository already records the 48 accepted centers, queries, row counts and
SHA-256 hashes.  This script re-runs only the first 12 indexed q3c queries,
re-applies the original 0.5-degree box cut and sorting, and refuses to write a
file unless its canonical and gzip hashes exactly match the recorded
provenance.  No field selection is repeated and no mock fallback exists.
"""
from __future__ import annotations
import gzip, hashlib, io, json, time
from pathlib import Path
import pandas as pd
from dl import queryClient as qc

PROV=Path('data/real/dr11/expanded48/provenance.json')
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

def main():
    p=json.loads(PROV.read_text())
    if p.get('status')!='REAL_DR11' or len(p.get('regions',[]))<N: raise RuntimeError('REAL_DR11 expanded48 provenance required')
    for i,r in enumerate(p['regions'][:N]):
        print(f"[materialize12] {i+1}/{N} {r['name']}",flush=True)
        d=pd.read_csv(io.StringIO(query(r['query']))); d.columns=[str(c).lower() for c in d.columns]
        if list(d.columns)!=['ra','dec']: raise RuntimeError(f"bad columns {list(d.columns)}")
        ra0=float(r['center_ra_deg']); dec0=float(r['center_dec_deg']); half=float(r.get('box_width_deg',.5))/2
        dra=((d.ra.astype(float)-ra0+180)%360)-180
        keep=dra.abs().le(half)&d.dec.astype(float).ge(dec0-half)&d.dec.astype(float).lt(dec0+half)
        d=d.loc[keep,['ra','dec']].sort_values(['ra','dec'],kind='mergesort').reset_index(drop=True)
        raw=d.to_csv(index=False,lineterminator='\n').encode(); gz=gzip.compress(raw,compresslevel=9,mtime=0)
        if len(d)!=int(r['rows']): raise RuntimeError(f"row mismatch {r['name']}: {len(d)} != {r['rows']}")
        if sha(raw)!=r['canonical_csv_sha256']: raise RuntimeError(f"canonical SHA mismatch {r['name']}")
        if sha(gz)!=r['stored_gzip_sha256']: raise RuntimeError(f"gzip SHA mismatch {r['name']}")
        path=Path(r['file']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(gz)
    print(json.dumps({'status':'REAL_DR11','materialized_fields':N,'hash_verified':True},indent=2))

if __name__=='__main__': main()
