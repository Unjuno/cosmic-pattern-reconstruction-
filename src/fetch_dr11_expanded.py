#!/usr/bin/env python3
"""Acquire 48 widely separated REAL DR11-South RA/Dec fields.

Candidate centers are a fixed all-sky grid shuffled with a fixed seed before any
survey query.  A candidate is accepted only if the exact 0.5 deg square has at
least MIN_ROWS primary DR11 sources and is >= MIN_SEP_DEG from previously
accepted centers.  Accepted/rejected candidates and all hashes are recorded.
There is no simulation fallback.
"""
from __future__ import annotations

import argparse, gzip, hashlib, io, json, math, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dl import queryClient as qc

TABLE='ls_dr11.tractor_s'; BOX_DEG=.50; QUERY_RADIUS_DEG=.40; TARGET_FIELDS=48; MIN_ROWS=5000; MIN_SEP_DEG=6.0


def sha256(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def angular_sep(ra1,dec1,ra2,dec2):
    r1,r2=np.deg2rad([ra1,ra2]); d1,d2=np.deg2rad([dec1,dec2]); c=np.sin(d1)*np.sin(d2)+np.cos(d1)*np.cos(d2)*np.cos(r1-r2); return float(np.rad2deg(np.arccos(np.clip(c,-1,1))))

def query(sql:str, attempts:int=4)->str:
    last=None
    for i in range(attempts):
        try:
            out=qc.query(sql=sql,fmt='csv',async_=False)
            if isinstance(out,bytes): out=out.decode('utf-8')
            if not isinstance(out,str) or len(out)<10: raise RuntimeError(f'unexpected response {type(out)}')
            return out
        except Exception as exc:
            last=exc
            if i+1<attempts: time.sleep(5*(i+1))
    raise RuntimeError(f'DR11 query failed: {last}')

def candidate_centers():
    cand=[(float(ra),float(dec)) for dec in [-55,-45,-35,-25,-15,-5,5,15,25] for ra in np.arange(0,360,12)]
    rng=np.random.default_rng(20260824); rng.shuffle(cand); return cand

def fetch_candidate(ra0,dec0):
    sql=f"SELECT ra,dec FROM {TABLE} WHERE brick_primary=1 AND q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS_DEG:.8f})"
    df=pd.read_csv(io.StringIO(query(sql))); df.columns=[str(c).lower() for c in df.columns]
    if list(df.columns)!=['ra','dec']: raise RuntimeError(f'unexpected columns {list(df.columns)}')
    half=BOX_DEG/2; dra=((df.ra.astype(float)-ra0+180)%360)-180
    keep=dra.abs().le(half)&df.dec.astype(float).ge(dec0-half)&df.dec.astype(float).lt(dec0+half)
    df=df.loc[keep,['ra','dec']].sort_values(['ra','dec'],kind='mergesort').reset_index(drop=True)
    return df,sql

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/real/dr11/expanded48'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    accepted=[]; trials=[]; started=datetime.now(timezone.utc).isoformat()
    for idx,(ra0,dec0) in enumerate(candidate_centers()):
        if len(accepted)>=TARGET_FIELDS: break
        if any(angular_sep(ra0,dec0,a['center_ra_deg'],a['center_dec_deg'])<MIN_SEP_DEG for a in accepted):
            trials.append({'candidate_index':idx,'ra':ra0,'dec':dec0,'status':'prequery_separation_reject'}); continue
        print(f'[DR11-48] candidate {idx}: ({ra0:.1f},{dec0:.1f})',flush=True)
        df,sql=fetch_candidate(ra0,dec0); n=len(df)
        if n<MIN_ROWS:
            trials.append({'candidate_index':idx,'ra':ra0,'dec':dec0,'status':'coverage_reject','rows':int(n)}); continue
        name=f'f{len(accepted):02d}_ra{int(ra0):03d}_{"p" if dec0>=0 else "m"}{abs(int(dec0)):02d}'
        raw=df.to_csv(index=False,lineterminator='\n').encode(); gz=gzip.compress(raw,compresslevel=9,mtime=0); path=out/f'{name}.csv.gz'; path.write_bytes(gz)
        rec={'name':name,'candidate_index':idx,'center_ra_deg':ra0,'center_dec_deg':dec0,'box_width_deg':BOX_DEG,'query_radius_deg':QUERY_RADIUS_DEG,'table':TABLE,'query':sql,'rows':int(n),'canonical_csv_sha256':sha256(raw),'stored_gzip_sha256':sha256(gz),'file':str(path)}
        accepted.append(rec); trials.append({'candidate_index':idx,'ra':ra0,'dec':dec0,'status':'accepted','rows':int(n),'name':name}); print(f'[DR11-48] accepted {name}: {n:,}',flush=True)
    if len(accepted)!=TARGET_FIELDS: raise RuntimeError(f'only {len(accepted)} acceptable fields; expected {TARGET_FIELDS}')
    prov={'status':'REAL_DR11','dataset':'DESI Legacy Imaging Surveys DR11','table':TABLE,'model_input_columns':['ra','dec'],'retrieved_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'field_selection':'fixed shuffled RA/Dec candidate grid; coverage threshold and pre-query angular-separation rule','candidate_seed':20260824,'target_fields':TARGET_FIELDS,'min_rows':MIN_ROWS,'min_separation_deg':MIN_SEP_DEG,'regions':accepted,'candidate_trials':trials,'total_rows':int(sum(r['rows'] for r in accepted))}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':'REAL_DR11','fields':len(accepted),'rows':prov['total_rows'],'trials':len(trials)},indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
