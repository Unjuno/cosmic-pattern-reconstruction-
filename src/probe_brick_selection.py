#!/usr/bin/env python3
"""Probe one official DR11 coadd brick's selection products.

Queries only the brick name from Data Lab, then reads official NERSC-hosted
coadd products. No science statistic is computed here.
"""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path
import pandas as pd
import requests
from astropy.io import fits
from dl import queryClient as qc

BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd'
SQL="""SELECT brickname,COUNT(*) AS n FROM ls_dr11.tractor_s
WHERE brick_primary=1 AND q3c_radial_query(ra,dec,156.0,5.0,0.08)
GROUP BY brickname ORDER BY n DESC"""

def query():
    text=qc.query(sql=SQL,fmt='csv',async_=False)
    if isinstance(text,bytes): text=text.decode()
    from io import StringIO
    return pd.read_csv(StringIO(text))

def inspect_fits(url):
    r=requests.get(url,timeout=180); r.raise_for_status(); b=r.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'not FITS {url}: {b[:80]!r}')
    hs=[]
    with fits.open(io.BytesIO(b),memmap=False) as H:
        for i,h in enumerate(H):
            a=h.data
            hs.append({'index':i,'name':h.name,'shape':None if a is None else list(a.shape),'dtype':None if a is None else str(a.dtype),'extname':h.header.get('EXTNAME')})
    return {'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'hdus':hs}

def main():
    out=Path('results/real_dr11/selection_probe'); out.mkdir(parents=True,exist_ok=True)
    tab=query()
    if len(tab)==0: raise RuntimeError('no DR11 brick found')
    brick=str(tab.iloc[0].brickname).strip(); pre=brick[:3]
    products={
      'depth_r':f'{BASE}/{pre}/{brick}/legacysurvey-{brick}-depth-r.fits.fz',
      'maskbits':f'{BASE}/{pre}/{brick}/legacysurvey-{brick}-maskbits.fits.fz',
      'nexp_r':f'{BASE}/{pre}/{brick}/legacysurvey-{brick}-nexp-r.fits.fz',
      'psfsize_r':f'{BASE}/{pre}/{brick}/legacysurvey-{brick}-psfsize-r.fits.fz',
    }
    heads={}
    for name,url in products.items():
        rr=requests.head(url,allow_redirects=True,timeout=60)
        heads[name]={'url':url,'status':rr.status_code,'content_length':rr.headers.get('content-length'),'content_type':rr.headers.get('content-type')}
    # Download depth and maskbits to verify actual FITS decoding and size.
    decoded={}
    for name in ['depth_r','maskbits']:
        decoded[name]=inspect_fits(products[name])
    summary={'status':'REAL_DR11_BRICK_SELECTION_PROBE','query':SQL,'brick_candidates':tab.head(10).to_dict(orient='records'),'chosen_brick':brick,'heads':heads,'decoded':decoded}
    (out/'brick_probe.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
