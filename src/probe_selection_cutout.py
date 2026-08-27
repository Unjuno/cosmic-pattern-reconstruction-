#!/usr/bin/env python3
"""Probe the official DR11 viewer FITS cutout with invvar+maskbits.

This is an acquisition/format test only. No science result is emitted.
"""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path
import requests
from astropy.io import fits

URL='https://www.legacysurvey.org/viewer/fits-cutout'
PARAMS={
    'ra':'156.0','dec':'5.0','layer':'ls-dr11-south',
    'pixscale':'3.515625','size':'512','bands':'r',
    'invvar':'1','maskbits':'1',
}

def main():
    out=Path('results/real_dr11/selection_probe'); out.mkdir(parents=True,exist_ok=True)
    r=requests.get(URL,params=PARAMS,timeout=180)
    r.raise_for_status(); b=r.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'viewer did not return FITS: status={r.status_code} content-type={r.headers.get("content-type")} head={b[:80]!r}')
    hdus=[]
    with fits.open(io.BytesIO(b),memmap=False) as H:
        for i,h in enumerate(H):
            data=h.data
            hdus.append({
                'index':i,'name':h.name,'shape':None if data is None else list(data.shape),
                'dtype':None if data is None else str(data.dtype),
                'extname':h.header.get('EXTNAME'),
                'bands':h.header.get('BANDS'),
            })
    summary={
        'status':'REAL_DR11_SELECTION_PROBE',
        'source_url':r.url,
        'sha256':hashlib.sha256(b).hexdigest(),
        'bytes':len(b),'hdus':hdus,
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
