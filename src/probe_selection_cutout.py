#!/usr/bin/env python3
"""Probe official DR11 FITS cutout paths for invvar/maskbits.

This is an acquisition/format test only. No science result is emitted.
"""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path
import requests
from astropy.io import fits

URL='https://www.legacysurvey.org/viewer/fits-cutout'
BASE={
    'ra':'156.0','dec':'5.0','layer':'ls-dr11-south',
    'pixscale':'14.0625','size':'128','bands':'r',
    'invvar':'1','maskbits':'1',
}

def inspect(params):
    r=requests.get(URL,params=params,timeout=120)
    r.raise_for_status(); b=r.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'viewer did not return FITS: status={r.status_code} content-type={r.headers.get("content-type")} head={b[:80]!r}')
    hdus=[]
    with fits.open(io.BytesIO(b),memmap=False) as H:
        for i,h in enumerate(H):
            data=h.data
            finite=None if data is None else data[data==data]
            hdus.append({
                'index':i,'name':h.name,'shape':None if data is None else list(data.shape),
                'dtype':None if data is None else str(data.dtype),
                'extname':h.header.get('EXTNAME'),'imagetype':h.header.get('IMAGETYP'),
                'bands':h.header.get('BANDS'),
                'min':None if finite is None or finite.size==0 else float(finite.min()),
                'max':None if finite is None or finite.size==0 else float(finite.max()),
            })
    return {'source_url':r.url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'hdus':hdus}

def main():
    out=Path('results/real_dr11/selection_probe'); out.mkdir(parents=True,exist_ok=True)
    variants={
        'resampled':dict(BASE),
        'subimage':{**BASE,'subimage':'1'},
    }
    results={}
    for name,params in variants.items():
        print('[probe]',name,flush=True)
        try: results[name]=inspect(params)
        except Exception as exc: results[name]={'error':repr(exc)}
    summary={'status':'REAL_DR11_SELECTION_PROBE','variants':results}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    if all('error' in x for x in results.values()): raise RuntimeError('all selection-map probe paths failed')
if __name__=='__main__': main()
