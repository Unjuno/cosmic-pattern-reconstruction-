#!/usr/bin/env python3
"""Test whether boundary gradients continue through a REAL DR11 center hole."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from analyze_dr11 import normalize_grid, region_grid, verify_and_load
from lofo_validate import exact_signflip_p

GRID=64; PATCH=8


def patches(g:np.ndarray)->np.ndarray:
    return np.asarray([g[i*PATCH:(i+1)*PATCH,j*PATCH:(j+1)*PATCH] for i in range(GRID//PATCH) for j in range(GRID//PATCH)])


def vectors(A:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    left=A[:,2:6,1].mean(1); right=A[:,2:6,6].mean(1); top=A[:,1,2:6].mean(1); bottom=A[:,6,2:6].mean(1)
    pred=np.column_stack([right-left,bottom-top])
    inner=A[:,2:6,2:6]
    true=np.column_stack([inner[:,:,2:4].mean((1,2))-inner[:,:,0:2].mean((1,2)),inner[:,2:4,:].mean((1,2))-inner[:,0:2,:].mean((1,2))])
    return pred,true


def cosine(a:np.ndarray,b:np.ndarray)->np.ndarray:
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1); out=np.full(len(a),np.nan); k=den>1e-10; out[k]=(a[k]*b[k]).sum(1)/den[k]; return out


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/pilot'); ap.add_argument('--out',default='results/real_dr11/latest'); args=ap.parse_args()
    data,out=Path(args.data),Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((data/'provenance.json').read_text())
    if prov.get('status')!='REAL_DR11' or prov.get('model_input_columns')!=['ra','dec']: raise RuntimeError('REAL_DR11 RA/Dec-only provenance required')
    rows=[]
    for m in prov['regions']:
        g,_=normalize_grid(region_grid(verify_and_load(m),m)); A=patches(g); pred,true=vectors(A); real=cosine(pred,true)
        shifted=np.roll(A.reshape(8,8,8,8),shift=(2,3),axis=(0,1)).reshape(64,8,8); pred_null,_=vectors(shifted); null=cosine(pred_null,true)
        rows.append({'field':m['name'],'real_median_cosine':float(np.nanmedian(real)),'real_mean_cosine':float(np.nanmean(real)),'matched_shift_median_cosine':float(np.nanmedian(null)),'matched_shift_mean_cosine':float(np.nanmean(null)),'real_positive_fraction':float(np.nanmean(real>0)),'matched_shift_positive_fraction':float(np.nanmean(null>0))})
    df=pd.DataFrame(rows); df.to_csv(out/'gradient_continuity_field_metrics.csv',index=False)
    diff=df.real_mean_cosine-df.matched_shift_mean_cosine
    summary={'status':'REAL_DR11','validation':'12-field boundary-to-interior gradient continuity','model_input_columns':['ra','dec'],'real_median_of_field_mean_cosine':float(df.real_mean_cosine.median()),'matched_shift_median_of_field_mean_cosine':float(df.matched_shift_mean_cosine.median()),'positive_fields':int((diff>0).sum()),'mean_paired_advantage':float(diff.mean()),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))}
    (out/'gradient_continuity_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
