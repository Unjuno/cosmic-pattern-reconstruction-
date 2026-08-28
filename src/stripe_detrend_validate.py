#!/usr/bin/env python3
"""Aggressive RA/Dec stripe-nuisance removal on 48 REAL_DR11 fields.

For each square tangent-plane log-count map, remove all additive row and column
means: z_res = z - row_mean - column_mean + global_mean.  This deliberately
removes both possible survey striping and some real large-scale structure.  We
then ask whether local ring-to-hidden correlation still beats a within-field
shift null.  No simulated catalog is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,binomtest,wilcoxon
from analyze_dr11 import verify_and_load

GRID=64; PATCH=16; STRIDE=8; HALF=.25
H=np.zeros((PATCH,PATCH),bool); H[4:12,4:12]=True
R=np.zeros((PATCH,PATCH),bool); R[3,3:13]=True; R[12,3:13]=True; R[4:12,3]=True; R[4:12,12]=True


def tangent_grid(df,meta):
    ra0=float(meta['center_ra_deg']); dec0=float(meta['center_dec_deg']); c=float(np.cos(np.deg2rad(dec0))); half=HALF*c
    dra=((df.ra.to_numpy(float)-ra0+180.0)%360.0)-180.0; x=dra*c; y=df.dec.to_numpy(float)-dec0
    keep=(np.abs(x)<half)&(np.abs(y)<half)
    g,_,_=np.histogram2d(y[keep],x[keep],bins=GRID,range=[[-half,half],[-half,half]])
    z=np.log1p(g); med=np.median(z); sc=np.median(np.abs(z-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6:sc=np.std(z)
    if not np.isfinite(sc) or sc<1e-6:sc=1.0
    return (z-med)/sc


def detrend(z): return z-z.mean(axis=1,keepdims=True)-z.mean(axis=0,keepdims=True)+z.mean()

def patch_corr(z):
    rr=[];hh=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            p=z[y:y+PATCH,x:x+PATCH]; rr.append(float(p[R].mean())); hh.append(float(p[H].mean()))
    return float(spearmanr(rr,hh).statistic),np.asarray(rr),np.asarray(hh)

def paired_gt0(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; pos=int((d>0).sum()); n=len(d)
    return {'n_fields':n,'positive_fields':pos,'exact_sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),
            'wilcoxon_one_sided_p':float(wilcoxon(d,alternative='greater').pvalue),'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/expanded48'); ap.add_argument('--out',default='results/real_dr11/anisotropy48'); args=ap.parse_args()
    root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((root/'provenance.json').read_text()); regs=prov.get('regions',[])
    if prov.get('status')!='REAL_DR11' or len(regs)!=48:raise RuntimeError('48-field REAL_DR11 provenance required')
    rows=[]
    for r in regs:
        z=tangent_grid(verify_and_load(r),r); raw,_,_=patch_corr(z); res,rr,hh=patch_corr(detrend(z)); null=float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic)
        rows.append({'field':r['name'],'raw_ring_hidden_spearman':raw,'stripe_detrended_spearman':res,'shift_null_spearman':null,
                     'detrended_minus_shift':res-null,'detrended_minus_raw':res-raw})
    df=pd.DataFrame(rows); df.to_csv(out/'stripe_detrend_field_metrics.csv',index=False)
    summary={'status':'REAL_DR11','validation':'48-field square-tangent-plane row/column nuisance removal','model_input_columns':['ra','dec'],
             'median_raw_spearman':float(np.median(df.raw_ring_hidden_spearman)),
             'median_stripe_detrended_spearman':float(np.median(df.stripe_detrended_spearman)),
             'median_shift_null_spearman':float(np.median(df.shift_null_spearman)),
             'detrended_minus_shift':paired_gt0(df.detrended_minus_shift),
             'interpretation':'A drop after row/column removal quantifies an axis-separable contribution. Significant residual locality above shift null shows that simple additive striping cannot explain the full signal.'}
    (out/'stripe_detrend_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
