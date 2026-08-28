#!/usr/bin/env python3
"""Sky-axis anisotropy test on 48 REAL_DR11 position-only fields.

Question: is the measured local density continuity preferentially aligned with
RA/Dec axes, as a survey/image-processing artifact might be?  Each field is
regridded onto a physically square local tangent plane before comparing RA,
Dec and the two diagonal correlations at identical cell shifts.  Inputs are
provenance-verified RA/Dec only; no simulated catalog is used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from analyze_dr11 import verify_and_load

GRID=64
HALF_DEG=0.25
SHIFTS=[2,4,8,16]


def tangent_grid(df: pd.DataFrame, meta: dict) -> tuple[np.ndarray,float]:
    ra0=float(meta['center_ra_deg']); dec0=float(meta['center_dec_deg'])
    c=float(np.cos(np.deg2rad(dec0)))
    half=HALF_DEG*c
    dra=((df.ra.to_numpy(float)-ra0+180.0)%360.0)-180.0
    x=dra*c; y=df.dec.to_numpy(float)-dec0
    keep=(np.abs(x)<half)&(np.abs(y)<half)
    h,_,_=np.histogram2d(y[keep],x[keep],bins=GRID,range=[[-half,half],[-half,half]])
    z=np.log1p(h); med=np.median(z); sc=np.median(np.abs(z-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(z)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    return (z-med)/sc, (2.0*half/GRID*60.0)


def corr_shift(a: np.ndarray, dy: int, dx: int) -> float:
    if dy>=0:
        y0=slice(0,a.shape[0]-dy if dy else None); y1=slice(dy,None)
    else:
        y0=slice(-dy,None); y1=slice(0,dy)
    if dx>=0:
        x0=slice(0,a.shape[1]-dx if dx else None); x1=slice(dx,None)
    else:
        x0=slice(-dx,None); x1=slice(0,dx)
    u=a[y0,x0].ravel(); v=a[y1,x1].ravel()
    if len(u)<20 or np.std(u)==0 or np.std(v)==0: return float('nan')
    return float(spearmanr(u,v).statistic)


def paired_zero_summary(d: np.ndarray) -> dict:
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d)
    if n==0:return {'n_fields':0}
    pos=int((d>0).sum()); neg=int((d<0).sum()); nt=pos+neg
    signp=float(binomtest(pos,nt,.5,alternative='two-sided').pvalue) if nt else 1.0
    try: wp=float(wilcoxon(d,alternative='two-sided').pvalue)
    except Exception: wp=float('nan')
    return {'n_fields':n,'positive_fields':pos,'negative_fields':neg,
            'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d)),
            'exact_sign_test_two_sided_p':signp,'wilcoxon_two_sided_p':wp}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/expanded48'); ap.add_argument('--out',default='results/real_dr11/anisotropy48'); args=ap.parse_args()
    datadir=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((datadir/'provenance.json').read_text()); regs=prov.get('regions',[])
    if prov.get('status')!='REAL_DR11' or len(regs)!=48: raise RuntimeError('48-field REAL_DR11 provenance required')
    rows=[]
    for r in regs:
        df=verify_and_load(r); z,cell_arcmin=tangent_grid(df,r)
        for s in SHIFTS:
            ds=max(1,int(round(s/np.sqrt(2))))
            r_ra=corr_shift(z,0,s); r_dec=corr_shift(z,s,0)
            r_d1=corr_shift(z,ds,ds); r_d2=corr_shift(z,ds,-ds)
            rows.append({'field':r['name'],'dec_deg':float(r['center_dec_deg']),'shift_cells':s,
                         'actual_sep_arcmin':s*cell_arcmin,'cell_arcmin':cell_arcmin,
                         'rho_ra':r_ra,'rho_dec':r_dec,'rho_diag_pos':r_d1,'rho_diag_neg':r_d2,
                         'delta_ra_minus_dec':r_ra-r_dec,'delta_diag':r_d1-r_d2,
                         'mean_axis_rho':0.5*(r_ra+r_dec),'mean_diag_rho':0.5*(r_d1+r_d2)})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    scales=[]
    for shift,g in df.groupby('shift_cells'):
        axis=paired_zero_summary(g.delta_ra_minus_dec.to_numpy()); diag=paired_zero_summary(g.delta_diag.to_numpy())
        med_axis=float(np.nanmedian(g.mean_axis_rho)); med_delta=float(np.nanmedian(g.delta_ra_minus_dec))
        scales.append({'shift_cells':int(shift),'median_actual_sep_arcmin':float(np.nanmedian(g.actual_sep_arcmin)),
                       'n_fields':int(len(g)),'median_rho_ra':float(np.nanmedian(g.rho_ra)),
                       'median_rho_dec':float(np.nanmedian(g.rho_dec)),
                       'median_rho_diag_pos':float(np.nanmedian(g.rho_diag_pos)),
                       'median_rho_diag_neg':float(np.nanmedian(g.rho_diag_neg)),
                       'median_mean_axis_rho':med_axis,
                       'median_axis_delta_fraction_of_mean':float(med_delta/(abs(med_axis)+1e-12)),
                       'axis_anisotropy':axis,'diagonal_anisotropy':diag})
    summary={'status':'REAL_DR11','validation':'48 independent fields; square-tangent-plane sky-axis anisotropy',
             'model_input_columns':['ra','dec'],'scales':scales,
             'interpretation_rule':'A consistent signed RA-vs-Dec difference supports a sky-axis-aligned survey/processing contribution. It does not quantify how much of the isotropic component is cosmological.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
