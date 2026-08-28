#!/usr/bin/env python3
"""Sky-axis anisotropy test on 48 REAL_DR11 position-only fields.

Question: is the measured local density continuity preferentially aligned with
RA/Dec axes, as a survey/image-processing artifact might be?  At matched
angular separations we compare RA, Dec, and the two diagonal correlations.
Inputs are provenance-verified RA/Dec only; no simulated catalog is used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from analyze_dr11 import verify_and_load, region_grid, normalize_grid

GRID=64
CELL_DEC_ARCMIN=0.5/GRID*60.0
Y_SHIFTS=[2,4,8,16]


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
    prov=json.loads((datadir/'provenance.json').read_text())
    regs=prov.get('regions',[])
    if prov.get('status')!='REAL_DR11' or len(regs)!=48: raise RuntimeError('48-field REAL_DR11 provenance required')
    rows=[]
    for r in regs:
        df=verify_and_load(r); g=region_grid(df,r); z,_=normalize_grid(g)
        dec=float(r['center_dec_deg']); c=max(0.25,float(np.cos(np.deg2rad(dec))))
        for sy in Y_SHIFTS:
            # Match the RA physical distance to sy Dec cells.
            sx=max(1,int(round(sy/c)))
            # Diagonal components target the same total separation.
            ddy=max(1,int(round(sy/np.sqrt(2))))
            ddx=max(1,int(round(sy/(np.sqrt(2)*c))))
            r_ra=corr_shift(z,0,sx); r_dec=corr_shift(z,sy,0)
            r_d1=corr_shift(z,ddy,ddx); r_d2=corr_shift(z,ddy,-ddx)
            sep_dec=sy*CELL_DEC_ARCMIN
            sep_ra=sx*CELL_DEC_ARCMIN*c
            sep_diag=np.sqrt((ddy*CELL_DEC_ARCMIN)**2+(ddx*CELL_DEC_ARCMIN*c)**2)
            rows.append({'field':r['name'],'dec_deg':dec,'target_sep_arcmin':sep_dec,
                         'ra_shift_cells':sx,'dec_shift_cells':sy,'diag_dx_cells':ddx,'diag_dy_cells':ddy,
                         'actual_ra_sep_arcmin':sep_ra,'actual_diag_sep_arcmin':sep_diag,
                         'rho_ra':r_ra,'rho_dec':r_dec,'rho_diag_pos':r_d1,'rho_diag_neg':r_d2,
                         'delta_ra_minus_dec':r_ra-r_dec,'delta_diag':r_d1-r_d2,
                         'mean_axis_rho':0.5*(r_ra+r_dec),'mean_diag_rho':0.5*(r_d1+r_d2)})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    scales=[]
    for sep,g in df.groupby('target_sep_arcmin'):
        axis=paired_zero_summary(g.delta_ra_minus_dec.to_numpy())
        diag=paired_zero_summary(g.delta_diag.to_numpy())
        scales.append({'target_sep_arcmin':float(sep),'n_fields':int(len(g)),
                       'median_rho_ra':float(np.nanmedian(g.rho_ra)),'median_rho_dec':float(np.nanmedian(g.rho_dec)),
                       'median_rho_diag_pos':float(np.nanmedian(g.rho_diag_pos)),'median_rho_diag_neg':float(np.nanmedian(g.rho_diag_neg)),
                       'median_mean_axis_rho':float(np.nanmedian(g.mean_axis_rho)),
                       'axis_anisotropy':axis,'diagonal_anisotropy':diag})
    summary={'status':'REAL_DR11','validation':'48 independent fields; sky-axis anisotropy at matched angular separation',
             'model_input_columns':['ra','dec'],'cell_dec_arcmin':CELL_DEC_ARCMIN,'scales':scales,
             'interpretation_rule':'A consistent signed RA-vs-Dec or diagonal difference would support a sky-axis-aligned survey artifact; near-zero differences do not prove cosmological origin.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
