#!/usr/bin/env python3
"""D4-symmetrized Gaussian hole reconstruction on 48 REAL_DR11 fields.

The empirical patch covariance is compared with a D4-symmetrized covariance
obtained by augmenting training patches with 90-degree rotations and
reflections.  If sky-axis anisotropy carries genuine reconstruction signal,
symmetrization should hurt; if it is nuisance/noise, symmetrization may help.
No simulated catalog is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest,wilcoxon
from analyze_dr11 import verify_and_load

GRID=64; PATCH=8; HALF=.25
H=np.zeros((PATCH,PATCH),bool); H[2:6,2:6]=True
HID=np.flatnonzero(H.ravel()); OBS=np.flatnonzero(~H.ravel())


def tangent_grid(df,meta):
    ra0=float(meta['center_ra_deg']); dec0=float(meta['center_dec_deg']); c=float(np.cos(np.deg2rad(dec0))); half=HALF*c
    dra=((df.ra.to_numpy(float)-ra0+180.0)%360.0)-180.0; x=dra*c; y=df.dec.to_numpy(float)-dec0
    keep=(np.abs(x)<half)&(np.abs(y)<half)
    g,_,_=np.histogram2d(y[keep],x[keep],bins=GRID,range=[[-half,half],[-half,half]])
    z=np.log1p(g); med=np.median(z); sc=np.median(np.abs(z-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6:sc=np.std(z)
    if not np.isfinite(sc) or sc<1e-6:sc=1.0
    return (z-med)/sc


def patchify(z):
    return np.asarray([z[y:y+PATCH,x:x+PATCH].ravel() for y in range(0,GRID,PATCH) for x in range(0,GRID,PATCH)])


def d4_augment(X):
    A=X.reshape(-1,PATCH,PATCH); out=[]
    for k in range(4):
        R=np.rot90(A,k,axes=(1,2)); out.append(R); out.append(np.flip(R,axis=2))
    return np.concatenate(out,axis=0).reshape(-1,PATCH*PATCH)


def predict(Xtr,Xte,d4=False):
    T=d4_augment(Xtr) if d4 else Xtr
    mu=T.mean(0); cov=np.cov(T,rowvar=False)
    coo=cov[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS)); coh=cov[np.ix_(OBS,HID)]
    return mu[HID]+(Xte[:,OBS]-mu[OBS])@np.linalg.solve(coo,coh)


def score(y,p):
    mse=float(np.mean((y-p)**2)); corr=float(np.corrcoef(y.ravel(),p.ravel())[0,1]) if np.std(p)>0 else float('nan')
    return mse,corr


def paired(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    return {'n_fields':n,'positive_fields':pos,'exact_sign_test_two_sided_p':float(binomtest(pos,n,.5).pvalue),
            'wilcoxon_two_sided_p':float(wilcoxon(d).pvalue),'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/expanded48'); ap.add_argument('--out',default='results/real_dr11/anisotropy48'); args=ap.parse_args()
    root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((root/'provenance.json').read_text()); regs=prov.get('regions',[])
    if prov.get('status')!='REAL_DR11' or len(regs)!=48:raise RuntimeError('48-field REAL_DR11 provenance required')
    fields={r['name']:patchify(tangent_grid(verify_and_load(r),r)) for r in regs}; names=list(fields); rows=[]
    for held in names:
        Xtr=np.concatenate([fields[f] for f in names if f!=held]); Xte=fields[held]; y=Xte[:,HID]
        pf=predict(Xtr,Xte,False); ps=predict(Xtr,Xte,True); pm=np.broadcast_to(Xtr.mean(0)[HID],y.shape)
        mf,cf=score(y,pf); ms,cs=score(y,ps); mm,cm=score(y,pm)
        rows.append({'field':held,'corr_empirical':cf,'corr_d4':cs,'corr_mean':cm,'mse_empirical':mf,'mse_d4':ms,'mse_mean':mm,
                     'delta_corr_d4_minus_empirical':cs-cf,'delta_mse_empirical_minus_d4':mf-ms})
    df=pd.DataFrame(rows); df.to_csv(out/'d4_reconstruction_field_metrics.csv',index=False)
    summary={'status':'REAL_DR11','validation':'48-field LOFO D4-symmetrized Gaussian reconstruction','model_input_columns':['ra','dec'],
             'median_corr_empirical':float(np.nanmedian(df.corr_empirical)),'median_corr_d4':float(np.nanmedian(df.corr_d4)),
             'median_corr_mean':float(np.nanmedian(df.corr_mean)),'median_mse_empirical':float(np.nanmedian(df.mse_empirical)),
             'median_mse_d4':float(np.nanmedian(df.mse_d4)),'median_mse_mean':float(np.nanmedian(df.mse_mean)),
             'corr_d4_minus_empirical':paired(df.delta_corr_d4_minus_empirical),'mse_empirical_minus_d4':paired(df.delta_mse_empirical_minus_d4),
             'interpretation':'D4 improvement means the sky-axis-aligned anisotropic component is not required for hole reconstruction and is better treated as nuisance/noise.'}
    (out/'d4_reconstruction_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
