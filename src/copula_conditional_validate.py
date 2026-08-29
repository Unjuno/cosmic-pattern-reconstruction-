#!/usr/bin/env python3
"""REAL DR11 Gaussian vs Gaussian-copula conditional hole reconstruction.

The copula uses empirical one-point marginals plus the training covariance
matrix only. Validation is whole-field leave-one-field-out on 48 real fields.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm,rankdata,binomtest,wilcoxon
from sklearn.metrics import roc_auc_score
GRID=64;PATCH=8;HALF=.25
H=np.zeros((8,8),bool);H[2:6,2:6]=1
R=np.zeros((8,8),bool);R[1,1:7]=1;R[6,1:7]=1;R[2:6,1]=1;R[2:6,6]=1
HID=np.flatnonzero(H.ravel());OBS=np.setdiff1d(np.arange(64),HID);RID=np.flatnonzero(R.ravel())
def tra(a,a0):return ((np.asarray(a,float)-a0+180)%360)-180
def grid(m,root):
 d=pd.read_csv(root/Path(m['file']).name);x=tra(d.ra,m['center_ra_deg']);y=d.dec-m['center_dec_deg'];k=(abs(x)<HALF)&(abs(y)<HALF);h,_,_=np.histogram2d(y[k],x[k],bins=GRID,range=[[-HALF,HALF],[-HALF,HALF]]);z=np.log1p(h);med=np.median(z);s=np.median(abs(z-med))*1.4826;s=s if s>1e-6 else np.std(z);return (z-med)/(s if s>1e-6 else 1)
def patch(z):return np.stack([z[y:y+8,x:x+8].ravel() for y in range(0,64,8) for x in range(0,64,8)])
def psd(a):a=(a+a.T)/2;w,v=np.linalg.eigh(a);return (v*np.clip(w,1e-8,None))@v.T
def gaussian(Xtr,Xte):
 mu=Xtr.mean(0);C=np.cov(Xtr,rowvar=False);Coo=C[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS));Coh=C[np.ix_(OBS,HID)];K=np.linalg.solve(Coo,Coh);mean=mu[HID]+(Xte[:,OBS]-mu[OBS])@K;cov=psd(C[np.ix_(HID,HID)]-C[np.ix_(HID,OBS)]@K);return mean,cov
def copula(Xtr,Xte):
 n=len(Xtr);S=np.sort(Xtr,axis=0);Ztr=norm.ppf(np.clip((rankdata(Xtr,axis=0,method='average')-.5)/n,1e-4,1-1e-4));Zte=np.empty_like(Xte)
 for j in range(Xtr.shape[1]):
  lo=np.searchsorted(S[:,j],Xte[:,j],side='left');hi=np.searchsorted(S[:,j],Xte[:,j],side='right');Zte[:,j]=norm.ppf(np.clip(((lo+hi)/2+.5)/n,1e-4,1-1e-4))
 mu=Ztr.mean(0);C=np.cov(Ztr,rowvar=False);Coo=C[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS));Coh=C[np.ix_(OBS,HID)];K=np.linalg.solve(Coo,Coh);mean=mu[HID]+(Zte[:,OBS]-mu[OBS])@K;cov=psd(C[np.ix_(HID,HID)]-C[np.ix_(HID,OBS)]@K);return mean,cov,S
def invemp(z,S):
 q=norm.cdf(z);n=len(S);xp=(np.arange(n)+.5)/n;out=np.empty_like(z)
 for k,j in enumerate(HID):out[...,k]=np.interp(q[...,k],xp,S[:,j],left=S[0,j],right=S[-1,j])
 return out
def auc(y,s):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def paired(d):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);p=int((d>0).sum());return {'n':n,'positive':p,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p_one_sided':float(binomtest(p,n,.5,alternative='greater').pvalue),'wilcoxon_p_one_sided':float(wilcoxon(d,alternative='greater').pvalue)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);P=json.loads((root/'provenance.json').read_text());names=[r['name'] for r in P['regions']];F={r['name']:patch(grid(r,root)) for r in P['regions']};rows=[];rng=np.random.default_rng(20260829)
 if P.get('status')!='REAL_DR11' or len(names)!=48:raise RuntimeError('48-field REAL_DR11 artifact required')
 for held in names:
  Xte=F[held];Xtr=np.concatenate([F[f] for f in names if f!=held]);truth=Xte[:,HID];tm=Xtr[:,HID].mean(1);em=Xte[:,HID].mean(1);tx=Xtr[:,HID].max(1);ex=Xte[:,HID].max(1);q25,q75=np.quantile(tm,[.25,.75]);qp=np.quantile(tx,.8);gm,gc=gaussian(Xtr,Xte);zm,zc,S=copula(Xtr,Xte);eps=rng.normal(size=(len(Xte),96,len(HID)));gs=gm[:,None,:]+eps@np.linalg.cholesky(gc).T;cs=invemp(zm[:,None,:]+eps@np.linalg.cholesky(zc).T,S);cm=cs.mean(1)
  rec={'field':held,'gaussian_corr':float(np.corrcoef(truth.ravel(),gm.ravel())[0,1]),'copula_corr':float(np.corrcoef(truth.ravel(),cm.ravel())[0,1]),'gaussian_mse':float(np.mean((truth-gm)**2)),'copula_mse':float(np.mean((truth-cm)**2)),'gaussian_cov90':float(np.mean((truth>=np.quantile(gs,.05,1))&(truth<=np.quantile(gs,.95,1)))),'copula_cov90':float(np.mean((truth>=np.quantile(cs,.05,1))&(truth<=np.quantile(cs,.95,1)))),'gaussian_width90':float(np.mean(np.quantile(gs,.95,1)-np.quantile(gs,.05,1))),'copula_width90':float(np.mean(np.quantile(cs,.95,1)-np.quantile(cs,.05,1)))};ring=Xte[:,RID].mean(1);gmean=gs.mean(2);cmean=cs.mean(2);gmax=gs.max(2);cmax=cs.max(2)
  for mot,y,rs,gp,cp in [('void',em<=q25,-ring,(gmean<=q25).mean(1),(cmean<=q25).mean(1)),('overdense',em>=q75,ring,(gmean>=q75).mean(1),(cmean>=q75).mean(1)),('peak',ex>=qp,ring,(gmax>=qp).mean(1),(cmax>=qp).mean(1))]:rec[f'{mot}_ring_auc']=auc(y,rs);rec[f'{mot}_gaussian_auc']=auc(y,gp);rec[f'{mot}_copula_auc']=auc(y,cp)
  rows.append(rec)
 D=pd.DataFrame(rows);D.to_csv(out/'field_metrics.csv',index=False);s={'status':'REAL_DR11_COPULA_CONDITIONAL','n_fields':48,'total_rows':int(P['total_rows']),'motifs':{},'exact':{}}
 for mot in ['void','overdense','peak']:s['motifs'][mot]={'ring_median_auc':float(np.nanmedian(D[f'{mot}_ring_auc'])),'gaussian_median_auc':float(np.nanmedian(D[f'{mot}_gaussian_auc'])),'copula_median_auc':float(np.nanmedian(D[f'{mot}_copula_auc'])),'copula_minus_ring':paired(D[f'{mot}_copula_auc']-D[f'{mot}_ring_auc']),'copula_minus_gaussian':paired(D[f'{mot}_copula_auc']-D[f'{mot}_gaussian_auc'])}
 for k in ['gaussian_corr','copula_corr','gaussian_mse','copula_mse','gaussian_cov90','copula_cov90','gaussian_width90','copula_width90']:s['exact'][k]=float(np.nanmedian(D[k]))
 (out/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s,indent=2,sort_keys=True))
if __name__=='__main__':main()
