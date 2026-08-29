#!/usr/bin/env python3
"""REAL DR11 two-point sufficiency stress test using IAAFT surrogates.

IAAFT preserves each field's one-point distribution exactly and approximately
preserves its Fourier power while scrambling phase. No mock cosmology is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,wilcoxon
from sklearn.metrics import roc_auc_score
GRID=64;PATCH=8;HALF=.25
H=np.zeros((8,8),bool);H[2:6,2:6]=1
R=np.zeros((8,8),bool);R[1,1:7]=1;R[6,1:7]=1;R[2:6,1]=1;R[2:6,6]=1
HID=np.flatnonzero(H.ravel());OBS=np.setdiff1d(np.arange(64),HID);RID=np.flatnonzero(R.ravel())
def tra(a,a0):return ((np.asarray(a,float)-a0+180)%360)-180
def grid(m,root):
 d=pd.read_csv(root/Path(m['file']).name);x=tra(d.ra,m['center_ra_deg']);y=d.dec-m['center_dec_deg'];k=(abs(x)<HALF)&(abs(y)<HALF);h,_,_=np.histogram2d(y[k],x[k],bins=GRID,range=[[-HALF,HALF],[-HALF,HALF]]);z=np.log1p(h);med=np.median(z);s=np.median(abs(z-med))*1.4826;s=s if s>1e-6 else np.std(z);return (z-med)/(s if s>1e-6 else 1)
def iaaft(z,seed,niter=100):
 rng=np.random.default_rng(seed);vals=np.sort(z.ravel());amp=np.abs(np.fft.fft2(z));x=rng.permutation(z.ravel()).reshape(z.shape)
 for _ in range(niter):
  F=np.fft.fft2(x);ph=F/np.where(abs(F)>0,abs(F),1);y=np.fft.ifft2(amp*ph).real;o=np.argsort(y.ravel());f=np.empty(y.size);f[o]=vals;x=f.reshape(z.shape)
 return x
def patch(z):return np.array([z[y:y+8,x:x+8].ravel() for y in range(0,64,8) for x in range(0,64,8)])
def auc(y,s):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def evalmot(fields):
 rows=[];names=list(fields)
 for held in names:
  tr=np.concatenate([fields[f] for f in names if f!=held]);te=fields[held];tm=tr[:,HID].mean(1);em=te[:,HID].mean(1);tx=tr[:,HID].max(1);ex=te[:,HID].max(1);r=te[:,RID].mean(1);q25,q75=np.quantile(tm,[.25,.75]);qp=np.quantile(tx,.8)
  for mot,y,s in [('void',em<=q25,-r),('overdense',em>=q75,r),('peak',ex>=qp,r)]:rows.append({'field':held,'motif':mot,'auc':auc(y,s)})
 return pd.DataFrame(rows)
def rec(fields):
 rows=[];names=list(fields)
 for held in names:
  tr=np.concatenate([fields[f] for f in names if f!=held]);te=fields[held];mu=tr.mean(0);C=np.cov(tr,rowvar=False);Coo=C[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS));Coh=C[np.ix_(OBS,HID)];pred=mu[HID]+(te[:,OBS]-mu[OBS])@np.linalg.solve(Coo,Coh);y=te[:,HID];rows.append({'field':held,'corr':float(np.corrcoef(y.ravel(),pred.ravel())[0,1]),'mse':float(np.mean((y-pred)**2))})
 return pd.DataFrame(rows)
def paired(d):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);p=int((d>0).sum());return {'n':n,'positive':p,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p_two_sided':float(binomtest(p,n,.5).pvalue),'wilcoxon_p_two_sided':float(wilcoxon(d).pvalue)}
def radial(a,b):
 A=abs(np.fft.fft2(a))**2;B=abs(np.fft.fft2(b))**2;ky=np.fft.fftfreq(GRID);kx=np.fft.fftfreq(GRID);KX,KY=np.meshgrid(kx,ky);rr=np.rint(np.hypot(KX,KY)*GRID).astype(int);pa=[];pb=[]
 for k in range(1,rr.max()+1):
  m=rr==k
  if m.sum():pa.append(A[m].mean());pb.append(B[m].mean())
 pa=np.asarray(pa);pb=np.asarray(pb);return float(np.mean(abs(pa-pb)/(pa+1e-9))),float(np.corrcoef(np.log1p(pa),np.log1p(pb))[0,1])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);P=json.loads((root/'provenance.json').read_text());raw={};sur={};qa=[]
 if P.get('status')!='REAL_DR11' or len(P.get('regions',[]))!=48:raise RuntimeError('48-field REAL_DR11 artifact required')
 for i,m in enumerate(P['regions']):
  z=grid(m,root);s=iaaft(z,20260829+i);raw[m['name']]=patch(z);sur[m['name']]=patch(s);pe,pc=radial(z,s);qa.append({'field':m['name'],'sorted_max_abs_diff':float(np.max(abs(np.sort(z.ravel())-np.sort(s.ravel())))),'radial_power_mean_rel_error':pe,'radial_logpower_corr':pc})
 real=evalmot(raw);ia=evalmot(sur);real['sample']='real';ia['sample']='iaaft';D=pd.concat([real,ia]);D.to_csv(out/'motif_field_metrics.csv',index=False);Rr=rec(raw).set_index('field');Ri=rec(sur).set_index('field');re=Rr.join(Ri,lsuffix='_real',rsuffix='_iaaft');re.to_csv(out/'reconstruction_field_metrics.csv');Q=pd.DataFrame(qa);Q.to_csv(out/'surrogate_quality.csv',index=False)
 s={'status':'REAL_DR11_IAAFT_TWO_POINT_TEST','n_fields':48,'total_rows':int(P['total_rows']),'quality':{'onepoint_max_abs_diff':float(Q.sorted_max_abs_diff.max()),'radial_power_mean_rel_error_mean':float(Q.radial_power_mean_rel_error.mean()),'radial_logpower_corr_median':float(Q.radial_logpower_corr.median())},'motifs':{},'reconstruction':{}}
 for mot in ['void','overdense','peak']:
  w=D[D.motif==mot].pivot(index='field',columns='sample',values='auc');s['motifs'][mot]={'real_median_auc':float(w.real.median()),'iaaft_median_auc':float(w.iaaft.median()),'real_minus_iaaft':paired(w.real-w.iaaft)}
 s['reconstruction']={'real_median_corr':float(re.corr_real.median()),'iaaft_median_corr':float(re.corr_iaaft.median()),'real_minus_iaaft_corr':paired(re.corr_real-re.corr_iaaft),'real_median_mse':float(re.mse_real.median()),'iaaft_median_mse':float(re.mse_iaaft.median()),'iaaft_minus_real_mse':paired(re.mse_iaaft-re.mse_real)};(out/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s,indent=2,sort_keys=True))
if __name__=='__main__':main()
