#!/usr/bin/env python3
"""REAL DR11 exact full-2D two-point phase-randomization stress test.

Each 64x64 real DR11 field is first mapped monotonically to Gaussian normal scores.
A control field is then made by multiplying the *exact* 2D Fourier amplitude of
that Gaussianized field by the Hermitian phase of independent Gaussian noise.
Thus the control preserves the full anisotropic two-point spectrum to machine
precision while scrambling phase. This test asks whether the small REAL-IAAFT
reconstruction advantage survives once the two-point constraint is exact.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,wilcoxon,rankdata,norm,ks_2samp
from sklearn.metrics import roc_auc_score

GRID=64;PATCH=8;HALF=.25
H=np.zeros((8,8),bool);H[2:6,2:6]=1
R=np.zeros((8,8),bool);R[1,1:7]=1;R[6,1:7]=1;R[2:6,1]=1;R[2:6,6]=1
HID=np.flatnonzero(H.ravel());OBS=np.setdiff1d(np.arange(64),HID);RID=np.flatnonzero(R.ravel())

def tra(a,a0):return ((np.asarray(a,float)-a0+180)%360)-180

def grid(m,root):
 d=pd.read_csv(root/Path(m['file']).name);x=tra(d.ra,m['center_ra_deg']);y=d.dec-m['center_dec_deg'];k=(abs(x)<HALF)&(abs(y)<HALF)
 h,_,_=np.histogram2d(y[k],x[k],bins=GRID,range=[[-HALF,HALF],[-HALF,HALF]])
 z=np.log1p(h);med=np.median(z);s=np.median(abs(z-med))*1.4826;s=s if s>1e-6 else np.std(z)
 return (z-med)/(s if s>1e-6 else 1)

def gaussianize(z):
 r=rankdata(z.ravel(),method='average');u=(r-.5)/len(r);return norm.ppf(u).reshape(z.shape)

def exact_phase(g,seed):
 rng=np.random.default_rng(seed);noise=rng.normal(size=g.shape);N=np.fft.fft2(noise);phase=N/np.where(abs(N)>0,abs(N),1)
 amp=abs(np.fft.fft2(g));s=np.fft.ifft2(amp*phase).real;s+=g.mean()-s.mean();return s

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
  tr=np.concatenate([fields[f] for f in names if f!=held]);te=fields[held];mu=tr.mean(0);C=np.cov(tr,rowvar=False);Coo=C[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS));Coh=C[np.ix_(OBS,HID)];pred=mu[HID]+(te[:,OBS]-mu[OBS])@np.linalg.solve(Coo,Coh);y=te[:,HID]
  rows.append({'field':held,'corr':float(np.corrcoef(y.ravel(),pred.ravel())[0,1]),'mse':float(np.mean((y-pred)**2))})
 return pd.DataFrame(rows).set_index('field')

def paired(d):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);p=int((d>0).sum());return {'n':n,'positive':p,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p_two_sided':float(binomtest(p,n,.5).pvalue),'wilcoxon_p_two_sided':float(wilcoxon(d).pvalue)}

def quality(g,s):
 A=abs(np.fft.fft2(g));B=abs(np.fft.fft2(s));m=np.ones_like(A,dtype=bool);m[0,0]=False
 return {'full2d_amp_max_relerr':float(np.max(abs(A-B))/(np.max(A)+1e-12)),'full2d_amp_l1_relerr':float(np.mean(abs(A[m]-B[m]))/(np.mean(A[m])+1e-12)),'marginal_ks':float(ks_2samp(g.ravel(),s.ravel()).statistic)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 P=json.loads((root/'provenance.json').read_text())
 if P.get('status')!='REAL_DR11' or len(P.get('regions',[]))!=48:raise RuntimeError('48-field REAL_DR11 artifact required')
 real={};phase={};qa=[]
 for i,m in enumerate(P['regions']):
  g=gaussianize(grid(m,root));s=exact_phase(g,20260830+i);real[m['name']]=patch(g);phase[m['name']]=patch(s);qa.append({'field':m['name'],**quality(g,s)})
 Rr=rec(real);Rp=rec(phase);re=Rr.join(Rp,lsuffix='_real_gaussianized',rsuffix='_phase_control');re.to_csv(out/'reconstruction_field_metrics.csv')
 Mr=evalmot(real);Mp=evalmot(phase);Mr['sample']='real_gaussianized';Mp['sample']='exact_phase';M=pd.concat([Mr,Mp]);M.to_csv(out/'motif_field_metrics.csv',index=False)
 Q=pd.DataFrame(qa);Q.to_csv(out/'surrogate_quality.csv',index=False)
 s={'status':'REAL_DR11_EXACT_FULL2D_PHASE_TEST','n_fields':48,'total_rows':int(P['total_rows']),'quality':{'full2d_amp_max_relerr_median':float(Q.full2d_amp_max_relerr.median()),'full2d_amp_l1_relerr_median':float(Q.full2d_amp_l1_relerr.median()),'marginal_ks_median':float(Q.marginal_ks.median())},'reconstruction':{},'motifs':{}}
 d=re.corr_real_gaussianized-re.corr_phase_control;s['reconstruction']={'real_gaussianized_median_corr':float(re.corr_real_gaussianized.median()),'exact_phase_median_corr':float(re.corr_phase_control.median()),'real_minus_phase_corr':paired(d),'real_gaussianized_median_mse':float(re.mse_real_gaussianized.median()),'exact_phase_median_mse':float(re.mse_phase_control.median()),'phase_minus_real_mse':paired(re.mse_phase_control-re.mse_real_gaussianized)}
 for mot in ['void','overdense','peak']:
  w=M[M.motif==mot].pivot(index='field',columns='sample',values='auc');s['motifs'][mot]={'real_gaussianized_median_auc':float(w.real_gaussianized.median()),'exact_phase_median_auc':float(w.exact_phase.median()),'real_minus_phase':paired(w.real_gaussianized-w.exact_phase)}
 (out/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s,indent=2,sort_keys=True))
if __name__=='__main__':main()
