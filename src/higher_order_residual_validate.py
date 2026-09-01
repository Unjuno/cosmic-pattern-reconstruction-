#!/usr/bin/env python3
"""Test for predictable higher-order structure beyond a Gaussian two-point model.

REAL_DR11 only. Each field is rank-Gaussianized. For every held-out field the
other 47 fields are split into 31 fields for the Gaussian conditional baseline
and 16 disjoint fields for a nonlinear residual corrector. The held-out field is
never used for either fit.

The identical protocol is applied to an exact-full-2D-amplitude, randomized-
phase surrogate. Evidence for higher-order structure requires the REAL residual
correction gain to exceed the surrogate gain across held-out fields.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm,binomtest,wilcoxon,spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

GRID=64;PATCH=8;HALF=.25
H=np.zeros((PATCH,PATCH),bool);H[2:6,2:6]=1
HID=np.flatnonzero(H.ravel());OBS=np.setdiff1d(np.arange(PATCH*PATCH),HID)

def tra(a,a0):return ((np.asarray(a,float)-a0+180)%360)-180

def grid(meta,root):
 d=pd.read_csv(root/Path(meta['file']).name)
 x=tra(d.ra,meta['center_ra_deg']);y=d.dec-meta['center_dec_deg'];k=(abs(x)<HALF)&(abs(y)<HALF)
 h,_,_=np.histogram2d(y[k],x[k],bins=GRID,range=[[-HALF,HALF],[-HALF,HALF]])
 z=np.log1p(h);med=np.median(z);s=np.median(abs(z-med))*1.4826;s=s if s>1e-6 else np.std(z)
 return (z-med)/(s if s>1e-6 else 1)

def gaussianize(z):
 r=rankdata(z.ravel(),method='average');u=(r-.5)/len(r);return norm.ppf(np.clip(u,1e-6,1-1e-6)).reshape(z.shape)

def exact_phase(g,seed):
 rng=np.random.default_rng(seed);n=rng.normal(size=g.shape);N=np.fft.fft2(n);ph=N/np.where(abs(N)>0,abs(N),1)
 amp=abs(np.fft.fft2(g));s=np.fft.ifft2(amp*ph).real;s+=g.mean()-s.mean();return s

def patch(z):return np.stack([z[y:y+PATCH,x:x+PATCH].ravel() for y in range(0,GRID,PATCH) for x in range(0,GRID,PATCH)])

def baseline_fit(X):
 mu=X.mean(0);C=np.cov(X,rowvar=False);Coo=C[np.ix_(OBS,OBS)]+.08*np.eye(len(OBS));Coh=C[np.ix_(OBS,HID)]
 return mu,np.linalg.solve(Coo,Coh)

def baseline_predict(model,X):
 mu,K=model;return (mu[HID]+(X[:,OBS]-mu[OBS])@K).mean(1)

def auc(y,s):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))

def paired(d,alternative='greater'):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);pos=int((d>0).sum())
 if n==0:return {'n':0}
 try:w=float(wilcoxon(d,alternative=alternative).pvalue)
 except Exception:w=float('nan')
 bp=float(binomtest(pos,n,.5,alternative=alternative if alternative in ('greater','less') else 'two-sided').pvalue)
 return {'n':n,'positive':pos,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p':bp,'wilcoxon_p':w}

def evaluate_sample(fields,names,sample_name):
 rows=[]
 for hi,held in enumerate(names):
  # Rotate field order so no fixed subset is always used for the residual model.
  order=names[hi+1:]+names[:hi]
  base_names=order[:31];corr_names=order[31:]
  Xbase=np.concatenate([fields[f] for f in base_names]);Xcorr=np.concatenate([fields[f] for f in corr_names]);Xte=fields[held]
  model=baseline_fit(Xbase);bc=baseline_predict(model,Xcorr);bt=baseline_predict(model,Xte)
  yc=Xcorr[:,HID].mean(1);yt=Xte[:,HID].mean(1);rc=yc-bc;rt=yt-bt
  Vc=Xcorr[:,OBS];Vt=Xte[:,OBS]
  hgb=HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=2.0,random_state=1700+hi).fit(Vc,rc)
  et=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=10,max_features=.75,n_jobs=-1,random_state=2700+hi).fit(Vc,rc)
  rh=hgb.predict(Vt);re=et.predict(Vt);ph=bt+rh;pe=bt+re
  bm=float(np.mean((yt-bt)**2));hm=float(np.mean((yt-ph)**2));em=float(np.mean((yt-pe)**2))
  # Label thresholds use all 47 non-test fields but not the test field.
  ytrain=np.concatenate([fields[f][:,HID].mean(1) for f in order]);q25,q75=np.quantile(ytrain,[.25,.75])
  rec={'sample':sample_name,'field':held,'baseline_mse':bm,'hgb_mse':hm,'et_mse':em,'hgb_gain':bm-hm,'et_gain':bm-em,
       'hgb_residual_rho':float(spearmanr(rt,rh).statistic),'et_residual_rho':float(spearmanr(rt,re).statistic),
       'void_baseline_auc':auc(yt<=q25,-bt),'void_hgb_auc':auc(yt<=q25,-ph),'void_et_auc':auc(yt<=q25,-pe),
       'overdense_baseline_auc':auc(yt>=q75,bt),'overdense_hgb_auc':auc(yt>=q75,ph),'overdense_et_auc':auc(yt>=q75,pe),
       'n_baseline_fields':len(base_names),'n_residual_train_fields':len(corr_names),'n_test_patches':len(Xte)}
  rows.append(rec)
 return pd.DataFrame(rows)

def summarize(D):
 out={}
 for sample,g in D.groupby('sample'):
  out[sample]={}
  for k in ['baseline_mse','hgb_mse','et_mse','hgb_gain','et_gain','hgb_residual_rho','et_residual_rho','void_baseline_auc','void_hgb_auc','void_et_auc','overdense_baseline_auc','overdense_hgb_auc','overdense_et_auc']:
   out[sample][f'{k}_median']=float(np.nanmedian(g[k]))
  out[sample]['hgb_gain_gt0']=paired(g.hgb_gain)
  out[sample]['et_gain_gt0']=paired(g.et_gain)
 R=D[D['sample']=='real'].set_index('field');P=D[D['sample']=='exact_phase'].set_index('field')
 out['paired_real_minus_phase']={
   'hgb_gain':paired(R.hgb_gain-P.hgb_gain),
   'et_gain':paired(R.et_gain-P.et_gain),
   'hgb_residual_rho':paired(R.hgb_residual_rho-P.hgb_residual_rho),
   'et_residual_rho':paired(R.et_residual_rho-P.et_residual_rho),
   'void_hgb_auc_gain':paired((R.void_hgb_auc-R.void_baseline_auc)-(P.void_hgb_auc-P.void_baseline_auc)),
   'overdense_hgb_auc_gain':paired((R.overdense_hgb_auc-R.overdense_baseline_auc)-(P.overdense_hgb_auc-P.overdense_baseline_auc)),
 }
 p=out['paired_real_minus_phase']['hgb_gain']
 out['primary_decision']='PASS_HIGHER_ORDER' if p.get('median',0)>0 and p.get('wilcoxon_p',1)<.05 else 'FAIL_OR_UNCERTAIN_HIGHER_ORDER'
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 P=json.loads((root/'provenance.json').read_text())
 if P.get('status')!='REAL_DR11' or len(P.get('regions',[]))!=48:raise RuntimeError('48-field REAL_DR11 artifact required')
 names=[r['name'] for r in P['regions']];real={};phase={}
 for i,m in enumerate(P['regions']):
  g=gaussianize(grid(m,root));real[m['name']]=patch(g);phase[m['name']]=patch(exact_phase(g,20260901+i))
 Dr=evaluate_sample(real,names,'real');Dp=evaluate_sample(phase,names,'exact_phase');D=pd.concat([Dr,Dp],ignore_index=True);D.to_csv(out/'field_metrics.csv',index=False)
 s={'status':'REAL_DR11_HIGHER_ORDER_RESIDUAL','n_fields':48,'total_rows':int(P['total_rows']),'protocol':'31-field Gaussian baseline + 16-field nonlinear residual fit + 1 held-out field; exact phase surrogate matched protocol','primary_model':'HistGradientBoostingRegressor','secondary_model':'ExtraTreesRegressor','summary':summarize(D)}
 (out/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s,indent=2,sort_keys=True))
if __name__=='__main__':main()
