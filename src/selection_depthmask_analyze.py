#!/usr/bin/env python3
"""Analyze the 48-field REAL_DR11 depth-r + MASKBITS sharded control.

Strict split: first 36 pre-registered fields train all selection models; final 12
are blind test. Selection features include the hidden center itself, deliberately
favoring the survey-selection counterhypothesis.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,spearmanr,wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,r2_score
GRID=64;PATCH=16;STRIDE=4
H=np.zeros((PATCH,PATCH),bool);H[4:12,4:12]=1
R=np.zeros((PATCH,PATCH),bool);R[3,3:13]=1;R[12,3:13]=1;R[4:12,3]=1;R[4:12,12]=1
FULL=np.ones((PATCH,PATCH),bool)
def normgrid(g,v,log=True):
 x=np.log1p(np.clip(g,0,None)) if log else np.asarray(g,float).copy();a=x[v];med=np.median(a);s=np.median(abs(a-med))*1.4826;s=s if np.isfinite(s) and s>1e-6 else np.std(a);s=s if np.isfinite(s) and s>1e-6 else 1.;z=(x-med)/s;z[~v]=0;return z
def patches(g,s,v):
 G=[];S=[];C=[]
 for y in range(0,GRID-PATCH+1,STRIDE):
  for x in range(0,GRID-PATCH+1,STRIDE):
   vv=v[y:y+PATCH,x:x+PATCH]
   if vv.mean()<.98 or not np.all(vv[H]) or not np.all(vv[R]):continue
   G.append(g[y:y+PATCH,x:x+PATCH]);S.append(s[y:y+PATCH,x:x+PATCH]);C.append((y,x))
 return np.asarray(G),np.asarray(S),C
def sf(s):
 out=[]
 for m in [H,R,FULL]:
  a=s[:,m,:];out += [a.mean(1),a.std(1)]
 return np.concatenate(out,1)
def auc(y,p):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,p))
def weights(y):
 y=np.asarray(y,int);n=len(y);p=max(1,int(y.sum()));q=max(1,n-p);return np.where(y,n/(2*p),n/(2*q))
def paired(d):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);p=int((d>0).sum())
 try:w=float(wilcoxon(d,alternative='greater').pvalue)
 except Exception:w=float('nan')
 return {'n':n,'positive':p,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p_one_sided':float(binomtest(p,n,.5,alternative='greater').pvalue),'wilcoxon_p_one_sided':w}
def corrshift(z,dy,dx):
 y0=max(0,-dy);y1=min(GRID,GRID-dy);x0=max(0,-dx);x1=min(GRID,GRID-dx);return float(spearmanr(z[y0:y1,x0:x1].ravel(),z[y0+dy:y1+dy,x0+dx:x1+dx].ravel()).statistic)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 manifests=[]
 for p in sorted(root.rglob('manifest_*.json')):manifests += json.loads(p.read_text())['regions']
 manifests=sorted(manifests,key=lambda r:r['index'])
 if len(manifests)!=48 or [r['index'] for r in manifests]!=list(range(48)):raise RuntimeError(f'need exactly 48 indexed fields, got {len(manifests)}')
 fields={}
 for m in manifests:
  p=next(root.rglob(f"{m['field']}.npz"));d=np.load(p);c=d['counts'];s=d['selection'];v=d['valid'].astype(bool);z=normgrid(c,v,True);P,S,C=patches(z,s,v)
  if len(P)<8:raise RuntimeError(f"too few patches {m['field']}: {len(P)}")
  fields[m['field']]={'counts':c,'sel':s,'valid':v,'z':z,'P':P,'S':S,'C':C}
 names=[m['field'] for m in manifests];train=names[:36];test=names[36:]
 Ptr=np.concatenate([fields[f]['P'] for f in train]);Str=np.concatenate([sf(fields[f]['S']) for f in train]);rtr=Ptr[:,R].mean(1)[:,None];hm=Ptr[:,H].mean(1);hx=Ptr[:,H].max(1);q25,q75=np.quantile(hm,[.25,.75]);qp=np.quantile(hx,.8);yl={'void':hm<=q25,'overdense':hm>=q75,'peak':hx>=qp};rm={};sm={}
 for mot,y in yl.items():
  rm[mot]=LogisticRegression(max_iter=1500,class_weight='balanced').fit(rtr,y.astype(int));sm[mot]=HistGradientBoostingClassifier(max_iter=100,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.5,random_state=31).fit(Str,y.astype(int),sample_weight=weights(y))
 X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]);Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
 if len(Y)>120000:
  ii=np.random.default_rng(20260830).choice(len(Y),120000,replace=False);X=X[ii];Y=Y[ii]
 reg=HistGradientBoostingRegressor(loss='poisson',max_iter=70,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=1.5,random_state=43).fit(X,np.clip(Y,1e-4,None))
 rows=[];anis=[]
 for f in test:
  d=fields[f];P=d['P'];S=sf(d['S']);r=P[:,R].mean(1)[:,None];hm=P[:,H].mean(1);hx=P[:,H].max(1);lab={'void':hm<=q25,'overdense':hm>=q75,'peak':hx>=qp};v=d['valid'];pred=np.full((GRID,GRID),np.nan);pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20);true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6);cr2=float(r2_score(true,pred[v]));csr=float(spearmanr(true,pred[v]).statistic);lam=pred*np.mean(d['counts'][v]);res=np.zeros((GRID,GRID));res[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1);res=normgrid(res,v,False)
  orr=[];oh=[];rr=[];rh=[];expected=[]
  for (y,x),p in zip(d['C'],P):
   rp=res[y:y+PATCH,x:x+PATCH];pp=pred[y:y+PATCH,x:x+PATCH];orr.append(p[R].mean());oh.append(p[H].mean());rr.append(rp[R].mean());rh.append(rp[H].mean());expected.append(np.log(np.clip(pp[H].mean(),1e-4,None)))
  obsrho=float(spearmanr(orr,oh).statistic);resrho=float(spearmanr(rr,rh).statistic);shiftrho=float(spearmanr(rr,np.roll(rh,max(1,len(rh)//3))).statistic)
  for mot,y in lab.items():
   oa=auc(y,rm[mot].predict_proba(r)[:,1]);sa=auc(y,sm[mot].predict_proba(S)[:,1]);e=np.asarray(expected);e=-e if mot=='void' else e;ea=auc(y,e);rows.append({'field':f,'motif':mot,'observed_ring_auc':oa,'selection_depthmask_auc':sa,'selection_expected_count_auc':ea,'selection_cell_r2':cr2,'selection_cell_spearman':csr,'observed_ring_hidden_rho':obsrho,'residual_ring_hidden_rho':resrho,'residual_shift_rho':shiftrho})
  pz=normgrid(np.nan_to_num(pred,nan=np.nanmedian(pred[v])),v,False)
  for lag in [1,2,4,8]:
   for sample,z in [('observed',d['z']),('selection_prediction',pz),('selection_residual',res)]:anis.append({'field':f,'sample':sample,'lag':lag,'rho_x':corrshift(z,0,lag),'rho_y':corrshift(z,lag,0)});anis[-1]['x_minus_y']=anis[-1]['rho_x']-anis[-1]['rho_y']
 df=pd.DataFrame(rows);df.to_csv(out/'blind_field_metrics.csv',index=False);ad=pd.DataFrame(anis);ad.to_csv(out/'anisotropy_field_metrics.csv',index=False);motifs=[]
 for mot,g in df.groupby('motif'):motifs.append({'motif':mot,'observed_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_depthmask_median_auc':float(np.nanmedian(g.selection_depthmask_auc)),'selection_expected_count_median_auc':float(np.nanmedian(g.selection_expected_count_auc)),'observed_minus_selection':paired(g.observed_ring_auc-g.selection_depthmask_auc),'observed_minus_expected_count':paired(g.observed_ring_auc-g.selection_expected_count_auc)})
 one=df.drop_duplicates('field');cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),'observed_ring_hidden_rho_median':float(np.nanmedian(one.observed_ring_hidden_rho)),'residual_ring_hidden_rho_median':float(np.nanmedian(one.residual_ring_hidden_rho)),'residual_shift_rho_median':float(np.nanmedian(one.residual_shift_rho)),'residual_minus_shift':paired(one.residual_ring_hidden_rho-one.residual_shift_rho)};anis_s=[]
 for (sample,lag),g in ad.groupby(['sample','lag']):anis_s.append({'sample':sample,'lag':int(lag),'median_rho_x':float(np.nanmedian(g.rho_x)),'median_rho_y':float(np.nanmedian(g.rho_y)),'x_minus_y':paired(g.x_minus_y)})
 summary={'status':'REAL_DR11_DEPTHMASK_48_SHARDED_BLIND','split':{'train':train,'blind_test':test},'selection_has_hidden_region_maps':True,'selection_features':manifests[0]['feature_names'],'motifs':motifs,'continuous_locality':cont,'directional_anisotropy':anis_s};(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':manifests},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
