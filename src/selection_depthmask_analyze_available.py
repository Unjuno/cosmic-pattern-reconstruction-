#!/usr/bin/env python3
"""Analyze availability-qualified 48-candidate REAL_DR11 depth+MASKBITS shards.

The original candidate index defines the split before seeing any outcome:
indices 0..35 are training candidates and 36..47 are blind-test candidates.
Missing official selection products are rejected solely for availability.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import selection_depthmask_analyze as a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);x=ap.parse_args();root=Path(x.input);out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
 allr=[]
 for p in sorted(root.rglob('manifest_*.json')): allr += json.loads(p.read_text())['regions']
 if len(allr)!=48 or sorted(r['index'] for r in allr)!=list(range(48)):raise RuntimeError(f'need all 48 candidate records, got {len(allr)}')
 accepted=sorted([r for r in allr if r.get('status')=='accepted'],key=lambda r:r['index']);rejected=sorted([r for r in allr if r.get('status')!='accepted'],key=lambda r:r['index'])
 train_meta=[r for r in accepted if r['index']<36];test_meta=[r for r in accepted if r['index']>=36]
 if len(train_meta)<24 or len(test_meta)<8:raise RuntimeError(f'insufficient objective coverage train={len(train_meta)} test={len(test_meta)}')
 fields={}
 for m in accepted:
  p=next(root.rglob(f"{m['field']}.npz"));d=np.load(p);cnt=d['counts'];sel=d['selection'];valid=d['valid'].astype(bool);z=a.normgrid(cnt,valid,True);P,S,C=a.patches(z,sel,valid)
  if len(P)<8:raise RuntimeError(f"too few patches {m['field']}: {len(P)}")
  fields[m['field']]={'counts':cnt,'sel':sel,'valid':valid,'z':z,'P':P,'S':S,'C':C}
 train=[r['field'] for r in train_meta];test=[r['field'] for r in test_meta]
 Ptr=np.concatenate([fields[f]['P'] for f in train]);Str=np.concatenate([a.sf(fields[f]['S']) for f in train]);rtr=Ptr[:,a.R].mean(1)[:,None];hm=Ptr[:,a.H].mean(1);hx=Ptr[:,a.H].max(1);q25,q75=np.quantile(hm,[.25,.75]);qp=np.quantile(hx,.8);yl={'void':hm<=q25,'overdense':hm>=q75,'peak':hx>=qp};rm={};sm={}
 for mot,y in yl.items():
  rm[mot]=LogisticRegression(max_iter=1500,class_weight='balanced').fit(rtr,y.astype(int));sm[mot]=HistGradientBoostingClassifier(max_iter=100,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.5,random_state=31).fit(Str,y.astype(int),sample_weight=a.weights(y))
 X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]);Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
 if len(Y)>120000:
  ii=np.random.default_rng(20260830).choice(len(Y),120000,replace=False);X=X[ii];Y=Y[ii]
 reg=HistGradientBoostingRegressor(loss='poisson',max_iter=70,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=1.5,random_state=43).fit(X,np.clip(Y,1e-4,None))
 rows=[];anis=[]
 for f in test:
  d=fields[f];P=d['P'];S=a.sf(d['S']);ring=P[:,a.R].mean(1)[:,None];hm=P[:,a.H].mean(1);hx=P[:,a.H].max(1);lab={'void':hm<=q25,'overdense':hm>=q75,'peak':hx>=qp};v=d['valid'];pred=np.full((a.GRID,a.GRID),np.nan);pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20);true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6);cr2=float(r2_score(true,pred[v]));csr=float(spearmanr(true,pred[v]).statistic);lam=pred*np.mean(d['counts'][v]);res=np.zeros((a.GRID,a.GRID));res[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1);res=a.normgrid(res,v,False);orr=[];oh=[];rr=[];rh=[];expected=[]
  for (yy,xx),p in zip(d['C'],P):
   rp=res[yy:yy+a.PATCH,xx:xx+a.PATCH];pp=pred[yy:yy+a.PATCH,xx:xx+a.PATCH];orr.append(p[a.R].mean());oh.append(p[a.H].mean());rr.append(rp[a.R].mean());rh.append(rp[a.H].mean());expected.append(np.log(np.clip(pp[a.H].mean(),1e-4,None)))
  obsrho=float(spearmanr(orr,oh).statistic);resrho=float(spearmanr(rr,rh).statistic);shiftrho=float(spearmanr(rr,np.roll(rh,max(1,len(rh)//3))).statistic)
  for mot,y in lab.items():
   oa=a.auc(y,rm[mot].predict_proba(ring)[:,1]);sa=a.auc(y,sm[mot].predict_proba(S)[:,1]);e=np.asarray(expected);e=-e if mot=='void' else e;ea=a.auc(y,e);rows.append({'field':f,'motif':mot,'observed_ring_auc':oa,'selection_depthmask_auc':sa,'selection_expected_count_auc':ea,'selection_cell_r2':cr2,'selection_cell_spearman':csr,'observed_ring_hidden_rho':obsrho,'residual_ring_hidden_rho':resrho,'residual_shift_rho':shiftrho})
  pz=a.normgrid(np.nan_to_num(pred,nan=np.nanmedian(pred[v])),v,False)
  for lag in [1,2,4,8]:
   for sample,z in [('observed',d['z']),('selection_prediction',pz),('selection_residual',res)]:
    rx=a.corrshift(z,0,lag);ry=a.corrshift(z,lag,0);anis.append({'field':f,'sample':sample,'lag':lag,'rho_x':rx,'rho_y':ry,'x_minus_y':rx-ry})
 df=pd.DataFrame(rows);df.to_csv(out/'blind_field_metrics.csv',index=False);ad=pd.DataFrame(anis);ad.to_csv(out/'anisotropy_field_metrics.csv',index=False);motifs=[]
 for mot,g in df.groupby('motif'):motifs.append({'motif':mot,'observed_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_depthmask_median_auc':float(np.nanmedian(g.selection_depthmask_auc)),'selection_expected_count_median_auc':float(np.nanmedian(g.selection_expected_count_auc)),'observed_minus_selection':a.paired(g.observed_ring_auc-g.selection_depthmask_auc),'observed_minus_expected_count':a.paired(g.observed_ring_auc-g.selection_expected_count_auc)})
 one=df.drop_duplicates('field');cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),'observed_ring_hidden_rho_median':float(np.nanmedian(one.observed_ring_hidden_rho)),'residual_ring_hidden_rho_median':float(np.nanmedian(one.residual_ring_hidden_rho)),'residual_shift_rho_median':float(np.nanmedian(one.residual_shift_rho)),'residual_minus_shift':a.paired(one.residual_ring_hidden_rho-one.residual_shift_rho)};anis_s=[]
 for (sample,lag),g in ad.groupby(['sample','lag']):anis_s.append({'sample':sample,'lag':int(lag),'median_rho_x':float(np.nanmedian(g.rho_x)),'median_rho_y':float(np.nanmedian(g.rho_y)),'x_minus_y':a.paired(g.x_minus_y)})
 summary={'status':'REAL_DR11_DEPTHMASK_AVAILABILITY_GATED_BLIND','candidate_count':48,'accepted_count':len(accepted),'rejected_count':len(rejected),'train_count':len(train),'blind_test_count':len(test),'split_rule':'original candidate index <36 train, >=36 blind test; availability only','selection_has_hidden_region_maps':True,'motifs':motifs,'continuous_locality':cont,'directional_anisotropy':anis_s,'availability_rejections':rejected};(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':summary['status'],'accepted':accepted,'rejected':rejected},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
