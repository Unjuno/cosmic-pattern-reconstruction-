#!/usr/bin/env python3
"""48-field leave-one-field-out replication on REAL DR11 positions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from analyze_dr11 import mask_indices,normalize_grid,patchify,reconstruction_predictions,region_grid,ring_features,score_reconstruction,verify_and_load
from scale_null_validate import matched_shift_hybrid


def motif_scores(Xtr,Xte):
    hidden=mask_indices('center25'); tm=Xtr[:,hidden].mean(1); em=Xte[:,hidden].mean(1); tx=Xtr[:,hidden].max(1); ex=Xte[:,hidden].max(1); q25,q75=np.quantile(tm,[.25,.75]); qp=np.quantile(tx,.8); Ftr,Fte=ring_features(Xtr),ring_features(Xte)
    labels={'void':(tm<=q25,em<=q25),'overdense':(tm>=q75,em>=q75),'peak':(tx>=qp,ex>=qp)}; out={}
    for motif,(ytr,yte) in labels.items():
        for feature,cols in [('ring_mean',[0]),('full8',list(range(8)))]:
            if len(np.unique(yte))<2: auc=float('nan')
            else:
                clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Ftr[:,cols],ytr.astype(int)); auc=float(roc_auc_score(yte.astype(int),clf.predict_proba(Fte[:,cols])[:,1]))
            out[f'{motif}_{feature}_auc']=auc
    return out

def paired_summary(diff):
    x=np.asarray(diff,float); x=x[np.isfinite(x)]; n=len(x); pos=int((x>0).sum())
    if n==0:return {'n_fields':0}
    rng=np.random.default_rng(20260824); boots=np.mean(rng.choice(x,(20000,n),replace=True),axis=1)
    try:w=float(wilcoxon(x,alternative='greater',zero_method='wilcox').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'mean_advantage':float(np.mean(x)),'median_advantage':float(np.median(x)),'bootstrap_mean_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]}

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/real/dr11/expanded48');ap.add_argument('--out',default='results/real_dr11/expanded48');args=ap.parse_args();data,out=Path(args.data),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((data/'provenance.json').read_text());
    if prov.get('status')!='REAL_DR11' or prov.get('model_input_columns')!=['ra','dec'] or len(prov.get('regions',[]))!=48:raise RuntimeError('48-field REAL_DR11 RA/Dec provenance required')
    meta={r['name']:r for r in prov['regions']}; fields=list(meta); parts={}
    for f,m in meta.items():g,_=normalize_grid(region_grid(verify_and_load(m),m));parts[f]=patchify(g,f)[0]
    shifted={f:matched_shift_hybrid(parts[f],8) for f in fields};rows=[]
    for k,held in enumerate(fields):
        train_fields=[f for f in fields if f!=held]
        for sample,use in [('real',parts),('matched_shift',shifted)]:
            tr0=np.concatenate([use[f] for f in train_fields]);te0=use[held];scaler=StandardScaler().fit(tr0);tr,te=scaler.transform(tr0),scaler.transform(te0);m=motif_scores(tr,te);hidden=mask_indices('center25');pred=reconstruction_predictions(tr,te,hidden)['gaussian'];mse,corr=score_reconstruction(te[:,hidden],pred);rows.append({'field':held,'sample':sample,'gaussian_corr':corr,'gaussian_mse':mse,**m})
        if (k+1)%8==0:print(f'[DR11-48] validated {k+1}/48 fields',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'field_metrics.csv',index=False);w=df.pivot(index='field',columns='sample')
    comps=[]
    metrics=['gaussian_corr','void_ring_mean_auc','overdense_ring_mean_auc','peak_ring_mean_auc','void_full8_auc','overdense_full8_auc','peak_full8_auc']
    for metric in metrics:
        real=w[metric]['real'];null=w[metric]['matched_shift'];d=real-null;comps.append({'metric':metric,'real_median':float(np.nanmedian(real)),'matched_shift_median':float(np.nanmedian(null)),**paired_summary(d.to_numpy())})
    real=w['gaussian_mse']['real'];null=w['gaussian_mse']['matched_shift'];comps.append({'metric':'gaussian_mse','real_median':float(np.nanmedian(real)),'matched_shift_median':float(np.nanmedian(null)),**paired_summary((null-real).to_numpy())})
    cdf=pd.DataFrame(comps);cdf.to_csv(out/'comparisons.csv',index=False)
    # Full8 minus ring mean in real observations: tests whether geometry adds beyond local mean.
    abl=[]
    realrows=df[df['sample']=='real'].set_index('field')
    for motif in ['void','overdense','peak']:
        d=realrows[f'{motif}_full8_auc']-realrows[f'{motif}_ring_mean_auc'];abl.append({'motif':motif,'ring_mean_median':float(np.nanmedian(realrows[f'{motif}_ring_mean_auc'])),'full8_median':float(np.nanmedian(realrows[f'{motif}_full8_auc'])),**paired_summary(d.to_numpy())})
    pd.DataFrame(abl).to_csv(out/'boundary_ablation.csv',index=False)
    summary={'status':'REAL_DR11','validation':'48 independent 0.5-degree fields, leave-one-field-out','total_rows':int(prov['total_rows']),'field_selection':prov['field_selection'],'comparisons':comps,'boundary_ablation':abl}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
