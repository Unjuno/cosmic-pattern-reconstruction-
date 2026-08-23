#!/usr/bin/env python3
"""Ablate boundary features for REAL DR11 center-hole motif prediction.

Question: is the observed boundary/interior signal a rich geometric pattern or
mostly simple local-density continuity? Entire sky fields are held out.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from analyze_dr11 import mask_indices, normalize_grid, patchify, region_grid, ring_features, verify_and_load
from lofo_validate import exact_signflip_p

FEATURE_SETS = {
    'ring_mean': [0],
    'ring_mean_std': [0,1],
    'ring_mean_gradient': [0,6,7],
    'four_sides': [2,3,4,5],
    'full8': list(range(8)),
}


def aucs(Xtr: np.ndarray, Xte: np.ndarray) -> dict[str, dict[str,float]]:
    hidden = mask_indices('center25')
    tm, em = Xtr[:,hidden].mean(1), Xte[:,hidden].mean(1)
    tx, ex = Xtr[:,hidden].max(1), Xte[:,hidden].max(1)
    q25,q75=np.quantile(tm,[.25,.75]); qp=np.quantile(tx,.8)
    Ftr,Fte=ring_features(Xtr),ring_features(Xte)
    labels={'void':(tm<=q25,em<=q25),'overdense':(tm>=q75,em>=q75),'peak':(tx>=qp,ex>=qp)}
    out={}
    for motif,(ytr,yte) in labels.items():
        out[motif]={}
        for name,cols in FEATURE_SETS.items():
            if len(np.unique(yte))<2:
                out[motif][name]=float('nan'); continue
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Ftr[:,cols],ytr.astype(int))
            out[motif][name]=float(roc_auc_score(yte.astype(int),clf.predict_proba(Fte[:,cols])[:,1]))
    return out


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/pilot'); ap.add_argument('--out',default='results/real_dr11/latest'); args=ap.parse_args()
    data,out=Path(args.data),Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((data/'provenance.json').read_text())
    if prov.get('status')!='REAL_DR11' or prov.get('model_input_columns')!=['ra','dec']: raise RuntimeError('REAL_DR11 RA/Dec-only provenance required')
    meta={r['name']:r for r in prov['regions']}; fields=list(meta)
    parts={}
    for f,m in meta.items():
        g,_=normalize_grid(region_grid(verify_and_load(m),m)); parts[f]=patchify(g,f)[0]
    rows=[]
    for held in fields:
        train_fields=[f for f in fields if f!=held]
        tr0=np.concatenate([parts[f] for f in train_fields]); te0=parts[held]
        scaler=StandardScaler().fit(tr0); scores=aucs(scaler.transform(tr0),scaler.transform(te0))
        for motif,d in scores.items():
            for features,auc in d.items(): rows.append({'field':held,'motif':motif,'features':features,'auc':auc})
    df=pd.DataFrame(rows); df.to_csv(out/'boundary_ablation_field_metrics.csv',index=False)
    comps=[]
    for motif in ['void','overdense','peak']:
        w=df[df.motif==motif].pivot(index='field',columns='features',values='auc')
        diff=w['full8']-w['ring_mean']; finite=np.isfinite(diff.to_numpy())
        comps.append({'motif':motif,'n_fields':int(finite.sum()),'ring_mean_median_auc':float(np.nanmedian(w['ring_mean'])),'full8_median_auc':float(np.nanmedian(w['full8'])),'full8_better_fields':int((diff[finite]>0).sum()),'mean_full8_minus_ring_mean':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
    pd.DataFrame(comps).to_csv(out/'boundary_ablation_comparisons.csv',index=False)
    summary={'status':'REAL_DR11','validation':'12-field LOFO boundary-feature ablation','model_input_columns':['ra','dec'],'feature_sets':FEATURE_SETS,'comparisons':comps}
    (out/'boundary_ablation_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
