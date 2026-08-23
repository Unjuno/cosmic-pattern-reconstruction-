#!/usr/bin/env python3
"""Compare artificial-hole geometries on REAL DR11 with matched-shift nulls."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from analyze_dr11 import mask_indices, normalize_grid, patchify, reconstruction_predictions, region_grid, score_reconstruction, verify_and_load
from lofo_validate import exact_signflip_p

MASKS=['center25','corner25','stripe25','random25']
METHODS=['mean','gaussian','pca','knn20']


def shifted_visible_hybrid(X:np.ndarray,hidden:np.ndarray)->np.ndarray:
    shifted=np.roll(X.reshape(8,8,64),shift=(2,3),axis=(0,1)).reshape(64,64)
    observed=np.setdiff1d(np.arange(64),hidden); out=X.copy(); out[:,observed]=shifted[:,observed]; return out


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/real/dr11/pilot'); ap.add_argument('--out',default='results/real_dr11/latest'); args=ap.parse_args()
    data,out=Path(args.data),Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((data/'provenance.json').read_text())
    if prov.get('status')!='REAL_DR11' or prov.get('model_input_columns')!=['ra','dec']: raise RuntimeError('REAL_DR11 RA/Dec-only provenance required')
    meta={r['name']:r for r in prov['regions']}; fields=list(meta); parts={}
    for f,m in meta.items():
        g,_=normalize_grid(region_grid(verify_and_load(m),m)); parts[f]=patchify(g,f)[0]
    rows=[]
    for mask in MASKS:
        hidden=mask_indices(mask)
        for held in fields:
            train_fields=[f for f in fields if f!=held]
            for sample in ['real','matched_shift']:
                use=parts if sample=='real' else {f:shifted_visible_hybrid(parts[f],hidden) for f in fields}
                tr0=np.concatenate([use[f] for f in train_fields]); te0=use[held]; scaler=StandardScaler().fit(tr0); tr,te=scaler.transform(tr0),scaler.transform(te0)
                preds=reconstruction_predictions(tr,te,hidden); y=te[:,hidden]
                for method in METHODS:
                    mse,corr=score_reconstruction(y,preds[method]); rows.append({'mask':mask,'field':held,'sample':sample,'method':method,'hidden_fraction':float(len(hidden)/64),'mse':mse,'corr':corr})
    df=pd.DataFrame(rows); df.to_csv(out/'mask_geometry_field_metrics.csv',index=False)
    comps=[]
    for mask in MASKS:
        for method in METHODS:
            s=df[(df['mask']==mask)&(df.method==method)]; w=s.pivot(index='field',columns='sample')
            for metric in ['corr','mse']:
                real=w[metric]['real']; null=w[metric]['matched_shift']; diff=(null-real) if metric=='mse' else (real-null)
                comps.append({'mask':mask,'method':method,'metric':metric,'real_median':float(np.nanmedian(real)),'matched_shift_median':float(np.nanmedian(null)),'positive_fields':int((diff>0).sum()),'mean_paired_advantage':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
    pd.DataFrame(comps).to_csv(out/'mask_geometry_comparisons.csv',index=False)
    summary={'status':'REAL_DR11','validation':'12-field LOFO artificial-mask geometry sweep','model_input_columns':['ra','dec'],'masks':MASKS,'comparisons':comps}
    (out/'mask_geometry_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
