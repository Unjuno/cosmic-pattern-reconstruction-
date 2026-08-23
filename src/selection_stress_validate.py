#!/usr/bin/env python3
"""Stress REAL DR11 local predictability against catalog cleaning and thinning.

Selections use observing-condition columns only to choose which observed sources
remain. The pattern model itself still receives RA/Dec only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from analyze_dr11 import mask_indices, motif_metrics, normalize_grid, patchify, reconstruction_predictions, region_grid, score_reconstruction
from lofo_validate import exact_signflip_p
from scale_null_validate import matched_shift_hybrid
from systematics_validate import load_qc

SELECTIONS=['all','maskbits0','allbands','clean_allbands','thin50','equal11000_clean']


def select(df:pd.DataFrame,name:str,seed:int)->pd.DataFrame:
    if name=='all': return df
    allbands=(df.nobs_g>0)&(df.nobs_r>0)&(df.nobs_i>0)&(df.nobs_z>0)
    clean=(df.maskbits==0)
    if name=='maskbits0': return df.loc[clean]
    if name=='allbands': return df.loc[allbands]
    if name=='clean_allbands': return df.loc[clean&allbands]
    if name=='thin50': return df.sample(frac=.5,random_state=seed)
    if name=='equal11000_clean':
        d=df.loc[clean&allbands]; return d.sample(n=min(11000,len(d)),random_state=seed)
    raise ValueError(name)


def metrics(Xtr:np.ndarray,Xte:np.ndarray)->dict:
    motifs={m['motif']:m['auc'] for m in motif_metrics(Xtr,Xte)}; hidden=mask_indices('center25'); pred=reconstruction_predictions(Xtr,Xte,hidden)['gaussian']; mse,corr=score_reconstruction(Xte[:,hidden],pred)
    return {'void_auc':motifs['void'],'overdense_auc':motifs['overdense'],'peak_auc':motifs['peak'],'gaussian_corr':corr,'gaussian_mse':mse}


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',default='data/real/dr11/pilot/provenance.json'); ap.add_argument('--qc',default='data/real/dr11/systematics/provenance.json'); ap.add_argument('--out',default='results/real_dr11/latest'); args=ap.parse_args()
    base=json.loads(Path(args.baseline).read_text()); qprov=json.loads(Path(args.qc).read_text()); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if base.get('status')!='REAL_DR11' or base.get('model_input_columns')!=['ra','dec'] or qprov.get('status')!='REAL_DR11_SYSTEMATICS_QC': raise RuntimeError('REAL_DR11 baseline and QC provenance required')
    bm={r['name']:r for r in base['regions']}; qm={r['name']:r for r in qprov['regions']}; fields=list(bm)
    if set(fields)!=set(qm): raise RuntimeError('field mismatch')
    qc={f:load_qc(qm[f]) for f in fields}
    rows=[]; retention=[]
    for selection in SELECTIONS:
        parts={}
        for j,f in enumerate(fields):
            d=select(qc[f],selection,1000+j if selection!='equal11000_clean' else 2000+j)
            retention.append({'selection':selection,'field':f,'rows':int(len(d)),'fraction_of_all':float(len(d)/len(qc[f]))})
            g,_=normalize_grid(region_grid(d,bm[f])); parts[f]=patchify(g,f)[0]
        shifted={f:matched_shift_hybrid(parts[f],8) for f in fields}
        for held in fields:
            train_fields=[f for f in fields if f!=held]
            for sample,use in [('real',parts),('matched_shift',shifted)]:
                tr0=np.concatenate([use[f] for f in train_fields]); te0=use[held]; scaler=StandardScaler().fit(tr0); m=metrics(scaler.transform(tr0),scaler.transform(te0)); rows.append({'selection':selection,'field':held,'sample':sample,**m})
    df=pd.DataFrame(rows); df.to_csv(out/'selection_stress_field_metrics.csv',index=False); pd.DataFrame(retention).to_csv(out/'selection_retention.csv',index=False)
    comps=[]
    for selection in SELECTIONS:
        s=df[df.selection==selection]
        for metric in ['void_auc','overdense_auc','peak_auc','gaussian_corr']:
            w=s.pivot(index='field',columns='sample',values=metric); diff=w['real']-w['matched_shift']; finite=np.isfinite(diff.to_numpy())
            comps.append({'selection':selection,'metric':metric,'n_fields':int(finite.sum()),'real_median':float(np.nanmedian(w['real'])),'matched_shift_median':float(np.nanmedian(w['matched_shift'])),'positive_fields':int((diff[finite]>0).sum()),'mean_paired_advantage':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
        w=s.pivot(index='field',columns='sample',values='gaussian_mse'); diff=w['matched_shift']-w['real']; finite=np.isfinite(diff.to_numpy()); comps.append({'selection':selection,'metric':'gaussian_mse','n_fields':int(finite.sum()),'real_median':float(np.nanmedian(w['real'])),'matched_shift_median':float(np.nanmedian(w['matched_shift'])),'positive_fields':int((diff[finite]>0).sum()),'mean_paired_advantage':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
    pd.DataFrame(comps).to_csv(out/'selection_stress_comparisons.csv',index=False)
    summary={'status':'REAL_DR11','validation':'12-field LOFO tracer-selection/thinning stress','model_input_columns':['ra','dec'],'selections':SELECTIONS,'comparisons':comps}
    (out/'selection_stress_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
