#!/usr/bin/env python3
"""Availability-gated REAL_DR11 PSF-vs-extended locality replication.

Uses the fixed first-48 expanded48 candidate order. A candidate is rejected only
for acquisition failure, duplicate brick, or insufficient tracer availability.
No outcome metric is inspected during acceptance.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

import tracer_split_validate as base

TARGET_FIELDS=36
MAX_CANDIDATES=48
MIN_PER_TRACER=1000


def build_data(centers):
    regs=centers.get('regions',[])[:MAX_CANDIDATES]
    data={}; prov=[]; rejected=[]; used=set(); rng=np.random.default_rng(20260831)
    for j,r in enumerate(regs):
        if len(data)>=TARGET_FIELDS: break
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
        try:
            brick,bq=base.choose_brick(ra,dec)
        except Exception as e:
            rejected.append({'field':name,'reason':'brick_resolution','error':str(e)}); continue
        if brick in used:
            rejected.append({'field':name,'brick':brick,'reason':'duplicate'}); continue
        used.add(brick); print(f'[tracer-split36] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}',flush=True)
        try:
            d,p=base.get_tractor(brick)
        except Exception as e:
            rejected.append({'field':name,'brick':brick,'reason':'tractor_acquisition','error':str(e)}); continue
        psf=d[d.type.eq('PSF')].copy(); ext=d[d.type.isin(base.EXT_TYPES)].copy(); n=min(len(psf),len(ext))
        if n<MIN_PER_TRACER:
            rejected.append({'field':name,'brick':brick,'reason':'tracer_availability','n_psf':int(len(psf)),'n_extended':int(len(ext)),'min_required':MIN_PER_TRACER}); continue
        psf_eq=psf.iloc[rng.choice(len(psf),n,replace=False)]; ext_eq=ext.iloc[rng.choice(len(ext),n,replace=False)]
        subsets={'all':d,'psf':psf,'extended':ext,'psf_equal':psf_eq,'extended_equal':ext_eq}
        data[name]={k:base.scores(base.norm(base.grid_from(v))) for k,v in subsets.items()}
        prov.append({'field':name,'brick':brick,'brick_choice_query':bq,'tractor':p,'n_psf':int(len(psf)),'n_extended':int(len(ext)),'equalized_n':int(n),'n_gaia_ref':int(d.ref_cat.eq('G3').sum())})
        print(f'[tracer-split36] accept {len(data)}/{TARGET_FIELDS}: PSF={len(psf)} EXT={len(ext)} EQ={n}',flush=True)
    return data,prov,rejected


def evaluate(data):
    fields=list(data); rows=[]
    for held in fields:
        train=[f for f in fields if f!=held]
        for tracer in ['all','psf','extended','psf_equal','extended_equal']:
            tr=np.concatenate([data[f][tracer] for f in train]); te=data[held][tracer]
            q25,q75=np.quantile(tr[:,1],[.25,.75]); qpk=np.quantile(tr[:,2],.8)
            labs={'void':(tr[:,1]<=q25,te[:,1]<=q25),'overdense':(tr[:,1]>=q75,te[:,1]>=q75),'peak':(tr[:,2]>=qpk,te[:,2]>=qpk)}
            for motif,(ytr,yte) in labs.items():
                real=base.fit_score(tr[:,0],ytr,te[:,0],yte)
                null=base.fit_score(np.roll(tr[:,0],max(1,len(tr)//3)),ytr,np.roll(te[:,0],max(1,len(te)//3)),yte)
                rows.append({'field':held,'tracer':tracer,'motif':motif,'auc':real,'matched_shift_auc':null})
        for src,tgt,label in [('psf_equal','extended_equal','psf_to_extended'),('extended_equal','psf_equal','extended_to_psf')]:
            src_tr=np.concatenate([data[f][src] for f in train]); tgt_tr=np.concatenate([data[f][tgt] for f in train]); src_te=data[held][src]; tgt_te=data[held][tgt]
            q25,q75=np.quantile(tgt_tr[:,1],[.25,.75]); qpk=np.quantile(tgt_tr[:,2],.8)
            for motif,ytr,yte in [('void',tgt_tr[:,1]<=q25,tgt_te[:,1]<=q25),('overdense',tgt_tr[:,1]>=q75,tgt_te[:,1]>=q75),('peak',tgt_tr[:,2]>=qpk,tgt_te[:,2]>=qpk)]:
                real=base.fit_score(src_tr[:,0],ytr,src_te[:,0],yte)
                null=base.fit_score(np.roll(src_tr[:,0],max(1,len(src_tr)//3)),ytr,np.roll(src_te[:,0],max(1,len(src_te)//3)),yte)
                rows.append({'field':held,'tracer':label,'motif':motif,'auc':real,'matched_shift_auc':null})
    return pd.DataFrame(rows)


def summarize(df):
    out=[]
    for (tracer,motif),g in df.groupby(['tracer','motif']):
        out.append({'tracer':tracer,'motif':motif,'median_auc':float(np.nanmedian(g.auc)),'matched_shift_median':float(np.nanmedian(g.matched_shift_auc)),'real_minus_shift':base.paired(g.auc-g.matched_shift_auc)})
    comparisons=[]
    for motif in ['void','overdense','peak']:
        ext=df[(df.tracer=='extended_equal')&(df.motif==motif)].set_index('field').auc
        psf=df[(df.tracer=='psf_equal')&(df.motif==motif)].set_index('field').auc
        cross=df[(df.tracer=='psf_to_extended')&(df.motif==motif)].set_index('field').auc
        rev=df[(df.tracer=='extended_to_psf')&(df.motif==motif)].set_index('field').auc
        comparisons.append({'motif':motif,'extended_self_median':float(np.nanmedian(ext)),'psf_self_median':float(np.nanmedian(psf)),'psf_to_extended_median':float(np.nanmedian(cross)),'extended_to_psf_median':float(np.nanmedian(rev)),'extended_self_minus_psf_cross':base.paired(ext-cross),'psf_self_minus_extended_cross':base.paired(psf-rev)})
    return out,comparisons


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/tracer_split36'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text())
    if centers.get('status')!='REAL_DR11': raise RuntimeError('REAL_DR11 centers required')
    data,prov,rejected=build_data(centers)
    if len(data)!=TARGET_FIELDS:
        (out/'availability_rejections.json').write_text(json.dumps(rejected,indent=2,sort_keys=True)+'\n')
        raise RuntimeError(f'only {len(data)} availability-qualified fields from {MAX_CANDIDATES}')
    df=evaluate(data); df.to_csv(out/'field_metrics.csv',index=False); summary_rows,comparisons=summarize(df)
    summary={'status':'REAL_DR11_TRACTOR_TRACER_SPLIT_36','validation':'36 availability-qualified fields from fixed first-48 candidate order; whole-field LOFO','availability_gate':{'min_per_tracer':MIN_PER_TRACER,'target_fields':TARGET_FIELDS,'max_candidates':MAX_CANDIDATES},'subsets':'PSF vs REX/EXP/DEV/SER; equalized per brick; downstream models use positions only','summary':summary_rows,'cross_tracer_comparisons':comparisons,'availability_rejections':rejected}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':prov,'rejections':rejected},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
