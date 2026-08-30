#!/usr/bin/env python3
"""Coverage-gated REAL_DR11 pixel-level selection null.

Candidate sky fields are the first 24 entries of the already pre-registered
expanded48 sequence. We inspect official DR11 depth/NEXP/MASKBITS only and
accept the first 12 fields that contain at least 8 fully valid analysis
patches. Scientific targets (void/overdense/peak performance) are never used
for acceptance. Source positions come from SHA-verified cached REAL_DR11
catalogs via ``selection_null_validate_cached``.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

import selection_null_validate_cached as cached
core=cached.core
TARGET_FIELDS=12; MAX_CANDIDATES=24; MIN_PATCHES=8


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null12'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); centers=json.loads(Path(args.centers).read_text())
    regs=centers.get('regions',[])[:MAX_CANDIDATES]
    if centers.get('status')!='REAL_DR11' or len(regs)<TARGET_FIELDS: raise RuntimeError('REAL_DR11 fixed centers required')
    fields={}; provenance=[]; coverage_rejections=[]; used=set()
    for j,r in enumerate(regs):
        if len(fields)>=TARGET_FIELDS: break
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg']); brick,bq,near=core.choose_brick_from_center(ra,dec)
        if brick in used:
            coverage_rejections.append({'field':name,'brick':brick,'reason':'duplicate_brick'}); continue
        used.add(brick); print(f'[selection-null-coverage] candidate {j+1}/{MAX_CANDIDATES} {name} -> {brick}',flush=True)
        urls=core.product_urls(brick); depth,hdr,dp=core.read_image(urls['depth_r'],'DEPTH_R'); mask,_,mp=core.read_image(urls['maskbits'],'MASKBITS'); nexp,_,npv=core.read_image(urls['nexp_r']); psf,_,pp=core.read_image(urls['psfsize_r'])
        if not(depth.shape==mask.shape==nexp.shape==psf.shape): raise RuntimeError(f'shape mismatch {brick}')
        src,sq=core.source_catalog(brick); counts,ninside=core.count_grid(src,WCS(hdr),depth.shape)
        feat=[]; fn=[]
        for arr,names in [(core.continuous_features(depth,True,True),['log_depth_mean','log_depth_std','depth_positive_frac']),(core.continuous_features(nexp,False,True),['nexp_mean','nexp_std','nexp_positive_frac']),(core.continuous_features(psf,False,True),['psfsize_mean','psfsize_std','psfsize_positive_frac'])]: feat+=arr; fn+=names
        feat+=core.mask_features(mask); fn+=['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(core.BIT_COUNT)]
        sel=np.stack(feat,-1); primary=sel[:,:,fn.index('primary_frac')]
        valid=(primary>=.9375)&(sel[:,:,fn.index('depth_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_positive_frac')]>=.9375)
        corrected=counts/np.clip(primary,.25,1); norm=core.robust_norm(corrected,valid,True); patches,sp,coords=core.make_patches(norm,sel,valid)
        if len(patches)<MIN_PATCHES:
            rec={'field':name,'brick':brick,'reason':'insufficient_selection_coverage','valid_cell_fraction':float(valid.mean()),'n_patches':int(len(patches))}; coverage_rejections.append(rec); print(f"[selection-null-coverage] reject {name}: {len(patches)} patches",flush=True); continue
        fields[name]={'brick':brick,'counts':corrected,'norm':norm,'sel':sel,'valid':valid,'patches':patches,'sp':sp,'coords':coords,'fn':fn}
        provenance.append({'field':name,'brick':brick,'center_ra_deg':ra,'center_dec_deg':dec,'nearest_primary_source_deg':near,'brick_choice_query':bq,'source_query':sq,'source_rows':int(len(src)),'sources_inside_wcs':ninside,'valid_cell_fraction':float(valid.mean()),'n_patches':int(len(patches)),'products':{'depth_r':dp,'maskbits':mp,'nexp_r':npv,'psfsize_r':pp}})
        print(f"[selection-null-coverage] accept {len(fields)}/{TARGET_FIELDS} {name}: {len(patches)} patches",flush=True)
    if len(fields)!=TARGET_FIELDS: raise RuntimeError(f'coverage gate yielded {len(fields)} fields; expected {TARGET_FIELDS}')

    names=list(fields); rows=[]
    for held in names:
        train=[f for f in names if f!=held]; Ptr=np.concatenate([fields[f]['patches'] for f in train]); Pte=fields[held]['patches']; Str=np.concatenate([core.selection_patch_features(fields[f]['sp']) for f in train]); Ste=core.selection_patch_features(fields[held]['sp'])
        rtr=Ptr[:,core.RING].mean(1)[:,None]; rte=Pte[:,core.RING].mean(1)[:,None]; hmtr=Ptr[:,core.HIDDEN].mean(1); hmte=Pte[:,core.HIDDEN].mean(1); hxtr=Ptr[:,core.HIDDEN].max(1); hxte=Pte[:,core.HIDDEN].max(1); q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(rtr,ytr.astype(int)); obs=core.auc(yte,lr.predict_proba(rte)[:,1])
            clf=HistGradientBoostingClassifier(max_iter=80,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=20,l2_regularization=1,random_state=31).fit(Str,ytr.astype(int),sample_weight=core.balanced_weights(ytr)); selauc=core.auc(yte,clf.predict_proba(Ste)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obs,'selection_only_auc':selauc})
        X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]); Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=50,learning_rate=.07,max_leaf_nodes=12,min_samples_leaf=60,l2_regularization=1,random_state=43).fit(X,np.clip(Y,1e-4,None))
        d=fields[held]; v=d['valid']; pred=np.full((core.GRID,core.GRID),np.nan); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20); true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6); r2=float(r2_score(true,pred[v])); sr=float(spearmanr(true,pred[v]).statistic)
        lam=pred*np.mean(d['counts'][v]); resid=np.zeros((core.GRID,core.GRID)); resid[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1); resid=core.robust_norm(resid,v,False)
        rr=[];hh=[];orr=[];oh=[]
        for (y,x),p in zip(d['coords'],d['patches']):
            rp=resid[y:y+core.PATCH,x:x+core.PATCH]; rr.append(rp[core.RING].mean()); hh.append(rp[core.HIDDEN].mean()); orr.append(p[core.RING].mean()); oh.append(p[core.HIDDEN].mean())
        resrho=float(spearmanr(rr,hh).statistic); nullrho=float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic); obsrho=float(spearmanr(orr,oh).statistic)
        for rec in rows:
            if rec['field']==held: rec.update({'selection_cell_r2':r2,'selection_cell_spearman':sr,'observed_ring_hidden_spearman':obsrho,'residual_ring_hidden_spearman':resrho,'residual_shift_spearman':nullrho})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False); sums=[]
    for motif,g in df.groupby('motif'):
        sums.append({'motif':motif,'observed_ring_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_only_median_auc':float(np.nanmedian(g.selection_only_auc)),'observed_minus_selection':core.paired(g.observed_ring_auc-g.selection_only_auc)})
    one=df.drop_duplicates('field'); cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),'observed_ring_hidden_spearman_median':float(np.nanmedian(one.observed_ring_hidden_spearman)),'residual_ring_hidden_spearman_median':float(np.nanmedian(one.residual_ring_hidden_spearman)),'residual_shift_spearman_median':float(np.nanmedian(one.residual_shift_spearman)),'residual_minus_shift':core.paired(one.residual_ring_hidden_spearman-one.residual_shift_spearman)}
    prov={'status':'REAL_DR11_PIXEL_SELECTION_NULL','n_bricks':TARGET_FIELDS,'candidate_pool':MAX_CANDIDATES,'coverage_gate':{'min_valid_patches':MIN_PATCHES,'criterion':'official depth/NEXP/MASKBITS support only; no outcome metric'},'coverage_rejections':coverage_rejections,'selection_products':['depth-r','maskbits','nexp-r','psfsize-r'],'regions':provenance,'total_source_rows':int(sum(x['source_rows'] for x in provenance))}; (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','validation':'12 coverage-qualified DR11 regions selected in fixed expanded48 order; whole-region LOFO','motifs':sums,'continuous_locality':cont,'coverage_rejections':coverage_rejections}; (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
