#!/usr/bin/env python3
"""Multiband REAL_DR11 pixel-level selection-null stress test.

Uses the same pre-registered candidate order and r-band coverage gate as the
successful 12-brick test, but gives the nuisance model official g/r/i/z
coadd depth, NEXP, PSFSIZE plus MASKBITS.  It also compares selection-based
residual locality against a constant-expectation residual processed with the
same transform, isolating transform effects from actual selection correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

import run_selection_null_direct_coverage as direct

core = direct.core
BANDS = "griz"
TARGET_FIELDS = 12
MAX_CANDIDATES = 24
MIN_PATCHES = 8
STRIDE_FINE = 4


def product_urls_all(brick: str) -> dict:
    root = f"{core.BASE}/{brick[:3]}/{brick}"
    out = {"maskbits": f"{root}/legacysurvey-{brick}-maskbits.fits.fz"}
    for b in BANDS:
        out[f"depth_{b}"] = f"{root}/legacysurvey-{brick}-depth-{b}.fits.fz"
        out[f"nexp_{b}"] = f"{root}/legacysurvey-{brick}-nexp-{b}.fits.fz"
        out[f"psfsize_{b}"] = f"{root}/legacysurvey-{brick}-psfsize-{b}.fits.fz"
    return out


def make_patches_stride(grid, sel, valid):
    gp=[]; sf=[]; coords=[]
    for y in range(0, core.GRID-core.PATCH+1, STRIDE_FINE):
        for x in range(0, core.GRID-core.PATCH+1, STRIDE_FINE):
            v=valid[y:y+core.PATCH,x:x+core.PATCH]
            if v.mean()<.98 or not np.all(v[core.HIDDEN]) or not np.all(v[core.RING]): continue
            gp.append(grid[y:y+core.PATCH,x:x+core.PATCH]); sf.append(sel[y:y+core.PATCH,x:x+core.PATCH]); coords.append((y,x))
    return np.asarray(gp),np.asarray(sf),coords


def patch_features(s):
    out=[]
    for m in [core.HIDDEN,core.RING,core.FULL]:
        a=s[:,m,:]; out += [a.mean(1),a.std(1)]
    return np.concatenate(out,1)


def paired(d, alt="greater"):
    a=np.asarray(d,float); a=a[np.isfinite(a) & (a!=0)]
    if len(a)==0:return {"n":0}
    pos=int((a>0).sum())
    try:w=float(wilcoxon(a,alternative=alt).pvalue)
    except Exception:w=float("nan")
    return {"n":int(len(a)),"positive":pos,"median":float(np.median(a)),"mean":float(np.mean(a)),
            "sign_p":float(binomtest(pos,len(a),.5,alternative=alt).pvalue),"wilcoxon_p":w}


def locality_from_residual(counts, pred_rel, valid, coords):
    mu=float(np.mean(counts[valid])); lam=np.clip(pred_rel,0.03,20)*mu
    z=np.zeros_like(counts,float); z[valid]=(counts[valid]-lam[valid])/np.sqrt(lam[valid]+1.0)
    z=core.robust_norm(z,valid,False)
    rr=[];hh=[]
    for y,x in coords:
        p=z[y:y+core.PATCH,x:x+core.PATCH]; rr.append(float(p[core.RING].mean())); hh.append(float(p[core.HIDDEN].mean()))
    return float(spearmanr(rr,hh).statistic),float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null_multiband12'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text()); regs=centers.get('regions',[])[:MAX_CANDIDATES]
    if centers.get('status')!='REAL_DR11' or len(regs)<TARGET_FIELDS: raise RuntimeError('REAL_DR11 centers required')
    fields={}; rejected=[]; provenance=[]; used=set()
    for j,r in enumerate(regs):
        if len(fields)>=TARGET_FIELDS:break
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg']); brick,bq,near=core.choose_brick_from_center(ra,dec)
        if brick in used: rejected.append({'field':name,'brick':brick,'reason':'duplicate'}); continue
        used.add(brick); print(f'[multiband] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}',flush=True)
        urls=product_urls_all(brick); mask,_,pmask=core.read_image(urls['maskbits'],'MASKBITS')
        arrays={}; product_prov={'maskbits':pmask}; hdr=None
        for b in BANDS:
            depth,h,pdepth=core.read_image(urls[f'depth_{b}'],f'DEPTH_{b.upper()}'); nexp,_,pnexp=core.read_image(urls[f'nexp_{b}']); psf,_,ppsf=core.read_image(urls[f'psfsize_{b}'])
            arrays[(b,'depth')]=depth;arrays[(b,'nexp')]=nexp;arrays[(b,'psf')]=psf;product_prov.update({f'depth_{b}':pdepth,f'nexp_{b}':pnexp,f'psfsize_{b}':ppsf})
            if b=='r':hdr=h
        shape=mask.shape
        if any(a.shape!=shape for a in arrays.values()):raise RuntimeError(f'shape mismatch {brick}')
        src,sq=core.source_catalog(brick); counts,ninside=core.count_grid(src,WCS(hdr),shape)
        feat=[];fn=[];band_indices={}
        for b in BANDS:
            start=len(feat)
            for arr,names in [
                (core.continuous_features(arrays[(b,'depth')],True,True),[f'log_depth_{b}_mean',f'log_depth_{b}_std',f'depth_{b}_positive_frac']),
                (core.continuous_features(arrays[(b,'nexp')],False,True),[f'nexp_{b}_mean',f'nexp_{b}_std',f'nexp_{b}_positive_frac']),
                (core.continuous_features(arrays[(b,'psf')],False,True),[f'psfsize_{b}_mean',f'psfsize_{b}_std',f'psfsize_{b}_positive_frac'])]: feat+=arr;fn+=names
            band_indices[b]=list(range(start,len(feat)))
        mask_start=len(feat);feat+=core.mask_features(mask);fn+=['primary_frac','clean_frac']+[f'maskbit_{k}_frac' for k in range(core.BIT_COUNT)];mask_indices=list(range(mask_start,len(feat)))
        sel=np.stack(feat,-1); primary=sel[:,:,fn.index('primary_frac')]
        valid=(primary>=.9375)&(sel[:,:,fn.index('depth_r_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_r_positive_frac')]>=.9375)
        corrected=counts/np.clip(primary,.25,1);norm=core.robust_norm(corrected,valid,True);patches,sp,coords=make_patches_stride(norm,sel,valid)
        if len(patches)<MIN_PATCHES: rejected.append({'field':name,'brick':brick,'reason':'coverage','n_patches':int(len(patches))});continue
        r_idx=band_indices['r']+mask_indices
        fields[name]={'brick':brick,'counts':corrected,'norm':norm,'sel':sel,'valid':valid,'patches':patches,'sp':sp,'coords':coords,'r_idx':r_idx,'fn':fn}
        provenance.append({'field':name,'brick':brick,'source_rows':int(len(src)),'source_provenance':sq,'n_patches':int(len(patches)),'valid_cell_fraction':float(valid.mean()),'products':product_prov,'brick_choice':bq,'nearest_primary_source_deg':near})
        print(f'[multiband] accept {len(fields)}/{TARGET_FIELDS} patches={len(patches)}',flush=True)
    if len(fields)!=TARGET_FIELDS:raise RuntimeError(f'only {len(fields)} accepted')

    names=list(fields);rows=[]
    for held in names:
        train=[f for f in names if f!=held];Ptr=np.concatenate([fields[f]['patches'] for f in train]);Pte=fields[held]['patches']
        Salltr=np.concatenate([patch_features(fields[f]['sp']) for f in train]);Sallte=patch_features(fields[held]['sp'])
        Srtr=np.concatenate([patch_features(fields[f]['sp'][...,fields[f]['r_idx']]) for f in train]);Srte=patch_features(fields[held]['sp'][...,fields[held]['r_idx']])
        rtr=Ptr[:,core.RING].mean(1)[:,None];rte=Pte[:,core.RING].mean(1)[:,None];hmtr=Ptr[:,core.HIDDEN].mean(1);hmte=Pte[:,core.HIDDEN].mean(1);hxtr=Ptr[:,core.HIDDEN].max(1);hxte=Pte[:,core.HIDDEN].max(1);q25,q75=np.quantile(hmtr,[.25,.75]);qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(rtr,ytr.astype(int));obs=core.auc(yte,lr.predict_proba(rte)[:,1])
            def fit_auc(Xtr,Xte):
                clf=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=12,min_samples_leaf=30,l2_regularization=1.5,random_state=31).fit(Xtr,ytr.astype(int),sample_weight=core.balanced_weights(ytr));return core.auc(yte,clf.predict_proba(Xte)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obs,'selection_r_auc':fit_auc(Srtr,Srte),'selection_griz_auc':fit_auc(Salltr,Sallte)})
        d=fields[held];v=d['valid'];Xall=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]);Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
        Xr=np.concatenate([fields[f]['sel'][fields[f]['valid']][:,fields[f]['r_idx']] for f in train])
        def fit_count(X, Xte):
            reg=HistGradientBoostingRegressor(loss='poisson',max_iter=60,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=70,l2_regularization=1.5,random_state=43).fit(X,np.clip(Y,1e-4,None));return np.clip(reg.predict(Xte),.03,20)
        pr=fit_count(Xr,d['sel'][v][:,d['r_idx']]);pa=fit_count(Xall,d['sel'][v]);true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6)
        pred_r=np.ones((core.GRID,core.GRID));pred_a=np.ones((core.GRID,core.GRID));pred_r[v]=pr;pred_a[v]=pa
        const=np.ones((core.GRID,core.GRID));rho_const,null_const=locality_from_residual(d['counts'],const,v,d['coords']);rho_r,null_r=locality_from_residual(d['counts'],pred_r,v,d['coords']);rho_a,null_a=locality_from_residual(d['counts'],pred_a,v,d['coords'])
        for rec in rows:
            if rec['field']==held:rec.update({'count_r_r2':float(r2_score(true,pr)),'count_griz_r2':float(r2_score(true,pa)),'constant_residual_rho':rho_const,'selection_r_residual_rho':rho_r,'selection_griz_residual_rho':rho_a,'selection_griz_shift_rho':null_a})
    df=pd.DataFrame(rows);df.to_csv(out/'field_metrics.csv',index=False)
    motifs=[]
    for motif,g in df.groupby('motif'):
        motifs.append({'motif':motif,'observed_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_r_median_auc':float(np.nanmedian(g.selection_r_auc)),'selection_griz_median_auc':float(np.nanmedian(g.selection_griz_auc)),'observed_minus_griz':paired(g.observed_ring_auc-g.selection_griz_auc),'griz_minus_r':paired(g.selection_griz_auc-g.selection_r_auc),'observed_minus_chance':paired(g.observed_ring_auc-.5),'griz_minus_chance':paired(g.selection_griz_auc-.5)})
    one=df.drop_duplicates('field');continuous={'count_r_r2_median':float(np.nanmedian(one.count_r_r2)),'count_griz_r2_median':float(np.nanmedian(one.count_griz_r2)),'constant_residual_rho_median':float(np.nanmedian(one.constant_residual_rho)),'selection_r_residual_rho_median':float(np.nanmedian(one.selection_r_residual_rho)),'selection_griz_residual_rho_median':float(np.nanmedian(one.selection_griz_residual_rho)),'selection_griz_shift_rho_median':float(np.nanmedian(one.selection_griz_shift_rho)),'constant_minus_griz_residual':paired(one.constant_residual_rho-one.selection_griz_residual_rho),'griz_residual_minus_shift':paired(one.selection_griz_residual_rho-one.selection_griz_shift_rho),'griz_r2_minus_r_r2':paired(one.count_griz_r2-one.count_r_r2)}
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL_MULTIBAND','validation':'12 coverage-qualified whole-brick LOFO; stride4 sensitivity','bands':list(BANDS),'selection_has_hidden_region_maps':True,'motifs':motifs,'continuous_locality':continuous,'coverage_rejections':rejected};(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':'REAL_DR11_PIXEL_SELECTION_NULL_MULTIBAND','regions':provenance,'rejections':rejected},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
