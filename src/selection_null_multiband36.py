#!/usr/bin/env python3
"""Availability-gated 36-field REAL_DR11 multiband selection-null replication.

Candidates are the first 48 pre-registered expanded48 fields. Rejection is
allowed only for objective acquisition/coverage conditions: duplicate brick,
missing official coadd product, product shape mismatch, or insufficient valid
patch coverage. No outcome metric is inspected during acceptance.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from astropy.wcs import WCS
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

import selection_null_multiband as mb

core=mb.core
BANDS=mb.BANDS
TARGET_FIELDS=36
MAX_CANDIDATES=48
MIN_PATCHES=8
STRIDE=8


def make_patches(grid,sel,valid):
    gp=[];sf=[];coords=[]
    for y in range(0,core.GRID-core.PATCH+1,STRIDE):
        for x in range(0,core.GRID-core.PATCH+1,STRIDE):
            v=valid[y:y+core.PATCH,x:x+core.PATCH]
            if v.mean()<.98 or not np.all(v[core.HIDDEN]) or not np.all(v[core.RING]): continue
            gp.append(grid[y:y+core.PATCH,x:x+core.PATCH]);sf.append(sel[y:y+core.PATCH,x:x+core.PATCH]);coords.append((y,x))
    return np.asarray(gp),np.asarray(sf),coords


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json');ap.add_argument('--out',default='results/real_dr11/selection_null_multiband36');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text());regs=centers.get('regions',[])[:MAX_CANDIDATES]
    if centers.get('status')!='REAL_DR11' or len(regs)<TARGET_FIELDS: raise RuntimeError('REAL_DR11 expanded48 centers required')
    fields={};rejected=[];provenance=[];used=set()
    for j,r in enumerate(regs):
        if len(fields)>=TARGET_FIELDS: break
        name=r['name'];ra=float(r['center_ra_deg']);dec=float(r['center_dec_deg'])
        try:
            brick,bq,near=core.choose_brick_from_center(ra,dec)
        except Exception as e:
            rejected.append({'field':name,'reason':'brick_resolution','error':str(e)});continue
        if brick in used:
            rejected.append({'field':name,'brick':brick,'reason':'duplicate'});continue
        used.add(brick);print(f'[multiband36] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}',flush=True)
        try:
            urls=mb.product_urls_all(brick);mask,_,pmask=core.read_image(urls['maskbits'],'MASKBITS')
            arrays={};product_prov={'maskbits':pmask};hdr=None
            for b in BANDS:
                depth,h,pdepth=core.read_image(urls[f'depth_{b}'],f'DEPTH_{b.upper()}')
                nexp,_,pnexp=core.read_image(urls[f'nexp_{b}'])
                psf,_,ppsf=core.read_image(urls[f'psfsize_{b}'])
                arrays[(b,'depth')]=depth;arrays[(b,'nexp')]=nexp;arrays[(b,'psf')]=psf
                product_prov.update({f'depth_{b}':pdepth,f'nexp_{b}':pnexp,f'psfsize_{b}':ppsf})
                if b=='r': hdr=h
        except Exception as e:
            rejected.append({'field':name,'brick':brick,'reason':'missing_coadd','error':str(e)});continue
        shape=mask.shape
        if any(a.shape!=shape for a in arrays.values()):
            rejected.append({'field':name,'brick':brick,'reason':'shape_mismatch'});continue
        try:
            src,sq=core.source_catalog(brick);counts,ninside=core.count_grid(src,WCS(hdr),shape)
        except Exception as e:
            rejected.append({'field':name,'brick':brick,'reason':'source_materialization','error':str(e)});continue
        feat=[];fn=[];band_indices={}
        for b in BANDS:
            start=len(feat)
            for arr,names in [
                (core.continuous_features(arrays[(b,'depth')],True,True),[f'log_depth_{b}_mean',f'log_depth_{b}_std',f'depth_{b}_positive_frac']),
                (core.continuous_features(arrays[(b,'nexp')],False,True),[f'nexp_{b}_mean',f'nexp_{b}_std',f'nexp_{b}_positive_frac']),
                (core.continuous_features(arrays[(b,'psf')],False,True),[f'psfsize_{b}_mean',f'psfsize_{b}_std',f'psfsize_{b}_positive_frac'])]:
                feat+=arr;fn+=names
            band_indices[b]=list(range(start,len(feat)))
        mask_start=len(feat);feat+=core.mask_features(mask);fn+=['primary_frac','clean_frac']+[f'maskbit_{k}_frac' for k in range(core.BIT_COUNT)];mask_indices=list(range(mask_start,len(feat)))
        sel=np.stack(feat,-1);primary=sel[:,:,fn.index('primary_frac')]
        valid=(primary>=.9375)&(sel[:,:,fn.index('depth_r_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_r_positive_frac')]>=.9375)
        corrected=counts/np.clip(primary,.25,1);norm=core.robust_norm(corrected,valid,True);patches,sp,coords=make_patches(norm,sel,valid)
        if len(patches)<MIN_PATCHES:
            rejected.append({'field':name,'brick':brick,'reason':'coverage','n_patches':int(len(patches))});continue
        r_idx=band_indices['r']+mask_indices
        fields[name]={'brick':brick,'counts':corrected,'sel':sel,'valid':valid,'patches':patches,'sp':sp,'coords':coords,'r_idx':r_idx}
        provenance.append({'field':name,'brick':brick,'source_rows':int(len(src)),'source_provenance':sq,'n_patches':int(len(patches)),'valid_cell_fraction':float(valid.mean()),'products':product_prov,'brick_choice':bq,'nearest_primary_source_deg':near})
        print(f'[multiband36] accept {len(fields)}/{TARGET_FIELDS} patches={len(patches)}',flush=True)
    if len(fields)!=TARGET_FIELDS:
        (out/'availability_rejections.json').write_text(json.dumps(rejected,indent=2,sort_keys=True)+'\n')
        raise RuntimeError(f'only {len(fields)} accepted from {MAX_CANDIDATES}')

    names=list(fields);rows=[]
    for held in names:
        train=[f for f in names if f!=held]
        Ptr=np.concatenate([fields[f]['patches'] for f in train]);Pte=fields[held]['patches']
        Salltr=np.concatenate([mb.patch_features(fields[f]['sp']) for f in train]);Sallte=mb.patch_features(fields[held]['sp'])
        Srtr=np.concatenate([mb.patch_features(fields[f]['sp'][...,fields[f]['r_idx']]) for f in train]);Srte=mb.patch_features(fields[held]['sp'][...,fields[held]['r_idx']])
        rtr=Ptr[:,core.RING].mean(1)[:,None];rte=Pte[:,core.RING].mean(1)[:,None]
        hmtr=Ptr[:,core.HIDDEN].mean(1);hmte=Pte[:,core.HIDDEN].mean(1);hxtr=Ptr[:,core.HIDDEN].max(1);hxte=Pte[:,core.HIDDEN].max(1)
        q25,q75=np.quantile(hmtr,[.25,.75]);qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(rtr,ytr.astype(int));obs=core.auc(yte,lr.predict_proba(rte)[:,1])
            def fit_auc(Xtr,Xte):
                clf=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=12,min_samples_leaf=30,l2_regularization=1.5,random_state=31).fit(Xtr,ytr.astype(int),sample_weight=core.balanced_weights(ytr));return core.auc(yte,clf.predict_proba(Xte)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obs,'selection_r_auc':fit_auc(Srtr,Srte),'selection_griz_auc':fit_auc(Salltr,Sallte)})
        d=fields[held];v=d['valid'];Xall=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]);Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train]);Xr=np.concatenate([fields[f]['sel'][fields[f]['valid']][:,fields[f]['r_idx']] for f in train])
        def fit_count(X,Xte):
            reg=HistGradientBoostingRegressor(loss='poisson',max_iter=60,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=70,l2_regularization=1.5,random_state=43).fit(X,np.clip(Y,1e-4,None));return np.clip(reg.predict(Xte),.03,20)
        pr=fit_count(Xr,d['sel'][v][:,d['r_idx']]);pa=fit_count(Xall,d['sel'][v]);true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6)
        pred_r=np.ones((core.GRID,core.GRID));pred_a=np.ones((core.GRID,core.GRID));pred_r[v]=pr;pred_a[v]=pa;const=np.ones((core.GRID,core.GRID))
        rho_const,null_const=mb.locality_from_residual(d['counts'],const,v,d['coords']);rho_r,null_r=mb.locality_from_residual(d['counts'],pred_r,v,d['coords']);rho_a,null_a=mb.locality_from_residual(d['counts'],pred_a,v,d['coords'])
        for rec in rows:
            if rec['field']==held: rec.update({'count_r_r2':float(r2_score(true,pr)),'count_griz_r2':float(r2_score(true,pa)),'constant_residual_rho':rho_const,'selection_r_residual_rho':rho_r,'selection_griz_residual_rho':rho_a,'selection_griz_shift_rho':null_a})
    df=pd.DataFrame(rows);df.to_csv(out/'field_metrics.csv',index=False)
    motifs=[]
    for motif,g in df.groupby('motif'):
        motifs.append({'motif':motif,'observed_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_r_median_auc':float(np.nanmedian(g.selection_r_auc)),'selection_griz_median_auc':float(np.nanmedian(g.selection_griz_auc)),'observed_minus_griz':mb.paired(g.observed_ring_auc-g.selection_griz_auc),'griz_minus_r':mb.paired(g.selection_griz_auc-g.selection_r_auc),'observed_minus_chance':mb.paired(g.observed_ring_auc-.5),'griz_minus_chance':mb.paired(g.selection_griz_auc-.5)})
    one=df.drop_duplicates('field');continuous={'count_r_r2_median':float(np.nanmedian(one.count_r_r2)),'count_griz_r2_median':float(np.nanmedian(one.count_griz_r2)),'constant_residual_rho_median':float(np.nanmedian(one.constant_residual_rho)),'selection_r_residual_rho_median':float(np.nanmedian(one.selection_r_residual_rho)),'selection_griz_residual_rho_median':float(np.nanmedian(one.selection_griz_residual_rho)),'selection_griz_shift_rho_median':float(np.nanmedian(one.selection_griz_shift_rho)),'constant_minus_griz_residual':mb.paired(one.constant_residual_rho-one.selection_griz_residual_rho),'griz_residual_minus_shift':mb.paired(one.selection_griz_residual_rho-one.selection_griz_shift_rho),'griz_r2_minus_r_r2':mb.paired(one.count_griz_r2-one.count_r_r2)}
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL_MULTIBAND_36','validation':'36 availability-qualified fields from fixed first-48 candidate order; whole-field LOFO; stride8','candidate_count':MAX_CANDIDATES,'accepted_fields':TARGET_FIELDS,'selection_has_hidden_region_maps':True,'motifs':motifs,'continuous_locality':continuous,'availability_rejections':rejected}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':provenance,'rejections':rejected},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
