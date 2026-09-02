#!/usr/bin/env python3
"""REAL_DR11 pixel-level selection-null test on 12 independent bricks.

Observed target: real DR11 Tractor RA/Dec from bounded field queries.
Selection controls: official DR11 depth-r, maskbits, nexp-r, psfsize-r maps.
Brick naming uses only a tiny cone query; source rows use the same bounded 0.4-deg
query pattern already validated by the expanded48 REAL_DR11 acquisition.
No simulated cosmology or mock catalog is used.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

from selection_null_validate_fast import (
    GRID, PATCH, STRIDE, N_FIELDS, BIT_COUNT, TABLE, HIDDEN, RING,
    q, query_df, read_image, product_urls, tangent_sep2,
    choose_brick_from_center, continuous_features, mask_features,
    count_grid, robust_norm, make_patches, selection_patch_features,
    balanced_weights, auc, paired,
)


def source_catalog_box(ra0: float, dec0: float):
    sql=(f"SELECT ra,dec FROM {TABLE} WHERE brick_primary=1 "
         f"AND q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},0.40000000)")
    d=query_df(sql)
    if list(d.columns)!=['ra','dec']:
        raise RuntimeError(f'bad source columns: {list(d.columns)}')
    raw=d.sort_values(['ra','dec'],kind='mergesort').to_csv(index=False,lineterminator='\n').encode()
    return d, {'query':sql,'rows':int(len(d)),'canonical_csv_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--out',default='results/real_dr11/selection_null12')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text()); regs=centers.get('regions',[])[:N_FIELDS]
    if centers.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS:
        raise RuntimeError('REAL_DR11 fixed centers required')

    fields={}; provenance=[]; used=set()
    for j,r in enumerate(regs):
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
        brick,bq,near=choose_brick_from_center(ra,dec)
        if brick in used: raise RuntimeError(f'duplicate brick {brick}')
        used.add(brick)
        print(f'[selection-null12-v2] {j+1}/{N_FIELDS} {name} -> {brick}',flush=True)
        urls=product_urls(brick)
        depth,hdr,dp=read_image(urls['depth_r'],'DEPTH_R')
        mask,_,mp=read_image(urls['maskbits'],'MASKBITS')
        nexp,_,npv=read_image(urls['nexp_r'])
        psf,_,pp=read_image(urls['psfsize_r'])
        if not(depth.shape==mask.shape==nexp.shape==psf.shape):
            raise RuntimeError(f'shape mismatch {brick}')
        src,spv=source_catalog_box(ra,dec)
        counts,ninside=count_grid(src,WCS(hdr),depth.shape)
        if ninside < 1000:
            raise RuntimeError(f'too few bounded-query sources inside brick WCS {brick}: {ninside}')

        feat=[]; fn=[]
        for arr,names in [
            (continuous_features(depth,True,True),['log_depth_mean','log_depth_std','depth_positive_frac']),
            (continuous_features(nexp,False,True),['nexp_mean','nexp_std','nexp_positive_frac']),
            (continuous_features(psf,False,True),['psfsize_mean','psfsize_std','psfsize_positive_frac']),
        ]:
            feat += arr; fn += names
        feat += mask_features(mask)
        fn += ['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(BIT_COUNT)]
        sel=np.stack(feat,-1)
        primary=sel[:,:,fn.index('primary_frac')]
        valid=(primary>=.9375)&(sel[:,:,fn.index('depth_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_positive_frac')]>=.9375)
        corrected=counts/np.clip(primary,.25,1)
        norm=robust_norm(corrected,valid,True)
        patches,sp,coords=make_patches(norm,sel,valid)
        if len(patches)<8: raise RuntimeError(f'too few valid patches {brick}: {len(patches)}')
        fields[name]={'brick':brick,'counts':corrected,'norm':norm,'sel':sel,'valid':valid,'patches':patches,'sp':sp,'coords':coords,'fn':fn}
        provenance.append({
            'field':name,'brick':brick,'center_ra_deg':ra,'center_dec_deg':dec,
            'nearest_primary_source_deg':near,'brick_choice_query':bq,
            'source_query':spv,'source_rows':int(len(src)),'sources_inside_wcs':ninside,
            'valid_cell_fraction':float(valid.mean()),'n_patches':int(len(patches)),
            'products':{'depth_r':dp,'maskbits':mp,'nexp_r':npv,'psfsize_r':pp},
        })

    names=list(fields); rows=[]
    for held in names:
        train=[f for f in names if f!=held]
        Ptr=np.concatenate([fields[f]['patches'] for f in train]); Pte=fields[held]['patches']
        Str=np.concatenate([selection_patch_features(fields[f]['sp']) for f in train]); Ste=selection_patch_features(fields[held]['sp'])
        rtr=Ptr[:,RING].mean(1)[:,None]; rte=Pte[:,RING].mean(1)[:,None]
        hmtr=Ptr[:,HIDDEN].mean(1); hmte=Pte[:,HIDDEN].mean(1)
        hxtr=Ptr[:,HIDDEN].max(1); hxte=Pte[:,HIDDEN].max(1)
        q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(rtr,ytr.astype(int))
            obs=auc(yte,lr.predict_proba(rte)[:,1])
            clf=HistGradientBoostingClassifier(max_iter=80,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=20,l2_regularization=1,random_state=31)
            clf.fit(Str,ytr.astype(int),sample_weight=balanced_weights(ytr))
            selauc=auc(yte,clf.predict_proba(Ste)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obs,'selection_only_auc':selauc})

        X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train])
        Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=50,learning_rate=.07,max_leaf_nodes=12,min_samples_leaf=60,l2_regularization=1,random_state=43)
        reg.fit(X,np.clip(Y,1e-4,None))
        d=fields[held]; v=d['valid']
        pred=np.full((GRID,GRID),np.nan); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20)
        true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6)
        r2=float(r2_score(true,pred[v])); sr=float(spearmanr(true,pred[v]).statistic)
        lam=pred*np.mean(d['counts'][v])
        resid=np.zeros((GRID,GRID)); resid[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1)
        resid=robust_norm(resid,v,False)
        rr=[]; hh=[]; orr=[]; oh=[]
        for (y,x),p in zip(d['coords'],d['patches']):
            rp=resid[y:y+PATCH,x:x+PATCH]
            rr.append(rp[RING].mean()); hh.append(rp[HIDDEN].mean())
            orr.append(p[RING].mean()); oh.append(p[HIDDEN].mean())
        resrho=float(spearmanr(rr,hh).statistic)
        nullrho=float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic)
        obsrho=float(spearmanr(orr,oh).statistic)
        for rec in rows:
            if rec['field']==held:
                rec.update({'selection_cell_r2':r2,'selection_cell_spearman':sr,
                            'observed_ring_hidden_spearman':obsrho,
                            'residual_ring_hidden_spearman':resrho,
                            'residual_shift_spearman':nullrho})

    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    sums=[]
    for motif,g in df.groupby('motif'):
        sums.append({'motif':motif,
                     'observed_ring_median_auc':float(np.nanmedian(g.observed_ring_auc)),
                     'selection_only_median_auc':float(np.nanmedian(g.selection_only_auc)),
                     'observed_minus_selection':paired(g.observed_ring_auc-g.selection_only_auc)})
    one=df.drop_duplicates('field')
    cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),
          'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),
          'observed_ring_hidden_spearman_median':float(np.nanmedian(one.observed_ring_hidden_spearman)),
          'residual_ring_hidden_spearman_median':float(np.nanmedian(one.residual_ring_hidden_spearman)),
          'residual_shift_spearman_median':float(np.nanmedian(one.residual_shift_spearman)),
          'residual_minus_shift':paired(one.residual_ring_hidden_spearman-one.residual_shift_spearman)}
    prov={'status':'REAL_DR11_PIXEL_SELECTION_NULL','n_bricks':N_FIELDS,
          'brick_selection':'fixed sky center -> nearest BRICK_PRIMARY source in small cone; no density ranking',
          'source_acquisition':'bounded 0.4-deg Data Lab query; then WCS filter to selected brick',
          'selection_products':['depth-r','maskbits','nexp-r','psfsize-r'],
          'regions':provenance,'total_source_rows':int(sum(x['source_rows'] for x in provenance))}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','validation':'12 independent DR11 brick regions; whole-brick LOFO; bounded-query source acquisition','motifs':sums,'continuous_locality':cont}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
