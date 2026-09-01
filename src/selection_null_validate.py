#!/usr/bin/env python3
"""Fast pixel-level DR11 selection-null validation on independent bricks.

Science target: observed DR11 source positions only.
Selection controls: official DR11 coadd depth-r, maskbits, nexp-r and psfsize-r.
Each held-out brick is evaluated only by models trained on other bricks.
No simulated cosmology and no mock fallback are used.
"""
from __future__ import annotations
import argparse, hashlib, io, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS
from dl import queryClient as qc
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

GRID=64; PATCH=16; STRIDE=8
BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd'
TABLE='ls_dr11.tractor_s'; BIT_COUNT=20
HIDDEN=np.zeros((PATCH,PATCH),bool); HIDDEN[4:12,4:12]=True
RING=np.zeros((PATCH,PATCH),bool)
RING[3,3:13]=True; RING[12,3:13]=True; RING[4:12,3]=True; RING[4:12,12]=True
FULL=np.ones((PATCH,PATCH),bool)

def sha256(b): return hashlib.sha256(b).hexdigest()

def q(sql, attempts=4):
    last=None
    for i in range(attempts):
        try:
            out=qc.query(sql=sql,fmt='csv',async_=False)
            if isinstance(out,bytes): out=out.decode()
            if not isinstance(out,str): raise RuntimeError(type(out))
            return out
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(3*(i+1))
    raise RuntimeError(f'query failed: {last}')

def query_df(sql):
    from io import StringIO
    return pd.read_csv(StringIO(q(sql)))

def get_bytes(url, attempts=4):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,timeout=120); r.raise_for_status(); b=r.content
            if not b.startswith(b'SIMPLE'): raise RuntimeError(f'not FITS: {b[:50]!r}')
            return b
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(3*(i+1))
    raise RuntimeError(f'download failed {url}: {last}')

def read_image(url, wanted=None):
    b=get_bytes(url)
    with fits.open(io.BytesIO(b),memmap=False) as H:
        pick=None
        for h in H:
            if h.data is None or getattr(h.data,'ndim',0)!=2: continue
            if wanted is None or str(h.name).upper()==wanted.upper() or str(h.header.get('EXTNAME','')).upper()==wanted.upper():
                pick=h; break
        if pick is None: pick=next(h for h in H if h.data is not None and getattr(h.data,'ndim',0)==2)
        a=np.asarray(pick.data).copy(); hdr=pick.header.copy()
    return a,hdr,{'url':url,'sha256':sha256(b),'bytes':len(b),'shape':list(a.shape),'dtype':str(a.dtype),'extname':hdr.get('EXTNAME')}

def product_urls(brick):
    pre=brick[:3]; root=f'{BASE}/{pre}/{brick}'
    return {
      'depth_r':f'{root}/legacysurvey-{brick}-depth-r.fits.fz',
      'maskbits':f'{root}/legacysurvey-{brick}-maskbits.fits.fz',
      'nexp_r':f'{root}/legacysurvey-{brick}-nexp-r.fits.fz',
      'psfsize_r':f'{root}/legacysurvey-{brick}-psfsize-r.fits.fz'}

def choose_brick(ra,dec):
    # Geometrically local lookup: pick the brick of the nearest primary source.
    # Radius expansion is deterministic and does not optimize on source counts.
    for rad in (0.03,0.06,0.12):
        sql=(f"SELECT brickname,ra,dec FROM {TABLE} WHERE brick_primary=1 "
             f"AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{rad:.8f})")
        d=query_df(sql)
        if len(d):
            dra=((d.ra.to_numpy(float)-ra+180)%360)-180
            dsq=(dra*np.cos(np.deg2rad(dec)))**2+(d.dec.to_numpy(float)-dec)**2
            i=int(np.argmin(dsq)); return str(d.brickname.iloc[i]).strip(),float(np.sqrt(dsq[i])),sql
    raise RuntimeError(f'no primary source near requested center {ra},{dec}')

def source_catalog(brick):
    sql=f"SELECT ra,dec FROM {TABLE} WHERE brick_primary=1 AND brickname='{brick}'"
    d=query_df(sql)
    if list(d.columns)!=['ra','dec']: raise RuntimeError(f'bad source columns: {list(d.columns)}')
    return d,sql

def sample_cells(a):
    ny,nx=a.shape; offs=np.array([.125,.375,.625,.875])
    yy=np.clip(((np.arange(GRID)[:,None]+offs[None,:])/GRID*ny).astype(int),0,ny-1)
    xx=np.clip(((np.arange(GRID)[:,None]+offs[None,:])/GRID*nx).astype(int),0,nx-1)
    return a[yy[:,None,:,None],xx[None,:,None,:]]

def continuous_features(a, log=False, positive_only=False):
    s=sample_cells(np.asarray(a,float)); good=np.isfinite(s)&((s>0) if positive_only else True)
    if log: s=np.log1p(np.clip(s,0,None))
    vals=np.where(good,s,np.nan)
    mean=np.nan_to_num(np.nanmean(vals,axis=(2,3)),nan=0.0)
    std=np.nan_to_num(np.nanstd(vals,axis=(2,3)),nan=0.0)
    return [mean,std,good.mean(axis=(2,3))]

def mask_features(mask):
    s=sample_cells(np.asarray(mask,np.int64))
    feats=[((s&1)==0).mean(axis=(2,3)),(s==0).mean(axis=(2,3))]
    feats += [((s&(1<<bit))!=0).mean(axis=(2,3)) for bit in range(BIT_COUNT)]
    return feats

def count_grid(df,wcs,shape):
    x,y=wcs.world_to_pixel_values(df.ra.to_numpy(float),df.dec.to_numpy(float)); ny,nx=shape
    keep=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny)
    bx=np.floor(x[keep]/nx*GRID).astype(int); by=np.floor(y[keep]/ny*GRID).astype(int)
    g=np.zeros((GRID,GRID),float); np.add.at(g,(by,bx),1.0)
    return g,int(keep.sum())

def normalize_field(g,valid):
    x=np.log1p(np.clip(g,0,None)); vals=x[valid]; med=np.median(vals); sc=np.median(np.abs(vals-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(vals)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    z=(x-med)/sc; z[~valid]=0.0; return z

def robust_standardize(g,valid):
    vals=g[valid]; med=np.median(vals); sc=np.median(np.abs(vals-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(vals)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    z=(g-med)/sc; z[~valid]=0.0; return z

def make_patches(grid,sel,valid):
    gp=[]; sf=[]; coords=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
      for x in range(0,GRID-PATCH+1,STRIDE):
        v=valid[y:y+PATCH,x:x+PATCH]
        if v.mean()<.98 or not np.all(v[HIDDEN]) or not np.all(v[RING]): continue
        gp.append(grid[y:y+PATCH,x:x+PATCH]); sf.append(sel[y:y+PATCH,x:x+PATCH,:]); coords.append((y,x))
    return np.asarray(gp),np.asarray(sf),coords

def selection_patch_features(s):
    out=[]
    for mask in (HIDDEN,RING,FULL):
        a=s[:,mask,:]; out += [a.mean(axis=1),a.std(axis=1)]
    return np.concatenate(out,axis=1)

def obs_ring(p): return p[:,RING].mean(axis=1)
def hidden_mean(p): return p[:,HIDDEN].mean(axis=1)
def hidden_max(p): return p[:,HIDDEN].max(axis=1)
def balanced_weights(y):
    y=np.asarray(y,int); n=len(y); p=max(1,int(y.sum())); q=max(1,n-p); return np.where(y==1,n/(2*p),n/(2*q))
def auc_or_nan(y,score): return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,score))

def paired_summary(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    if not n:return {'n_fields':0}
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'mean_difference':float(np.mean(d)),'median_difference':float(np.median(d))}

def acquire_field(r):
    name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
    brick,nearest_dist,brick_lookup_sql=choose_brick(ra,dec); urls=product_urls(brick)
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut={k:ex.submit(read_image,u,('DEPTH_r' if k=='depth_r' else 'MASKBITS' if k=='maskbits' else None)) for k,u in urls.items()}
        got={k:v.result() for k,v in fut.items()}
    depth,hdr,dp=got['depth_r']; mask,_,mp=got['maskbits']; nexp,_,npv=got['nexp_r']; psf,_,pp=got['psfsize_r']
    if not(depth.shape==mask.shape==nexp.shape==psf.shape): raise RuntimeError(f'shape mismatch {brick}')
    src,sq=source_catalog(brick); counts,ninside=count_grid(src,WCS(hdr),depth.shape)
    feat=[]; fn=[]
    for arr,names in [
      (continuous_features(depth,True,True),['log_depth_mean','log_depth_std','depth_positive_frac']),
      (continuous_features(nexp,False,True),['nexp_mean','nexp_std','nexp_positive_frac']),
      (continuous_features(psf,False,True),['psfsize_mean','psfsize_std','psfsize_positive_frac'])]:
        feat.extend(arr); fn.extend(names)
    feat.extend(mask_features(mask)); fn.extend(['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(BIT_COUNT)])
    sel=np.stack(feat,axis=-1); primary=sel[:,:,fn.index('primary_frac')]
    valid=(primary>=.9375)&(sel[:,:,fn.index('depth_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_positive_frac')]>=.9375)
    corrected=counts/np.clip(primary,.25,1.0); norm=normalize_field(corrected,valid); patches,selfeat,coords=make_patches(norm,sel,valid)
    if len(patches)<8: raise RuntimeError(f'too few valid patches {brick}: {len(patches)}')
    source_raw=src.sort_values(['ra','dec'],kind='mergesort').to_csv(index=False,lineterminator='\n').encode()
    data={'brick':brick,'counts':corrected,'sel':sel,'valid':valid,'patches':patches,'selpatch':selfeat,'coords':coords,'feature_names':fn}
    prov={'field':name,'requested_center_ra_deg':ra,'requested_center_dec_deg':dec,'brick':brick,'nearest_primary_source_distance_deg':nearest_dist,'brick_lookup_query':brick_lookup_sql,'source_query':sq,'source_rows':int(len(src)),'source_canonical_sha256':sha256(source_raw),'sources_inside_wcs':ninside,'products':{'depth_r':dp,'maskbits':mp,'nexp_r':npv,'psfsize_r':pp},'valid_cell_fraction':float(valid.mean()),'n_patches':int(len(patches))}
    return name,data,prov

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null48'); ap.add_argument('--n-fields',type=int,default=24); ap.add_argument('--folds',type=int,default=4); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); centers=json.loads(Path(args.centers).read_text())
    regs=centers.get('regions',[])[:args.n_fields]
    if centers.get('status')!='REAL_DR11' or len(regs)!=args.n_fields: raise RuntimeError('provenance-verified REAL_DR11 centers required')
    fields={}; provenance=[]; selected=set()
    for j,r in enumerate(regs):
        name,d,p=acquire_field(r)
        if d['brick'] in selected: raise RuntimeError(f'duplicate selected brick {d["brick"]}')
        selected.add(d['brick']); fields[name]=d; provenance.append(p)
        print(f'[selection-null] acquired {j+1}/{len(regs)} {name} -> {d["brick"]}: {p["source_rows"]} sources, {p["n_patches"]} patches',flush=True)
    names=list(fields); feature_names=fields[names[0]]['feature_names']
    rows=[]; nfold=max(2,min(args.folds,len(names)))
    fold_id={n:i%nfold for i,n in enumerate(names)}
    for fold in range(nfold):
        test=[n for n in names if fold_id[n]==fold]; train=[n for n in names if fold_id[n]!=fold]
        Ptr=np.concatenate([fields[f]['patches'] for f in train]); Str=np.concatenate([selection_patch_features(fields[f]['selpatch']) for f in train]); Rtr=obs_ring(Ptr)[:,None]
        hmtr=hidden_mean(Ptr); hxtr=hidden_max(Ptr); q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        train_labels={'void':hmtr<=q25,'overdense':hmtr>=q75,'peak':hxtr>=qpk}
        ring_models={m:LogisticRegression(max_iter=1000,class_weight='balanced').fit(Rtr,y.astype(int)) for m,y in train_labels.items()}
        sel_models={}
        for m,y in train_labels.items():
            h=HistGradientBoostingClassifier(max_iter=80,learning_rate=.07,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=1.0,random_state=31)
            h.fit(Str,y.astype(int),sample_weight=balanced_weights(y)); sel_models[m]=h
        # One count-selection model per fold, not per brick.
        X=[]; y=[]
        for f in train:
            d=fields[f]; v=d['valid']; X.append(d['sel'][v]); yy=d['counts'][v]; y.append(yy/(np.mean(yy)+1e-6))
        X=np.concatenate(X); y=np.concatenate(y)
        if len(y)>80000:
            rng=np.random.default_rng(20260901+fold); ii=rng.choice(len(y),80000,replace=False); X=X[ii]; y=y[ii]
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=60,learning_rate=.07,max_leaf_nodes=15,min_samples_leaf=60,l2_regularization=1.0,random_state=43).fit(X,np.clip(y,1e-4,None))
        for held in test:
            d=fields[held]; Pte=d['patches']; Ste=selection_patch_features(d['selpatch']); Rte=obs_ring(Pte)[:,None]
            hmte=hidden_mean(Pte); hxte=hidden_max(Pte); ytest={'void':hmte<=q25,'overdense':hmte>=q75,'peak':hxte>=qpk}
            v=d['valid']; pred=np.full((GRID,GRID),np.nan,float); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20)
            true_rel=d['counts'][v]/(np.mean(d['counts'][v])+1e-6); r2=float(r2_score(true_rel,pred[v])); sr=float(spearmanr(true_rel,pred[v]).statistic)
            lam=pred*np.mean(d['counts'][v]); resid=np.zeros((GRID,GRID)); resid[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1.0); resid=robust_standardize(resid,v)
            rv=[]; hv=[]; rrv=[]; hhv=[]; sv=[]
            for (yy0,xx0),p in zip(d['coords'],Pte):
                rp=resid[yy0:yy0+PATCH,xx0:xx0+PATCH]; lp=np.log(np.clip(pred[yy0:yy0+PATCH,xx0:xx0+PATCH],1e-4,None))
                rv.append(float(p[RING].mean())); hv.append(float(p[HIDDEN].mean())); rrv.append(float(rp[RING].mean())); hhv.append(float(rp[HIDDEN].mean())); sv.append(float(lp[HIDDEN].mean()))
            obsrho=float(spearmanr(rv,hv).statistic); resrho=float(spearmanr(rrv,hhv).statistic); nullrho=float(spearmanr(rrv,np.roll(hhv,max(1,len(hhv)//3))).statistic)
            for motif in ('void','overdense','peak'):
                yt=ytest[motif]; obsauc=auc_or_nan(yt,ring_models[motif].predict_proba(Rte)[:,1]); selauc=auc_or_nan(yt,sel_models[motif].predict_proba(Ste)[:,1])
                score=np.asarray(sv); expauc=auc_or_nan(yt,-score if motif=='void' else score)
                rows.append({'field':held,'fold':fold,'motif':motif,'observed_ring_auc':obsauc,'selection_only_auc':selauc,'selection_expected_count_auc':expauc,'selection_cell_r2':r2,'selection_cell_spearman':sr,'observed_ring_hidden_spearman':obsrho,'residual_ring_hidden_spearman':resrho,'residual_shift_spearman':nullrho})
        print(f'[selection-null] cross-fit fold {fold+1}/{nfold}: train={len(train)}, test={len(test)}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    summaries=[]
    for motif,g in df.groupby('motif'):
        obs=g.observed_ring_auc.to_numpy(); sel=g.selection_only_auc.to_numpy(); exp=g.selection_expected_count_auc.to_numpy()
        summaries.append({'motif':motif,'observed_ring_median_auc':float(np.nanmedian(obs)),'selection_only_median_auc':float(np.nanmedian(sel)),'selection_expected_count_median_auc':float(np.nanmedian(exp)),'observed_minus_selection':paired_summary(obs-sel),'observed_minus_expected_count':paired_summary(obs-exp)})
    one=df.drop_duplicates('field'); cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),'observed_ring_hidden_spearman_median':float(np.nanmedian(one.observed_ring_hidden_spearman)),'residual_ring_hidden_spearman_median':float(np.nanmedian(one.residual_ring_hidden_spearman)),'residual_shift_spearman_median':float(np.nanmedian(one.residual_shift_spearman)),'residual_minus_shift':paired_summary(one.residual_ring_hidden_spearman.to_numpy()-one.residual_shift_spearman.to_numpy())}
    prov={'status':'REAL_DR11_PIXEL_SELECTION_NULL','dataset':'DESI Legacy Imaging Surveys DR11','table':TABLE,'n_bricks':len(names),'selection_products':['depth-r','maskbits','nexp-r','psfsize-r'],'selection_features':feature_names,'brick_selection':'brickname of nearest BRICK_PRIMARY source to fixed predeclared field center; deterministic radius expansion; not optimized on source count','cross_validation':f'{nfold}-fold grouped by whole brick','regions':provenance,'total_source_rows':int(sum(x['source_rows'] for x in provenance))}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','validation':f'{len(names)} independent brick-primary regions; {nfold}-fold whole-brick cross-fitting','motifs':summaries,'continuous_locality':cont}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
