#!/usr/bin/env python3
"""Pixel-level DR11 selection-null validation on 48 independent bricks.

Science target: observed DR11 source positions only.
Selection controls: official DR11 coadd depth-r, maskbits, nexp-r and psfsize-r.
All source-count predictions are cross-field. No simulated cosmology is used.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, time
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
TABLE='ls_dr11.tractor_s'
BRICK_SUMMARY_URL='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/survey-bricks-dr11-south.fits.gz'
BIT_COUNT=20

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
            r=requests.get(url,timeout=180); r.raise_for_status(); b=r.content
            if not b.startswith(b'SIMPLE'): raise RuntimeError(f'not FITS: {b[:50]!r}')
            return b
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(3*(i+1))
    raise RuntimeError(f'download failed {url}: {last}')

def get_bytes_any(url, attempts=4):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,timeout=180); r.raise_for_status(); b=r.content
            if len(b)<100: raise RuntimeError(f'too-short response: {b[:80]!r}')
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
        if pick is None:
            pick=next(h for h in H if h.data is not None and getattr(h.data,'ndim',0)==2)
        a=np.asarray(pick.data).copy(); hdr=pick.header.copy()
    return a,hdr,{'url':url,'sha256':sha256(b),'bytes':len(b),'shape':list(a.shape),'dtype':str(a.dtype),'extname':hdr.get('EXTNAME')}

def product_urls(brick):
    pre=brick[:3]; root=f'{BASE}/{pre}/{brick}'
    return {
        'depth_r':f'{root}/legacysurvey-{brick}-depth-r.fits.fz',
        'maskbits':f'{root}/legacysurvey-{brick}-maskbits.fits.fz',
        'nexp_r':f'{root}/legacysurvey-{brick}-nexp-r.fits.fz',
        'psfsize_r':f'{root}/legacysurvey-{brick}-psfsize-r.fits.fz',
    }

def load_brick_summary():
    b=get_bytes_any(BRICK_SUMMARY_URL)
    raw=gzip.decompress(b) if b[:2]==b'\x1f\x8b' else b
    with fits.open(io.BytesIO(raw),memmap=False) as H:
        tab=H[1].data
        names=np.asarray(tab['brickname']).astype(str)
        ras=np.asarray(tab['ra'],float); decs=np.asarray(tab['dec'],float)
    return {'brickname':names,'ra':ras,'dec':decs}, {'url':BRICK_SUMMARY_URL,'sha256':sha256(b),'bytes':len(b),'rows':int(len(names))}

def choose_brick(ra,dec,bricks):
    dra=((bricks['ra']-ra+180.0)%360.0)-180.0
    d2=(dra*np.cos(np.deg2rad(dec)))**2+(bricks['dec']-dec)**2
    i=int(np.argmin(d2)); dist=float(np.sqrt(d2[i]))
    if dist>0.30: raise RuntimeError(f'nearest covered brick too far from {ra},{dec}: {dist:.3f} deg')
    return str(bricks['brickname'][i]).strip(), float(bricks['ra'][i]), float(bricks['dec'][i]), dist

def source_catalog(brick):
    sql=f"SELECT ra,dec FROM {TABLE} WHERE brick_primary=1 AND brickname='{brick}'"
    d=query_df(sql)
    if list(d.columns)!=['ra','dec']: raise RuntimeError(f'bad source columns: {list(d.columns)}')
    return d,sql

def sample_cells(a):
    """Sample each 64x64 cell on a fixed 4x4 subgrid."""
    ny,nx=a.shape
    offs=np.array([.125,.375,.625,.875])
    yy=np.clip(((np.arange(GRID)[:,None]+offs[None,:])/GRID*ny).astype(int),0,ny-1)
    xx=np.clip(((np.arange(GRID)[:,None]+offs[None,:])/GRID*nx).astype(int),0,nx-1)
    return a[yy[:,None,:,None],xx[None,:,None,:]]

def continuous_features(a, log=False, positive_only=False):
    s=sample_cells(np.asarray(a,float))
    if positive_only:
        good=np.isfinite(s)&(s>0)
    else:
        good=np.isfinite(s)
    if log: s=np.log1p(np.clip(s,0,None))
    vals=np.where(good,s,np.nan)
    mean=np.nanmean(vals,axis=(2,3)); std=np.nanstd(vals,axis=(2,3)); valid=good.mean(axis=(2,3))
    mean=np.nan_to_num(mean,nan=0.0); std=np.nan_to_num(std,nan=0.0)
    return [mean,std,valid]

def mask_features(mask):
    s=sample_cells(np.asarray(mask,np.int64))
    # DR11 MASKBITS bit 0 is NPRIMARY; bit unset means BRICK_PRIMARY support.
    feats=[((s & 1)==0).mean(axis=(2,3)), (s==0).mean(axis=(2,3))]
    for bit in range(BIT_COUNT): feats.append(((s & (1<<bit))!=0).mean(axis=(2,3)))
    return feats

def count_grid(df,wcs,shape):
    x,y=wcs.world_to_pixel_values(df.ra.to_numpy(float),df.dec.to_numpy(float))
    ny,nx=shape
    keep=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny)
    bx=np.floor(x[keep]/nx*GRID).astype(int); by=np.floor(y[keep]/ny*GRID).astype(int)
    g=np.zeros((GRID,GRID),float); np.add.at(g,(by,bx),1.0)
    return g,int(keep.sum())

def normalize_field(g,valid):
    x=np.log1p(np.clip(g,0,None)); vals=x[valid]
    med=np.median(vals); sc=np.median(np.abs(vals-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(vals)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    z=(x-med)/sc; z[~valid]=0.0
    return z

def robust_standardize(g,valid):
    vals=g[valid]; med=np.median(vals); sc=np.median(np.abs(vals-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(vals)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    z=(g-med)/sc; z[~valid]=0.0
    return z

def make_patches(grid,sel,valid):
    gp=[]; sf=[]; coords=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            v=valid[y:y+PATCH,x:x+PATCH]
            if v.mean()<0.98 or not np.all(v[HIDDEN]) or not np.all(v[RING]): continue
            gp.append(grid[y:y+PATCH,x:x+PATCH])
            sf.append(sel[y:y+PATCH,x:x+PATCH,:])
            coords.append((y,x))
    return np.asarray(gp),np.asarray(sf),coords

def selection_patch_features(s):
    out=[]
    for mask in [HIDDEN,RING,FULL]:
        a=s[:,mask,:]
        out.extend([a.mean(axis=1),a.std(axis=1)])
    return np.concatenate(out,axis=1)

def obs_ring(p): return p[:,RING].mean(axis=1)
def hidden_mean(p): return p[:,HIDDEN].mean(axis=1)
def hidden_max(p): return p[:,HIDDEN].max(axis=1)

def balanced_weights(y):
    y=np.asarray(y,int); n=len(y); p=max(1,int(y.sum())); q=max(1,n-p)
    return np.where(y==1,n/(2*p),n/(2*q))

def auc_or_nan(y,score):
    return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,score))

def paired_summary(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    if not n:return {'n_fields':0}
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'mean_difference':float(np.mean(d)),'median_difference':float(np.median(d))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null48'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text())
    if centers.get('status')!='REAL_DR11' or len(centers.get('regions',[]))!=48: raise RuntimeError('48-field REAL_DR11 centers required')

    bricks, brick_summary_prov=load_brick_summary()
    fields={}; provenance=[]; selected_bricks=set()
    for j,r in enumerate(centers['regions']):
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
        brick,brick_ra,brick_dec,brick_distance=choose_brick(ra,dec,bricks)
        if brick in selected_bricks: raise RuntimeError(f'duplicate selected brick: {brick}')
        selected_bricks.add(brick); urls=product_urls(brick)
        print(f'[selection-null] {j+1}/48 {name} -> {brick}',flush=True)
        depth,hdr,dp=read_image(urls['depth_r'],'DEPTH_r')
        mask,_,mp=read_image(urls['maskbits'],'MASKBITS')
        nexp,_,npv=read_image(urls['nexp_r'],None)
        psf,_,pp=read_image(urls['psfsize_r'],None)
        if not (depth.shape==mask.shape==nexp.shape==psf.shape): raise RuntimeError(f'shape mismatch {brick}')
        w=WCS(hdr)
        src,sq=source_catalog(brick); counts,ninside=count_grid(src,w,depth.shape)
        mf=mask_features(mask); feat=[]; fn=[]
        for arr,names in [
            (continuous_features(depth,log=True,positive_only=True),['log_depth_mean','log_depth_std','depth_positive_frac']),
            (continuous_features(nexp,log=False,positive_only=True),['nexp_mean','nexp_std','nexp_positive_frac']),
            (continuous_features(psf,log=False,positive_only=True),['psfsize_mean','psfsize_std','psfsize_positive_frac']),
        ]:
            feat.extend(arr); fn.extend(names)
        feat.extend(mf); fn.extend(['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(BIT_COUNT)])
        sel=np.stack(feat,axis=-1)
        primary=sel[:,:,fn.index('primary_frac')]
        depth_cov=sel[:,:,fn.index('depth_positive_frac')]
        nexp_cov=sel[:,:,fn.index('nexp_positive_frac')]
        valid=(primary>=0.9375)&(depth_cov>=0.9375)&(nexp_cov>=0.9375)
        area=np.clip(primary,0.25,1.0); corrected=counts/area
        norm=normalize_field(corrected,valid)
        patches,selfeat,coords=make_patches(norm,sel,valid)
        if len(patches)<8: raise RuntimeError(f'too few valid patches {brick}: {len(patches)}')
        fields[name]={'brick':brick,'counts':corrected,'norm':norm,'sel':sel,'valid':valid,'patches':patches,'selpatch':selfeat,'coords':coords,'feature_names':fn}
        source_raw=src.sort_values(['ra','dec'],kind='mergesort').to_csv(index=False,lineterminator='\n').encode()
        provenance.append({'field':name,'requested_center_ra_deg':ra,'requested_center_dec_deg':dec,'brick':brick,'brick_center_ra_deg':brick_ra,'brick_center_dec_deg':brick_dec,'brick_center_distance_deg':brick_distance,'source_query':sq,'source_rows':int(len(src)),'source_canonical_sha256':sha256(source_raw),'sources_inside_wcs':ninside,'products':{'depth_r':dp,'maskbits':mp,'nexp_r':npv,'psfsize_r':pp},'valid_cell_fraction':float(valid.mean()),'n_patches':int(len(patches))})

    names=list(fields); feature_names=fields[names[0]]['feature_names']
    if any(fields[f]['feature_names']!=feature_names for f in names): raise RuntimeError('feature name mismatch')

    rows=[]
    for held in names:
        train=[f for f in names if f!=held]
        Ptr=np.concatenate([fields[f]['patches'] for f in train]); Pte=fields[held]['patches']
        Str=np.concatenate([selection_patch_features(fields[f]['selpatch']) for f in train]); Ste=selection_patch_features(fields[held]['selpatch'])
        Rtr=obs_ring(Ptr)[:,None]; Rte=obs_ring(Pte)[:,None]
        hmtr=hidden_mean(Ptr); hmte=hidden_mean(Pte); hxtr=hidden_max(Ptr); hxte=hidden_max(Pte)
        q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Rtr,ytr.astype(int))
            obsauc=auc_or_nan(yte,lr.predict_proba(Rte)[:,1])
            h=HistGradientBoostingClassifier(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,random_state=31)
            h.fit(Str,ytr.astype(int),sample_weight=balanced_weights(ytr))
            qcauc=auc_or_nan(yte,h.predict_proba(Ste)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obsauc,'selection_only_auc':qcauc})

    for held in names:
        train=[f for f in names if f!=held]; X=[]; y=[]
        for f in train:
            d=fields[f]; v=d['valid']; xx=d['sel'][v]; yy=d['counts'][v]; yy=yy/(np.mean(yy)+1e-6)
            X.append(xx); y.append(yy)
        X=np.concatenate(X); y=np.concatenate(y)
        if len(y)>120000:
            rng=np.random.default_rng(20260828 + names.index(held)); ii=rng.choice(len(y),120000,replace=False); X=X[ii]; y=y[ii]
        model=HistGradientBoostingRegressor(loss='poisson',max_iter=80,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=1.0,random_state=43)
        model.fit(X,np.clip(y,1e-4,None))
        d=fields[held]; v=d['valid']; pred=np.full((GRID,GRID),np.nan,float); pred[v]=np.clip(model.predict(d['sel'][v]),.03,20)
        true_rel=d['counts'][v]/(np.mean(d['counts'][v])+1e-6)
        r2=float(r2_score(true_rel,pred[v])); rho=float(spearmanr(true_rel,pred[v]).statistic)
        lam=pred*np.mean(d['counts'][v]); resid=np.zeros((GRID,GRID)); resid[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1.0); resid=robust_standardize(resid,v)
        rvals=[]; hvals=[]; rrvals=[]; hhvals=[]; svals=[]
        for (yy0,xx0),p in zip(d['coords'],d['patches']):
            rp=resid[yy0:yy0+PATCH,xx0:xx0+PATCH]
            lp=np.log(np.clip(pred[yy0:yy0+PATCH,xx0:xx0+PATCH],1e-4,None))
            rvals.append(float(p[RING].mean())); hvals.append(float(p[HIDDEN].mean()))
            rrvals.append(float(rp[RING].mean())); hhvals.append(float(rp[HIDDEN].mean())); svals.append(float(lp[HIDDEN].mean()))
        obs_rho=float(spearmanr(rvals,hvals).statistic); res_rho=float(spearmanr(rrvals,hhvals).statistic)
        null_rho=float(spearmanr(rrvals,np.roll(hhvals,max(1,len(hhvals)//3))).statistic)
        for rec in rows:
            if rec['field']==held:
                rec.update({'selection_cell_r2':r2,'selection_cell_spearman':rho,'observed_ring_hidden_spearman':obs_rho,'residual_ring_hidden_spearman':res_rho,'residual_shift_spearman':null_rho})
        Ptr=np.concatenate([fields[f]['patches'] for f in train]); Pte=d['patches']; hmtr=hidden_mean(Ptr); hmte=hidden_mean(Pte); hxtr=hidden_max(Ptr); hxte=hidden_max(Pte); q25,q75=np.quantile(hmtr,[.25,.75]);qpk=np.quantile(hxtr,.8)
        ytest={'void':hmte<=q25,'overdense':hmte>=q75,'peak':hxte>=qpk}
        for rec in rows:
            if rec['field']==held:
                score=np.asarray(svals)
                if rec['motif']=='void': score=-score
                rec['selection_expected_count_auc']=auc_or_nan(ytest[rec['motif']],score)

    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    summaries=[]
    for motif,g in df.groupby('motif'):
        obs=g.observed_ring_auc.to_numpy(); qcauc=g.selection_only_auc.to_numpy(); expauc=g.selection_expected_count_auc.to_numpy()
        summaries.append({'motif':motif,'observed_ring_median_auc':float(np.nanmedian(obs)),'selection_only_median_auc':float(np.nanmedian(qcauc)),'selection_expected_count_median_auc':float(np.nanmedian(expauc)),'observed_minus_selection':paired_summary(obs-qcauc),'observed_minus_expected_count':paired_summary(obs-expauc)})
    one=df.drop_duplicates('field')
    cont={
      'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),
      'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),
      'observed_ring_hidden_spearman_median':float(np.nanmedian(one.observed_ring_hidden_spearman)),
      'residual_ring_hidden_spearman_median':float(np.nanmedian(one.residual_ring_hidden_spearman)),
      'residual_shift_spearman_median':float(np.nanmedian(one.residual_shift_spearman)),
      'residual_minus_shift':paired_summary(one.residual_ring_hidden_spearman.to_numpy()-one.residual_shift_spearman.to_numpy()),
    }
    prov={'status':'REAL_DR11_PIXEL_SELECTION_NULL','dataset':'DESI Legacy Imaging Surveys DR11','table':TABLE,'n_bricks':48,'selection_products':['depth-r','maskbits','nexp-r','psfsize-r'],'selection_features':feature_names,'brick_selection':'nearest covered south brick center from official survey-bricks-dr11-south.fits.gz; independent of source counts','brick_summary':brick_summary_prov,'regions':provenance,'total_source_rows':int(sum(x['source_rows'] for x in provenance))}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','validation':'48 independent brick-primary regions; whole-brick LOFO','angular_patch_note':'64 grid per ~0.25-degree brick; 16-cell patches ~3.75 arcmin; hidden center ~1.875 arcmin','motifs':summaries,'continuous_locality':cont}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
