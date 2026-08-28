#!/usr/bin/env python3
"""Fast REAL_DR11 pixel-level selection-null test on 12 independent bricks.

Observed target: DR11 brick-primary RA/Dec counts.
Selection controls: official DR11 depth-r, maskbits, nexp-r, psfsize-r maps.
No cosmological simulation or mock catalog is used.
"""
from __future__ import annotations
import argparse, hashlib, io, json, time
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

GRID=64; PATCH=16; STRIDE=8; N_FIELDS=12; BIT_COUNT=20
BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd'
TABLE='ls_dr11.tractor_s'
HIDDEN=np.zeros((PATCH,PATCH),bool); HIDDEN[4:12,4:12]=True
RING=np.zeros((PATCH,PATCH),bool); RING[3,3:13]=True; RING[12,3:13]=True; RING[4:12,3]=True; RING[4:12,12]=True
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
            if i+1<attempts: time.sleep(2*(i+1))
    raise RuntimeError(f'query failed: {last}')
def query_df(sql):
    from io import StringIO
    return pd.read_csv(StringIO(q(sql)))
def get_bytes(url, attempts=4):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,timeout=90); r.raise_for_status(); b=r.content
            if not b.startswith(b'SIMPLE'): raise RuntimeError(f'not FITS: {b[:40]!r}')
            return b
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(2*(i+1))
    raise RuntimeError(f'download failed {url}: {last}')
def read_image(url, wanted=None):
    b=get_bytes(url)
    with fits.open(io.BytesIO(b),memmap=False) as H:
        candidates=[h for h in H if h.data is not None and getattr(h.data,'ndim',0)==2]
        if wanted:
            pick=next((h for h in candidates if str(h.name).upper()==wanted.upper() or str(h.header.get('EXTNAME','')).upper()==wanted.upper()),candidates[0])
        else: pick=candidates[0]
        a=np.asarray(pick.data).copy(); hdr=pick.header.copy()
    return a,hdr,{'url':url,'sha256':sha256(b),'bytes':len(b),'shape':list(a.shape),'extname':hdr.get('EXTNAME')}
def product_urls(brick):
    root=f'{BASE}/{brick[:3]}/{brick}'
    return {k:f'{root}/legacysurvey-{brick}-{s}.fits.fz' for k,s in {
        'depth_r':'depth-r','maskbits':'maskbits','nexp_r':'nexp-r','psfsize_r':'psfsize-r'}.items()}
def tangent_sep2(ra,dec,ra0,dec0):
    dra=((np.asarray(ra,float)-ra0+180)%360)-180
    return (dra*np.cos(np.deg2rad(dec0)))**2+(np.asarray(dec,float)-dec0)**2
def choose_brick_from_center(ra,dec):
    for radius in [0.03,0.06,0.12]:
        sql=(f"SELECT brickname,ra,dec FROM {TABLE} WHERE brick_primary=1 "
             f"AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{radius:.8f})")
        d=query_df(sql)
        if len(d):
            i=int(np.argmin(tangent_sep2(d.ra,d.dec,ra,dec)))
            brick=str(d.iloc[i].brickname).strip()
            return brick,sql,float(np.sqrt(tangent_sep2([d.iloc[i].ra],[d.iloc[i].dec],ra,dec)[0]))
    raise RuntimeError(f'no BRICK_PRIMARY source near fixed center {ra},{dec}')
def source_catalog(brick):
    sql=f"SELECT ra,dec FROM {TABLE} WHERE brick_primary=1 AND brickname='{brick}'"
    d=query_df(sql)
    if list(d.columns)!=['ra','dec']: raise RuntimeError(f'bad source columns {list(d.columns)}')
    return d,sql
def sample_cells(a):
    ny,nx=a.shape; offs=np.array([.125,.375,.625,.875])
    yy=np.clip(((np.arange(GRID)[:,None]+offs)/GRID*ny).astype(int),0,ny-1)
    xx=np.clip(((np.arange(GRID)[:,None]+offs)/GRID*nx).astype(int),0,nx-1)
    return a[yy[:,None,:,None],xx[None,:,None,:]]
def continuous_features(a,log=False,positive_only=False):
    s=sample_cells(np.asarray(a,float)); good=np.isfinite(s)&((s>0) if positive_only else True)
    if log:s=np.log1p(np.clip(s,0,None))
    vals=np.where(good,s,np.nan)
    mean=np.nan_to_num(np.nanmean(vals,axis=(2,3)),nan=0.0)
    std=np.nan_to_num(np.nanstd(vals,axis=(2,3)),nan=0.0)
    return [mean,std,good.mean(axis=(2,3))]
def mask_features(mask):
    s=sample_cells(np.asarray(mask,np.int64)); feats=[((s&1)==0).mean((2,3)),(s==0).mean((2,3))]
    feats.extend([((s&(1<<b))!=0).mean((2,3)) for b in range(BIT_COUNT)])
    return feats
def count_grid(df,wcs,shape):
    x,y=wcs.world_to_pixel_values(df.ra.to_numpy(float),df.dec.to_numpy(float)); ny,nx=shape
    keep=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny)
    bx=np.floor(x[keep]/nx*GRID).astype(int); by=np.floor(y[keep]/ny*GRID).astype(int)
    g=np.zeros((GRID,GRID),float); np.add.at(g,(by,bx),1.0); return g,int(keep.sum())
def robust_norm(g,valid,log=True):
    x=np.log1p(np.clip(g,0,None)) if log else g.copy(); vals=x[valid]
    med=np.median(vals); sc=np.median(np.abs(vals-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(vals)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    z=(x-med)/sc; z[~valid]=0; return z
def make_patches(grid,sel,valid):
    gp=[]; sf=[]; coords=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            v=valid[y:y+PATCH,x:x+PATCH]
            if v.mean()<.98 or not np.all(v[HIDDEN]) or not np.all(v[RING]):continue
            gp.append(grid[y:y+PATCH,x:x+PATCH]); sf.append(sel[y:y+PATCH,x:x+PATCH]); coords.append((y,x))
    return np.asarray(gp),np.asarray(sf),coords
def selection_patch_features(s):
    out=[]
    for m in [HIDDEN,RING,FULL]:
        a=s[:,m,:]; out += [a.mean(1),a.std(1)]
    return np.concatenate(out,1)
def balanced_weights(y):
    y=np.asarray(y,int); n=len(y); p=max(1,int(y.sum())); q=max(1,n-p); return np.where(y,n/(2*p),n/(2*q))
def auc(y,s): return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def paired(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'mean_difference':float(np.mean(d)),'median_difference':float(np.median(d))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null12'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); centers=json.loads(Path(args.centers).read_text())
    regs=centers.get('regions',[])[:N_FIELDS]
    if centers.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS:raise RuntimeError('REAL_DR11 fixed centers required')
    fields={}; provenance=[]; used=set()
    for j,r in enumerate(regs):
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg']); brick,bq,near=choose_brick_from_center(ra,dec)
        if brick in used:raise RuntimeError(f'duplicate brick {brick}')
        used.add(brick); print(f'[selection-null12] {j+1}/{N_FIELDS} {name} -> {brick}',flush=True)
        urls=product_urls(brick); depth,hdr,dp=read_image(urls['depth_r'],'DEPTH_R'); mask,_,mp=read_image(urls['maskbits'],'MASKBITS'); nexp,_,npv=read_image(urls['nexp_r']); psf,_,pp=read_image(urls['psfsize_r'])
        if not(depth.shape==mask.shape==nexp.shape==psf.shape):raise RuntimeError(f'shape mismatch {brick}')
        src,sq=source_catalog(brick); counts,ninside=count_grid(src,WCS(hdr),depth.shape)
        feat=[]; fn=[]
        for arr,names in [(continuous_features(depth,True,True),['log_depth_mean','log_depth_std','depth_positive_frac']),(continuous_features(nexp,False,True),['nexp_mean','nexp_std','nexp_positive_frac']),(continuous_features(psf,False,True),['psfsize_mean','psfsize_std','psfsize_positive_frac'])]:feat+=arr; fn+=names
        feat+=mask_features(mask); fn+=['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(BIT_COUNT)]
        sel=np.stack(feat,-1); primary=sel[:,:,fn.index('primary_frac')]; valid=(primary>=.9375)&(sel[:,:,fn.index('depth_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_positive_frac')]>=.9375)
        corrected=counts/np.clip(primary,.25,1); norm=robust_norm(corrected,valid,True); patches,sp,coords=make_patches(norm,sel,valid)
        if len(patches)<8:raise RuntimeError(f'too few valid patches {brick}: {len(patches)}')
        fields[name]={'brick':brick,'counts':corrected,'norm':norm,'sel':sel,'valid':valid,'patches':patches,'sp':sp,'coords':coords,'fn':fn}
        provenance.append({'field':name,'brick':brick,'center_ra_deg':ra,'center_dec_deg':dec,'nearest_primary_source_deg':near,'brick_choice_query':bq,'source_query':sq,'source_rows':int(len(src)),'sources_inside_wcs':ninside,'valid_cell_fraction':float(valid.mean()),'n_patches':len(patches),'products':{'depth_r':dp,'maskbits':mp,'nexp_r':npv,'psfsize_r':pp}})
    names=list(fields); rows=[]
    for held in names:
        train=[f for f in names if f!=held]; Ptr=np.concatenate([fields[f]['patches'] for f in train]); Pte=fields[held]['patches']; Str=np.concatenate([selection_patch_features(fields[f]['sp']) for f in train]); Ste=selection_patch_features(fields[held]['sp'])
        rtr=Ptr[:,RING].mean(1)[:,None]; rte=Pte[:,RING].mean(1)[:,None]; hmtr=Ptr[:,HIDDEN].mean(1); hmte=Pte[:,HIDDEN].mean(1); hxtr=Ptr[:,HIDDEN].max(1); hxte=Pte[:,HIDDEN].max(1); q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        labels={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for motif,(ytr,yte) in labels.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(rtr,ytr.astype(int)); obs=auc(yte,lr.predict_proba(rte)[:,1])
            clf=HistGradientBoostingClassifier(max_iter=80,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=20,l2_regularization=1,random_state=31).fit(Str,ytr.astype(int),sample_weight=balanced_weights(ytr)); selauc=auc(yte,clf.predict_proba(Ste)[:,1])
            rows.append({'field':held,'motif':motif,'observed_ring_auc':obs,'selection_only_auc':selauc})
        # cell-level selection expectation and residual locality
        X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]); Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(np.mean(fields[f]['counts'][fields[f]['valid']])+1e-6) for f in train])
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=50,learning_rate=.07,max_leaf_nodes=12,min_samples_leaf=60,l2_regularization=1,random_state=43).fit(X,np.clip(Y,1e-4,None))
        d=fields[held]; v=d['valid']; pred=np.full((GRID,GRID),np.nan); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20); true=d['counts'][v]/(np.mean(d['counts'][v])+1e-6); r2=float(r2_score(true,pred[v])); sr=float(spearmanr(true,pred[v]).statistic)
        lam=pred*np.mean(d['counts'][v]); resid=np.zeros((GRID,GRID)); resid[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1); resid=robust_norm(resid,v,False)
        rr=[]; hh=[]; orr=[]; oh=[]
        for (y,x),p in zip(d['coords'],d['patches']):
            rp=resid[y:y+PATCH,x:x+PATCH]; rr.append(rp[RING].mean()); hh.append(rp[HIDDEN].mean()); orr.append(p[RING].mean()); oh.append(p[HIDDEN].mean())
        resrho=float(spearmanr(rr,hh).statistic); nullrho=float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic); obsrho=float(spearmanr(orr,oh).statistic)
        for rec in rows:
            if rec['field']==held:rec.update({'selection_cell_r2':r2,'selection_cell_spearman':sr,'observed_ring_hidden_spearman':obsrho,'residual_ring_hidden_spearman':resrho,'residual_shift_spearman':nullrho})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False); sums=[]
    for motif,g in df.groupby('motif'):
        sums.append({'motif':motif,'observed_ring_median_auc':float(np.nanmedian(g.observed_ring_auc)),'selection_only_median_auc':float(np.nanmedian(g.selection_only_auc)),'observed_minus_selection':paired(g.observed_ring_auc-g.selection_only_auc)})
    one=df.drop_duplicates('field'); cont={'selection_cell_r2_median':float(np.nanmedian(one.selection_cell_r2)),'selection_cell_spearman_median':float(np.nanmedian(one.selection_cell_spearman)),'observed_ring_hidden_spearman_median':float(np.nanmedian(one.observed_ring_hidden_spearman)),'residual_ring_hidden_spearman_median':float(np.nanmedian(one.residual_ring_hidden_spearman)),'residual_shift_spearman_median':float(np.nanmedian(one.residual_shift_spearman)),'residual_minus_shift':paired(one.residual_ring_hidden_spearman-one.residual_shift_spearman)}
    prov={'status':'REAL_DR11_PIXEL_SELECTION_NULL','n_bricks':N_FIELDS,'brick_selection':'fixed sky center -> nearest BRICK_PRIMARY source in small cone; no density ranking','selection_products':['depth-r','maskbits','nexp-r','psfsize-r'],'regions':provenance,'total_source_rows':int(sum(x['source_rows'] for x in provenance))}; (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','validation':'12 independent DR11 brick-primary regions; whole-brick LOFO','motifs':sums,'continuous_locality':cont}; (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
