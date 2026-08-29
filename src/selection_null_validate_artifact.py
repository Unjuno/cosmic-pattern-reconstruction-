#!/usr/bin/env python3
"""Pixel-level selection null using the frozen 48-field REAL_DR11 artifact.

No Data Lab source re-query is performed. Source RA/Dec are read from the
provenance-verified expanded48 artifact; only official DR11 coadd maps are
fetched here. First 12 independent fields are used as a fast gate.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

GRID=64; PATCH=16; STRIDE=8; N_FIELDS=12; BIT_COUNT=20
BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd'
HIDDEN=np.zeros((PATCH,PATCH),bool); HIDDEN[4:12,4:12]=True
RING=np.zeros((PATCH,PATCH),bool); RING[3,3:13]=True; RING[12,3:13]=True; RING[4:12,3]=True; RING[4:12,12]=True
FULL=np.ones((PATCH,PATCH),bool)

def sha256(b): return hashlib.sha256(b).hexdigest()
def get(url,attempts=3):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,timeout=90); r.raise_for_status(); b=r.content
            if not b.startswith(b'SIMPLE'): raise RuntimeError(f'not FITS {b[:30]!r}')
            return b
        except Exception as e:
            last=e; time.sleep(i+1)
    raise RuntimeError(f'download failed {url}: {last}')
def read_image(url,wanted=None):
    b=get(url)
    with fits.open(io.BytesIO(b),memmap=False) as H:
        hs=[h for h in H if h.data is not None and getattr(h.data,'ndim',0)==2]
        p=next((h for h in hs if wanted and (str(h.name).upper()==wanted.upper() or str(h.header.get('EXTNAME','')).upper()==wanted.upper())),hs[0])
        a=np.asarray(p.data).copy(); hdr=p.header.copy()
    return a,hdr,{'url':url,'sha256':sha256(b),'bytes':len(b),'shape':list(a.shape),'extname':hdr.get('EXTNAME')}
def urls(brick):
    root=f'{BASE}/{brick[:3]}/{brick}'
    return {k:f'{root}/legacysurvey-{brick}-{s}.fits.fz' for k,s in {'depth':'depth-r','mask':'maskbits','nexp':'nexp-r','psf':'psfsize-r'}.items()}
def candidate_names(ra,dec):
    ir=int(round((ra%360)*10)); idec=int(round(abs(dec)*10)); sign='p' if dec>=0 else 'm'
    offs=[]
    for dr in range(-4,5):
        for dd in range(-4,5): offs.append((dr*dr+dd*dd,dr,dd))
    for _,dr,dd in sorted(offs):
        j=idec+dd
        if j<0: continue
        yield f'{(ir+dr)%3600:04d}{sign}{j:03d}'
def resolve_brick(ra,dec):
    failures=[]
    for brick in candidate_names(ra,dec):
        u=urls(brick)['depth']
        try:
            a,h,p=read_image(u,'DEPTH_R'); w=WCS(h); x,y=w.world_to_pixel_values(ra,dec)
            if np.isfinite(x) and np.isfinite(y) and 0<=x<a.shape[1] and 0<=y<a.shape[0]:
                return brick,a,h,p
        except Exception as e:
            failures.append(f'{brick}:{type(e).__name__}')
    raise RuntimeError(f'no coadd brick containing center {ra},{dec}; attempts={failures[:8]}...')
def verify_source(meta):
    p=Path(meta['file']); gz=p.read_bytes(); raw=gzip.decompress(gz)
    if sha256(gz)!=meta['stored_gzip_sha256'] or sha256(raw)!=meta['canonical_csv_sha256']: raise RuntimeError(f'hash mismatch {p}')
    d=pd.read_csv(p)
    if list(d.columns)!=['ra','dec'] or len(d)!=int(meta['rows']): raise RuntimeError(f'bad frozen source {p}')
    return d
def sample(a):
    ny,nx=a.shape; o=np.array([.125,.375,.625,.875])
    yy=np.clip(((np.arange(GRID)[:,None]+o)/GRID*ny).astype(int),0,ny-1); xx=np.clip(((np.arange(GRID)[:,None]+o)/GRID*nx).astype(int),0,nx-1)
    return a[yy[:,None,:,None],xx[None,:,None,:]]
def continuous(a,log=False,pos=False):
    s=sample(np.asarray(a,float)); good=np.isfinite(s)&((s>0) if pos else True)
    if log:s=np.log1p(np.clip(s,0,None))
    v=np.where(good,s,np.nan)
    return [np.nan_to_num(np.nanmean(v,(2,3))),np.nan_to_num(np.nanstd(v,(2,3))),good.mean((2,3))]
def maskfeat(a):
    s=sample(np.asarray(a,np.int64)); out=[((s&1)==0).mean((2,3)),(s==0).mean((2,3))]
    out += [((s&(1<<b))!=0).mean((2,3)) for b in range(BIT_COUNT)]
    return out
def count_grid(d,w,shape):
    x,y=w.world_to_pixel_values(d.ra.to_numpy(float),d.dec.to_numpy(float)); ny,nx=shape
    k=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny)
    bx=np.floor(x[k]/nx*GRID).astype(int); by=np.floor(y[k]/ny*GRID).astype(int); g=np.zeros((GRID,GRID)); np.add.at(g,(by,bx),1)
    return g,int(k.sum())
def norm(g,v,log=True):
    a=np.log1p(np.clip(g,0,None)) if log else g.copy(); q=a[v]; m=np.median(q); s=np.median(np.abs(q-m))*1.4826
    if not np.isfinite(s) or s<1e-6:s=np.std(q)
    if not np.isfinite(s) or s<1e-6:s=1
    z=(a-m)/s; z[~v]=0; return z
def patches(g,s,v):
    gp=[]; sp=[]; xy=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            vv=v[y:y+PATCH,x:x+PATCH]
            if vv.mean()<.98 or not np.all(vv[HIDDEN]) or not np.all(vv[RING]): continue
            gp.append(g[y:y+PATCH,x:x+PATCH]); sp.append(s[y:y+PATCH,x:x+PATCH]); xy.append((y,x))
    return np.asarray(gp),np.asarray(sp),xy
def selfeat(s):
    out=[]
    for m in [HIDDEN,RING,FULL]:
        a=s[:,m,:]; out += [a.mean(1),a.std(1)]
    return np.concatenate(out,1)
def auc(y,s): return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def weights(y):
    y=np.asarray(y,int); n=len(y); p=max(1,y.sum()); q=max(1,n-y.sum()); return np.where(y,n/(2*p),n/(2*q))
def paired(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'mean_difference':float(d.mean()),'median_difference':float(np.median(d))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_null12_artifact'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); P=json.loads(Path(a.centers).read_text()); regs=P['regions'][:N_FIELDS]
    fields={}; pv=[]
    for j,r in enumerate(regs):
        ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg']); brick,depth,hdr,dp=resolve_brick(ra,dec); u=urls(brick)
        print(f'[selection-artifact] {j+1}/{N_FIELDS} {r["name"]} -> {brick}',flush=True)
        mask,_,mp=read_image(u['mask'],'MASKBITS'); nexp,_,npv=read_image(u['nexp']); psf,_,pp=read_image(u['psf']); src=verify_source(r)
        counts,nin=count_grid(src,WCS(hdr),depth.shape); feat=[]; fn=[]
        for arr,names in [(continuous(depth,True,True),['log_depth_mean','log_depth_std','depth_positive_frac']),(continuous(nexp,False,True),['nexp_mean','nexp_std','nexp_positive_frac']),(continuous(psf,False,True),['psfsize_mean','psfsize_std','psfsize_positive_frac'])]: feat+=arr; fn+=names
        feat+=maskfeat(mask); fn+=['primary_frac','clean_frac']+[f'maskbit_{b}_frac' for b in range(BIT_COUNT)]
        sel=np.stack(feat,-1); primary=sel[:,:,fn.index('primary_frac')]; valid=(primary>=.9375)&(sel[:,:,fn.index('depth_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_positive_frac')]>=.9375)
        corr=counts/np.clip(primary,.25,1); z=norm(corr,valid,True); gp,sp,xy=patches(z,sel,valid)
        if len(gp)<8: raise RuntimeError(f'too few patches {brick}: {len(gp)}')
        fields[r['name']]={'counts':corr,'z':z,'sel':sel,'valid':valid,'gp':gp,'sp':sp,'xy':xy}
        pv.append({'field':r['name'],'brick':brick,'frozen_source_file':r['file'],'frozen_source_gzip_sha256':r['stored_gzip_sha256'],'source_rows_box':int(len(src)),'sources_inside_brick_wcs':nin,'valid_cell_fraction':float(valid.mean()),'n_patches':int(len(gp)),'products':{'depth':dp,'mask':mp,'nexp':npv,'psf':pp}})
    names=list(fields); rows=[]
    for held in names:
        train=[f for f in names if f!=held]; tr=np.concatenate([fields[f]['gp'] for f in train]); te=fields[held]['gp']; st=np.concatenate([selfeat(fields[f]['sp']) for f in train]); se=selfeat(fields[held]['sp'])
        ringtr=tr[:,RING].mean(1)[:,None]; ringte=te[:,RING].mean(1)[:,None]; hmtr=tr[:,HIDDEN].mean(1); hmte=te[:,HIDDEN].mean(1); hxtr=tr[:,HIDDEN].max(1); hxte=te[:,HIDDEN].max(1); q25,q75=np.quantile(hmtr,[.25,.75]); qpk=np.quantile(hxtr,.8)
        labs={'void':(hmtr<=q25,hmte<=q25),'overdense':(hmtr>=q75,hmte>=q75),'peak':(hxtr>=qpk,hxte>=qpk)}
        for mot,(yt,ye) in labs.items():
            lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(ringtr,yt.astype(int)); oa=auc(ye,lr.predict_proba(ringte)[:,1])
            clf=HistGradientBoostingClassifier(max_iter=80,learning_rate=.06,max_leaf_nodes=12,min_samples_leaf=20,l2_regularization=1,random_state=31).fit(st,yt.astype(int),sample_weight=weights(yt)); sa=auc(ye,clf.predict_proba(se)[:,1]); rows.append({'field':held,'motif':mot,'observed_ring_auc':oa,'selection_only_auc':sa})
        X=np.concatenate([fields[f]['sel'][fields[f]['valid']] for f in train]); Y=np.concatenate([fields[f]['counts'][fields[f]['valid']]/(fields[f]['counts'][fields[f]['valid']].mean()+1e-6) for f in train])
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=50,learning_rate=.07,max_leaf_nodes=12,min_samples_leaf=60,l2_regularization=1,random_state=43).fit(X,np.clip(Y,1e-4,None)); d=fields[held]; v=d['valid']; pred=np.full((GRID,GRID),np.nan); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20); true=d['counts'][v]/(d['counts'][v].mean()+1e-6)
        r2=float(r2_score(true,pred[v])); sr=float(spearmanr(true,pred[v]).statistic); lam=pred*d['counts'][v].mean(); res=np.zeros((GRID,GRID)); res[v]=(d['counts'][v]-lam[v])/np.sqrt(lam[v]+1); res=norm(res,v,False)
        rr=[]; hh=[]; orr=[]; oh=[]
        for (y,x),p in zip(d['xy'],d['gp']):
            rp=res[y:y+PATCH,x:x+PATCH]; rr.append(rp[RING].mean()); hh.append(rp[HIDDEN].mean()); orr.append(p[RING].mean()); oh.append(p[HIDDEN].mean())
        resrho=float(spearmanr(rr,hh).statistic); nullrho=float(spearmanr(rr,np.roll(hh,max(1,len(hh)//3))).statistic); obsrho=float(spearmanr(orr,oh).statistic)
        for rec in rows:
            if rec['field']==held: rec.update({'selection_cell_r2':r2,'selection_cell_spearman':sr,'observed_ring_hidden_spearman':obsrho,'residual_ring_hidden_spearman':resrho,'residual_shift_spearman':nullrho})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False); ms=[]
    for mot,g in df.groupby('motif'): ms.append({'motif':mot,'observed_ring_median_auc':float(g.observed_ring_auc.median()),'selection_only_median_auc':float(g.selection_only_auc.median()),'observed_minus_selection':paired(g.observed_ring_auc-g.selection_only_auc)})
    one=df.drop_duplicates('field'); cont={'selection_cell_r2_median':float(one.selection_cell_r2.median()),'selection_cell_spearman_median':float(one.selection_cell_spearman.median()),'observed_ring_hidden_spearman_median':float(one.observed_ring_hidden_spearman.median()),'residual_ring_hidden_spearman_median':float(one.residual_ring_hidden_spearman.median()),'residual_shift_spearman_median':float(one.residual_shift_spearman.median()),'residual_minus_shift':paired(one.residual_ring_hidden_spearman-one.residual_shift_spearman)}
    summary={'status':'REAL_DR11_PIXEL_SELECTION_NULL','source':'frozen expanded48 GitHub Actions artifact 9563504957','n_fields':N_FIELDS,'motifs':ms,'continuous_locality':cont}; (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':pv},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
