#!/usr/bin/env python3
"""REAL_DR11 tracer-split locality test using official Tractor FITS.

Subsets are selected with Tractor morphology, but models see positions only.
Tests whether extended-source locality survives separately from PSF-like sources,
and whether PSF boundary density predicts extended hidden density (cross-tracer null).
"""
from __future__ import annotations
import argparse, hashlib, io, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from astropy.io import fits
from dl import queryClient as qc
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TABLE='ls_dr11.tractor_s'; N_FIELDS=12; GRID=64; PATCH=16; STRIDE=8
TRACTOR_BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor'
HIDDEN=np.zeros((PATCH,PATCH),bool); HIDDEN[4:12,4:12]=True
RING=np.zeros((PATCH,PATCH),bool); RING[3,3:13]=True; RING[12,3:13]=True; RING[4:12,3]=True; RING[4:12,12]=True
EXT_TYPES={'REX','EXP','DEV','SER'}

def q(sql,attempts=4):
    last=None
    for i in range(attempts):
        try:
            out=qc.query(sql=sql,fmt='csv',async_=False)
            if isinstance(out,bytes):out=out.decode()
            if not isinstance(out,str):raise RuntimeError(type(out))
            return out
        except Exception as e:
            last=e
            if i+1<attempts:time.sleep(2*(i+1))
    raise RuntimeError(f'query failed: {last}')
def qdf(sql):
    from io import StringIO
    return pd.read_csv(StringIO(q(sql)))
def sep2(ra,dec,ra0,dec0):
    dra=((np.asarray(ra,float)-ra0+180)%360)-180
    return (dra*np.cos(np.deg2rad(dec0)))**2+(np.asarray(dec,float)-dec0)**2
def choose_brick(ra,dec):
    for rad in [.03,.06,.12]:
        sql=f"SELECT brickname,ra,dec FROM {TABLE} WHERE brick_primary=1 AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{rad:.8f})"
        d=qdf(sql)
        if len(d):
            i=int(np.argmin(sep2(d.ra,d.dec,ra,dec)))
            return str(d.iloc[i].brickname).strip(),sql
    raise RuntimeError(f'no brick near {ra},{dec}')
def get_tractor(brick):
    url=f'{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits'
    r=requests.get(url,timeout=180); r.raise_for_status(); b=r.content
    if not b.startswith(b'SIMPLE'):raise RuntimeError(f'not FITS {url}: {b[:40]!r}')
    with fits.open(io.BytesIO(b),memmap=False) as H:
        t=H[1].data
        cols={c.lower():c for c in t.names}
        need=['brick_primary','type','ra','dec','bx','by','ref_cat']
        miss=[c for c in need if c not in cols]
        if miss:raise RuntimeError(f'missing columns {miss}')
        d=pd.DataFrame({c:np.asarray(t[cols[c]]) for c in need})
    for c in ['type','ref_cat']:
        d[c]=d[c].map(lambda x:x.decode().strip() if isinstance(x,(bytes,bytearray)) else str(x).strip())
    d=d[d.brick_primary.astype(bool)].copy()
    return d,{'url':url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'rows_primary':int(len(d)),'type_counts':{str(k):int(v) for k,v in d.type.value_counts().items()}}
def grid_from(df):
    x=pd.to_numeric(df.bx,errors='coerce').to_numpy(float); y=pd.to_numeric(df.by,errors='coerce').to_numpy(float)
    keep=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<3600)&(y>=0)&(y<3600)
    ix=np.floor(x[keep]/3600*GRID).astype(int); iy=np.floor(y[keep]/3600*GRID).astype(int)
    g=np.zeros((GRID,GRID),float); np.add.at(g,(iy,ix),1); return g
def norm(g):
    x=np.log1p(g); m=np.median(x); s=np.median(np.abs(x-m))*1.4826
    if not np.isfinite(s) or s<1e-6:s=np.std(x)
    return (x-m)/(s if s>1e-6 else 1)
def scores(g):
    out=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            p=g[y:y+PATCH,x:x+PATCH]; out.append((p[RING].mean(),p[HIDDEN].mean(),p[HIDDEN].max()))
    return np.asarray(out)
def auc(y,s):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def fit_score(xtr,ytr,xte,yte):
    m=LogisticRegression(max_iter=1000,class_weight='balanced').fit(np.asarray(xtr)[:,None],np.asarray(ytr,int))
    return auc(yte,m.predict_proba(np.asarray(xte)[:,None])[:,1])
def paired(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum())
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d))}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/tracer_split12'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    centers=json.loads(Path(args.centers).read_text()); regs=centers.get('regions',[])[:N_FIELDS]
    if centers.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS:raise RuntimeError('fixed REAL_DR11 centers required')
    data={}; prov=[]; used=set(); rng=np.random.default_rng(20260829)
    for j,r in enumerate(regs):
        brick,bq=choose_brick(float(r['center_ra_deg']),float(r['center_dec_deg']))
        if brick in used:raise RuntimeError(f'duplicate {brick}')
        used.add(brick); print(f'[tracer-split] {j+1}/{N_FIELDS} {brick}',flush=True); d,p=get_tractor(brick)
        psf=d[d.type.eq('PSF')].copy(); ext=d[d.type.isin(EXT_TYPES)].copy(); n=min(len(psf),len(ext))
        if n<1500:raise RuntimeError(f'too few equalized tracers {brick}: PSF={len(psf)} EXT={len(ext)}')
        psf_eq=psf.iloc[rng.choice(len(psf),n,replace=False)]; ext_eq=ext.iloc[rng.choice(len(ext),n,replace=False)]
        subsets={'all':d,'psf':psf,'extended':ext,'psf_equal':psf_eq,'extended_equal':ext_eq}
        data[r['name']]={k:scores(norm(grid_from(v))) for k,v in subsets.items()}
        prov.append({'field':r['name'],'brick':brick,'brick_choice_query':bq,'tractor':p,'n_psf':int(len(psf)),'n_extended':int(len(ext)),'equalized_n':int(n),'n_gaia_ref':int(d.ref_cat.eq('G3').sum())})
    fields=list(data); rows=[]
    for held in fields:
        train=[f for f in fields if f!=held]
        for tracer in ['all','psf','extended','psf_equal','extended_equal']:
            tr=np.concatenate([data[f][tracer] for f in train]); te=data[held][tracer]; q25,q75=np.quantile(tr[:,1],[.25,.75]); qpk=np.quantile(tr[:,2],.8)
            labs={'void':(tr[:,1]<=q25,te[:,1]<=q25),'overdense':(tr[:,1]>=q75,te[:,1]>=q75),'peak':(tr[:,2]>=qpk,te[:,2]>=qpk)}
            for motif,(ytr,yte) in labs.items():
                real=fit_score(tr[:,0],ytr,te[:,0],yte)
                # matched shift preserves the field's ring distribution but destroys local pairing
                trn=[]; ytn=[]
                for f in train:
                    a=data[f][tracer]; q1,q3=np.quantile(tr[:,1],[.25,.75]); qp=np.quantile(tr[:,2],.8)
                    if motif=='void':yy=a[:,1]<=q1
                    elif motif=='overdense':yy=a[:,1]>=q3
                    else:yy=a[:,2]>=qp
                    trn.extend(np.roll(a[:,0],max(1,len(a)//3))); ytn.extend(yy)
                null=fit_score(trn,ytn,np.roll(te[:,0],max(1,len(te)//3)),yte)
                rows.append({'field':held,'tracer':tracer,'motif':motif,'self_auc':real,'matched_shift_auc':null})
        # cross-tracer: PSF ring -> extended hidden labels, and reverse
        for src,tgt,label in [('psf_equal','extended_equal','psf_to_extended'),('extended_equal','psf_equal','extended_to_psf')]:
            src_tr=np.concatenate([data[f][src] for f in train]); tgt_tr=np.concatenate([data[f][tgt] for f in train]); src_te=data[held][src]; tgt_te=data[held][tgt]
            q25,q75=np.quantile(tgt_tr[:,1],[.25,.75]); qpk=np.quantile(tgt_tr[:,2],.8)
            for motif,ytr,yte in [('void',tgt_tr[:,1]<=q25,tgt_te[:,1]<=q25),('overdense',tgt_tr[:,1]>=q75,tgt_te[:,1]>=q75),('peak',tgt_tr[:,2]>=qpk,tgt_te[:,2]>=qpk)]:
                rows.append({'field':held,'tracer':label,'motif':motif,'self_auc':fit_score(src_tr[:,0],ytr,src_te[:,0],yte),'matched_shift_auc':float('nan')})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False); sums=[]
    for (tracer,motif),g in df.groupby(['tracer','motif']):
        rec={'tracer':tracer,'motif':motif,'median_auc':float(np.nanmedian(g.self_auc))}
        if np.isfinite(g.matched_shift_auc).any():rec.update({'matched_shift_median':float(np.nanmedian(g.matched_shift_auc)),'self_minus_shift':paired(g.self_auc-g.matched_shift_auc)})
        sums.append(rec)
    # Compare equalized extended self vs PSF->extended cross-prediction field by field.
    comp=[]
    for motif in ['void','overdense','peak']:
        a=df[(df.tracer=='extended_equal')&(df.motif==motif)].set_index('field').self_auc
        b=df[(df.tracer=='psf_to_extended')&(df.motif==motif)].set_index('field').self_auc
        comp.append({'motif':motif,'extended_self_median':float(np.nanmedian(a)),'psf_to_extended_median':float(np.nanmedian(b)),'extended_self_minus_psf_cross':paired(a-b)})
    summary={'status':'REAL_DR11_TRACTOR_TRACER_SPLIT','validation':'12 independent official Tractor bricks; whole-brick LOFO','subsets':'PSF vs REX/EXP/DEV/SER; equalized per brick; downstream models use positions only','summary':sums,'cross_tracer_comparisons':comp}; (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'provenance.json').write_text(json.dumps({'status':'REAL_DR11_TRACTOR_TRACER_SPLIT','regions':prov},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
