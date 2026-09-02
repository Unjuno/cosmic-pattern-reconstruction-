#!/usr/bin/env python3
"""REAL_DR11 bright/faint extended-source cross-population locality test.

Catalog properties are used only to define tracer subsets. Downstream predictors
use positions only. A random half/half split of the same extended population is
the positive-control benchmark for a shared underlying angular field.
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

TABLE='ls_dr11.tractor_s'; TRACTOR_BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor'
N_FIELDS=12; GRID=64; PATCH=16; STRIDE=8; MIN_EXT=1200
EXT_TYPES={'REX','EXP','DEV','SER'}
HIDDEN=np.zeros((PATCH,PATCH),bool); HIDDEN[4:12,4:12]=True
RING=np.zeros((PATCH,PATCH),bool); RING[3,3:13]=True; RING[12,3:13]=True; RING[4:12,3]=True; RING[4:12,12]=True

def native(a):
    a=np.asarray(a)
    if a.dtype.kind in 'iufcb' and a.dtype.byteorder not in ('=','|'): return a.astype(a.dtype.newbyteorder('='),copy=True)
    return a.copy()

def q(sql,attempts=4):
    last=None
    for i in range(attempts):
        try:
            x=qc.query(sql=sql,fmt='csv',async_=False)
            if isinstance(x,bytes):x=x.decode()
            if not isinstance(x,str):raise RuntimeError(type(x))
            return x
        except Exception as e:
            last=e
            if i+1<attempts:time.sleep(2*(i+1))
    raise RuntimeError(last)
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
            i=int(np.argmin(sep2(d.ra,d.dec,ra,dec))); return str(d.iloc[i].brickname).strip(),sql
    raise RuntimeError(f'no brick near {ra},{dec}')
def get_extended(brick):
    url=f'{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits'; last=None
    for i in range(4):
        try:
            r=requests.get(url,timeout=180);r.raise_for_status();b=r.content
            if not b.startswith(b'SIMPLE'):raise RuntimeError(f'not FITS {b[:40]!r}')
            with fits.open(io.BytesIO(b),memmap=False) as H:
                t=H[1].data; cols={c.lower():c for c in t.names}
                need=['brick_primary','type','bx','by','flux_r','mw_transmission_r']
                miss=[c for c in need if c not in cols]
                if miss:raise RuntimeError(f'missing {miss}')
                v={c:native(t[cols[c]]) for c in need}
            d=pd.DataFrame(v); d['type']=d.type.map(lambda x:x.decode().strip() if isinstance(x,(bytes,bytearray)) else str(x).strip())
            d=d[d.brick_primary.astype(bool)&d.type.isin(EXT_TYPES)].copy()
            fr=pd.to_numeric(d.flux_r,errors='coerce').to_numpy(float); mw=pd.to_numeric(d.mw_transmission_r,errors='coerce').to_numpy(float)
            good=np.isfinite(fr)&np.isfinite(mw)&(fr>0)&(mw>0); d=d.loc[good].copy(); d['dered_flux_r']=fr[good]/mw[good]
            return d,{'url':url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'n_extended_positive_r':int(len(d))}
        except Exception as e:
            last=e
            if i+1<4:time.sleep(2*(i+1))
    raise RuntimeError(f'Tractor read failed {url}: {last}')
def grid(df):
    x=pd.to_numeric(df.bx,errors='coerce').to_numpy(float); y=pd.to_numeric(df.by,errors='coerce').to_numpy(float)
    keep=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<3600)&(y>=0)&(y<3600); ix=np.floor(x[keep]/3600*GRID).astype(int);iy=np.floor(y[keep]/3600*GRID).astype(int)
    g=np.zeros((GRID,GRID),float);np.add.at(g,(iy,ix),1);return g
def norm(g):
    x=np.log1p(g);m=np.median(x);s=np.median(np.abs(x-m))*1.4826
    if not np.isfinite(s) or s<1e-6:s=np.std(x)
    return (x-m)/(s if s>1e-6 else 1)
def scores(g):
    out=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            p=g[y:y+PATCH,x:x+PATCH];out.append((p[RING].mean(),p[HIDDEN].mean(),p[HIDDEN].max()))
    return np.asarray(out)
def auc(y,s):return float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def fit_score(xtr,ytr,xte,yte):
    if len(np.unique(ytr))<2:return float('nan')
    m=LogisticRegression(max_iter=1000,class_weight='balanced').fit(np.asarray(xtr)[:,None],np.asarray(ytr,int));return auc(yte,m.predict_proba(np.asarray(xte)[:,None])[:,1])
def paired(d):
    d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);pos=int((d>0).sum())
    try:w=float(wilcoxon(d,alternative='greater').pvalue)
    except Exception:w=float('nan')
    return {'n_fields':n,'positive_fields':pos,'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_one_sided_p':w,'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d))}
def eval_pair(data,fields,held,src,tgt,label):
    trf=[f for f in fields if f!=held]; sx=np.concatenate([data[f][src] for f in trf]); ty=np.concatenate([data[f][tgt] for f in trf]); ste=data[held][src];tte=data[held][tgt]
    q25,q75=np.quantile(ty[:,1],[.25,.75]);qpk=np.quantile(ty[:,2],.8);out=[]
    for motif,ytr,yte in [('void',ty[:,1]<=q25,tte[:,1]<=q25),('overdense',ty[:,1]>=q75,tte[:,1]>=q75),('peak',ty[:,2]>=qpk,tte[:,2]>=qpk)]:
        real=fit_score(sx[:,0],ytr,ste[:,0],yte); null=fit_score(np.roll(sx[:,0],max(1,len(sx)//3)),ytr,np.roll(ste[:,0],max(1,len(ste)//3)),yte)
        out.append({'field':held,'pair':label,'motif':motif,'auc':real,'matched_shift_auc':null})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json');ap.add_argument('--out',default='results/real_dr11/magnitude_split12');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path(args.centers).read_text());regs=c.get('regions',[])[:N_FIELDS]
    if c.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS:raise RuntimeError('fixed REAL_DR11 centers required')
    data={};prov=[];used=set()
    for j,r in enumerate(regs):
        brick,bq=choose_brick(float(r['center_ra_deg']),float(r['center_dec_deg']))
        if brick in used:raise RuntimeError(f'duplicate {brick}')
        used.add(brick);d,p=get_extended(brick)
        if len(d)<MIN_EXT:raise RuntimeError(f'too few extended positive-r sources {brick}: {len(d)}')
        order=np.argsort(d.dered_flux_r.to_numpy(float),kind='mergesort'); n=2*(len(order)//2);order=order[:n];half=n//2
        faint=d.iloc[order[:half]].copy();bright=d.iloc[order[half:]].copy()
        seed=int(hashlib.sha256(brick.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);perm=rng.permutation(n);a=d.iloc[order[perm[:half]]];b=d.iloc[order[perm[half:]]]
        subsets={'bright':bright,'faint':faint,'random_a':a,'random_b':b}
        data[r['name']]={k:scores(norm(grid(v))) for k,v in subsets.items()}
        prov.append({'field':r['name'],'brick':brick,'brick_choice_query':bq,'tractor':p,'n_used':n,'n_each':half,'median_dered_flux_r':float(np.median(d.iloc[order].dered_flux_r))})
        print(f'[magnitude-split] {j+1}/{N_FIELDS} {brick} each={half}',flush=True)
    fields=list(data);rows=[]
    for held in fields:
        for src,tgt,label in [('bright','bright','bright_self'),('faint','faint','faint_self'),('bright','faint','bright_to_faint'),('faint','bright','faint_to_bright'),('random_a','random_b','random_cross_ab'),('random_b','random_a','random_cross_ba')]:
            rows+=eval_pair(data,fields,held,src,tgt,label)
    df=pd.DataFrame(rows);df.to_csv(out/'field_metrics.csv',index=False);s=[]
    for (pair,motif),g in df.groupby(['pair','motif']):s.append({'pair':pair,'motif':motif,'median_auc':float(np.nanmedian(g.auc)),'matched_shift_median':float(np.nanmedian(g.matched_shift_auc)),'real_minus_shift':paired(g.auc-g.matched_shift_auc)})
    comp=[]
    for motif in ['void','overdense','peak']:
        bf=df[(df.pair=='bright_to_faint')&(df.motif==motif)].set_index('field').auc; rc=df[(df.pair=='random_cross_ab')&(df.motif==motif)].set_index('field').auc
        fb=df[(df.pair=='faint_to_bright')&(df.motif==motif)].set_index('field').auc; rb=df[(df.pair=='random_cross_ba')&(df.motif==motif)].set_index('field').auc
        comp.append({'motif':motif,'bright_to_faint_median':float(np.nanmedian(bf)),'random_ab_median':float(np.nanmedian(rc)),'random_minus_bright_faint':paired(rc-bf),'faint_to_bright_median':float(np.nanmedian(fb)),'random_ba_median':float(np.nanmedian(rb)),'random_minus_faint_bright':paired(rb-fb)})
    summary={'status':'REAL_DR11_EXTENDED_MAGNITUDE_SPLIT_12','validation':'12 fixed official Tractor bricks; whole-field LOFO','split':'REX/EXP/DEV/SER with positive r flux, exact equal bright/faint halves by within-brick dereddened r flux; random equal halves are positive control','summary':s,'comparisons':comp};(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':prov},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
