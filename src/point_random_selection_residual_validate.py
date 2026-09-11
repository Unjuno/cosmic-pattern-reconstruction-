#!/usr/bin/env python3
"""REAL_DR11 locality after official point-random selection residualization.

Independent replication of the accepted 36-field pixel-level selection-null,
but selection covariates are reconstructed only from official DR11 point-random
catalogs. The official files are randomly ordered, so a deterministic prefix is
an unbiased row sample according to the DR11 file documentation.

Primary question
----------------
After a cross-field nuisance model built from point-random MASKBITS, NOBS,
depth, PSF-size, GALDEPTH, EBV and PHOTSYS surfaces is subtracted from observed
source counts, does local visible-ring -> hidden-center coupling remain above a
matched-shift control?

No simulated cosmology is used. A PASS rejects only this sampled point-random
selection explanation; it does not establish a cosmological origin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import analyze_dr11 as adr
import locality_validate as loc
import bispectrum_phase_validate as bp

BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/randoms"
N_FILES = 20
GRID = 64
PREFIX_FRACTION = 0.02
KNN = 4
MAX_KTH_RADIUS_DEG = 0.08
N_FOLDS = 6
BLOCK = 2880
CARD = 80
HEADER_PROBE_BYTES = 262_144

# Exact 36 fields used by the accepted direct-coadd selection residual locality test.
LOCKED_FIELDS = [
    'f00_ra156_p05','f01_ra132_m05','f02_ra144_p15','f03_ra036_m35','f04_ra180_m05',
    'f06_ra072_m05','f07_ra216_m05','f08_ra336_m25','f09_ra216_p15','f10_ra240_m35',
    'f11_ra060_m35','f12_ra312_m05','f13_ra228_p05','f15_ra144_m25','f17_ra048_m15',
    'f19_ra132_p25','f20_ra120_p25','f22_ra036_p05','f23_ra012_m35','f24_ra012_m55',
    'f25_ra228_m35','f26_ra312_p05','f27_ra060_m05','f28_ra192_p05','f29_ra312_m25',
    'f30_ra000_m45','f31_ra132_p05','f32_ra024_m35','f33_ra348_p15','f34_ra000_m15',
    'f35_ra168_m25','f36_ra168_m45','f37_ra336_m55','f39_ra084_m55','f40_ra192_m35',
    'f41_ra348_m25',
]

RANDOM_COLUMNS = [
    'RA','DEC','PHOTSYS','MASKBITS',
    'NOBS_G','NOBS_R','NOBS_I','NOBS_Z',
    'PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_I','PSFDEPTH_Z',
    'GALDEPTH_G','GALDEPTH_R','GALDEPTH_I','GALDEPTH_Z',
    'PSFSIZE_G','PSFSIZE_R','PSFSIZE_I','PSFSIZE_Z','EBV',
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wrap_deg(x: np.ndarray | float) -> np.ndarray:
    return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0


def fetch_range(url: str, start: int, end: int, timeout: int = 360) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{end}',
        'Accept-Encoding': 'identity',
        'Connection': 'close',
        'User-Agent': 'cosmic-pattern-reconstruction-point-random-selection/1.0',
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = int(getattr(r, 'status', 0) or 0)
        cr = r.headers.get('Content-Range')
        cl = r.headers.get('Content-Length')
        expected = end - start + 1
        if status != 206 or not cr:
            raise RuntimeError(f'bounded Range not honored: status={status} content-range={cr} url={url}')
        if cl is not None and int(cl) != expected:
            raise RuntimeError(f'Content-Length mismatch {cl} != {expected}: {url}')
        data = r.read(expected + 1)
        if len(data) != expected:
            raise RuntimeError(f'Range byte mismatch {len(data)} != {expected}: {url}')
        meta = {
            'status': status, 'content_range': cr,
            'content_length': int(cl) if cl is not None else None,
            'etag': r.headers.get('ETag'), 'last_modified': r.headers.get('Last-Modified'),
        }
    return data, meta


def split_value_comment(s: str) -> str:
    s = s.rstrip()
    if not s:
        return ''
    if s.lstrip().startswith("'"):
        q = s.find("'")
        q = s.find("'", q + 1)
        while q >= 0 and q + 1 < len(s) and s[q + 1] == "'":
            q = s.find("'", q + 2)
        if q >= 0:
            return s[:q+1].strip()
    return s.split('/', 1)[0].strip()


def parse_value(v: str):
    v = v.strip()
    if not v:
        return None
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1].replace("''", "'").strip()
    if v == 'T': return True
    if v == 'F': return False
    try: return int(v)
    except ValueError: pass
    try: return float(v.replace('D','E'))
    except ValueError: return v


def parse_header(blob: bytes, offset: int) -> tuple[dict, int]:
    h = {}
    pos = offset
    end = None
    while pos + CARD <= len(blob):
        card = blob[pos:pos+CARD].decode('ascii', errors='strict')
        key = card[:8].strip()
        if key == 'END':
            end = pos + CARD
            break
        if card[8:10] == '= ' and key:
            h[key] = parse_value(split_value_comment(card[10:]))
        pos += CARD
    if end is None:
        raise RuntimeError(f'FITS END not found at offset {offset}')
    n = int(math.ceil((end-offset)/BLOCK)*BLOCK)
    return h, offset+n


def hdu_data_bytes(h: dict) -> int:
    if str(h.get('XTENSION','')).upper() == 'BINTABLE':
        return int(h['NAXIS1'])*int(h['NAXIS2']) + int(h.get('PCOUNT',0) or 0)
    naxis = int(h.get('NAXIS',0) or 0)
    if naxis <= 0: return 0
    n = 1
    for i in range(1,naxis+1): n *= int(h[f'NAXIS{i}'])
    return n*abs(int(h.get('BITPIX',8) or 8))//8


def inspect_fits(url: str) -> tuple[dict, dict]:
    b, http = fetch_range(url,0,HEADER_PROBE_BYTES-1,120)
    p, pend = parse_header(b,0)
    ext_off = pend + (int(math.ceil(hdu_data_bytes(p)/BLOCK)*BLOCK) if hdu_data_bytes(p) else 0)
    e, data_off = parse_header(b,ext_off)
    if str(e.get('XTENSION','')).upper() != 'BINTABLE':
        raise RuntimeError(f'first extension not BINTABLE: {url}')
    return e, {
        'url':url,'data_offset':int(data_off),'row_bytes':int(e['NAXIS1']),
        'rows':int(e['NAXIS2']),'tfields':int(e['TFIELDS']),
        'header_prefix_sha256':sha256(b[:data_off]),'header_http':http,
    }


def tform_width(tform: str) -> tuple[int,int,str]:
    m = re.match(r'^(\d*)([LXBIJKAEDCMPQ])(?:\([^)]*\))?$', str(tform).strip().upper())
    if not m: raise RuntimeError(f'unsupported TFORM {tform}')
    rep = int(m.group(1) or 1); code = m.group(2)
    unit = {'L':1,'X':0,'B':1,'I':2,'J':4,'K':8,'A':1,'E':4,'D':8,'C':8,'M':16,'P':8,'Q':16}[code]
    width = int(math.ceil(rep/8)) if code=='X' else rep*unit
    return width,rep,code


def field_specs(h: dict) -> tuple[dict[str,dict],int]:
    specs={};off=0
    for i in range(1,int(h['TFIELDS'])+1):
        name=str(h[f'TTYPE{i}']).strip().upper(); form=str(h[f'TFORM{i}']).strip().upper()
        width,rep,code=tform_width(form)
        specs[name]={'offset':off,'width':width,'repeat':rep,'code':code,'tform':form}
        off += width
    if off != int(h['NAXIS1']): raise RuntimeError(f'row layout mismatch {off}!={h["NAXIS1"]}')
    return specs,off


def extract(data: bytes,row_bytes:int,spec:dict)->np.ndarray:
    code=spec['code'];rep=int(spec['repeat']);n=len(data)//row_bytes;off=int(spec['offset'])
    if rep!=1 and code!='A': raise RuntimeError(f'non-scalar field unsupported: {spec}')
    if code=='A': dt=np.dtype(f'S{rep}')
    else:
        dt={'L':np.dtype('S1'),'B':np.dtype('u1'),'I':np.dtype('>i2'),'J':np.dtype('>i4'),
            'K':np.dtype('>i8'),'E':np.dtype('>f4'),'D':np.dtype('>f8')}.get(code)
        if dt is None: raise RuntimeError(f'extract code unsupported: {code}')
    return np.ndarray((n,),dtype=dt,buffer=data,offset=off,strides=(row_bytes,)).copy()


def load_regions(path:Path)->dict[str,dict]:
    p=json.loads(path.read_text());regs=p.get('regions',[])
    if p.get('status')!='REAL_DR11' or len(regs)!=48: raise RuntimeError('48-field REAL_DR11 provenance required')
    m={r['name']:r for r in regs}
    if any(n not in m for n in LOCKED_FIELDS): raise RuntimeError('locked 36 field set missing from provenance')
    return {n:m[n] for n in LOCKED_FIELDS}


def acquire_points(regions:dict[str,dict],fraction:float)->tuple[dict[str,pd.DataFrame],list[dict]]:
    store={n:[] for n in regions};prov=[];schema0=None
    for fi in range(N_FILES):
        url=f'{BASE}/randoms-1-{fi}.fits';hdr,meta=inspect_fits(url);specs,row_bytes=field_specs(hdr)
        missing=sorted(set(RANDOM_COLUMNS)-set(specs))
        if missing: raise RuntimeError(f'file {fi} missing {missing}')
        schema={k:specs[k]['tform'] for k in sorted(specs)}
        if schema0 is None: schema0=schema
        elif schema!=schema0: raise RuntimeError(f'schema mismatch file {fi}')
        ntot=int(meta['rows']);ntake=max(1,int(math.floor(ntot*fraction)))
        start=int(meta['data_offset']);end=start+ntake*row_bytes-1
        raw,http=fetch_range(url,start,end,480)
        cols={c:extract(raw,row_bytes,specs[c]) for c in RANDOM_COLUMNS}
        ra=cols['RA'].astype(float);dec=cols['DEC'].astype(float);hits=0
        for name,r in regions.items():
            ra0=float(r['center_ra_deg']);dec0=float(r['center_dec_deg']);half=float(r.get('box_width_deg',.5))/2
            m=(np.abs(wrap_deg(ra-ra0))<=half)&(dec>=dec0-half)&(dec<dec0+half)
            ii=np.flatnonzero(m)
            if not len(ii): continue
            hits += len(ii)
            d={}
            for c in RANDOM_COLUMNS:
                a=cols[c][ii]
                if c=='PHOTSYS': d[c]=[bytes(x).decode('ascii',errors='ignore').strip() for x in a]
                else: d[c]=a.astype(float if c not in ['MASKBITS','NOBS_G','NOBS_R','NOBS_I','NOBS_Z'] else np.int64)
            d['FILE_INDEX']=np.full(len(ii),fi,dtype=np.int16)
            d['PREFIX_ROW']=ii.astype(np.int64)
            store[name].append(pd.DataFrame(d))
        meta.update({'file_index':fi,'rows_sampled':ntake,'sample_fraction_effective':ntake/ntot,
                     'sample_data_bytes':len(raw),'sample_data_sha256':sha256(raw),'data_http':http,
                     'target_hits_locked36':int(hits)})
        prov.append(meta)
        print(f'[point-selection] random {fi+1:02d}/20 rows={ntake} bytes={len(raw)} hits36={hits}',flush=True)
        del raw,cols
    out={}
    for n,parts in store.items():
        out[n]=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=RANDOM_COLUMNS)
    return out,prov


def point_feature_matrix(d:pd.DataFrame)->tuple[np.ndarray,list[str]]:
    mb=d.MASKBITS.to_numpy(np.int64)
    arr=[];names=[]
    def add(name,v): names.append(name);arr.append(np.asarray(v,float))
    add('maskbits_zero',(mb==0).astype(float))
    add('primary_bit_clear',((mb & 1)==0).astype(float))
    add('bright_bit_clear',((mb & 2)==0).astype(float))
    allobs=np.ones(len(d),bool)
    for b in 'GRIZ': allobs &= d[f'NOBS_{b}'].to_numpy(float)>0
    add('all_griz_nobs_positive',allobs.astype(float))
    for prefix in ['NOBS','PSFDEPTH','GALDEPTH','PSFSIZE']:
        for b in 'GRIZ':
            v=np.clip(d[f'{prefix}_{b}'].to_numpy(float),0,None)
            add(f'log1p_{prefix.lower()}_{b.lower()}',np.log1p(v))
    add('ebv',np.clip(d.EBV.to_numpy(float),0,None))
    add('photsys_s',(d.PHOTSYS.astype(str).to_numpy()=='S').astype(float))
    return np.column_stack(arr),names


def selection_grid(d:pd.DataFrame,meta:dict)->tuple[np.ndarray,np.ndarray,dict,list[str]]:
    if len(d)<KNN: raise RuntimeError(f'{meta["name"]}: only {len(d)} point randoms')
    ra0=float(meta['center_ra_deg']);dec0=float(meta['center_dec_deg'])
    xy=np.column_stack([wrap_deg(d.RA.to_numpy(float)-ra0),d.DEC.to_numpy(float)-dec0])
    feat,names=point_feature_matrix(d)
    axis=-.25 + .5*(np.arange(GRID)+.5)/GRID
    yy,xx=np.meshgrid(axis,axis,indexing='ij');q=np.column_stack([xx.ravel(),yy.ravel()])
    tree=cKDTree(xy);dist,idx=tree.query(q,k=KNN)
    dist=np.asarray(dist,float);idx=np.asarray(idx,int)
    if KNN==1: dist=dist[:,None];idx=idx[:,None]
    w=1.0/np.maximum(dist,1e-4)**2;w/=w.sum(axis=1,keepdims=True)
    vals=np.einsum('nk,nkf->nf',w,feat[idx])
    kth=dist[:,-1]
    vals=np.column_stack([vals,kth])
    names=names+['knn4_radius_deg']
    valid=(kth<=MAX_KTH_RADIUS_DEG)&np.all(np.isfinite(vals),axis=1)
    return vals.reshape(GRID,GRID,-1),valid.reshape(GRID,GRID),{
        'n_point_randoms':int(len(d)),'valid_cell_fraction':float(valid.mean()),
        'median_knn4_radius_deg':float(np.median(kth)),'p95_knn4_radius_deg':float(np.quantile(kth,.95)),
    },names


def rho(a,b)->float:
    r=float(spearmanr(np.asarray(a,float),np.asarray(b,float)).statistic)
    return r if np.isfinite(r) else float('nan')


def fit_model(data:dict,train:list[str],seed:int):
    X=[];y=[]
    for n in train:
        d=data[n];v=d['valid'];counts=d['counts'];mu=float(np.mean(counts[v]))+1e-6
        X.append(d['sel'][v]);y.append(counts[v]/mu)
    X=np.concatenate(X);y=np.concatenate(y)
    if len(y)>120000:
        rng=np.random.default_rng(seed);ii=rng.choice(len(y),120000,replace=False);X,y=X[ii],y[ii]
    model=HistGradientBoostingRegressor(loss='poisson',max_iter=70,learning_rate=.06,max_leaf_nodes=15,
                                        min_samples_leaf=70,l2_regularization=1.5,random_state=seed)
    model.fit(X,np.clip(y,1e-4,None));return model


def fill_invalid(a,valid,seed):
    z=np.asarray(a,float).copy();vals=z[valid]
    if not len(vals): raise RuntimeError('no valid cells')
    rng=np.random.default_rng(seed);z[~valid]=rng.choice(vals,size=int((~valid).sum()),replace=True);return z


def robust_norm(a):
    x=np.asarray(a,float);med=np.median(x);s=np.median(np.abs(x-med))*1.4826
    if not np.isfinite(s) or s<1e-6: s=np.std(x)
    if not np.isfinite(s) or s<1e-6: s=1.0
    return (x-med)/s


def normalized_maps(counts,expected,valid,seed):
    raw=robust_norm(fill_invalid(np.log1p(counts),valid,seed))
    z=np.full_like(counts,np.nan,float);z[valid]=(counts[valid]-expected[valid])/np.sqrt(expected[valid]+1.0)
    residual=robust_norm(fill_invalid(z,valid,seed+1));return raw,residual


def context_local(grid):
    c=loc.contexts_from_grid(grid);return c[:,0],c[:,1]


def decision(primary:dict,n_valid:int)->str:
    if n_valid<30: return 'UNCERTAIN'
    med=float(primary.get('median',np.nan));sp=float(primary.get('sign_p_one_sided',np.nan));wp=float(primary.get('wilcoxon_p_one_sided',np.nan))
    if med>0 and sp<.05 and wp<.05: return 'PASS'
    if med<=0 or sp>=.10 or wp>=.10: return 'FAIL'
    return 'UNCERTAIN'


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--provenance',default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--fraction',type=float,default=PREFIX_FRACTION)
    ap.add_argument('--out',default='results/real_dr11/point_random_selection_residual36')
    args=ap.parse_args()
    if not (0<args.fraction<=.05): raise RuntimeError('fraction must be in (0,.05]')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    regions=load_regions(Path(args.provenance))
    points,remote_prov=acquire_points(regions,args.fraction)

    data={};qc=[];feature_names0=None
    for i,name in enumerate(LOCKED_FIELDS):
        m=regions[name];src=adr.verify_and_load(m);counts=adr.region_grid(src,m).astype(float)
        sel,valid,q,names=selection_grid(points[name],m)
        if feature_names0 is None: feature_names0=names
        elif names!=feature_names0: raise RuntimeError('feature name mismatch')
        q.update({'field':name,'source_rows':int(len(src))})
        qc.append(q)
        if q['valid_cell_fraction']>=.85:
            data[name]={'counts':counts,'sel':sel,'valid':valid}
        print(f'[point-selection] field {i+1:02d}/36 {name} randoms={len(points[name])} valid={valid.mean():.3f}',flush=True)
    pd.DataFrame(qc).to_csv(out/'point_random_qc.csv',index=False)
    valid_names=[n for n in LOCKED_FIELDS if n in data]

    fold_id={n:i%N_FOLDS for i,n in enumerate(valid_names)};rows=[];model_rows=[]
    for fold in range(N_FOLDS):
        test=[n for n in valid_names if fold_id[n]==fold];train=[n for n in valid_names if fold_id[n]!=fold]
        if not test or len(train)<20: continue
        model=fit_model(data,train,33000+fold)
        for name in test:
            d=data[name];v=d['valid'];counts=d['counts'];mu=float(np.mean(counts[v]))+1e-6
            pred=np.full((GRID,GRID),np.nan,float);pred[v]=np.clip(model.predict(d['sel'][v]),.03,20)
            true_rel=counts[v]/mu
            model_rows.append({'field':name,'fold':fold,'selection_r2':float(r2_score(true_rel,pred[v])),
                               'selection_spearman':rho(true_rel,pred[v])})
            expected=pred*mu;idx=LOCKED_FIELDS.index(name);raw,res=normalized_maps(counts,expected,v,720000+idx*4)
            for sample,g in [('raw',raw),('residual',res)]:
                hidden,visible=context_local(g);shift=loc.shifted_feature(visible)
                rows.append({'field':name,'fold':fold,'sample':sample,'rho':rho(visible,hidden),
                             'matched_shift_rho':rho(shift,hidden)})
        print(f'[point-selection] fold {fold+1}/{N_FOLDS} train={len(train)} test={len(test)}',flush=True)

    D=pd.DataFrame(rows);M=pd.DataFrame(model_rows);D.to_csv(out/'field_metrics.csv',index=False);M.to_csv(out/'selection_model_metrics.csv',index=False)
    comparisons={}
    for sample in ['raw','residual']:
        g=D[D['sample']==sample].copy();diff=(g.rho-g.matched_shift_rho).to_numpy(float)
        comparisons[sample]={
            'rho_median':float(np.nanmedian(g.rho)),'matched_shift_median':float(np.nanmedian(g.matched_shift_rho)),
            'real_minus_shift':bp.paired(diff),
        }
    primary=comparisons['residual']['real_minus_shift'];dec=decision(primary,len(valid_names))
    summary={
        'status':'REAL_DR11_POINT_RANDOM_SELECTION_RESIDUAL36','decision':dec,
        'n_locked_fields':36,'n_valid_fields':len(valid_names),'prefix_fraction_per_file':float(args.fraction),
        'files_sampled':20,'grid':64,'field_width_deg':.5,'knn':KNN,'max_knn_radius_deg':MAX_KTH_RADIUS_DEG,
        'point_selection_features':feature_names0,'cross_validation':'6-fold grouped by whole field',
        'selection_model':'HistGradientBoostingRegressor Poisson; same core hyperparameters as accepted direct-coadd selection residual model',
        'selection_model_r2_median':float(np.nanmedian(M.selection_r2)) if len(M) else float('nan'),
        'selection_model_spearman_median':float(np.nanmedian(M.selection_spearman)) if len(M) else float('nan'),
        'comparisons':comparisons,
        'H':'after official point-random selection residualization, local visible-to-hidden coupling remains above matched shift',
        'T':'same 36 fields as accepted direct-coadd selection test; first 2% of each of 20 randomized official point-random files; k=4 IDW selection surfaces; 6-fold whole-field CV',
        'D':'PASS if n_valid>=30, residual median real-minus-shift>0 and one-sided sign/Wilcoxon p<0.05; FAIL if n_valid>=30 and median<=0 or either p>=0.10; otherwise UNCERTAIN',
        'C':'sampled point-level survey selection explains the local source-density continuity',
        'U':'finite point-random sampling, kNN smoothing (~arcmin scale), unmodeled deblending/tracer effects, random-catalog construction; no cosmological inference',
        'historical_direct_coadd_comparator':{
            'n_fields':36,'selection_model_r2_median':0.00594,'residual_rho_median':0.368,
            'matched_shift_median':0.0347,'positive_fields':27,'sign_p_one_sided':0.00197,'wilcoxon_p_one_sided':0.000180,
            'role':'context only; not part of this experiment decision rule',
        },
        'remote_files':[{
            'file_index':p['file_index'],'url':p['url'],'rows':p['rows'],'rows_sampled':p['rows_sampled'],
            'sample_fraction_effective':p['sample_fraction_effective'],'sample_data_bytes':p['sample_data_bytes'],
            'sample_data_sha256':p['sample_data_sha256'],'etag':p['data_http']['etag'],
            'content_range':p['data_http']['content_range'],'header_prefix_sha256':p['header_prefix_sha256'],
        } for p in remote_prov],
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='remote_files' and k!='point_selection_features'},indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
