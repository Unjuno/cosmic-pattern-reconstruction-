#!/usr/bin/env python3
"""Gaia-match contamination stress test on provenance-fixed REAL_DR11 fields.

This is not a star/galaxy classifier.  We split DR11 Tractor sources only by
whether the catalog has a plausible Gaia EDR3 G magnitude, calling the groups
``gaia_matched`` and ``gaia_unmatched``.  The scientific estimator in each
subset still uses positions only.  We also randomly thin the larger unmatched
sample to the matched sample size within each field so that a difference in
point density cannot by itself explain locality differences.

No mock catalog is generated.
"""
from __future__ import annotations
import argparse, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from dl import queryClient as qc
from scipy.stats import binomtest, spearmanr, wilcoxon

TABLE='ls_dr11.tractor_s'; N_FIELDS=12; GRID=64; PATCH=16; STRIDE=8; HALF=.25
H=np.zeros((PATCH,PATCH),bool); H[4:12,4:12]=True
R=np.zeros((PATCH,PATCH),bool); R[3,3:13]=True; R[12,3:13]=True; R[4:12,3]=True; R[4:12,12]=True


def query(sql,attempts=5):
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


def fetch_field(meta):
    ra0=float(meta['center_ra_deg']); dec0=float(meta['center_dec_deg']); rad=float(meta.get('query_radius_deg',.4)); half=float(meta.get('box_width_deg',.5))/2
    sql=(f"SELECT ra,dec,gaia_phot_g_mean_mag FROM {TABLE} WHERE brick_primary=1 AND "
         f"q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},{rad:.8f})")
    d=pd.read_csv(io.StringIO(query(sql))); d.columns=[str(c).lower() for c in d.columns]
    if list(d.columns)!=['ra','dec','gaia_phot_g_mean_mag']: raise RuntimeError(f'bad columns {list(d.columns)}')
    dra=((d.ra.astype(float)-ra0+180)%360)-180
    keep=dra.abs().le(half)&d.dec.astype(float).ge(dec0-half)&d.dec.astype(float).lt(dec0+half)
    return d.loc[keep].reset_index(drop=True),sql


def tangent_grid(ra,dec,ra0,dec0):
    c=float(np.cos(np.deg2rad(dec0))); half=HALF*c
    dra=((np.asarray(ra,float)-ra0+180.0)%360.0)-180.0; x=dra*c; y=np.asarray(dec,float)-dec0
    keep=(np.abs(x)<half)&(np.abs(y)<half)
    g,_,_=np.histogram2d(y[keep],x[keep],bins=GRID,range=[[-half,half],[-half,half]])
    z=np.log1p(g); med=np.median(z); sc=np.median(np.abs(z-med))*1.4826
    if not np.isfinite(sc) or sc<1e-6: sc=np.std(z)
    if not np.isfinite(sc) or sc<1e-6: sc=1.0
    return (z-med)/sc


def locality(z):
    rr=[];hh=[]
    for y in range(0,GRID-PATCH+1,STRIDE):
        for x in range(0,GRID-PATCH+1,STRIDE):
            p=z[y:y+PATCH,x:x+PATCH]; rr.append(float(p[R].mean())); hh.append(float(p[H].mean()))
    return float(spearmanr(rr,hh).statistic)


def axis_delta(z,shift=8):
    ura=z[:,:-shift].ravel(); vra=z[:,shift:].ravel(); udec=z[:-shift,:].ravel(); vdec=z[shift:,:].ravel()
    return float(spearmanr(ura,vra).statistic-spearmanr(udec,vdec).statistic)


def paired(d,alternative='two-sided'):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; n=len(d); pos=int((d>0).sum()); neg=int((d<0).sum())
    if alternative=='greater': sp=float(binomtest(pos,n,.5,alternative='greater').pvalue)
    elif alternative=='less': sp=float(binomtest(neg,n,.5,alternative='greater').pvalue)
    else: sp=float(binomtest(pos,n,.5,alternative='two-sided').pvalue)
    try: wp=float(wilcoxon(d,alternative=alternative).pvalue)
    except Exception: wp=float('nan')
    return {'n_fields':n,'positive_fields':pos,'negative_fields':neg,'median_difference':float(np.median(d)),'mean_difference':float(np.mean(d)),
            'exact_sign_test_p':sp,'wilcoxon_p':wp,'alternative':alternative}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--provenance',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/gaia_split12'); args=ap.parse_args()
    p=json.loads(Path(args.provenance).read_text()); regs=p.get('regions',[])[:N_FIELDS]
    if p.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS: raise RuntimeError('REAL_DR11 fixed provenance required')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rows=[]; rng=np.random.default_rng(20260829)
    for i,r in enumerate(regs):
        print(f"[gaia-split] {i+1}/{N_FIELDS} {r['name']}",flush=True)
        d,sql=fetch_field(r); g=pd.to_numeric(d.gaia_phot_g_mean_mag,errors='coerce').to_numpy(float)
        gm=np.isfinite(g)&(g>0)&(g<40); gu=~gm
        if gm.sum()<500 or gu.sum()<500: raise RuntimeError(f"too few split sources {r['name']}: {gm.sum()}/{gu.sum()}")
        ra0=float(r['center_ra_deg']); dec0=float(r['center_dec_deg'])
        za=tangent_grid(d.ra,d.dec,ra0,dec0); zm=tangent_grid(d.ra[gm],d.dec[gm],ra0,dec0); zu=tangent_grid(d.ra[gu],d.dec[gu],ra0,dec0)
        n=int(min(gm.sum(),gu.sum())); reps=[]; ad=[]
        iu=np.flatnonzero(gu)
        for rep in range(24):
            pick=rng.choice(iu,n,replace=False); z=tangent_grid(d.ra.iloc[pick],d.dec.iloc[pick],ra0,dec0); reps.append(locality(z)); ad.append(axis_delta(z))
        rows.append({'field':r['name'],'query':sql,'n_all':len(d),'n_gaia_matched':int(gm.sum()),'n_gaia_unmatched':int(gu.sum()),'matched_fraction':float(gm.mean()),
                     'locality_all':locality(za),'locality_gaia_matched':locality(zm),'locality_gaia_unmatched':locality(zu),
                     'locality_gaia_unmatched_count_matched_mean':float(np.mean(reps)),'locality_gaia_unmatched_count_matched_sd':float(np.std(reps)),
                     'axis_delta_all':axis_delta(za),'axis_delta_gaia_matched':axis_delta(zm),'axis_delta_gaia_unmatched':axis_delta(zu),
                     'axis_delta_gaia_unmatched_count_matched_mean':float(np.mean(ad))})
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    summary={'status':'REAL_DR11','validation':'12 fixed fields; Gaia-match split and equal-count thinning; positions-only estimator',
             'labels_warning':'gaia_matched and gaia_unmatched are catalog-match groups, not pure star/galaxy classes',
             'median_matched_fraction':float(np.median(df.matched_fraction)),
             'median_locality_all':float(np.median(df.locality_all)),'median_locality_gaia_matched':float(np.median(df.locality_gaia_matched)),
             'median_locality_gaia_unmatched':float(np.median(df.locality_gaia_unmatched)),
             'median_locality_gaia_unmatched_count_matched':float(np.median(df.locality_gaia_unmatched_count_matched_mean)),
             'unmatched_minus_matched':paired(df.locality_gaia_unmatched-df.locality_gaia_matched),
             'count_matched_unmatched_minus_matched':paired(df.locality_gaia_unmatched_count_matched_mean-df.locality_gaia_matched),
             'median_axis_delta_matched':float(np.median(df.axis_delta_gaia_matched)),'median_axis_delta_unmatched':float(np.median(df.axis_delta_gaia_unmatched)),
             'axis_delta_unmatched_minus_matched':paired(df.axis_delta_gaia_unmatched-df.axis_delta_gaia_matched),
             'interpretation_rule':'If locality survives in Gaia-unmatched sources after equal-count thinning, simple dominance by Gaia-matched stellar-like sources is disfavored. This does not establish a pure-galaxy sample.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
