#!/usr/bin/env python3
"""Compare source-count anisotropy with official DR11 selection-map anisotropy.

Uses the first 12 provenance-fixed fields employed by the multiband stress test.
For each matched brick, measure adjacent-cell Spearman correlation separately
along coadd x and y for source counts and g/r/i/z depth, NEXP, PSFSIZE, plus
BRICK_PRIMARY support.  This is a survey-systematics QC diagnostic, not a
cosmological discovery test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.stats import binomtest, spearmanr, wilcoxon

import selection_null_multiband as mb

core = mb.core
N_FIELDS = 12


def rho(a, b):
    r = float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).statistic)
    return r if np.isfinite(r) else 0.0


def xy_stats(g):
    a = np.asarray(g, float)
    return rho(a[:, :-1], a[:, 1:]), rho(a[:-1, :], a[1:, :])


def paired(a):
    x = np.asarray(a, float); x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0: return {"n": 0}
    pos = int((x > 0).sum())
    try: w = float(wilcoxon(x, alternative="two-sided").pvalue)
    except Exception: w = float("nan")
    return {"n": int(len(x)), "positive": pos, "median": float(np.median(x)), "mean": float(np.mean(x)),
            "sign_p_two_sided": float(binomtest(pos, len(x), .5, alternative="two-sided").pvalue),
            "wilcoxon_p_two_sided": w}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/selection_directionality12'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    p=json.loads(Path(args.centers).read_text()); regs=p.get('regions',[])[:N_FIELDS]
    if p.get('status')!='REAL_DR11' or len(regs)!=N_FIELDS: raise RuntimeError('12 REAL_DR11 fixed centers required')
    rows=[]; provenance=[]
    for i,r in enumerate(regs):
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg']); brick,bq,near=core.choose_brick_from_center(ra,dec)
        urls=mb.product_urls_all(brick); mask,_,pm=core.read_image(urls['maskbits'],'MASKBITS')
        products={'maskbits':pm}; maps={}
        hdr=None
        for b in mb.BANDS:
            depth,h,pd=core.read_image(urls[f'depth_{b}'],f'DEPTH_{b.upper()}'); nexp,_,pn=core.read_image(urls[f'nexp_{b}']); psf,_,pp=core.read_image(urls[f'psfsize_{b}'])
            maps[f'depth_{b}']=core.continuous_features(depth,True,True)[0]
            maps[f'nexp_{b}']=core.continuous_features(nexp,False,True)[0]
            maps[f'psfsize_{b}']=core.continuous_features(psf,False,True)[0]
            products.update({f'depth_{b}':pd,f'nexp_{b}':pn,f'psfsize_{b}':pp})
            if b=='r': hdr=h
        mf=core.mask_features(mask); maps['primary_frac']=mf[0]; maps['clean_frac']=mf[1]
        src,sq=core.source_catalog(brick); counts,ninside=core.count_grid(src,WCS(hdr),mask.shape)
        primary=np.clip(maps['primary_frac'],.25,1.0); count_corr=counts/primary
        count_norm=core.robust_norm(count_corr,maps['primary_frac']>=.9375,True)
        sx,sy=xy_stats(count_norm)
        rows.append({'field':name,'dec':dec,'brick':brick,'map':'source_count','rho_x':sx,'rho_y':sy,'x_minus_y':sx-sy})
        for k,g in maps.items():
            x,y=xy_stats(g); rows.append({'field':name,'dec':dec,'brick':brick,'map':k,'rho_x':x,'rho_y':y,'x_minus_y':x-y})
        provenance.append({'field':name,'brick':brick,'center_ra_deg':ra,'center_dec_deg':dec,'source_rows':int(len(src)),'sources_inside_wcs':ninside,'source_provenance':sq,'brick_choice':bq,'nearest_primary_source_deg':near,'products':products})
        print(f'[selection-directionality] {i+1}/{N_FIELDS} {name}->{brick} source_dxdy={sx-sy:+.4f}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False)
    wide=df.pivot(index='field',columns='map',values='x_minus_y')
    src=wide['source_count']
    diagnostics=[]
    for col in wide.columns:
        if col=='source_count': continue
        a=wide[col]
        diagnostics.append({'map':col,'median_x_minus_y':float(np.median(a)),'x_minus_y_test':paired(a),
                            'spearman_source_vs_map_x_minus_y':float(spearmanr(src,a).statistic)})
    south=df[(df['map']=='source_count') & (df.dec<0)].x_minus_y
    north=df[(df['map']=='source_count') & (df.dec>=0)].x_minus_y
    summary={'status':'REAL_DR11_SELECTION_DIRECTIONALITY','validation':'12 matched official DR11 brick products; post-hoc QC',
             'source_count':{'median_x_minus_y':float(np.median(src)),'all_fields':paired(src),'south_dec_lt_0':paired(south),'north_dec_ge_0':paired(north)},
             'selection_maps':diagnostics}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'provenance.json').write_text(json.dumps({'status':'REAL_DR11_SELECTION_DIRECTIONALITY','regions':provenance},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
