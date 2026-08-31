#!/usr/bin/env python3
"""Independent-sky replication of the REAL DR11 x DESI DR1 redshift-view gate.

Candidate sky centers come from a fixed shuffled grid and must be >=6 deg from
all original expanded48 fields and from other accepted replication fields.
Fields are accepted using only survey/data-availability thresholds, never the
cross-view correlation outcome. The null permutes observed DESI redshifts
within field while leaving angular positions fixed.
"""
from __future__ import annotations

import argparse, hashlib, io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from redshift_view_validate import (
    HALF, QUERY_RADIUS, ZEDGES, N_PERM, SEED, query, square_clip,
    count_grid_from_offsets, local_map, broad_map, slice_stat,
)

TARGET_FIELDS = 24
MIN_SEP_DEG = 6.0
MIN_DR11_RADIAL = 5000
MIN_DESI_RADIAL = 80
MIN_DESI_SQUARE = 60
MIN_BIN_REP = 12
TABLE_IMG = 'ls_dr11.tractor'
TABLE_Z = 'desi_dr1.zpix'
REP_SEED = 20260831


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sep(ra1, dec1, ra2, dec2):
    r1, r2, d1, d2 = map(np.deg2rad, [ra1, ra2, dec1, dec2])
    c = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def candidates():
    pts = [(float(ra), float(dec)) for dec in [-10,0,10,20,30,40,50,60]
           for ra in np.arange(0, 360, 10)]
    rng = np.random.default_rng(REP_SEED)
    rng.shuffle(pts)
    return pts


def one_value(sql):
    d = pd.read_csv(io.StringIO(query(sql)))
    return int(d.iloc[0,0])


def z_query(ra, dec, count_only=False):
    sel = 'COUNT(*) AS n' if count_only else 'mean_fiber_ra AS ra, mean_fiber_dec AS dec, z'
    return (
        f"SELECT {sel} FROM {TABLE_Z} WHERE zwarn=0 AND zcat_primary=TRUE "
        "AND spectype='GALAXY' "
        f"AND z>={ZEDGES[0]:.4f} AND z<{ZEDGES[-1]:.4f} "
        f"AND q3c_radial_query(mean_fiber_ra,mean_fiber_dec,{ra:.8f},{dec:.8f},{QUERY_RADIUS:.8f})"
    )


def img_query(ra, dec, count_only=False):
    sel = 'COUNT(*) AS n' if count_only else 'ra,dec'
    return (f"SELECT {sel} FROM {TABLE_IMG} WHERE brick_primary=1 AND "
            f"q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{QUERY_RADIUS:.8f})")


def paired_summary(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n=len(x)
    if not n: return {'n_fields':0}
    pos=int((x>0).sum())
    try: wp=float(wilcoxon(x,alternative='greater').pvalue)
    except Exception: wp=float('nan')
    return {'n_fields':n,'positive_fields':pos,
            'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),
            'wilcoxon_one_sided_p':wp,'mean_difference':float(np.mean(x)),
            'median_difference':float(np.median(x))}


def slice_stat_rep(img_local, img_broad, dra, ddec, z):
    # Same statistic as primary gate but slightly lower per-bin count threshold.
    rows=[]
    for k in range(len(ZEDGES)-1):
        lo,hi=ZEDGES[k],ZEDGES[k+1]
        m=(z>=lo)&(z<hi); n=int(m.sum())
        if n < MIN_BIN_REP: continue
        g=count_grid_from_offsets(dra[m],ddec[m])
        from redshift_view_validate import corr
        rows.append({'zlo':float(lo),'zhi':float(hi),'n':n,
                     'local_corr':corr(img_local,local_map(g)),
                     'broad_corr':corr(img_broad,broad_map(g))})
    if len(rows)<2: return None,rows
    w=np.array([r['n'] for r in rows],float)
    lc=np.array([r['local_corr'] for r in rows],float)
    bc=np.array([r['broad_corr'] for r in rows],float)
    gl=np.isfinite(lc); gb=np.isfinite(bc)
    if gl.sum()<2:return None,rows
    return {'local':float(np.average(lc[gl],weights=w[gl])),
            'broad':float(np.average(bc[gb],weights=w[gb])) if gb.any() else float('nan'),
            'max_local':float(np.nanmax(lc))},rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--original',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/redshift_view_replication24'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(args.original).read_text())
    old=[(float(r['center_ra_deg']),float(r['center_dec_deg'])) for r in original['regions']]
    accepted=[]; trials=[]
    for idx,(ra,dec) in enumerate(candidates()):
        if len(accepted)>=TARGET_FIELDS: break
        if any(sep(ra,dec,a,b)<MIN_SEP_DEG for a,b in old+[(r['ra'],r['dec']) for r in accepted]):
            trials.append({'candidate_index':idx,'ra':ra,'dec':dec,'status':'separation_reject'}); continue
        nz=one_value(z_query(ra,dec,True))
        if nz<MIN_DESI_RADIAL:
            trials.append({'candidate_index':idx,'ra':ra,'dec':dec,'status':'desi_coverage_reject','n_desi_radial':nz}); continue
        ni=one_value(img_query(ra,dec,True))
        if ni<MIN_DR11_RADIAL:
            trials.append({'candidate_index':idx,'ra':ra,'dec':dec,'status':'dr11_coverage_reject','n_desi_radial':nz,'n_dr11_radial':ni}); continue
        accepted.append({'name':f'r{len(accepted):02d}_ra{int(ra):03d}_{"p" if dec>=0 else "m"}{abs(int(dec)):02d}', 'ra':ra,'dec':dec,'n_desi_radial':nz,'n_dr11_radial':ni})
        trials.append({'candidate_index':idx,'ra':ra,'dec':dec,'status':'accepted','n_desi_radial':nz,'n_dr11_radial':ni,'name':accepted[-1]['name']})
        print(f"[rep-select] accepted {accepted[-1]['name']} DESI={nz} DR11={ni}",flush=True)
    if len(accepted)<TARGET_FIELDS:
        raise RuntimeError(f'only {len(accepted)} replication fields; expected {TARGET_FIELDS}')

    rng=np.random.default_rng(REP_SEED); rows=[]; prov=[]; nulls={}
    for j,f in enumerate(accepted):
        name,ra0,dec0=f['name'],f['ra'],f['dec']
        iraw=query(img_query(ra0,dec0,False)); im=pd.read_csv(io.StringIO(iraw)); im.columns=[c.lower() for c in im.columns]
        im=square_clip(im,ra0,dec0,'ra','dec')
        icount=count_grid_from_offsets(im['_dra'].to_numpy(),im['_ddec'].to_numpy()); iloc,ibroad=local_map(icount),broad_map(icount)
        zsql=z_query(ra0,dec0,False); zraw=query(zsql); zdf=pd.read_csv(io.StringIO(zraw)); zdf.columns=[c.lower() for c in zdf.columns]
        zdf=square_clip(zdf,ra0,dec0,'ra','dec'); zdf['z']=pd.to_numeric(zdf.z,errors='coerce'); zdf=zdf[np.isfinite(zdf.z)].sort_values(['ra','dec','z'],kind='mergesort').reset_index(drop=True)
        zcounts=[int(((zdf.z>=ZEDGES[k])&(zdf.z<ZEDGES[k+1])).sum()) for k in range(len(ZEDGES)-1)]
        prec={'field':name,'center_ra_deg':ra0,'center_dec_deg':dec0,'n_dr11_square':int(len(im)),'n_desi_square':int(len(zdf)),'zbin_counts':zcounts,'imaging_sql':img_query(ra0,dec0,False),'desi_sql':zsql,'imaging_sha256':sha256(im[['ra','dec']].to_csv(index=False,lineterminator='\n').encode()),'desi_sha256':sha256(zdf[['ra','dec','z']].to_csv(index=False,lineterminator='\n').encode())}; prov.append(prec)
        print(f"[rep-run] {j+1}/{len(accepted)} {name}: DR11={len(im)} DESI={len(zdf)} bins={zcounts}",flush=True)
        if len(zdf)<MIN_DESI_SQUARE:
            rows.append({'field':name,'status':'rejected_low_square','n_desi':len(zdf)}); continue
        dra=zdf['_dra'].to_numpy(float); ddec=zdf['_ddec'].to_numpy(float); z=zdf.z.to_numpy(float)
        actual,slices=slice_stat_rep(iloc,ibroad,dra,ddec,z)
        if actual is None:
            rows.append({'field':name,'status':'rejected_bins','n_desi':len(zdf)}); continue
        nl=[]; nb=[]; nm=[]
        for b in range(N_PERM):
            st,_=slice_stat_rep(iloc,ibroad,dra,ddec,rng.permutation(z))
            if st is not None: nl.append(st['local']); nb.append(st['broad']); nm.append(st['max_local'])
        nl=np.asarray(nl); nb=np.asarray(nb); nm=np.asarray(nm)
        if len(nl)<N_PERM//2:
            rows.append({'field':name,'status':'rejected_null','n_desi':len(zdf)}); continue
        rows.append({'field':name,'status':'accepted','n_desi':len(zdf),'n_slices':len(slices),'actual_local':actual['local'],'null_local_mean':float(nl.mean()),'delta_local':actual['local']-float(nl.mean()),'field_perm_p_local':float((1+(nl>=actual['local']).sum())/(len(nl)+1)),'actual_broad':actual['broad'],'null_broad_mean':float(nb.mean()),'delta_broad':actual['broad']-float(nb.mean()),'actual_max_local':actual['max_local'],'null_max_mean':float(nm.mean()),'delta_max_local':actual['max_local']-float(nm.mean()),'slice_details_json':json.dumps(slices,sort_keys=True)})
        nulls[name]=nl
    df=pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv',index=False); a=df[df.status=='accepted'].copy()
    if len(a)<12: raise RuntimeError(f'too few accepted independent fields: {len(a)}')
    B=min(len(nulls[f]) for f in a.field); actual_global=float(a.actual_local.mean()); gnull=np.array([np.mean([nulls[f][b] for f in a.field]) for b in range(B)])
    summary={'status':'REAL_DR11_DESI_DR1_INDEPENDENT_REPLICATION','candidate_seed':REP_SEED,'candidate_rule':'fixed shuffled 10-deg RA x listed-Dec grid; >=6 deg from original expanded48 and replication peers; coverage thresholds only','target_fields':TARGET_FIELDS,'accepted_stat_fields':int(len(a)),'z_edges':ZEDGES.tolist(),'local_actual_median':float(a.actual_local.median()),'local_null_median':float(a.null_local_mean.median()),'local_delta':paired_summary(a.delta_local.to_numpy()),'broad_delta':paired_summary(a.delta_broad.to_numpy()),'max_slice_delta':paired_summary(a.delta_max_local.to_numpy()),'global_mean_local_actual':actual_global,'global_null_mean':float(gnull.mean()),'global_redshift_permutation_p':float((1+(gnull>=actual_global).sum())/(B+1))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'selection_trials':trials,'fields':prov,'seed':REP_SEED,'n_permutations':N_PERM},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
