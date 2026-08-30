#!/usr/bin/env python3
"""Target-program-stratified redshift permutation stress for REAL DR11 x DESI DR1.

This preserves DESI angular positions AND the observed z distribution separately
within each (survey, program) group in a field. It is stricter than the primary
unstratified z shuffle and tests whether the exploratory signal was caused by
redshift-dependent DESI observing/target-program composition.
"""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from redshift_view_validate import (
    HALF, QUERY_RADIUS, ZEDGES, query, square_clip, count_grid_from_offsets,
    local_map, broad_map, slice_stat, verify_and_load,
)

GRID_PERM=300
SEED=20260901
MIN_TOTAL=60
MIN_BIN=12


def paired(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if not n:return {'n_fields':0}
    pos=int((x>0).sum())
    try:wp=float(wilcoxon(x,alternative='greater').pvalue)
    except Exception:wp=float('nan')
    return {'n_fields':n,'positive_fields':pos,
            'sign_test_one_sided_p':float(binomtest(pos,n,.5,alternative='greater').pvalue),
            'wilcoxon_one_sided_p':wp,'mean_difference':float(x.mean()),'median_difference':float(np.median(x))}


def stat_custom(img_local,img_broad,dra,ddec,z):
    rows=[]
    for k in range(len(ZEDGES)-1):
        lo,hi=ZEDGES[k],ZEDGES[k+1]; m=(z>=lo)&(z<hi); n=int(m.sum())
        if n<MIN_BIN:continue
        g=count_grid_from_offsets(dra[m],ddec[m])
        from redshift_view_validate import corr
        rows.append((n,corr(img_local,local_map(g)),corr(img_broad,broad_map(g))))
    if len(rows)<2:return None
    w=np.array([r[0] for r in rows],float); l=np.array([r[1] for r in rows]); b=np.array([r[2] for r in rows])
    gl=np.isfinite(l); gb=np.isfinite(b)
    if gl.sum()<2:return None
    return {'local':float(np.average(l[gl],weights=w[gl])),
            'broad':float(np.average(b[gb],weights=w[gb])) if gb.any() else float('nan'),
            'max':float(np.nanmax(l))}


def shuffle_stratified(z,survey,program,rng):
    out=np.asarray(z,float).copy()
    key=np.char.add(np.char.add(np.asarray(survey,str).astype('U32'),'|'),np.asarray(program,str).astype('U32'))
    for k in np.unique(key):
        ii=np.flatnonzero(key==k)
        if len(ii)>1:out[ii]=rng.permutation(out[ii])
    return out


def eval_centers(center_rows,label,outdir):
    rng=np.random.default_rng(SEED + (0 if label=='primary' else 1000)); rows=[]; nulls={}
    for j,c in enumerate(center_rows):
        name=c['name']; ra0=float(c['ra']); dec0=float(c['dec'])
        # DR11 imaging is queried directly for both sets to keep treatment identical.
        isql=("SELECT ra,dec FROM ls_dr11.tractor WHERE brick_primary=1 AND "
              f"q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS:.8f})")
        im=pd.read_csv(io.StringIO(query(isql))); im.columns=[x.lower() for x in im.columns]; im=square_clip(im,ra0,dec0,'ra','dec')
        ic=count_grid_from_offsets(im['_dra'].to_numpy(),im['_ddec'].to_numpy()); il,ib=local_map(ic),broad_map(ic)
        zsql=("SELECT mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, survey, program FROM desi_dr1.zpix "
              "WHERE zwarn=0 AND zcat_primary=TRUE AND spectype='GALAXY' "
              f"AND z>={ZEDGES[0]:.4f} AND z<{ZEDGES[-1]:.4f} AND "
              f"q3c_radial_query(mean_fiber_ra,mean_fiber_dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS:.8f})")
        d=pd.read_csv(io.StringIO(query(zsql))); d.columns=[x.lower() for x in d.columns]; d=square_clip(d,ra0,dec0,'ra','dec'); d['z']=pd.to_numeric(d.z,errors='coerce'); d=d[np.isfinite(d.z)].reset_index(drop=True)
        if len(d)<MIN_TOTAL:
            rows.append({'set':label,'field':name,'status':'low_n','n':len(d)});continue
        dra=d['_dra'].to_numpy(); ddec=d['_ddec'].to_numpy(); z=d.z.to_numpy(); survey=d.survey.fillna('NA').astype(str).to_numpy(); program=d.program.fillna('NA').astype(str).to_numpy()
        actual=stat_custom(il,ib,dra,ddec,z)
        if actual is None:
            rows.append({'set':label,'field':name,'status':'bins','n':len(d)});continue
        nl=[]; nb=[]; nm=[]
        for b in range(GRID_PERM):
            st=stat_custom(il,ib,dra,ddec,shuffle_stratified(z,survey,program,rng))
            if st is not None:nl.append(st['local']);nb.append(st['broad']);nm.append(st['max'])
        if len(nl)<GRID_PERM//2:
            rows.append({'set':label,'field':name,'status':'null','n':len(d)});continue
        nl=np.asarray(nl);nb=np.asarray(nb);nm=np.asarray(nm);nulls[name]=nl
        rows.append({'set':label,'field':name,'status':'accepted','n':len(d),'n_groups':len(set(zip(survey,program))),
                     'actual_local':actual['local'],'null_local':float(nl.mean()),'delta_local':actual['local']-float(nl.mean()),
                     'actual_broad':actual['broad'],'null_broad':float(nb.mean()),'delta_broad':actual['broad']-float(nb.mean()),
                     'actual_max':actual['max'],'null_max':float(nm.mean()),'delta_max':actual['max']-float(nm.mean()),
                     'perm_p':float((1+(nl>=actual['local']).sum())/(len(nl)+1))})
        print(f"[strat-z] {label} {j+1}/{len(center_rows)} {name}: n={len(d)} delta={rows[-1]['delta_local']:.4f}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(outdir/f'{label}_field_metrics.csv',index=False); a=df[df.status=='accepted']
    B=min((len(nulls[f]) for f in a.field),default=0)
    if B and len(a):
        gnull=np.array([np.mean([nulls[f][b] for f in a.field]) for b in range(B)]); ga=float(a.actual_local.mean())
        gp=float((1+(gnull>=ga).sum())/(B+1))
    else:ga=gp=float('nan')
    return {'set':label,'accepted_fields':int(len(a)),'local_delta':paired(a.delta_local.to_numpy()),
            'broad_delta':paired(a.delta_broad.to_numpy()),'max_delta':paired(a.delta_max.to_numpy()),
            'actual_local_median':float(a.actual_local.median()) if len(a) else float('nan'),
            'null_local_median':float(a.null_local.median()) if len(a) else float('nan'),
            'global_actual_mean':ga,'global_perm_p':gp},df


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',default='data/real/dr11/expanded48/provenance.json');ap.add_argument('--replication',default='results/real_dr11/redshift_view_replication24/provenance.json');ap.add_argument('--out',default='results/real_dr11/redshift_view_stratified');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    op=json.loads(Path(args.original).read_text()); rp=json.loads(Path(args.replication).read_text())
    primary=[{'name':r['name'],'ra':r['center_ra_deg'],'dec':r['center_dec_deg']} for r in op['regions']]
    repl=[{'name':r['field'],'ra':r['center_ra_deg'],'dec':r['center_dec_deg']} for r in rp['fields']]
    ps,pdf=eval_centers(primary,'primary',out);rs,rdf=eval_centers(repl,'replication',out)
    comb=pd.concat([pdf[pdf.status=='accepted'],rdf[rdf.status=='accepted']],ignore_index=True)
    cs={'n_fields':int(len(comb)),'local_delta':paired(comb.delta_local.to_numpy()),'broad_delta':paired(comb.delta_broad.to_numpy()),'max_delta':paired(comb.delta_max.to_numpy())}
    summary={'status':'REAL_DR11_DESI_DR1_SURVEY_PROGRAM_STRATIFIED_Z_NULL','null':'within-field z permutation separately within exact (survey,program) groups; all angular positions unchanged','primary':ps,'replication':rs,'combined':cs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
