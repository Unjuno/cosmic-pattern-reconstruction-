#!/usr/bin/env python3
"""Selection-aware higher-order phase-coupling test on REAL DR11.

Uses the 36 availability-qualified bricks from the accepted multiband pixel
selection-null experiment.  For each grouped holdout fold, a count model is
trained only on other bricks using official g/r/i/z depth, NEXP, PSF-size and
MASKBITS/primary maps.  The held-out observed count map is residualized by that
selection expectation, then tested for bispectrum phase coupling against an
exact-Fourier-amplitude random-phase surrogate.

No simulated cosmological target or mock fallback is used.
"""
from __future__ import annotations
import argparse,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import selection_null_multiband as mb
import bispectrum_phase_validate as bp

core=mb.core; BANDS=mb.BANDS; GRID=64

def acquire(rec):
    name=rec['field']; brick=rec['brick']; urls=mb.product_urls_all(brick)
    jobs={'maskbits':(urls['maskbits'],'MASKBITS')}
    for b in BANDS:
        jobs[f'depth_{b}']=(urls[f'depth_{b}'],f'DEPTH_{b.upper()}')
        jobs[f'nexp_{b}']=(urls[f'nexp_{b}'],None)
        jobs[f'psfsize_{b}']=(urls[f'psfsize_{b}'],None)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={k:ex.submit(core.read_image,u,w) for k,(u,w) in jobs.items()}
        got={k:v.result() for k,v in fut.items()}
    mask,_,_=got['maskbits']; shape=mask.shape; arrays={}; hdr=None
    for b in BANDS:
        depth,h,_=got[f'depth_{b}']; nexp,_,_=got[f'nexp_{b}']; psf,_,_=got[f'psfsize_{b}']
        if depth.shape!=shape or nexp.shape!=shape or psf.shape!=shape: raise RuntimeError(f'shape mismatch {brick}')
        arrays[(b,'depth')]=depth; arrays[(b,'nexp')]=nexp; arrays[(b,'psf')]=psf
        if b=='r': hdr=h
    src,sq=core.source_catalog(brick); counts,ninside=core.count_grid(src,WCS(hdr),shape)
    feat=[]; fn=[]
    for b in BANDS:
        for arr,names in [
          (core.continuous_features(arrays[(b,'depth')],True,True),[f'log_depth_{b}_mean',f'log_depth_{b}_std',f'depth_{b}_positive_frac']),
          (core.continuous_features(arrays[(b,'nexp')],False,True),[f'nexp_{b}_mean',f'nexp_{b}_std',f'nexp_{b}_positive_frac']),
          (core.continuous_features(arrays[(b,'psf')],False,True),[f'psfsize_{b}_mean',f'psfsize_{b}_std',f'psfsize_{b}_positive_frac'])]:
            feat+=arr; fn+=names
    feat+=core.mask_features(mask); fn+=['primary_frac','clean_frac']+[f'maskbit_{k}_frac' for k in range(core.BIT_COUNT)]
    sel=np.stack(feat,-1); primary=sel[:,:,fn.index('primary_frac')]
    valid=(primary>=.9375)&(sel[:,:,fn.index('depth_r_positive_frac')]>=.9375)&(sel[:,:,fn.index('nexp_r_positive_frac')]>=.9375)
    corrected=counts/np.clip(primary,.25,1.0)
    return name,{'brick':brick,'counts':corrected,'sel':sel,'valid':valid,'feature_names':fn,'source_rows':int(len(src)),'source_query':sq,'sources_inside_wcs':int(ninside),'valid_fraction':float(valid.mean())}

def fill_invalid(a,valid,seed):
    z=np.asarray(a,float).copy(); vals=z[valid]
    if not len(vals): raise RuntimeError('no valid cells')
    rng=np.random.default_rng(seed); z[~valid]=rng.choice(vals,size=int((~valid).sum()),replace=True)
    return z

def pair_summary(real,control): return bp.paired(np.asarray(real,float)-np.asarray(control,float))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--selection-provenance',default='results/real_dr11/selection_null_multiband36/provenance.json'); ap.add_argument('--out',default='results/real_dr11/bispectrum_selection_residual36'); ap.add_argument('--folds',type=int,default=6); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    P=json.loads(Path(args.selection_provenance).read_text())
    regs=P.get('regions',[])
    if P.get('status')!='REAL_DR11_PIXEL_SELECTION_NULL_MULTIBAND_36' or len(regs)!=36: raise RuntimeError('accepted 36-field multiband selection provenance required')
    fields={}; prov=[]
    for i,r in enumerate(regs):
        name,d=acquire(r); fields[name]=d; prov.append({'field':name,'brick':d['brick'],'source_rows':d['source_rows'],'sources_inside_wcs':d['sources_inside_wcs'],'valid_fraction':d['valid_fraction'],'source_query':d['source_query']})
        print(f'[bispec-selection] acquired {i+1}/36 {name}->{d["brick"]} valid={d["valid_fraction"]:.3f}',flush=True)
    names=list(fields); nfold=max(2,min(args.folds,len(names))); fold_id={n:i%nfold for i,n in enumerate(names)}
    tri=bp.build_triangles(seed=20260901,max_pairs=16000); rows=[]; model_rows=[]
    for fold in range(nfold):
        test=[n for n in names if fold_id[n]==fold]; train=[n for n in names if fold_id[n]!=fold]
        X=[]; y=[]
        for f in train:
            d=fields[f]; v=d['valid']; yy=d['counts'][v]; X.append(d['sel'][v]); y.append(yy/(np.mean(yy)+1e-6))
        X=np.concatenate(X); y=np.concatenate(y)
        if len(y)>120000:
            rng=np.random.default_rng(20260901+fold); ii=rng.choice(len(y),120000,replace=False); X=X[ii]; y=y[ii]
        reg=HistGradientBoostingRegressor(loss='poisson',max_iter=70,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=70,l2_regularization=1.5,random_state=61).fit(X,np.clip(y,1e-4,None))
        for name in test:
            d=fields[name]; v=d['valid']; pred=np.full((GRID,GRID),np.nan,float); pred[v]=np.clip(reg.predict(d['sel'][v]),.03,20)
            mean_count=np.mean(d['counts'][v])+1e-6; true_rel=d['counts'][v]/mean_count
            r2=float(r2_score(true_rel,pred[v])); rho=float(spearmanr(true_rel,pred[v]).statistic)
            expected=pred*mean_count; residual=np.zeros((GRID,GRID),float); residual[v]=d['counts'][v]-expected[v]
            raw_f=fill_invalid(d['counts'],v,20260901+names.index(name)*11)
            res_f=fill_invalid(residual,v,20260902+names.index(name)*11)
            exp_f=fill_invalid(expected,v,20260903+names.index(name)*11)
            raw_g=bp.gaussianize(raw_f); res_g=bp.gaussianize(res_f); exp_g=bp.gaussianize(exp_f)
            raw_p=bp.exact_phase(raw_g,20261001+names.index(name)); res_p=bp.exact_phase(res_g,20262001+names.index(name)); exp_p=bp.exact_phase(exp_g,20263001+names.index(name))
            stats={'raw':bp.stats_for_field(raw_g,tri),'raw_phase':bp.stats_for_field(raw_p,tri),'residual':bp.stats_for_field(res_g,tri),'residual_phase':bp.stats_for_field(res_p,tri),'selection_expected':bp.stats_for_field(exp_g,tri),'selection_phase':bp.stats_for_field(exp_p,tri)}
            for fam in tri:
                for sample in stats:
                    rows.append({'field':name,'fold':fold,'family':fam,'sample':sample,**stats[sample][fam]})
            model_rows.append({'field':name,'fold':fold,'selection_cell_r2':r2,'selection_cell_spearman':rho,'valid_fraction':d['valid_fraction']})
        print(f'[bispec-selection] fold {fold+1}/{nfold}: train={len(train)} test={len(test)}',flush=True)
    D=pd.DataFrame(rows); M=pd.DataFrame(model_rows); D.to_csv(out/'field_metrics.csv',index=False); M.to_csv(out/'selection_model_metrics.csv',index=False)
    summary={'status':'REAL_DR11_BISPECTRUM_SELECTION_RESIDUAL','n_fields':36,'cross_validation':f'{nfold}-fold grouped by whole brick','selection_features':'official g/r/i/z depth, NEXP, PSF-size, MASKBITS and primary support','invalid_pixel_handling':'deterministic resampling from each field valid-cell distribution before rank-Gaussianization','selection_model_r2_median':float(np.nanmedian(M.selection_cell_r2)),'selection_model_spearman_median':float(np.nanmedian(M.selection_cell_spearman)),'primary_family':'all','families':{}}
    for fam,g in D.groupby('family'):
        W=g.pivot(index='field',columns='sample',values=['phase_lock','bicoherence','signed_bicoherence']); fs={}
        for metric in ['phase_lock','bicoherence','signed_bicoherence']:
            q={s:W[metric][s] for s in ['raw','raw_phase','residual','residual_phase','selection_expected','selection_phase']}
            fs[metric]={s+'_median':float(np.nanmedian(v)) for s,v in q.items()}
            fs[metric].update({'raw_minus_phase':pair_summary(q['raw'],q['raw_phase']),'residual_minus_phase':pair_summary(q['residual'],q['residual_phase']),'selection_minus_phase':pair_summary(q['selection_expected'],q['selection_phase']),'residual_minus_raw':bp.paired(q['residual']-q['raw'])})
        fs['n_triangles']=int(g.n_triangles.iloc[0]); summary['families'][fam]=fs
    p=summary['families']['all']['bicoherence']['residual_minus_phase']
    summary['primary_decision']='PASS_SELECTION_RESISTANT_PHASE_COUPLING' if p['median']>0 and p['wilcoxon_p_one_sided']<.05 and p['sign_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_SELECTION_RESISTANT_PHASE_COUPLING'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'source_selection_provenance':str(args.selection_provenance),'regions':prov},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
