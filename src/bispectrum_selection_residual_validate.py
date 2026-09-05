#!/usr/bin/env python3
"""Selection-aware higher-order phase-coupling test on REAL DR11.

Reconstructs the exact *protocol* of the accepted 36-field multiband
selection-null experiment from the committed, pre-registered expanded48 sky
centers: scan the first 48 candidates in fixed order and accept the first 36
that pass only objective brick/coadd/source/coverage gates.  No outcome metric
is inspected during acceptance.

For each grouped holdout fold, a count model is trained only on other accepted
bricks using official g/r/i/z depth, NEXP, PSF-size and MASKBITS/primary maps.
The held-out observed count map is residualized by that selection expectation,
then tested for bispectrum phase coupling against an exact-Fourier-amplitude
random-phase surrogate.  No simulated cosmological target or mock fallback is
used.
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
import selection_null_multiband36 as mb36
import bispectrum_phase_validate as bp

core=mb.core; BANDS=mb.BANDS; GRID=64
TARGET_FIELDS=36; MAX_CANDIDATES=48; MIN_PATCHES=8

def acquire_candidate(rec):
    name=rec['name']; ra=float(rec['center_ra_deg']); dec=float(rec['center_dec_deg'])
    brick,bq,near=core.choose_brick_from_center(ra,dec)
    urls=mb.product_urls_all(brick)
    jobs={'maskbits':(urls['maskbits'],'MASKBITS')}
    for b in BANDS:
        jobs[f'depth_{b}']=(urls[f'depth_{b}'],f'DEPTH_{b.upper()}')
        jobs[f'nexp_{b}']=(urls[f'nexp_{b}'],None)
        jobs[f'psfsize_{b}']=(urls[f'psfsize_{b}'],None)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={k:ex.submit(core.read_image,u,w) for k,(u,w) in jobs.items()}
        got={k:v.result() for k,v in fut.items()}
    mask,_,pmask=got['maskbits']; shape=mask.shape; arrays={}; product_prov={'maskbits':pmask}; hdr=None
    for b in BANDS:
        depth,h,pdepth=got[f'depth_{b}']; nexp,_,pnexp=got[f'nexp_{b}']; psf,_,ppsf=got[f'psfsize_{b}']
        if depth.shape!=shape or nexp.shape!=shape or psf.shape!=shape: raise ValueError('shape_mismatch')
        arrays[(b,'depth')]=depth; arrays[(b,'nexp')]=nexp; arrays[(b,'psf')]=psf
        product_prov.update({f'depth_{b}':pdepth,f'nexp_{b}':pnexp,f'psfsize_{b}':ppsf})
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
    # Reproduce the accepted multiband36 coverage gate exactly.
    norm=core.robust_norm(corrected,valid,True)
    patches,_,_=mb36.make_patches(norm,sel,valid)
    if len(patches)<MIN_PATCHES: raise RuntimeError(f'coverage:{len(patches)}')
    data={'brick':brick,'counts':corrected,'sel':sel,'valid':valid,'feature_names':fn,'source_rows':int(len(src)),'source_query':sq,'sources_inside_wcs':int(ninside),'valid_fraction':float(valid.mean()),'n_patches':int(len(patches))}
    prov={'field':name,'requested_center_ra_deg':ra,'requested_center_dec_deg':dec,'brick':brick,'brick_choice':bq,'nearest_primary_source_deg':near,'source_rows':int(len(src)),'sources_inside_wcs':int(ninside),'source_query':sq,'valid_fraction':float(valid.mean()),'n_patches':int(len(patches)),'products':product_prov}
    return name,data,prov

def fill_invalid(a,valid,seed):
    z=np.asarray(a,float).copy(); vals=z[valid]
    if not len(vals): raise RuntimeError('no valid cells')
    rng=np.random.default_rng(seed); z[~valid]=rng.choice(vals,size=int((~valid).sum()),replace=True)
    return z

def pair_summary(real,control): return bp.paired(np.asarray(real,float)-np.asarray(control,float))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/bispectrum_selection_residual36'); ap.add_argument('--folds',type=int,default=6); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    C=json.loads(Path(args.centers).read_text()); regs=C.get('regions',[])[:MAX_CANDIDATES]
    if C.get('status')!='REAL_DR11' or len(regs)<TARGET_FIELDS: raise RuntimeError('committed REAL_DR11 expanded48 centers required')
    fields={}; prov=[]; rejected=[]; used=set()
    for j,r in enumerate(regs):
        if len(fields)>=TARGET_FIELDS: break
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
        try:
            brick,bq,near=core.choose_brick_from_center(ra,dec)
        except Exception as e:
            rejected.append({'field':name,'reason':'brick_resolution','error':str(e)}); continue
        if brick in used:
            rejected.append({'field':name,'brick':brick,'reason':'duplicate'}); continue
        # Match multiband36: reserve the resolved brick before downstream availability tests.
        used.add(brick)
        print(f'[bispec-selection] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}',flush=True)
        try:
            name2,d,p=acquire_candidate(r)
        except ValueError as e:
            rejected.append({'field':name,'brick':brick,'reason':'shape_mismatch','error':str(e)}); continue
        except RuntimeError as e:
            msg=str(e); reason='coverage' if msg.startswith('coverage:') else 'source_or_coverage'
            rejected.append({'field':name,'brick':brick,'reason':reason,'error':msg}); continue
        except Exception as e:
            rejected.append({'field':name,'brick':brick,'reason':'missing_coadd_or_source','error':str(e)}); continue
        if d['brick']!=brick or name2!=name: raise RuntimeError('brick/name reconstruction mismatch')
        fields[name]=d; prov.append(p)
        print(f'[bispec-selection] accept {len(fields)}/{TARGET_FIELDS} patches={d["n_patches"]} valid={d["valid_fraction"]:.3f}',flush=True)
    if len(fields)!=TARGET_FIELDS:
        (out/'availability_rejections.json').write_text(json.dumps(rejected,indent=2,sort_keys=True)+'\n')
        raise RuntimeError(f'only {len(fields)} accepted from fixed {MAX_CANDIDATES}')

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
            idx=names.index(name)
            raw_g=bp.gaussianize(fill_invalid(d['counts'],v,20260901+idx*11))
            res_g=bp.gaussianize(fill_invalid(residual,v,20260902+idx*11))
            exp_g=bp.gaussianize(fill_invalid(expected,v,20260903+idx*11))
            raw_p=bp.exact_phase(raw_g,20261001+idx); res_p=bp.exact_phase(res_g,20262001+idx); exp_p=bp.exact_phase(exp_g,20263001+idx)
            stats={'raw':bp.stats_for_field(raw_g,tri),'raw_phase':bp.stats_for_field(raw_p,tri),'residual':bp.stats_for_field(res_g,tri),'residual_phase':bp.stats_for_field(res_p,tri),'selection_expected':bp.stats_for_field(exp_g,tri),'selection_phase':bp.stats_for_field(exp_p,tri)}
            for fam in tri:
                for sample in stats: rows.append({'field':name,'fold':fold,'family':fam,'sample':sample,**stats[sample][fam]})
            model_rows.append({'field':name,'fold':fold,'selection_cell_r2':r2,'selection_cell_spearman':rho,'valid_fraction':d['valid_fraction']})
        print(f'[bispec-selection] fold {fold+1}/{nfold}: train={len(train)} test={len(test)}',flush=True)
    D=pd.DataFrame(rows); M=pd.DataFrame(model_rows); D.to_csv(out/'field_metrics.csv',index=False); M.to_csv(out/'selection_model_metrics.csv',index=False)
    summary={'status':'REAL_DR11_BISPECTRUM_SELECTION_RESIDUAL','n_fields':36,'candidate_protocol':'first 36 availability-qualified fields from fixed first-48 expanded48 candidate order; same objective gates as selection_null_multiband36','availability_rejections':rejected,'cross_validation':f'{nfold}-fold grouped by whole brick','selection_features':'official g/r/i/z depth, NEXP, PSF-size, MASKBITS and primary support','invalid_pixel_handling':'deterministic resampling from each field valid-cell distribution before rank-Gaussianization','selection_model_r2_median':float(np.nanmedian(M.selection_cell_r2)),'selection_model_spearman_median':float(np.nanmedian(M.selection_cell_spearman)),'primary_family':'all','families':{}}
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
    (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'source_centers':str(args.centers),'regions':prov,'rejections':rejected},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
