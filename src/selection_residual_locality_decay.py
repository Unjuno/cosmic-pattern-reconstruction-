#!/usr/bin/env python3
"""REAL DR11 locality decay after official pixel-level selection residualization.

Uses exactly the 36 selection-qualified bricks from PR18/PR5.  Full
BRICK_PRIMARY source count maps are modeled from official g/r/i/z depth, NEXP,
PSF-size, MASKBITS and BRICK_PRIMARY support using grouped whole-brick CV.
For raw and selection-residualized maps, measure continuous Spearman coupling
between each 1.875-arcmin hidden center and context at four distances defined by
the accepted 48-field locality experiment.  A cyclic matched-shift context null
preserves the field's patch-level one-point structure while breaking local
pairing.

Primary: residual local-visible rho minus matched-shift rho must have positive
median and one-sided sign+Wilcoxon p<.05. Secondary: context coupling should
decay with distance after selection residualization. No simulated cosmology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import tracer_split_native_patch as native
native.install()
import cross_tracer_selection_residual_bispectrum as cs
import locality_validate as loc
import bispectrum_phase_validate as bp

GRID = 64
N_FOLDS = 6
LOCKED = [
 ('f00_ra156_p05','1560p050'), ('f01_ra132_m05','1319m050'), ('f02_ra144_p15','1439p150'),
 ('f03_ra036_m35','0360m350'), ('f04_ra180_m05','1801m050'), ('f06_ra072_m05','0720m050'),
 ('f07_ra216_m05','2159m050'), ('f08_ra336_m25','3359m250'), ('f09_ra216_p15','2160p150'),
 ('f10_ra240_m35','2398m350'), ('f11_ra060_m35','0601m350'), ('f12_ra312_m05','3119m050'),
 ('f13_ra228_p05','2280p050'), ('f15_ra144_m25','1440m250'), ('f17_ra048_m15','0479m150'),
 ('f19_ra132_p25','1319p250'), ('f20_ra120_p25','1198p250'), ('f22_ra036_p05','0359p050'),
 ('f23_ra012_m35','0120m350'), ('f24_ra012_m55','0119m550'), ('f25_ra228_m35','2279m350'),
 ('f26_ra312_p05','3119p050'), ('f27_ra060_m05','0600m050'), ('f28_ra192_p05','1919p050'),
 ('f29_ra312_m25','3119m250'), ('f30_ra000_m45','0001m450'), ('f31_ra132_p05','1319p050'),
 ('f32_ra024_m35','0239m350'), ('f33_ra348_p15','3479p150'), ('f34_ra000_m15','3598m150'),
 ('f35_ra168_m25','1680m250'), ('f36_ra168_m45','1678m450'), ('f37_ra336_m55','3359m550'),
 ('f39_ra084_m55','0839m550'), ('f40_ra192_m35','1920m350'), ('f41_ra348_m25','3480m250'),
]
FEATURES = ['local_visible','external_1_4cells','external_5_8cells','external_9_16cells']
DISTANCE_ORDER = {f:i for i,f in enumerate(FEATURES)}


def acquire():
    data, provenance = {}, []
    for i,(name,brick) in enumerate(LOCKED):
        print(f'[residual-locality] acquire {i+1}/{len(LOCKED)} {name}->{brick}', flush=True)
        tractor,tp = cs.tr.get_tractor(brick)
        counts = cs.tr.grid_from(tractor).astype(float)
        sel,valid,feature_names,product_prov,n_patches = cs.build_selection_features(brick)
        data[name] = {'brick':brick,'counts':counts,'sel':sel,'valid':valid,'feature_names':feature_names}
        provenance.append({'field':name,'brick':brick,'tractor':tp,'source_rows':int(len(tractor)),
                           'valid_cell_fraction':float(valid.mean()),'n_coverage_patches':int(n_patches),
                           'selection_products':product_prov})
    return data,provenance


def rho(a,b):
    r=float(spearmanr(np.asarray(a,float),np.asarray(b,float)).statistic)
    return r if np.isfinite(r) else float('nan')


def normalized_maps(counts,expected,valid,seed):
    raw = cs.fill_invalid(counts,valid,seed)
    raw = cs.core.robust_norm(raw,np.ones_like(valid,dtype=bool),True)
    z=np.zeros((GRID,GRID),float)
    z[valid]=(counts[valid]-expected[valid])/np.sqrt(expected[valid]+1.0)
    z=cs.fill_invalid(z,valid,seed+1)
    z=cs.core.robust_norm(z,np.ones_like(valid,dtype=bool),False)
    return raw,z


def context_rows(grid):
    c=loc.contexts_from_grid(grid)
    return {'hidden':c[:,0], **{f:c[:,j+1] for j,f in enumerate(FEATURES)}}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/real_dr11/selection_residual_locality36')
    ap.add_argument('--folds',type=int,default=N_FOLDS)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)

    data,provenance=acquire();names=list(data)
    if len(names)!=36: raise RuntimeError(f'locked field count {len(names)}')
    if len({tuple(data[n]['feature_names']) for n in names})!=1: raise RuntimeError('selection feature mismatch')
    nfold=max(2,min(args.folds,len(names)));fold_id={n:i%nfold for i,n in enumerate(names)}
    rows=[];model_rows=[]

    for fold in range(nfold):
        test=[n for n in names if fold_id[n]==fold];train=[n for n in names if fold_id[n]!=fold]
        model=cs.fit_selection_model(data,train,'counts',12000+fold)
        for name in test:
            d=data[name];v=d['valid'];counts=d['counts'];mu=float(np.mean(counts[v]))+1e-6
            pred_rel=np.full((GRID,GRID),np.nan,float);pred_rel[v]=np.clip(model.predict(d['sel'][v]),.03,20)
            expected=pred_rel*mu
            true_rel=counts[v]/mu
            model_rows.append({'field':name,'fold':fold,
                               'selection_r2':float(__import__('sklearn.metrics').metrics.r2_score(true_rel,pred_rel[v])),
                               'selection_spearman':rho(true_rel,pred_rel[v])})
            idx=names.index(name);raw,residual=normalized_maps(counts,expected,v,610000+idx*4)
            for sample,grid in [('raw',raw),('residual',residual)]:
                C=context_rows(grid);hidden=C['hidden']
                for feature in FEATURES:
                    x=C[feature];xs=loc.shifted_feature(x)
                    rows.append({'field':name,'fold':fold,'sample':sample,'feature':feature,
                                 'rho':rho(x,hidden),'matched_shift_rho':rho(xs,hidden)})
        print(f'[residual-locality] fold {fold+1}/{nfold}: train={len(train)} test={len(test)}',flush=True)

    D=pd.DataFrame(rows);M=pd.DataFrame(model_rows)
    D.to_csv(out/'field_metrics.csv',index=False);M.to_csv(out/'selection_model_metrics.csv',index=False)
    comparisons=[]
    for sample in ['raw','residual']:
        for feature in FEATURES:
            g=D[(D['sample']==sample)&(D.feature==feature)].set_index('field')
            diff=g.rho-g.matched_shift_rho
            comparisons.append({'sample':sample,'feature':feature,
                                'distance_definition':{
                                  'local_visible':'within same 3.75 arcmin patch excluding central 1.875 arcmin',
                                  'external_1_4cells':'0.469-1.875 arcmin beyond patch edge',
                                  'external_5_8cells':'2.344-3.750 arcmin beyond patch edge',
                                  'external_9_16cells':'4.219-7.500 arcmin beyond patch edge'}[feature],
                                'rho_median':float(np.nanmedian(g.rho)),
                                'matched_shift_median':float(np.nanmedian(g.matched_shift_rho)),
                                'real_minus_shift':bp.paired(diff.to_numpy())})

    # Per-field slope across increasing context distance; negative slope means locality decay.
    slopes=[]
    for name in names:
        g=D[(D.field==name)&(D['sample']=='residual')].copy()
        g['order']=g.feature.map(DISTANCE_ORDER);g=g.sort_values('order')
        adv=(g.rho-g.matched_shift_rho).to_numpy(float)
        slopes.append(float(np.polyfit(g['order'].to_numpy(float),adv,1)[0]))
    decay=bp.paired(-np.asarray(slopes))  # positive = decreasing advantage with distance

    summary={'status':'REAL_DR11_SELECTION_RESIDUAL_LOCALITY_DECAY','n_fields':36,
             'field_set':'exact locked PR18 selection-qualified bricks',
             'cross_validation':f'{nfold}-fold grouped by whole brick',
             'selection_features':'official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY support',
             'selection_model_r2_median':float(np.nanmedian(M.selection_r2)),
             'selection_model_spearman_median':float(np.nanmedian(M.selection_spearman)),
             'comparisons':comparisons,'residual_locality_decay_negative_slope_test':decay,
             'primary':'residual local_visible rho minus matched_shift',
             'predeclared_pass':'primary median>0 AND one-sided sign p<.05 AND Wilcoxon p<.05'}
    p=[x for x in comparisons if x['sample']=='residual' and x['feature']=='local_visible'][0]['real_minus_shift']
    summary['primary_decision']=('PASS_SELECTION_RESIDUAL_LOCALITY' if p['median']>0 and p['sign_p_one_sided']<.05 and p['wilcoxon_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_SELECTION_RESIDUAL_LOCALITY')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':provenance},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
