#!/usr/bin/env python3
"""REAL DR11 exact-power phase-surrogate locality after selection residualization.

Uses exactly the accepted PR24 selection-qualified 36 bricks and the same
6-fold grouped pixel-selection count model. For raw and selection-residualized
maps, preserve the exact 2-D Fourier amplitude spectrum while replacing phases
with 16 independent real-field phase realizations. Compare hidden-center vs
context Spearman in the observed map to the mean exact-power surrogate value.

This is a direct test of whether the observed locality statistic is fully
explained by two-point/power information. Primary PASS requires the residual
same-patch visible locality to exceed the exact-power phase surrogate with
positive median and one-sided sign+Wilcoxon p<.05. No simulated cosmology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import selection_residual_locality_decay as sl
import cross_tracer_selection_residual_bispectrum as cs
import bispectrum_phase_validate as bp

N_SURR = 16


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/real_dr11/selection_residual_phase_surrogate36')
    ap.add_argument('--folds',type=int,default=6)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data,provenance=sl.acquire();names=list(data)
    if len(names)!=36: raise RuntimeError(f'locked field count {len(names)}')
    nfold=max(2,min(args.folds,len(names)));fold_id={n:i%nfold for i,n in enumerate(names)}
    rows=[];model_rows=[]
    for fold in range(nfold):
        test=[n for n in names if fold_id[n]==fold];train=[n for n in names if fold_id[n]!=fold]
        model=cs.fit_selection_model(data,train,'counts',15000+fold)
        for name in test:
            d=data[name];v=d['valid'];counts=d['counts'];mu=float(np.mean(counts[v]))+1e-6
            pred_rel=np.full((sl.GRID,sl.GRID),np.nan,float);pred_rel[v]=np.clip(model.predict(d['sel'][v]),.03,20)
            expected=pred_rel*mu
            true_rel=counts[v]/mu
            model_rows.append({'field':name,'fold':fold,'selection_r2':float(__import__('sklearn.metrics').metrics.r2_score(true_rel,pred_rel[v])),'selection_spearman':sl.rho(true_rel,pred_rel[v])})
            idx=names.index(name);raw,residual=sl.normalized_maps(counts,expected,v,710000+idx*4)
            for si,(sample,grid) in enumerate([('raw',raw),('residual',residual)]):
                C=sl.context_rows(grid);hidden=C['hidden']
                observed={f:sl.rho(C[f],hidden) for f in sl.FEATURES}
                surrogate={f:[] for f in sl.FEATURES}
                for r in range(N_SURR):
                    sg=bp.exact_phase(grid,810000+idx*100+si*20+r)
                    S=sl.context_rows(sg);sh=S['hidden']
                    for f in sl.FEATURES: surrogate[f].append(sl.rho(S[f],sh))
                for f in sl.FEATURES:
                    a=np.asarray(surrogate[f],float)
                    rows.append({'field':name,'fold':fold,'sample':sample,'feature':f,'real_rho':observed[f],
                                 'phase_surrogate_mean_rho':float(np.nanmean(a)),'phase_surrogate_median_rho':float(np.nanmedian(a)),
                                 'phase_surrogate_sd':float(np.nanstd(a,ddof=1)),'real_minus_phase_mean':float(observed[f]-np.nanmean(a))})
        print(f'[phase-surrogate-locality] fold {fold+1}/{nfold}: train={len(train)} test={len(test)}',flush=True)
    D=pd.DataFrame(rows);M=pd.DataFrame(model_rows);D.to_csv(out/'field_metrics.csv',index=False);M.to_csv(out/'selection_model_metrics.csv',index=False)
    comps=[]
    for sample in ['raw','residual']:
        for f in sl.FEATURES:
            g=D[(D['sample']==sample)&(D.feature==f)].set_index('field');x=g.real_minus_phase_mean.to_numpy()
            comps.append({'sample':sample,'feature':f,'real_rho_median':float(np.nanmedian(g.real_rho)),
                          'phase_surrogate_mean_rho_median':float(np.nanmedian(g.phase_surrogate_mean_rho)),
                          'real_minus_exact_power_phase_surrogate':bp.paired(x)})
    summary={'status':'REAL_DR11_SELECTION_RESIDUAL_PHASE_SURROGATE_LOCALITY','n_fields':36,'n_surrogates_per_field':N_SURR,
             'field_set':'exact PR24/PR18 selection-qualified bricks','cross_validation':f'{nfold}-fold grouped by whole brick',
             'surrogate':'exact Fourier amplitude per field; random real-field phases; map mean preserved','comparisons':comps,
             'selection_model_r2_median':float(np.nanmedian(M.selection_r2)),'selection_model_spearman_median':float(np.nanmedian(M.selection_spearman)),
             'primary':'residual local_visible real rho minus mean exact-power phase-surrogate rho',
             'predeclared_pass':'primary median>0 AND one-sided sign p<.05 AND Wilcoxon p<.05'}
    p=[x for x in comps if x['sample']=='residual' and x['feature']=='local_visible'][0]['real_minus_exact_power_phase_surrogate']
    summary['primary_decision']=('PASS_HIGHER_ORDER_LOCALITY_BEYOND_POWER' if p['median']>0 and p['sign_p_one_sided']<.05 and p['wilcoxon_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_HIGHER_ORDER_LOCALITY_BEYOND_POWER')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':provenance},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
