#!/usr/bin/env python3
"""REAL DR11 repeated random-half cross-bispectrum stability on 48 fields.

Repeats the accepted random-half experiment for eight independent disjoint A/B
splits of every provenance-verified field.  The mixed bispectrum and exact
Fourier-amplitude symmetric phase-null are unchanged.  Inference is made on the
per-field mean effect across splits, while replicate-level summaries and
field-rank repeatability quantify split sensitivity.

Primary PASS: all-family bicoherence field-mean effect has positive median plus
one-sided sign and Wilcoxon p<.05, and at least 6/8 replicate medians are
positive. No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_dr11 import region_grid, verify_and_load
import bispectrum_phase_validate as bp
import cross_tracer_bispectrum_validate as cb

N_FIELDS = 48
N_REPS = 8


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/real/dr11/expanded48')
    ap.add_argument('--out',default='results/real_dr11/random_half_repeatability48')
    args=ap.parse_args();root=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((root/'provenance.json').read_text())
    if prov.get('status')!='REAL_DR11' or len(prov.get('regions',[]))!=N_FIELDS:
        raise RuntimeError('48-field REAL_DR11 provenance required')
    tri=bp.build_triangles(seed=20260901,max_pairs=16000)
    rows=[]
    for fi,meta in enumerate(prov['regions']):
        df=verify_and_load(meta)
        for rep in range(N_REPS):
            rng=np.random.default_rng(2026090600+rep*1000+fi)
            order=rng.permutation(len(df));n=len(df)//2
            a=df.iloc[order[:n]].copy();b=df.iloc[order[n:2*n]].copy()
            ga=bp.gaussianize(region_grid(a,meta));gb=bp.gaussianize(region_grid(b,meta))
            pa=bp.exact_phase(ga,700000+rep*1000+fi);pb=bp.exact_phase(gb,800000+rep*1000+fi)
            real=cb.mixed_stats(ga,gb,tri);n1=cb.mixed_stats(ga,pb,tri);n2=cb.mixed_stats(pa,gb,tri)
            for fam in tri:
                for metric in ['phase_lock','bicoherence','signed_bicoherence']:
                    null=float((n1[fam][metric]+n2[fam][metric])/2)
                    rv=float(real[fam][metric])
                    rows.append({'field':meta['name'],'rep':rep,'family':fam,'metric':metric,
                                 'real':rv,'phase_null':null,'effect':rv-null})
        if (fi+1)%8==0: print(f'[random-repeat] {fi+1}/{N_FIELDS} fields',flush=True)
    D=pd.DataFrame(rows);D.to_csv(out/'draw_metrics.csv',index=False)
    A=D.groupby(['field','family','metric'],as_index=False).agg(real=('real','mean'),phase_null=('phase_null','mean'),effect=('effect','mean'))
    A.to_csv(out/'field_mean_metrics.csv',index=False)
    summary={'status':'REAL_DR11_RANDOM_HALF_REPEATABILITY','n_fields':N_FIELDS,'n_reps':N_REPS,
             'primary_family':'all','primary_metric':'bicoherence','predeclared_pass':'field-mean median effect >0 AND one-sided sign p<.05 AND Wilcoxon p<.05 AND >=6/8 replicate medians >0','families':{}}
    for fam in tri:
        fs={'metrics':{}}
        for metric in ['phase_lock','bicoherence','signed_bicoherence']:
            g=D[(D.family==fam)&(D.metric==metric)]
            gm=A[(A.family==fam)&(A.metric==metric)].set_index('field')
            rep_summ=[]
            for rep in range(N_REPS):
                x=g[g.rep==rep].set_index('field').effect
                rep_summ.append({'rep':rep,**bp.paired(x.to_numpy())})
            wide=g.pivot(index='field',columns='rep',values='effect')
            cors=[]
            for i,j in combinations(range(N_REPS),2):
                r=float(spearmanr(wide[i],wide[j]).statistic);cors.append(r)
            fs['metrics'][metric]={
                'field_mean_real_median':float(np.nanmedian(gm.real)),
                'field_mean_phase_null_median':float(np.nanmedian(gm.phase_null)),
                'field_mean_effect':bp.paired(gm.effect.to_numpy()),
                'replicates':rep_summ,
                'positive_replicate_medians':int(sum(x['median']>0 for x in rep_summ)),
                'pairwise_field_rank_spearman_mean':float(np.nanmean(cors)),
                'pairwise_field_rank_spearman_median':float(np.nanmedian(cors)),
            }
        summary['families'][fam]=fs
    p=summary['families']['all']['metrics']['bicoherence']
    e=p['field_mean_effect']
    summary['primary_decision']=('PASS_ENSEMBLE_RANDOM_HALF_SHARED_PHASE_COUPLING' if e['median']>0 and e['sign_p_one_sided']<.05 and e['wilcoxon_p_one_sided']<.05 and p['positive_replicate_medians']>=6 else 'FAIL_OR_UNCERTAIN_ENSEMBLE_RANDOM_HALF_SHARED_PHASE_COUPLING')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'source_provenance':str(root/'provenance.json'),'split_seed_family':'2026090600 + rep*1000 + field_index'},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
