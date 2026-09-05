#!/usr/bin/env python3
"""REAL DR11 selection-residualized random-half cross-bispectrum.

Uses exactly the 36 objective selection-qualified bricks accepted by the
selection-residualized morphology cross-tracer experiment.  Within each brick,
all BRICK_PRIMARY Tractor sources are split into two disjoint equal-count random
halves (A/B) with a fixed RNG.  Separate grouped-CV selection count models for A
and B use official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY
support. Raw, selection-expected and selection-residualized A/B maps are tested
with the same symmetric mixed bispectrum and exact-Fourier-amplitude phase-null.

Primary question: does generic point-field shared phase coupling survive known
pixel-level survey-selection removal when morphology is not used? No simulated
cosmology or mock catalog is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import tracer_split_native_patch as native
native.install()
import cross_tracer_selection_residual_bispectrum as cs
import bispectrum_phase_validate as bp

GRID = 64
N_FOLDS = 6
SPLIT_SEED = 20260906

# Locked from the successful PR18 provenance.  This list was determined only by
# fixed candidate order, Tractor availability, morphology availability, official
# coadd-product availability and coverage; not by any bispectrum outcome.
LOCKED = [
 ('f00_ra156_p05','1560p050'), ('f01_ra132_m05','1319m050'),
 ('f02_ra144_p15','1439p150'), ('f03_ra036_m35','0360m350'),
 ('f04_ra180_m05','1801m050'), ('f06_ra072_m05','0720m050'),
 ('f07_ra216_m05','2159m050'), ('f08_ra336_m25','3359m250'),
 ('f09_ra216_p15','2160p150'), ('f10_ra240_m35','2398m350'),
 ('f11_ra060_m35','0601m350'), ('f12_ra312_m05','3119m050'),
 ('f13_ra228_p05','2280p050'), ('f15_ra144_m25','1440m250'),
 ('f17_ra048_m15','0479m150'), ('f19_ra132_p25','1319p250'),
 ('f20_ra120_p25','1198p250'), ('f22_ra036_p05','0359p050'),
 ('f23_ra012_m35','0120m350'), ('f24_ra012_m55','0119m550'),
 ('f25_ra228_m35','2279m350'), ('f26_ra312_p05','3119p050'),
 ('f27_ra060_m05','0600m050'), ('f28_ra192_p05','1919p050'),
 ('f29_ra312_m25','3119m250'), ('f30_ra000_m45','0001m450'),
 ('f31_ra132_p05','1319p050'), ('f32_ra024_m35','0239m350'),
 ('f33_ra348_p15','3479p150'), ('f34_ra000_m15','3598m150'),
 ('f35_ra168_m25','1680m250'), ('f36_ra168_m45','1678m450'),
 ('f37_ra336_m55','3359m550'), ('f39_ra084_m55','0839m550'),
 ('f40_ra192_m35','1920m350'), ('f41_ra348_m25','3480m250'),
]


def acquire():
    data, provenance = {}, []
    for idx, (name, brick) in enumerate(LOCKED):
        print(f'[random-half-selection] acquire {idx+1}/{len(LOCKED)} {name}->{brick}', flush=True)
        tractor, tractor_prov = cs.tr.get_tractor(brick)
        if len(tractor) < 4000:
            raise RuntimeError(f'too few all-primary sources in locked brick {brick}: {len(tractor)}')
        rng = np.random.default_rng(SPLIT_SEED + idx)
        order = rng.permutation(len(tractor))
        n = len(tractor) // 2
        a = tractor.iloc[order[:n]].copy()
        b = tractor.iloc[order[n:2*n]].copy()
        ga = cs.tr.grid_from(a).astype(float)
        gb = cs.tr.grid_from(b).astype(float)
        sel, valid, feature_names, product_prov, n_patches = cs.build_selection_features(brick)
        data[name] = {
            'brick': brick, 'a': ga, 'b': gb, 'sel': sel, 'valid': valid,
            'feature_names': feature_names,
        }
        provenance.append({
            'field': name, 'brick': brick, 'tractor': tractor_prov,
            'input_rows': int(len(tractor)), 'half_rows': int(n),
            'valid_cell_fraction': float(valid.mean()),
            'n_coverage_patches': int(n_patches), 'selection_products': product_prov,
        })
    return data, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/real_dr11/random_half_selection_residual36')
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = acquire()
    names = list(data)
    if len(names) != 36:
        raise RuntimeError(f'locked field count changed: {len(names)}')
    if len({tuple(data[n]['feature_names']) for n in names}) != 1:
        raise RuntimeError('selection feature mismatch')

    nfold = max(2, min(args.folds, len(names)))
    fold_id = {name: i % nfold for i, name in enumerate(names)}
    tri = bp.build_triangles(seed=20260901, max_pairs=16000)
    rows, model_rows = [], []

    for fold in range(nfold):
        test_names = [n for n in names if fold_id[n] == fold]
        train_names = [n for n in names if fold_id[n] != fold]
        models = {
            'a': cs.fit_selection_model(data, train_names, 'a', 9100 + fold),
            'b': cs.fit_selection_model(data, train_names, 'b', 10100 + fold),
        }
        for name in test_names:
            d = data[name]
            valid = d['valid']
            maps = {}
            metrics = {'field': name, 'fold': fold}
            idx = names.index(name)
            for ti, tracer in enumerate(['a', 'b']):
                counts = d[tracer]
                mean = float(np.mean(counts[valid])) + 1e-6
                pred_rel = np.full((GRID, GRID), np.nan, float)
                pred_rel[valid] = np.clip(models[tracer].predict(d['sel'][valid]), 0.03, 20)
                expected = pred_rel * mean
                residual = np.zeros((GRID, GRID), float)
                residual[valid] = counts[valid] - expected[valid]
                true_rel = counts[valid] / mean
                metrics[f'{tracer}_selection_r2'] = float(r2_score(true_rel, pred_rel[valid]))
                metrics[f'{tracer}_selection_spearman'] = float(spearmanr(true_rel, pred_rel[valid]).statistic)
                maps[(tracer, 'raw')] = bp.gaussianize(cs.fill_invalid(counts, valid, 410000 + idx*20 + ti))
                maps[(tracer, 'selection')] = bp.gaussianize(cs.fill_invalid(expected, valid, 420000 + idx*20 + ti))
                maps[(tracer, 'residual')] = bp.gaussianize(cs.fill_invalid(residual, valid, 430000 + idx*20 + ti))
            model_rows.append(metrics)

            for si, sample in enumerate(['raw', 'selection', 'residual']):
                real, null = cs.cross_stats_with_null(
                    maps[('a', sample)], maps[('b', sample)], tri,
                    500000 + idx*30 + si*2, 500001 + idx*30 + si*2,
                )
                for fam in tri:
                    rows.append({'field': name, 'fold': fold, 'family': fam, 'sample': sample, 'control': 'real', **real[fam]})
                    rows.append({'field': name, 'fold': fold, 'family': fam, 'sample': sample, 'control': 'phase_null', **null[fam]})
        print(f'[random-half-selection] fold {fold+1}/{nfold}: train={len(train_names)} test={len(test_names)}', flush=True)

    D = pd.DataFrame(rows)
    M = pd.DataFrame(model_rows)
    D.to_csv(out / 'field_metrics.csv', index=False)
    M.to_csv(out / 'selection_model_metrics.csv', index=False)

    summary = {
        'status': 'REAL_DR11_SELECTION_RESIDUAL_RANDOM_HALF_CROSS_BISPECTRUM',
        'n_fields': 36,
        'field_set': 'exact locked PR18 selection-qualified bricks',
        'split': 'all BRICK_PRIMARY Tractor sources -> two disjoint equal-count random halves; fixed seed',
        'cross_validation': f'{nfold}-fold grouped by whole brick',
        'selection_features': 'official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY support',
        'primary_family': 'all',
        'primary_metric': 'residual random-half cross bicoherence minus exact-amplitude phase-null',
        'predeclared_pass': 'median difference >0 AND one-sided sign p<.05 AND one-sided Wilcoxon p<.05',
        'selection_model': {
            'a_r2_median': float(np.nanmedian(M.a_selection_r2)),
            'a_spearman_median': float(np.nanmedian(M.a_selection_spearman)),
            'b_r2_median': float(np.nanmedian(M.b_selection_r2)),
            'b_spearman_median': float(np.nanmedian(M.b_selection_spearman)),
        },
        'families': {},
    }

    for fam, g in D.groupby('family'):
        fs = {'n_triangles': int(g.n_triangles.iloc[0])}
        for metric in ['phase_lock', 'bicoherence', 'signed_bicoherence']:
            tab = g.pivot(index='field', columns=['sample', 'control'], values=metric)
            rec = {}
            for sample in ['raw', 'selection', 'residual']:
                real = tab[(sample, 'real')]
                null = tab[(sample, 'phase_null')]
                rec[f'{sample}_real_median'] = float(np.nanmedian(real))
                rec[f'{sample}_phase_null_median'] = float(np.nanmedian(null))
                rec[f'{sample}_minus_phase'] = bp.paired(real - null)
            rec['residual_minus_raw'] = bp.paired(tab[('residual','real')] - tab[('raw','real')])
            fs[metric] = rec
        summary['families'][fam] = fs

    p = summary['families']['all']['bicoherence']['residual_minus_phase']
    summary['primary_decision'] = (
        'PASS_SELECTION_RESISTANT_RANDOM_HALF_SHARED_PHASE_COUPLING'
        if p['median'] > 0 and p['sign_p_one_sided'] < .05 and p['wilcoxon_p_one_sided'] < .05
        else 'FAIL_OR_UNCERTAIN_SELECTION_RESISTANT_RANDOM_HALF_SHARED_PHASE_COUPLING'
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'provenance.json').write_text(json.dumps({
        'status': summary['status'], 'split_seed': SPLIT_SEED, 'regions': provenance,
    }, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
