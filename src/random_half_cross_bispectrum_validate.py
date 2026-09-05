#!/usr/bin/env python3
"""REAL DR11 random-half cross-bispectrum control on 48 independent fields.

Each provenance-verified DR11 field is split into two disjoint equal-count random
halves with a fixed RNG. The two half catalogs are gridded independently,
rank-Gaussianized, and tested with the same symmetric mixed bispectrum and exact
Fourier-amplitude phase-null used by the PSF-vs-extended cross-tracer test.

Purpose: determine whether shared cross-tracer phase coupling is specific to a
morphology split or is a generic property shared by independent thinnings of the
same observed point field. No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_dr11 import region_grid, verify_and_load
import bispectrum_phase_validate as bp
import cross_tracer_bispectrum_validate as cb

TARGET_FIELDS = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real/dr11/expanded48')
    ap.add_argument('--out', default='results/real_dr11/random_half_cross_bispectrum48')
    args = ap.parse_args()
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = json.loads((root / 'provenance.json').read_text())
    if prov.get('status') != 'REAL_DR11' or len(prov.get('regions', [])) != TARGET_FIELDS:
        raise RuntimeError('48-field provenance-verified REAL_DR11 input required')

    rng = np.random.default_rng(20260906)
    tri = bp.build_triangles(seed=20260901, max_pairs=16000)
    rows = []
    split_prov = []

    for i, meta in enumerate(prov['regions']):
        df = verify_and_load(meta)
        order = rng.permutation(len(df))
        n = len(df) // 2
        a = df.iloc[order[:n]].copy()
        b = df.iloc[order[n:2*n]].copy()
        ga = bp.gaussianize(region_grid(a, meta))
        gb = bp.gaussianize(region_grid(b, meta))
        pa = bp.exact_phase(ga, 20267001 + i)
        pb = bp.exact_phase(gb, 20268001 + i)

        real = cb.mixed_stats(ga, gb, tri)
        n1 = cb.mixed_stats(ga, pb, tri)
        n2 = cb.mixed_stats(pa, gb, tri)
        for fam in tri:
            rows.append({'field': meta['name'], 'family': fam, 'sample': 'real', **real[fam]})
            rows.append({
                'field': meta['name'], 'family': fam, 'sample': 'phase_null',
                'phase_lock': float((n1[fam]['phase_lock'] + n2[fam]['phase_lock']) / 2),
                'bicoherence': float((n1[fam]['bicoherence'] + n2[fam]['bicoherence']) / 2),
                'signed_bicoherence': float((n1[fam]['signed_bicoherence'] + n2[fam]['signed_bicoherence']) / 2),
                'n_triangles': int(real[fam]['n_triangles']),
            })
        split_prov.append({'field': meta['name'], 'input_rows': int(len(df)), 'half_rows': int(n)})

    D = pd.DataFrame(rows)
    D.to_csv(out / 'field_metrics.csv', index=False)
    summary = {
        'status': 'REAL_DR11_RANDOM_HALF_CROSS_BISPECTRUM',
        'n_fields': TARGET_FIELDS,
        'split': 'two disjoint equal-count random halves per field; fixed RNG seed 20260906',
        'primary_family': 'all',
        'primary_metric': 'cross bicoherence minus exact-amplitude symmetric phase-null',
        'predeclared_pass': 'median difference > 0 AND one-sided sign p < .05 AND one-sided Wilcoxon p < .05',
        'families': {},
    }
    for fam, g in D.groupby('family'):
        W = g.pivot(index='field', columns='sample', values=['phase_lock', 'bicoherence', 'signed_bicoherence'])
        fs = {'n_triangles': int(g.n_triangles.iloc[0])}
        for metric in ['phase_lock', 'bicoherence', 'signed_bicoherence']:
            real = W[metric]['real']
            null = W[metric]['phase_null']
            fs[metric] = {
                'real_median': float(np.nanmedian(real)),
                'phase_null_median': float(np.nanmedian(null)),
                'real_minus_null': bp.paired(real - null),
            }
        summary['families'][fam] = fs

    p = summary['families']['all']['bicoherence']['real_minus_null']
    summary['primary_decision'] = (
        'PASS_RANDOM_HALF_SHARED_PHASE_COUPLING'
        if p['median'] > 0 and p['sign_p_one_sided'] < .05 and p['wilcoxon_p_one_sided'] < .05
        else 'FAIL_OR_UNCERTAIN_RANDOM_HALF_SHARED_PHASE_COUPLING'
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'provenance.json').write_text(json.dumps({
        'status': summary['status'], 'source_provenance': str(root / 'provenance.json'),
        'split_seed': 20260906, 'regions': split_prov,
    }, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
