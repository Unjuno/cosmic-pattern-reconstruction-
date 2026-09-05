#!/usr/bin/env python3
"""REAL DR11 fixed-count PSF-vs-extended cross-bispectrum stress.

Replicates the accepted morphology cross-tracer test while fixing each accepted
field to exactly 1000 PSF and 1000 extended (REX/EXP/DEV/SER) sources. This
removes field-to-field tracer-count variation as a signal-strength confound.
The fixed candidate order, objective availability gate, mixed-bispectrum
statistic, and exact Fourier-amplitude symmetric phase-null are unchanged.
No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bispectrum_phase_validate as bp
import cross_tracer_bispectrum_validate as cb
import tracer_split_validate as tr

TARGET_FIELDS = 36
MAX_CANDIDATES = 48
FIXED_N = 1000


def build_data(centers):
    regs = centers.get('regions', [])[:MAX_CANDIDATES]
    data, prov, rejected = {}, [], []
    used = set()
    rng = np.random.default_rng(20260906)
    for j, r in enumerate(regs):
        if len(data) >= TARGET_FIELDS:
            break
        name = r['name']
        ra = float(r['center_ra_deg'])
        dec = float(r['center_dec_deg'])
        try:
            brick, bq = tr.choose_brick(ra, dec)
        except Exception as e:
            rejected.append({'field': name, 'reason': 'brick_resolution', 'error': str(e)})
            continue
        if brick in used:
            rejected.append({'field': name, 'brick': brick, 'reason': 'duplicate'})
            continue
        used.add(brick)
        print(f'[fixed-cross] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}', flush=True)
        try:
            d, p = tr.get_tractor(brick)
        except Exception as e:
            rejected.append({'field': name, 'brick': brick, 'reason': 'tractor_acquisition', 'error': str(e)})
            continue
        psf = d[d.type.eq('PSF')].copy()
        ext = d[d.type.isin(tr.EXT_TYPES)].copy()
        if len(psf) < FIXED_N or len(ext) < FIXED_N:
            rejected.append({'field': name, 'brick': brick, 'reason': 'tracer_availability',
                             'n_psf': int(len(psf)), 'n_extended': int(len(ext))})
            continue
        psf_eq = psf.iloc[rng.choice(len(psf), FIXED_N, replace=False)]
        ext_eq = ext.iloc[rng.choice(len(ext), FIXED_N, replace=False)]
        gp = bp.gaussianize(tr.grid_from(psf_eq))
        ge = bp.gaussianize(tr.grid_from(ext_eq))
        data[name] = {'psf': gp, 'ext': ge}
        prov.append({'field': name, 'brick': brick, 'brick_choice_query': bq, 'tractor': p,
                     'n_psf': int(len(psf)), 'n_extended': int(len(ext)), 'fixed_n': FIXED_N})
        print(f'[fixed-cross] accept {len(data)}/{TARGET_FIELDS}: n={FIXED_N}', flush=True)
    return data, prov, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--out', default='results/real_dr11/fixed_count_cross_bispectrum36')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C = json.loads(Path(args.centers).read_text())
    if C.get('status') != 'REAL_DR11':
        raise RuntimeError('REAL_DR11 centers required')
    data, prov, rejected = build_data(C)
    if len(data) != TARGET_FIELDS:
        (out / 'availability_rejections.json').write_text(json.dumps(rejected, indent=2, sort_keys=True) + '\n')
        raise RuntimeError(f'only {len(data)} accepted')

    tri = bp.build_triangles(seed=20260901, max_pairs=16000)
    rows = []
    for i, (name, d) in enumerate(data.items()):
        psf, ext = d['psf'], d['ext']
        psf_phase = bp.exact_phase(psf, 20269001 + i)
        ext_phase = bp.exact_phase(ext, 20270001 + i)
        real = cb.mixed_stats(psf, ext, tri)
        n1 = cb.mixed_stats(psf, ext_phase, tri)
        n2 = cb.mixed_stats(psf_phase, ext, tri)
        for fam in tri:
            rows.append({'field': name, 'family': fam, 'sample': 'real', **real[fam]})
            rows.append({
                'field': name, 'family': fam, 'sample': 'phase_null',
                'phase_lock': float((n1[fam]['phase_lock'] + n2[fam]['phase_lock']) / 2),
                'bicoherence': float((n1[fam]['bicoherence'] + n2[fam]['bicoherence']) / 2),
                'signed_bicoherence': float((n1[fam]['signed_bicoherence'] + n2[fam]['signed_bicoherence']) / 2),
                'n_triangles': int(real[fam]['n_triangles']),
            })

    D = pd.DataFrame(rows)
    D.to_csv(out / 'field_metrics.csv', index=False)
    summary = {
        'status': 'REAL_DR11_FIXED_COUNT_CROSS_TRACER_BISPECTRUM',
        'n_fields': TARGET_FIELDS,
        'fixed_sources_per_tracer_per_field': FIXED_N,
        'candidate_protocol': 'same fixed-order objective availability gate as accepted morphology cross-tracer test',
        'primary_family': 'all',
        'primary_metric': 'cross bicoherence minus exact-amplitude symmetric phase-null',
        'predeclared_pass': 'median difference > 0 AND one-sided sign p < .05 AND one-sided Wilcoxon p < .05',
        'availability_rejections': rejected,
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
        'PASS_FIXED_COUNT_SHARED_CROSS_TRACER_PHASE_COUPLING'
        if p['median'] > 0 and p['sign_p_one_sided'] < .05 and p['wilcoxon_p_one_sided'] < .05
        else 'FAIL_OR_UNCERTAIN_FIXED_COUNT_SHARED_CROSS_TRACER_PHASE_COUPLING'
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'provenance.json').write_text(json.dumps({'status': summary['status'], 'regions': prov, 'rejections': rejected}, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
