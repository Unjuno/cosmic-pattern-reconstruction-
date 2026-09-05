#!/usr/bin/env python3
"""REAL DR11 sampling-density scaling of shared PSF-vs-extended bicoherence.

Selects a fixed high-density subset from the preregistered first-48 candidate
order using only tracer availability: the first 33 unique bricks with at least
2000 PSF and 2000 extended (REX/EXP/DEV/SER) primary Tractor sources.  On this
same field set, evaluate N={500,1000,1500,2000} sources per tracer with four
independent fixed-seed subsamples.  Per field and N, average the real-minus-
exact-amplitude-phase-null bicoherence across the four draws before inference.

Primary scaling test: fit a linear slope of field-level all-family effect versus
log(N) for every field, then require positive median slope and one-sided sign and
Wilcoxon p<.05. This distinguishes a genuine sampling/SNR threshold from a
field-count confound. No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bispectrum_phase_validate as bp
import cross_tracer_bispectrum_validate as cb
import tracer_split_native_patch as native
native.install()
import tracer_split_validate as tr

MAX_CANDIDATES = 48
TARGET_FIELDS = 33
MIN_AVAILABLE = 2000
N_LEVELS = [500, 1000, 1500, 2000]
N_REPS = 4


def acquire(centers):
    regs = centers.get('regions', [])[:MAX_CANDIDATES]
    data, prov, rejected = {}, [], []
    used = set()
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
        print(f'[sampling-scaling] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}', flush=True)
        try:
            d, p = tr.get_tractor(brick)
        except Exception as e:
            rejected.append({'field': name, 'brick': brick, 'reason': 'tractor_acquisition', 'error': str(e)})
            continue
        psf = d[d.type.eq('PSF')].copy()
        ext = d[d.type.isin(tr.EXT_TYPES)].copy()
        if len(psf) < MIN_AVAILABLE or len(ext) < MIN_AVAILABLE:
            rejected.append({'field': name, 'brick': brick, 'reason': 'high_density_gate',
                             'n_psf': int(len(psf)), 'n_extended': int(len(ext))})
            continue
        data[name] = {'brick': brick, 'psf': psf, 'ext': ext}
        prov.append({'field': name, 'brick': brick, 'brick_choice_query': bq, 'tractor': p,
                     'n_psf': int(len(psf)), 'n_extended': int(len(ext))})
        print(f'[sampling-scaling] accept {len(data)}/{TARGET_FIELDS}: psf={len(psf)} ext={len(ext)}', flush=True)
    return data, prov, rejected


def symmetric_null(a, b, tri, seed_a, seed_b):
    pa = bp.exact_phase(a, seed_a)
    pb = bp.exact_phase(b, seed_b)
    n1 = cb.mixed_stats(a, pb, tri)
    n2 = cb.mixed_stats(pa, b, tri)
    out = {}
    for fam in tri:
        out[fam] = {
            'bicoherence': float((n1[fam]['bicoherence'] + n2[fam]['bicoherence']) / 2),
            'phase_lock': float((n1[fam]['phase_lock'] + n2[fam]['phase_lock']) / 2),
            'signed_bicoherence': float((n1[fam]['signed_bicoherence'] + n2[fam]['signed_bicoherence']) / 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--out', default='results/real_dr11/cross_bispectrum_sampling_scaling33')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    centers = json.loads(Path(args.centers).read_text())
    if centers.get('status') != 'REAL_DR11':
        raise RuntimeError('REAL_DR11 centers required')
    data, prov, rejected = acquire(centers)
    if len(data) != TARGET_FIELDS:
        (out / 'availability_rejections.json').write_text(json.dumps(rejected, indent=2, sort_keys=True) + '\n')
        raise RuntimeError(f'only {len(data)} high-density fields; expected {TARGET_FIELDS}')

    tri = bp.build_triangles(seed=20260901, max_pairs=16000)
    rows = []
    names = list(data)
    for fi, name in enumerate(names):
        d = data[name]
        for N in N_LEVELS:
            for rep in range(N_REPS):
                rng = np.random.default_rng(2026090600 + fi * 100 + N + rep * 10000)
                psf = d['psf'].iloc[rng.choice(len(d['psf']), N, replace=False)]
                ext = d['ext'].iloc[rng.choice(len(d['ext']), N, replace=False)]
                gp = bp.gaussianize(tr.grid_from(psf))
                ge = bp.gaussianize(tr.grid_from(ext))
                real = cb.mixed_stats(gp, ge, tri)
                null = symmetric_null(gp, ge, tri,
                                      3000000 + fi*1000 + N + rep*2,
                                      3000001 + fi*1000 + N + rep*2)
                for fam in tri:
                    for metric in ['phase_lock', 'bicoherence', 'signed_bicoherence']:
                        rows.append({
                            'field': name, 'N': N, 'rep': rep, 'family': fam, 'metric': metric,
                            'real': float(real[fam][metric]), 'phase_null': float(null[fam][metric]),
                            'effect': float(real[fam][metric] - null[fam][metric]),
                        })
        print(f'[sampling-scaling] evaluated {fi+1}/{TARGET_FIELDS}', flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(out / 'draw_metrics.csv', index=False)
    A = D.groupby(['field', 'N', 'family', 'metric'], as_index=False).agg(
        real=('real', 'mean'), phase_null=('phase_null', 'mean'), effect=('effect', 'mean'))
    A.to_csv(out / 'field_level_metrics.csv', index=False)

    summary = {
        'status': 'REAL_DR11_CROSS_BISPECTRUM_SAMPLING_SCALING',
        'n_fields': TARGET_FIELDS,
        'availability_gate': f'first {TARGET_FIELDS} fixed-order unique bricks with >= {MIN_AVAILABLE} sources in each morphology tracer',
        'N_levels': N_LEVELS,
        'replicates_per_N': N_REPS,
        'primary_family': 'all',
        'primary_metric': 'bicoherence real-minus symmetric exact-amplitude phase-null',
        'predeclared_scaling_pass': 'field-level effect-vs-logN slope median >0 AND one-sided sign p<.05 AND Wilcoxon p<.05',
        'availability_rejections': rejected,
        'families': {},
    }

    for fam in tri:
        sf = {'metrics': {}}
        for metric in ['phase_lock', 'bicoherence', 'signed_bicoherence']:
            g = A[(A.family == fam) & (A.metric == metric)]
            level = {}
            for N in N_LEVELS:
                x = g[g.N == N].set_index('field')
                level[str(N)] = {
                    'real_median': float(np.nanmedian(x.real)),
                    'phase_null_median': float(np.nanmedian(x.phase_null)),
                    'effect': bp.paired(x.effect.to_numpy()),
                }
            slopes = []
            for name in names:
                x = g[g.field == name].sort_values('N')
                slopes.append(float(np.polyfit(np.log(x.N.to_numpy(float)), x.effect.to_numpy(float), 1)[0]))
            sf['metrics'][metric] = {'levels': level, 'logN_slope': bp.paired(np.asarray(slopes))}
        summary['families'][fam] = sf

    p = summary['families']['all']['metrics']['bicoherence']['logN_slope']
    summary['primary_decision'] = (
        'PASS_POSITIVE_SAMPLING_DENSITY_SCALING'
        if p['median'] > 0 and p['sign_p_one_sided'] < .05 and p['wilcoxon_p_one_sided'] < .05
        else 'FAIL_OR_UNCERTAIN_SAMPLING_DENSITY_SCALING'
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'provenance.json').write_text(json.dumps({'status': summary['status'], 'regions': prov, 'rejections': rejected}, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
