#!/usr/bin/env python3
"""Independent-sky redshift-slice spatial segregation test.

Motivation: in the matched-tracer redshift-view test, real fixed-z slices were
slightly *less* correlated with the full projected matched map than redshift-
shuffled slices. A redshift shuffle makes each bin a random thinning of the full
angular field, so high full-map correlation is expected under the null. This
script tests the more direct consequence of redshift localization: real z-slice
local maps should be less mutually correlated than survey/program-stratified
redshift-label shuffles.

The directional hypothesis was formulated after inspecting the earlier matched
redshift result. Therefore the decision here uses a new third independent sky
set selected without any outcome information. No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from redshift_view_validate import ZEDGES, count_grid_from_offsets, local_map, corr
from redshift_view_replication import sep, MIN_SEP_DEG, MIN_DR11_RADIAL, MIN_DESI_RADIAL
from matched_extended_redshift_validate import (
    acquire_matched, coverage_counts, select_replication, shuffle_stratified,
    MIN_BIN, N_PERM,
)

CONFIRM_SEED = 20260912
TARGET_CONFIRM = 24
MIN_MATCHED = 60


def confirm_candidates():
    pts = [(float(ra), float(dec)) for dec in [-10, 0, 10, 20, 30, 40, 50, 60]
           for ra in np.arange(0, 360, 10)]
    rng = np.random.default_rng(CONFIRM_SEED)
    rng.shuffle(pts)
    return pts


def select_confirmation(original: list[dict], prior_replication: list[dict]):
    blocked = [(float(r['center_ra_deg']), float(r['center_dec_deg'])) for r in original]
    blocked += [(float(r['ra']), float(r['dec'])) for r in prior_replication]
    accepted, trials = [], []
    for idx, (ra, dec) in enumerate(confirm_candidates()):
        if len(accepted) >= TARGET_CONFIRM:
            break
        previous = blocked + [(r['ra'], r['dec']) for r in accepted]
        if any(sep(ra, dec, a, b) < MIN_SEP_DEG for a, b in previous):
            trials.append({'candidate_index': idx, 'ra': ra, 'dec': dec, 'status': 'separation_reject'})
            continue
        nz, ni = coverage_counts(ra, dec)
        if nz < MIN_DESI_RADIAL:
            trials.append({'candidate_index': idx, 'ra': ra, 'dec': dec, 'status': 'desi_coverage_reject', 'n_desi_radial': nz})
            continue
        if ni < MIN_DR11_RADIAL:
            trials.append({'candidate_index': idx, 'ra': ra, 'dec': dec, 'status': 'dr11_coverage_reject', 'n_desi_radial': nz, 'n_dr11_radial': ni})
            continue
        name = f"c{len(accepted):02d}_ra{int(ra):03d}_{'p' if dec >= 0 else 'm'}{abs(int(dec)):02d}"
        rec = {'name': name, 'ra': float(ra), 'dec': float(dec), 'n_desi_radial': nz, 'n_dr11_radial': ni}
        accepted.append(rec)
        trials.append({'candidate_index': idx, **rec, 'status': 'accepted'})
        print(f"[z-seg-select] accept {len(accepted):02d}/{TARGET_CONFIRM} {name} DESI={nz} DR11={ni}", flush=True)
    if len(accepted) != TARGET_CONFIRM:
        raise RuntimeError(f'only {len(accepted)} confirmation centers selected')
    return accepted, trials


def pair_stat(dra, ddec, z):
    maps, counts = [], []
    for k in range(len(ZEDGES) - 1):
        lo, hi = ZEDGES[k], ZEDGES[k + 1]
        m = (z >= lo) & (z < hi)
        n = int(m.sum())
        if n < MIN_BIN:
            continue
        g = count_grid_from_offsets(dra[m], ddec[m])
        maps.append(local_map(g)); counts.append(n)
    if len(maps) < 2:
        return None
    vals, weights = [], []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            v = corr(maps[i], maps[j])
            if np.isfinite(v):
                vals.append(v)
                weights.append(float(np.sqrt(counts[i] * counts[j])))
    if not vals:
        return None
    return {'pair_corr': float(np.average(vals, weights=weights)), 'n_bins': int(len(maps)), 'n_pairs': int(len(vals))}


def full_projection_stat(dra, ddec, z):
    full = local_map(count_grid_from_offsets(dra, ddec))
    vals, weights = [], []
    for k in range(len(ZEDGES) - 1):
        lo, hi = ZEDGES[k], ZEDGES[k + 1]
        m = (z >= lo) & (z < hi)
        n = int(m.sum())
        if n < MIN_BIN:
            continue
        v = corr(full, local_map(count_grid_from_offsets(dra[m], ddec[m])))
        if np.isfinite(v):
            vals.append(v); weights.append(float(n))
    if len(vals) < 2:
        return None
    return float(np.average(vals, weights=weights))


def paired(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if not n:
        return {'n_fields': 0}
    pos = int((x > 0).sum())
    try: wp = float(wilcoxon(x, alternative='greater').pvalue)
    except Exception: wp = float('nan')
    return {'n_fields': n, 'positive_fields': pos,
            'sign_test_one_sided_p': float(binomtest(pos, n, .5, alternative='greater').pvalue),
            'wilcoxon_one_sided_p': wp,
            'mean_difference': float(np.mean(x)), 'median_difference': float(np.median(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--out', default='results/real_dr11/redshift_slice_segregation')
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = json.loads(Path(args.centers).read_text())
    if p.get('status') != 'REAL_DR11' or len(p.get('regions', [])) != 48:
        raise RuntimeError('fixed expanded48 REAL_DR11 centers required')

    prior_rep, prior_trials = select_replication(p['regions'])
    confirm, trials = select_confirmation(p['regions'], prior_rep)
    (out/'prior_replication_selection.json').write_text(json.dumps(prior_trials, indent=2, sort_keys=True)+'\n')
    (out/'confirmation_selection.json').write_text(json.dumps(trials, indent=2, sort_keys=True)+'\n')

    rng = np.random.default_rng(CONFIRM_SEED + 100000)
    rows, provenance, null_by_field = [], [], {}
    for j, c in enumerate(confirm):
        name, ra0, dec0 = c['name'], float(c['ra']), float(c['dec'])
        mi, mz, prov = acquire_matched(ra0, dec0)
        prov.update({'field': name, 'center_ra_deg': ra0, 'center_dec_deg': dec0, 'confirmation_seed': CONFIRM_SEED})
        provenance.append(prov)
        if len(mz) < MIN_MATCHED:
            rows.append({'field': name, 'status': 'low_match_n', 'n_matched': int(len(mz))}); continue
        dra = mz['_dra'].to_numpy(float); ddec = mz['_ddec'].to_numpy(float); z = mz.z.to_numpy(float)
        survey = mz.survey.fillna('NA').astype(str).to_numpy(); program = mz.program.fillna('NA').astype(str).to_numpy()
        actual = pair_stat(dra, ddec, z); actual_full = full_projection_stat(dra, ddec, z)
        if actual is None or actual_full is None:
            rows.append({'field': name, 'status': 'insufficient_bins', 'n_matched': int(len(mz))}); continue
        npair, nfull = [], []
        for _ in range(N_PERM):
            zs = shuffle_stratified(z, survey, program, rng)
            st = pair_stat(dra, ddec, zs); sf = full_projection_stat(dra, ddec, zs)
            if st is not None and sf is not None:
                npair.append(st['pair_corr']); nfull.append(sf)
        if len(npair) < N_PERM // 2:
            rows.append({'field': name, 'status': 'null_failure', 'n_matched': int(len(mz))}); continue
        npair = np.asarray(npair); nfull = np.asarray(nfull); null_by_field[name] = npair
        rows.append({
            'field': name, 'status': 'accepted', 'n_matched': int(len(mz)), 'n_bins': actual['n_bins'], 'n_pairs': actual['n_pairs'],
            'actual_pair_corr': actual['pair_corr'], 'null_pair_corr': float(npair.mean()),
            'segregation_effect': float(npair.mean() - actual['pair_corr']),
            'field_perm_p_lower': float((1 + (npair <= actual['pair_corr']).sum()) / (len(npair) + 1)),
            'actual_full_corr': actual_full, 'null_full_corr': float(nfull.mean()),
            'projection_segregation_effect': float(nfull.mean() - actual_full),
            'match_fraction': prov['desi_match_fraction'], 'median_sep_arcsec': prov['median_match_sep_arcsec'],
        })
        print(f"[z-seg] {j+1:02d}/{len(confirm)} {name} matched={len(mz)} effect={rows[-1]['segregation_effect']:+.4f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(out/'field_metrics.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    a = df[df.status.eq('accepted')].copy()
    B = min((len(null_by_field[f]) for f in a.field), default=0)
    if B and len(a):
        actual_global = float(a.actual_pair_corr.mean())
        gnull = np.asarray([np.mean([null_by_field[f][b] for f in a.field]) for b in range(B)])
        global_lower_p = float((1 + (gnull <= actual_global).sum()) / (B + 1))
    else:
        actual_global = global_lower_p = float('nan')
    primary = paired(a.segregation_effect.to_numpy(float))
    projection = paired(a.projection_segregation_effect.to_numpy(float))
    if len(a) < 12:
        decision = 'UNCERTAIN_REDSHIFT_SEGREGATION_COVERAGE'
    elif primary['median_difference'] >= 0.02 and primary['sign_test_one_sided_p'] < 0.01 and primary['wilcoxon_one_sided_p'] < 0.01 and global_lower_p < 0.01:
        decision = 'PASS_REDSHIFT_SLICE_SPATIAL_SEGREGATION'
    elif primary['median_difference'] <= 0 or primary['sign_test_one_sided_p'] >= 0.10 or primary['wilcoxon_one_sided_p'] >= 0.10 or global_lower_p >= 0.10:
        decision = 'FAIL_REDSHIFT_SLICE_SPATIAL_SEGREGATION'
    else:
        decision = 'UNCERTAIN_REDSHIFT_SLICE_SPATIAL_SEGREGATION'
    summary = {
        'status': 'REAL_DR11_DESI_DR1_MATCHED_REDSHIFT_SLICE_SEGREGATION',
        'decision': decision,
        'confirmation_set': 'new third independent sky set; coverage-only selection; >=6 deg from original expanded48 and prior independent replication centers',
        'n_accepted': int(len(a)),
        'primary_metric': 'mean stratified-null pairwise z-slice local-map correlation minus observed pairwise z-slice local-map correlation',
        'primary': primary,
        'secondary_projection_segregation': projection,
        'actual_pair_corr_median': float(a.actual_pair_corr.median()) if len(a) else float('nan'),
        'null_pair_corr_median': float(a.null_pair_corr.median()) if len(a) else float('nan'),
        'median_match_fraction': float(a.match_fraction.median()) if len(a) else float('nan'),
        'global_actual_pair_corr_mean': actual_global,
        'global_lower_tail_perm_p': global_lower_p,
        'predeclared_pass': '>=12 fields; median segregation >=0.02; one-sided sign and Wilcoxon p<0.01; global lower-tail permutation p<0.01',
        'guardrail': 'PASS would show that observed redshift labels partition the matched galaxy angular field into more spatially distinct slices than survey/program-stratified random labels. It would not establish that the original all-source hole-completion signal is cosmological, nor gravity or higher-order grammar.'
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
