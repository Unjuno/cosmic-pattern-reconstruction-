#!/usr/bin/env python3
"""REAL DR11 imaging vs DESI DR1 redshift-slice cross-view validation.

Science inputs:
- immutable observed Legacy Surveys DR11 RA/Dec catalogs for the fixed 48 fields;
- observed DESI DR1 zpix galaxy redshifts queried from Astro Data Lab.

No simulated cosmology is used. The primary null permutes redshift values among
DESI objects within each field, preserving the exact angular spectroscopic
sampling/target-selection pattern while destroying redshift-localized structure.
"""
from __future__ import annotations

import argparse, hashlib, io, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from dl import queryClient as qc
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, wilcoxon

from analyze_dr11 import verify_and_load

GRID = 32
HALF = 0.25
QUERY_RADIUS = 0.40
ZEDGES = np.array([0.05, 0.40, 0.80, 1.20, 1.80], float)
MIN_TOTAL = 80
MIN_BIN = 15
N_PERM = 200
SEED = 20260830
TABLE = 'desi_dr1.zpix'


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def query(sql: str, attempts: int = 4) -> str:
    last = None
    for i in range(attempts):
        try:
            out = qc.query(sql=sql, fmt='csv', async_=False)
            if isinstance(out, bytes):
                out = out.decode('utf-8')
            if not isinstance(out, str):
                raise RuntimeError(f'unexpected query result {type(out)}')
            return out
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(4 * (i + 1))
    raise RuntimeError(f'Data Lab query failed: {last}')


def tangent_ra(ra, ra0):
    return ((np.asarray(ra, float) - ra0 + 180.0) % 360.0) - 180.0


def square_clip(df: pd.DataFrame, ra0: float, dec0: float, ra_col: str, dec_col: str) -> pd.DataFrame:
    dra = tangent_ra(df[ra_col].to_numpy(float), ra0)
    ddec = df[dec_col].to_numpy(float) - dec0
    keep = (np.abs(dra) < HALF) & (np.abs(ddec) < HALF)
    out = df.loc[keep].copy()
    out['_dra'] = dra[keep]
    out['_ddec'] = ddec[keep]
    return out


def count_grid_from_offsets(dra: np.ndarray, ddec: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(ddec, dra, bins=GRID, range=[[-HALF, HALF], [-HALF, HALF]])
    return h.astype(float)


def standardized(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, float)
    x = x - np.mean(x)
    s = np.std(x)
    if not np.isfinite(s) or s < 1e-10:
        return np.zeros_like(x)
    return x / s


def local_map(counts: np.ndarray) -> np.ndarray:
    x = np.log1p(np.asarray(counts, float))
    return standardized(gaussian_filter(x, 1.2, mode='reflect') - gaussian_filter(x, 5.0, mode='reflect'))


def broad_map(counts: np.ndarray) -> np.ndarray:
    x = np.log1p(np.asarray(counts, float))
    return standardized(gaussian_filter(x, 5.0, mode='reflect'))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if np.std(aa) < 1e-10 or np.std(bb) < 1e-10:
        return float('nan')
    return float(np.corrcoef(aa, bb)[0, 1])


def slice_stat(img_local, img_broad, dra, ddec, z):
    rows = []
    for k in range(len(ZEDGES) - 1):
        lo, hi = ZEDGES[k], ZEDGES[k + 1]
        m = (z >= lo) & (z < hi)
        n = int(m.sum())
        if n < MIN_BIN:
            continue
        g = count_grid_from_offsets(dra[m], ddec[m])
        rows.append({'zlo': float(lo), 'zhi': float(hi), 'n': n,
                     'local_corr': corr(img_local, local_map(g)),
                     'broad_corr': corr(img_broad, broad_map(g))})
    if len(rows) < 2:
        return None, rows
    w = np.array([r['n'] for r in rows], float)
    lc = np.array([r['local_corr'] for r in rows], float)
    bc = np.array([r['broad_corr'] for r in rows], float)
    good_l = np.isfinite(lc); good_b = np.isfinite(bc)
    if good_l.sum() < 2:
        return None, rows
    local = float(np.average(lc[good_l], weights=w[good_l]))
    broad = float(np.average(bc[good_b], weights=w[good_b])) if good_b.any() else float('nan')
    return {'local': local, 'broad': broad, 'max_local': float(np.nanmax(lc))}, rows


def paired_summary(diff):
    x = np.asarray(diff, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return {'n_fields': 0}
    pos = int((x > 0).sum())
    try:
        wp = float(wilcoxon(x, alternative='greater').pvalue)
    except Exception:
        wp = float('nan')
    return {'n_fields': n, 'positive_fields': pos,
            'sign_test_one_sided_p': float(binomtest(pos, n, .5, alternative='greater').pvalue),
            'wilcoxon_one_sided_p': wp,
            'mean_difference': float(np.mean(x)), 'median_difference': float(np.median(x))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--out', default='results/real_dr11/redshift_view48')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    prov = json.loads(Path(args.centers).read_text())
    if prov.get('status') != 'REAL_DR11' or prov.get('model_input_columns') != ['ra', 'dec'] or len(prov.get('regions', [])) != 48:
        raise RuntimeError('immutable 48-field REAL_DR11 RA/Dec provenance required')

    rng = np.random.default_rng(SEED)
    field_rows, provenance, null_arrays = [], [], []

    for j, meta in enumerate(prov['regions']):
        name = meta['name']; ra0 = float(meta['center_ra_deg']); dec0 = float(meta['center_dec_deg'])
        imaging = verify_and_load(meta)
        im = square_clip(imaging, ra0, dec0, 'ra', 'dec')
        img_counts = count_grid_from_offsets(im['_dra'].to_numpy(), im['_ddec'].to_numpy())
        img_local, img_broad = local_map(img_counts), broad_map(img_counts)

        sql = (
            "SELECT mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zwarn, spectype, zcat_primary "
            f"FROM {TABLE} WHERE zwarn=0 AND zcat_primary=1 AND spectype='GALAXY' "
            f"AND z>={ZEDGES[0]:.4f} AND z<{ZEDGES[-1]:.4f} "
            f"AND q3c_radial_query(mean_fiber_ra,mean_fiber_dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS:.8f})"
        )
        raw = query(sql)
        desi = pd.read_csv(io.StringIO(raw))
        desi.columns = [str(c).lower() for c in desi.columns]
        if not {'ra', 'dec', 'z'}.issubset(desi.columns):
            raise RuntimeError(f'{name}: missing DESI columns')
        desi = square_clip(desi, ra0, dec0, 'ra', 'dec')
        desi['z'] = pd.to_numeric(desi['z'], errors='coerce')
        desi = desi[np.isfinite(desi.z) & (desi.z >= ZEDGES[0]) & (desi.z < ZEDGES[-1])]
        desi = desi.sort_values(['ra','dec','z'], kind='mergesort').reset_index(drop=True)
        canonical = desi[['ra','dec','z']].to_csv(index=False, lineterminator='\n').encode()

        rec = {'field': name, 'center_ra_deg': ra0, 'center_dec_deg': dec0,
               'sql': sql, 'rows_square': int(len(desi)), 'canonical_sha256': sha256(canonical),
               'zbin_counts': [int(((desi.z >= ZEDGES[k]) & (desi.z < ZEDGES[k+1])).sum()) for k in range(len(ZEDGES)-1)]}
        provenance.append(rec)
        print(f"[redshift-view] {j+1}/48 {name}: {len(desi)} useful DESI galaxies {rec['zbin_counts']}", flush=True)

        if len(desi) < MIN_TOTAL:
            field_rows.append({'field': name, 'status': 'rejected_low_n', 'n_desi': int(len(desi))})
            continue

        dra = desi['_dra'].to_numpy(float); ddec = desi['_ddec'].to_numpy(float); z = desi.z.to_numpy(float)
        actual, slices = slice_stat(img_local, img_broad, dra, ddec, z)
        if actual is None:
            field_rows.append({'field': name, 'status': 'rejected_bins', 'n_desi': int(len(desi))})
            continue

        null_local = np.empty(N_PERM, float); null_broad = np.empty(N_PERM, float); null_max = np.empty(N_PERM, float)
        for b in range(N_PERM):
            st, _ = slice_stat(img_local, img_broad, dra, ddec, rng.permutation(z))
            if st is None:
                null_local[b] = null_broad[b] = null_max[b] = np.nan
            else:
                null_local[b], null_broad[b], null_max[b] = st['local'], st['broad'], st['max_local']
        nl = null_local[np.isfinite(null_local)]; nb = null_broad[np.isfinite(null_broad)]; nm = null_max[np.isfinite(null_max)]
        if len(nl) < N_PERM // 2:
            field_rows.append({'field': name, 'status': 'rejected_null', 'n_desi': int(len(desi))})
            continue

        field_rows.append({'field': name, 'status': 'accepted', 'n_desi': int(len(desi)), 'n_slices': int(len(slices)),
                           'actual_local': actual['local'], 'null_local_mean': float(np.mean(nl)),
                           'delta_local': actual['local'] - float(np.mean(nl)),
                           'field_perm_p_local': float((1 + np.sum(nl >= actual['local'])) / (len(nl) + 1)),
                           'actual_broad': actual['broad'], 'null_broad_mean': float(np.mean(nb)),
                           'delta_broad': actual['broad'] - float(np.mean(nb)),
                           'actual_max_local': actual['max_local'], 'null_max_mean': float(np.mean(nm)),
                           'delta_max_local': actual['max_local'] - float(np.mean(nm)),
                           'slice_details_json': json.dumps(slices, sort_keys=True)})
        null_arrays.append({'field': name, 'local': nl.tolist(), 'broad': nb.tolist(), 'max_local': nm.tolist()})

    fdf = pd.DataFrame(field_rows); fdf.to_csv(out/'field_metrics.csv', index=False)
    accepted = fdf[fdf.status == 'accepted'].copy()
    if len(accepted) < 8:
        raise RuntimeError(f'too few DESI-covered fields: {len(accepted)}')

    null_by_field = {r['field']: r for r in null_arrays}
    B = min(len(null_by_field[f]['local']) for f in accepted.field)
    actual_global = float(accepted.actual_local.mean())
    global_null = np.array([np.mean([null_by_field[f]['local'][b] for f in accepted.field]) for b in range(B)])
    summary = {
        'status': 'REAL_DR11_DESI_DR1_REDSHIFT_VIEW',
        'dataset_imaging': 'DESI Legacy Imaging Surveys DR11 observed RA/Dec',
        'dataset_spectroscopy': 'DESI DR1 zpix, ZWARN=0, ZCAT_PRIMARY=1, SPECTYPE=GALAXY',
        'fixed_candidate_fields': 48, 'accepted_fields': int(len(accepted)), 'rejected_fields': int(len(fdf)-len(accepted)),
        'grid': GRID, 'z_edges': ZEDGES.tolist(),
        'null': 'within-field permutation of observed DESI redshifts; angular DESI positions unchanged',
        'local_actual_median': float(accepted.actual_local.median()),
        'local_null_median_of_field_means': float(accepted.null_local_mean.median()),
        'local_delta': paired_summary(accepted.delta_local.to_numpy()),
        'broad_delta': paired_summary(accepted.delta_broad.to_numpy()),
        'max_slice_delta': paired_summary(accepted.delta_max_local.to_numpy()),
        'global_mean_local_actual': actual_global, 'global_null_mean': float(global_null.mean()),
        'global_redshift_permutation_p': float((1 + np.sum(global_null >= actual_global)) / (B + 1))}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out/'provenance.json').write_text(json.dumps({'status': summary['status'],
        'imaging_provenance_sha256': sha256(Path(args.centers).read_bytes()), 'queries': provenance,
        'z_edges': ZEDGES.tolist(), 'seed': SEED, 'n_permutations': N_PERM}, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
