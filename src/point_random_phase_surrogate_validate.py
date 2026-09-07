#!/usr/bin/env python3
"""Exact-power phase-surrogate locality after point-random selection residualization.

Primary test is preregistered in docs/POINT_RANDOM_PHASE_SURROGATE.md.
No simulated cosmological field is used. Exact-power surrogates are statistical
controls generated from each held-out observed residual map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_dr11 as adr
import bispectrum_phase_validate as bp
import locality_validate as loc
import point_random_selection_residual_validate as pr

N_SURR = 32
DENSE_FRACTION = 0.04
POWER_TOL = 1e-10
EFFECT_FLOOR = 0.05
N_FOLDS = 6


def power_fidelity(real: np.ndarray, surrogate: np.ndarray) -> tuple[float, float]:
    a = np.abs(np.fft.fft2(np.asarray(real, float)))
    b = np.abs(np.fft.fft2(np.asarray(surrogate, float)))
    peak = max(float(np.max(a)), 1e-30)
    max_rel = float(np.max(np.abs(a - b)) / peak)
    l1_rel = float(np.sum(np.abs(a - b)) / max(float(np.sum(a)), 1e-30))
    return max_rel, l1_rel


def decision(primary: dict, n_valid: int, max_power_error: float) -> str:
    if not np.isfinite(max_power_error) or max_power_error > POWER_TOL:
        return 'INVALID_POWER_FIDELITY'
    if n_valid < 30:
        return 'UNCERTAIN_HIGHER_ORDER_LOCALITY'
    med = float(primary.get('median', np.nan))
    sp = float(primary.get('sign_p_one_sided', np.nan))
    wp = float(primary.get('wilcoxon_p_one_sided', np.nan))
    if med >= EFFECT_FLOOR and sp < .01 and wp < .01:
        return 'PASS_HIGHER_ORDER_LOCALITY'
    if med <= 0 or sp >= .10 or wp >= .10:
        return 'FAIL_HIGHER_ORDER_LOCALITY'
    return 'UNCERTAIN_HIGHER_ORDER_LOCALITY'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--provenance', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--fraction', type=float, default=DENSE_FRACTION)
    ap.add_argument('--surrogates', type=int, default=N_SURR)
    ap.add_argument('--out', default='results/real_dr11/point_random_phase_surrogate36')
    args = ap.parse_args()
    if abs(args.fraction - DENSE_FRACTION) > 1e-12:
        raise RuntimeError('primary experiment is fixed to fraction=0.04')
    if args.surrogates != N_SURR:
        raise RuntimeError(f'primary experiment is fixed to {N_SURR} surrogates per field')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions = pr.load_regions(Path(args.provenance))
    points, remote_prov = pr.acquire_points(regions, args.fraction)

    data: dict[str, dict] = {}
    qc_rows = []
    feature_names0 = None
    for i, name in enumerate(pr.LOCKED_FIELDS):
        m = regions[name]
        src = adr.verify_and_load(m)
        counts = adr.region_grid(src, m).astype(float)
        sel, valid, q, names = pr.selection_grid(points[name], m)
        if feature_names0 is None:
            feature_names0 = names
        elif names != feature_names0:
            raise RuntimeError('point-random feature-name mismatch')
        q.update({'field': name, 'source_rows': int(len(src))})
        qc_rows.append(q)
        if q['valid_cell_fraction'] >= .85:
            data[name] = {'counts': counts, 'sel': sel, 'valid': valid}
        print(
            f'[point-phase] field {i+1:02d}/36 {name} '
            f'randoms={len(points[name])} valid={valid.mean():.4f}',
            flush=True,
        )
    pd.DataFrame(qc_rows).to_csv(out / 'point_random_qc.csv', index=False)
    valid_names = [n for n in pr.LOCKED_FIELDS if n in data]

    fold_id = {n: i % N_FOLDS for i, n in enumerate(valid_names)}
    field_rows = []
    surrogate_rows = []
    model_rows = []

    for fold in range(N_FOLDS):
        test = [n for n in valid_names if fold_id[n] == fold]
        train = [n for n in valid_names if fold_id[n] != fold]
        if not test or len(train) < 20:
            continue
        model = pr.fit_model(data, train, 45000 + fold)
        for name in test:
            d = data[name]
            v = d['valid']
            counts = d['counts']
            mu = float(np.mean(counts[v])) + 1e-6
            pred_rel = np.full((pr.GRID, pr.GRID), np.nan, float)
            pred_rel[v] = np.clip(model.predict(d['sel'][v]), .03, 20)
            true_rel = counts[v] / mu
            from sklearn.metrics import r2_score
            model_rows.append({
                'field': name,
                'fold': fold,
                'selection_r2': float(r2_score(true_rel, pred_rel[v])),
                'selection_spearman': pr.rho(true_rel, pred_rel[v]),
            })

            expected = pred_rel * mu
            idx = pr.LOCKED_FIELDS.index(name)
            _, residual = pr.normalized_maps(counts, expected, v, 910000 + idx * 4)
            hidden, visible = pr.context_local(residual)
            real_rho = pr.rho(visible, hidden)
            shift_rho = pr.rho(loc.shifted_feature(visible), hidden)

            srhos = []
            for si in range(args.surrogates):
                seed = 2026090800 + idx * 100 + si
                sg = bp.exact_phase(residual, seed)
                sh, sv = pr.context_local(sg)
                srho = pr.rho(sv, sh)
                max_rel, l1_rel = power_fidelity(residual, sg)
                mean_abs_error = float(abs(np.mean(sg) - np.mean(residual)))
                srhos.append(srho)
                surrogate_rows.append({
                    'field': name,
                    'fold': fold,
                    'surrogate': si,
                    'seed': seed,
                    'rho': srho,
                    'power_max_rel_error': max_rel,
                    'power_l1_rel_error': l1_rel,
                    'mean_abs_error': mean_abs_error,
                })

            a = np.asarray(srhos, float)
            field_rows.append({
                'field': name,
                'fold': fold,
                'real_residual_rho': real_rho,
                'matched_shift_rho': shift_rho,
                'phase_surrogate_mean_rho': float(np.nanmean(a)),
                'phase_surrogate_median_rho': float(np.nanmedian(a)),
                'phase_surrogate_sd': float(np.nanstd(a, ddof=1)),
                'real_minus_phase_mean': float(real_rho - np.nanmean(a)),
                'real_minus_matched_shift': float(real_rho - shift_rho),
            })
        print(f'[point-phase] fold {fold+1}/{N_FOLDS} train={len(train)} test={len(test)}', flush=True)

    F = pd.DataFrame(field_rows)
    S = pd.DataFrame(surrogate_rows)
    M = pd.DataFrame(model_rows)
    F.to_csv(out / 'field_metrics.csv', index=False)
    S.to_csv(out / 'surrogate_qc.csv', index=False)
    M.to_csv(out / 'selection_model_metrics.csv', index=False)

    primary = bp.paired(F.real_minus_phase_mean.to_numpy(float))
    residual_vs_shift = bp.paired(F.real_minus_matched_shift.to_numpy(float))
    max_power_error = float(S.power_max_rel_error.max()) if len(S) else float('nan')
    max_l1_error = float(S.power_l1_rel_error.max()) if len(S) else float('nan')
    max_mean_error = float(S.mean_abs_error.max()) if len(S) else float('nan')
    dec = decision(primary, len(valid_names), max_power_error)

    rp = pd.DataFrame(remote_prov)
    remote_rows = []
    for p in remote_prov:
        remote_rows.append({
            'file_index': p['file_index'],
            'url': p['url'],
            'rows': p['rows'],
            'rows_sampled': p['rows_sampled'],
            'sample_fraction_effective': p['sample_fraction_effective'],
            'sample_data_bytes': p['sample_data_bytes'],
            'sample_data_sha256': p['sample_data_sha256'],
            'etag': p['data_http']['etag'],
            'content_range': p['data_http']['content_range'],
            'header_prefix_sha256': p['header_prefix_sha256'],
        })
    pd.DataFrame(remote_rows).to_csv(out / 'remote_file_provenance.csv', index=False)

    qcdf = pd.DataFrame(qc_rows)
    summary = {
        'status': 'REAL_DR11_POINT_RANDOM_RESIDUAL_EXACT_POWER_PHASE36',
        'decision': dec,
        'n_locked_fields': 36,
        'n_valid_fields': len(valid_names),
        'point_random_fraction_per_file': float(args.fraction),
        'files_sampled': 20,
        'n_surrogates_per_field': int(args.surrogates),
        'selection_features': feature_names0,
        'selection_model_r2_median': float(np.nanmedian(M.selection_r2)) if len(M) else float('nan'),
        'selection_model_spearman_median': float(np.nanmedian(M.selection_spearman)) if len(M) else float('nan'),
        'point_random_qc': {
            'total_points_locked_fields': int(qcdf.n_point_randoms.sum()),
            'median_points_per_field': float(np.median(qcdf.n_point_randoms)),
            'min_points_per_field': int(qcdf.n_point_randoms.min()),
            'median_valid_cell_fraction': float(np.median(qcdf.valid_cell_fraction)),
            'min_valid_cell_fraction': float(qcdf.valid_cell_fraction.min()),
            'median_knn4_radius_arcmin': float(np.median(qcdf.median_knn4_radius_deg) * 60),
            'median_p95_knn4_radius_arcmin': float(np.median(qcdf.p95_knn4_radius_deg) * 60),
        },
        'real_residual_rho_median': float(np.nanmedian(F.real_residual_rho)),
        'phase_surrogate_mean_rho_median': float(np.nanmedian(F.phase_surrogate_mean_rho)),
        'matched_shift_rho_median': float(np.nanmedian(F.matched_shift_rho)),
        'primary_real_minus_exact_power': primary,
        'residual_real_minus_matched_shift': residual_vs_shift,
        'surrogate_power_fidelity': {
            'max_power_max_rel_error': max_power_error,
            'max_power_l1_rel_error': max_l1_error,
            'max_mean_abs_error': max_mean_error,
            'required_max_rel_error': POWER_TOL,
        },
        'H': 'point-random-selection-residual locality exceeds exact full-2D power phase surrogates',
        'T': '36 fixed fields; first 4% of each of 20 randomized official point-random files; same dense4 residualization; 32 exact-power phase surrogates per field',
        'D': 'PASS_HIGHER_ORDER_LOCALITY iff n_valid>=30, median real-minus-phase>=0.05 and one-sided sign/Wilcoxon p<0.01; FAIL if n_valid>=30 and median<=0 or either p>=0.10; otherwise UNCERTAIN; power fidelity failure invalidates run',
        'C': 'current selection-stressed locality is adequately reproduced, for this statistic, by exact two-point/power information',
        'U': 'surrogates are not full discrete-process models; residual selection/deblending/tracer effects remain; no cosmological inference',
        'total_remote_rows_sampled': int(sum(int(p['rows_sampled']) for p in remote_prov)),
        'total_remote_sample_bytes': int(sum(int(p['sample_data_bytes']) for p in remote_prov)),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
