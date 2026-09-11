#!/usr/bin/env python3
"""Gaussianized exact-power test for dense4 point-random residual maps.

Primary protocol is preregistered in docs/POINT_RANDOM_GAUSSIAN_PHASE.md.
Also saves the deterministic held-out residual maps for future tests and runs a
secondary independent raw-phase ensemble as Monte-Carlo stability QC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import analyze_dr11 as adr
import bispectrum_phase_validate as bp
import locality_validate as loc
import point_random_selection_residual_validate as pr
import point_random_phase_surrogate_validate as pphase

N_SURR = 32
DENSE_FRACTION = 0.04
POWER_TOL = 1e-10
EFFECT_FLOOR = 0.05
N_FOLDS = 6


def paired_for_representation(maps: dict[str, np.ndarray], fold_ids: dict[str, int], gaussianized: bool,
                              seed_base: int, surrogate_rows: list[dict], field_rows: list[dict]):
    for name in maps:
        g0 = maps[name]
        g = bp.gaussianize(g0) if gaussianized else g0
        hidden, visible = pr.context_local(g)
        real_rho = pr.rho(visible, hidden)
        shift_rho = pr.rho(loc.shifted_feature(visible), hidden)
        idx = pr.LOCKED_FIELDS.index(name)
        srhos = []
        for si in range(N_SURR):
            seed = seed_base + idx * 100 + si
            sg = bp.exact_phase(g, seed)
            sh, sv = pr.context_local(sg)
            srho = pr.rho(sv, sh)
            max_rel, l1_rel = pphase.power_fidelity(g, sg)
            srhos.append(srho)
            surrogate_rows.append({
                'field': name,
                'fold': fold_ids[name],
                'representation': 'gaussianized_residual' if gaussianized else 'raw_residual_replication',
                'surrogate': si,
                'seed': seed,
                'rho': srho,
                'power_max_rel_error': max_rel,
                'power_l1_rel_error': l1_rel,
                'mean_abs_error': float(abs(np.mean(sg) - np.mean(g))),
            })
        a = np.asarray(srhos, float)
        field_rows.append({
            'field': name,
            'fold': fold_ids[name],
            'representation': 'gaussianized_residual' if gaussianized else 'raw_residual_replication',
            'real_rho': real_rho,
            'matched_shift_rho': shift_rho,
            'phase_surrogate_mean_rho': float(np.nanmean(a)),
            'phase_surrogate_median_rho': float(np.nanmedian(a)),
            'phase_surrogate_sd': float(np.nanstd(a, ddof=1)),
            'real_minus_phase_mean': float(real_rho - np.nanmean(a)),
            'real_minus_matched_shift': float(real_rho - shift_rho),
        })


def decision(primary: dict, n_valid: int, max_power_error: float) -> str:
    if not np.isfinite(max_power_error) or max_power_error > POWER_TOL:
        return 'INVALID_POWER_FIDELITY'
    if n_valid < 30:
        return 'UNCERTAIN_HIGHER_ORDER_GAUSSIANIZED_LOCALITY'
    med = float(primary.get('median', np.nan))
    sp = float(primary.get('sign_p_one_sided', np.nan))
    wp = float(primary.get('wilcoxon_p_one_sided', np.nan))
    if med >= EFFECT_FLOOR and sp < .01 and wp < .01:
        return 'PASS_HIGHER_ORDER_GAUSSIANIZED_LOCALITY'
    if med <= 0 or sp >= .10 or wp >= .10:
        return 'FAIL_HIGHER_ORDER_GAUSSIANIZED_LOCALITY'
    return 'UNCERTAIN_HIGHER_ORDER_GAUSSIANIZED_LOCALITY'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--provenance', default='data/real/dr11/expanded48/provenance.json')
    ap.add_argument('--fraction', type=float, default=DENSE_FRACTION)
    ap.add_argument('--surrogates', type=int, default=N_SURR)
    ap.add_argument('--out', default='results/real_dr11/point_random_gaussian_phase36')
    args = ap.parse_args()
    if abs(args.fraction - DENSE_FRACTION) > 1e-12:
        raise RuntimeError('primary experiment fixed to fraction=0.04')
    if args.surrogates != N_SURR:
        raise RuntimeError(f'primary experiment fixed to {N_SURR} surrogates')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions = pr.load_regions(Path(args.provenance))
    points, remote_prov = pr.acquire_points(regions, args.fraction)

    data = {}
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
        print(f'[gaussian-phase] field {i+1:02d}/36 {name} randoms={len(points[name])} valid={valid.mean():.4f}', flush=True)
    Q = pd.DataFrame(qc_rows)
    Q.to_csv(out / 'point_random_qc.csv', index=False)
    valid_names = [n for n in pr.LOCKED_FIELDS if n in data]

    fold_id = {n: i % N_FOLDS for i, n in enumerate(valid_names)}
    residual_maps: dict[str, np.ndarray] = {}
    model_rows = []
    for fold in range(N_FOLDS):
        test = [n for n in valid_names if fold_id[n] == fold]
        train = [n for n in valid_names if fold_id[n] != fold]
        if not test or len(train) < 20:
            continue
        model = pr.fit_model(data, train, 56000 + fold)
        for name in test:
            d = data[name]
            v = d['valid']
            counts = d['counts']
            mu = float(np.mean(counts[v])) + 1e-6
            pred_rel = np.full((pr.GRID, pr.GRID), np.nan, float)
            pred_rel[v] = np.clip(model.predict(d['sel'][v]), .03, 20)
            true_rel = counts[v] / mu
            model_rows.append({
                'field': name,
                'fold': fold,
                'selection_r2': float(r2_score(true_rel, pred_rel[v])),
                'selection_spearman': pr.rho(true_rel, pred_rel[v]),
            })
            expected = pred_rel * mu
            idx = pr.LOCKED_FIELDS.index(name)
            _, residual = pr.normalized_maps(counts, expected, v, 930000 + idx * 4)
            residual_maps[name] = residual.astype(np.float64)
        print(f'[gaussian-phase] residual fold {fold+1}/{N_FOLDS} train={len(train)} test={len(test)}', flush=True)

    if set(residual_maps) != set(valid_names):
        raise RuntimeError('not all valid fields received held-out residual maps')
    order = [n for n in valid_names if n in residual_maps]
    stack = np.stack([residual_maps[n] for n in order])
    np.savez_compressed(out / 'residual_maps.npz', fields=np.asarray(order), residual_maps=stack)
    (out / 'residual_map_manifest.json').write_text(json.dumps({
        'status': 'REAL_DR11_POINT_RANDOM_DENSE4_RESIDUAL_MAP_CACHE',
        'fields': order,
        'shape': list(stack.shape),
        'dtype': str(stack.dtype),
        'selection_fraction_per_file': args.fraction,
        'selection_features': feature_names0,
        'cross_validation': '6-fold grouped by whole field',
        'role': 'derived held-out nuisance-residual maps for future statistical stress tests; not raw survey data',
    }, indent=2, sort_keys=True) + '\n')

    surrogate_rows: list[dict] = []
    field_rows: list[dict] = []
    # Primary transformed representation.
    paired_for_representation(residual_maps, fold_id, True, 2026092800, surrogate_rows, field_rows)
    # Secondary independent raw-phase ensemble; never pooled into primary decision.
    paired_for_representation(residual_maps, fold_id, False, 2026091800, surrogate_rows, field_rows)

    F = pd.DataFrame(field_rows)
    S = pd.DataFrame(surrogate_rows)
    M = pd.DataFrame(model_rows)
    F.to_csv(out / 'field_metrics.csv', index=False)
    S.to_csv(out / 'surrogate_qc.csv', index=False)
    M.to_csv(out / 'selection_model_metrics.csv', index=False)

    comps = {}
    for rep in ['gaussianized_residual', 'raw_residual_replication']:
        g = F[F.representation == rep].copy()
        primary = bp.paired(g.real_minus_phase_mean.to_numpy(float))
        vs_shift = bp.paired(g.real_minus_matched_shift.to_numpy(float))
        comps[rep] = {
            'real_rho_median': float(np.nanmedian(g.real_rho)),
            'phase_surrogate_mean_rho_median': float(np.nanmedian(g.phase_surrogate_mean_rho)),
            'matched_shift_rho_median': float(np.nanmedian(g.matched_shift_rho)),
            'real_minus_exact_power': primary,
            'real_minus_matched_shift': vs_shift,
        }

    primary = comps['gaussianized_residual']['real_minus_exact_power']
    primary_qc = S[S.representation == 'gaussianized_residual']
    max_power_error = float(primary_qc.power_max_rel_error.max())
    dec = decision(primary, len(valid_names), max_power_error)

    remote_rows = []
    for p in remote_prov:
        remote_rows.append({
            'file_index': p['file_index'], 'url': p['url'], 'rows': p['rows'],
            'rows_sampled': p['rows_sampled'], 'sample_fraction_effective': p['sample_fraction_effective'],
            'sample_data_bytes': p['sample_data_bytes'], 'sample_data_sha256': p['sample_data_sha256'],
            'etag': p['data_http']['etag'], 'content_range': p['data_http']['content_range'],
            'header_prefix_sha256': p['header_prefix_sha256'],
        })
    pd.DataFrame(remote_rows).to_csv(out / 'remote_file_provenance.csv', index=False)

    summary = {
        'status': 'REAL_DR11_POINT_RANDOM_GAUSSIANIZED_EXACT_POWER_PHASE36',
        'decision': dec,
        'n_valid_fields': len(valid_names),
        'n_surrogates_per_field': N_SURR,
        'point_random_fraction_per_file': args.fraction,
        'selection_model_r2_median': float(np.nanmedian(M.selection_r2)),
        'selection_model_spearman_median': float(np.nanmedian(M.selection_spearman)),
        'point_random_qc': {
            'total_points_locked_fields': int(Q.n_point_randoms.sum()),
            'median_points_per_field': float(np.median(Q.n_point_randoms)),
            'min_points_per_field': int(Q.n_point_randoms.min()),
            'median_valid_cell_fraction': float(np.median(Q.valid_cell_fraction)),
            'min_valid_cell_fraction': float(Q.valid_cell_fraction.min()),
            'median_knn4_radius_arcmin': float(np.median(Q.median_knn4_radius_deg) * 60),
            'median_p95_knn4_radius_arcmin': float(np.median(Q.p95_knn4_radius_deg) * 60),
        },
        'comparisons': comps,
        'primary_representation': 'rank-Gaussianized point-random-selection residual map',
        'secondary_raw_phase_replication_role': 'Monte-Carlo stability QC only; not pooled with previous raw-phase result',
        'surrogate_power_fidelity': {
            'gaussianized_max_power_max_rel_error': max_power_error,
            'gaussianized_max_power_l1_rel_error': float(primary_qc.power_l1_rel_error.max()),
            'gaussianized_max_mean_abs_error': float(primary_qc.mean_abs_error.max()),
            'required_max_rel_error': POWER_TOL,
        },
        'residual_map_cache': 'residual_maps.npz',
        'H': 'after marginal control, point-random-selection-residual locality exceeds exact full-2D power phase surrogates',
        'T': '36 fixed fields; first 4% of each of 20 official randomized point-random files; dense4 residualization; rank-Gaussianization; 32 exact-power surrogates per field',
        'D': 'PASS if n_valid>=30, median gaussianized real-minus-phase>=0.05 and one-sided sign/Wilcoxon p<0.01; FAIL if median<=0 or either p>=0.10; otherwise UNCERTAIN; power fidelity failure invalidates run',
        'C': 'after marginal control, exact two-point/power information adequately reproduces the transformed locality statistic',
        'U': 'Gaussianization defines a transformed statistic; exact-power surrogates are not full discrete-process models; residual observational/tracer effects remain; no cosmological inference',
        'total_remote_rows_sampled': int(sum(int(p['rows_sampled']) for p in remote_prov)),
        'total_remote_sample_bytes': int(sum(int(p['sample_data_bytes']) for p in remote_prov)),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
