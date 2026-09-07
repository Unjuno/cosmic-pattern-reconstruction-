#!/usr/bin/env python3
"""IAAFT locality test on cached dense4 point-random selection residual maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import point_random_selection_residual_validate as pr
import locality_validate as loc

N_SURR = 32
N_ITER = 150
EFFECT_FLOOR = 0.05
MAX_ONEPOINT_DIFF = 1e-12
MIN_RADIAL_LOGPOWER_CORR_MEDIAN = 0.995
MAX_RADIAL_POWER_RELERR_MEAN = 0.05
GRID = 64


def iaaft(z: np.ndarray, seed: int, niter: int = N_ITER) -> np.ndarray:
    z = np.asarray(z, float)
    rng = np.random.default_rng(seed)
    target_values = np.sort(z.ravel())
    target_amp = np.abs(np.fft.fft2(z))
    x = rng.permutation(z.ravel()).reshape(z.shape)
    for _ in range(niter):
        F = np.fft.fft2(x)
        phase = F / np.where(np.abs(F) > 0, np.abs(F), 1)
        y = np.fft.ifft2(target_amp * phase).real
        order = np.argsort(y.ravel())
        flat = np.empty(y.size, float)
        flat[order] = target_values
        x = flat.reshape(z.shape)
    return x


def radial_power_quality(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    A = np.abs(np.fft.fft2(a)) ** 2
    B = np.abs(np.fft.fft2(b)) ** 2
    ky = np.fft.fftfreq(GRID)
    kx = np.fft.fftfreq(GRID)
    KX, KY = np.meshgrid(kx, ky)
    rr = np.rint(np.hypot(KX, KY) * GRID).astype(int)
    pa, pb = [], []
    for k in range(1, rr.max() + 1):
        m = rr == k
        if m.sum():
            pa.append(float(A[m].mean()))
            pb.append(float(B[m].mean()))
    pa = np.asarray(pa)
    pb = np.asarray(pb)
    rel = float(np.mean(np.abs(pa - pb) / (pa + 1e-9)))
    corr = float(np.corrcoef(np.log1p(pa), np.log1p(pb))[0, 1])
    return rel, corr


def full_amp_l1_rel(a: np.ndarray, b: np.ndarray) -> float:
    A = np.abs(np.fft.fft2(a))
    B = np.abs(np.fft.fft2(b))
    return float(np.sum(np.abs(A - B)) / max(float(np.sum(A)), 1e-30))


def paired(diff: np.ndarray) -> dict:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    pos = int((d > 0).sum())
    try:
        wp = float(wilcoxon(d, alternative='greater').pvalue)
    except Exception:
        wp = float('nan')
    return {
        'n': n,
        'positive': pos,
        'median': float(np.median(d)),
        'mean': float(np.mean(d)),
        'sign_p_one_sided': float(binomtest(pos, n, .5, alternative='greater').pvalue),
        'wilcoxon_p_one_sided': wp,
    }


def quality_gate(q: pd.DataFrame) -> tuple[bool, dict]:
    metrics = {
        'max_sorted_onepoint_abs_diff': float(q.sorted_onepoint_max_abs_diff.max()),
        'radial_logpower_corr_median': float(q.radial_logpower_corr.median()),
        'radial_power_relerr_mean': float(q.radial_power_mean_rel_error.mean()),
        'full_amp_l1_relerr_median': float(q.full_amp_l1_rel_error.median()),
        'full_amp_l1_relerr_max': float(q.full_amp_l1_rel_error.max()),
        'required_max_onepoint_diff': MAX_ONEPOINT_DIFF,
        'required_min_radial_logpower_corr_median': MIN_RADIAL_LOGPOWER_CORR_MEDIAN,
        'required_max_radial_power_relerr_mean': MAX_RADIAL_POWER_RELERR_MEAN,
    }
    ok = (
        metrics['max_sorted_onepoint_abs_diff'] <= MAX_ONEPOINT_DIFF
        and metrics['radial_logpower_corr_median'] >= MIN_RADIAL_LOGPOWER_CORR_MEDIAN
        and metrics['radial_power_relerr_mean'] <= MAX_RADIAL_POWER_RELERR_MEAN
    )
    return bool(ok), metrics


def decision(primary: dict, n_fields: int, quality_ok: bool) -> str:
    if not quality_ok:
        return 'INVALID_IAAFT_QUALITY'
    if n_fields < 30:
        return 'UNCERTAIN_HIGHER_ORDER_IAAFT_LOCALITY'
    med = float(primary['median'])
    sp = float(primary['sign_p_one_sided'])
    wp = float(primary['wilcoxon_p_one_sided'])
    if med >= EFFECT_FLOOR and sp < .01 and wp < .01:
        return 'PASS_HIGHER_ORDER_IAAFT_LOCALITY'
    if med <= 0 or sp >= .10 or wp >= .10:
        return 'FAIL_HIGHER_ORDER_IAAFT_LOCALITY'
    return 'UNCERTAIN_HIGHER_ORDER_IAAFT_LOCALITY'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--surrogates', type=int, default=N_SURR)
    ap.add_argument('--iterations', type=int, default=N_ITER)
    ap.add_argument('--out', default='results/real_dr11/point_random_residual_iaaft36')
    args = ap.parse_args()
    if args.surrogates != N_SURR or args.iterations != N_ITER:
        raise RuntimeError(f'primary test fixed to {N_SURR} surrogates and {N_ITER} iterations')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('status') != 'REAL_DR11_POINT_RANDOM_DENSE4_RESIDUAL_MAP_CACHE':
        raise RuntimeError('wrong residual-map cache status')
    if manifest.get('shape') != [36, 64, 64] or float(manifest.get('selection_fraction_per_file')) != .04:
        raise RuntimeError('unexpected residual-map cache shape/fraction')

    Z = np.load(args.cache, allow_pickle=False)
    fields = [str(x) for x in Z['fields'].tolist()]
    maps = np.asarray(Z['residual_maps'], float)
    if maps.shape != (36, 64, 64) or len(fields) != 36 or len(set(fields)) != 36:
        raise RuntimeError('invalid residual map cache')

    field_rows = []
    qrows = []
    for i, (name, z) in enumerate(zip(fields, maps)):
        hidden, visible = pr.context_local(z)
        real_rho = pr.rho(visible, hidden)
        matched_shift = pr.rho(loc.shifted_feature(visible), hidden)
        srhos = []
        for si in range(N_SURR):
            seed = 2026093800 + i * 100 + si
            s = iaaft(z, seed, N_ITER)
            sh, sv = pr.context_local(s)
            srho = pr.rho(sv, sh)
            onepoint = float(np.max(np.abs(np.sort(z.ravel()) - np.sort(s.ravel()))))
            radial_rel, radial_corr = radial_power_quality(z, s)
            full_rel = full_amp_l1_rel(z, s)
            srhos.append(srho)
            qrows.append({
                'field': name, 'surrogate': si, 'seed': seed, 'rho': srho,
                'sorted_onepoint_max_abs_diff': onepoint,
                'radial_power_mean_rel_error': radial_rel,
                'radial_logpower_corr': radial_corr,
                'full_amp_l1_rel_error': full_rel,
            })
        a = np.asarray(srhos, float)
        field_rows.append({
            'field': name,
            'real_rho': real_rho,
            'matched_shift_rho': matched_shift,
            'iaaft_mean_rho': float(np.mean(a)),
            'iaaft_median_rho': float(np.median(a)),
            'iaaft_sd_rho': float(np.std(a, ddof=1)),
            'real_minus_iaaft_mean': float(real_rho - np.mean(a)),
            'real_minus_matched_shift': float(real_rho - matched_shift),
        })
        print(f'[iaaft-residual] field {i+1:02d}/36 {name} real={real_rho:.4f} iaaft={np.mean(a):.4f}', flush=True)

    F = pd.DataFrame(field_rows)
    Q = pd.DataFrame(qrows)
    F.to_csv(out / 'field_metrics.csv', index=False)
    Q.to_csv(out / 'surrogate_quality.csv', index=False)

    primary = paired(F.real_minus_iaaft_mean.to_numpy(float))
    vs_shift = paired(F.real_minus_matched_shift.to_numpy(float))
    quality_ok, quality = quality_gate(Q)
    dec = decision(primary, len(F), quality_ok)
    summary = {
        'status': 'REAL_DR11_POINT_RANDOM_RESIDUAL_IAAFT_LOCALITY36',
        'decision': dec,
        'source_residual_artifact_run': 34144835351,
        'source_residual_artifact_id': 10027376732,
        'source_residual_artifact_digest': 'sha256:3c1c14603e3bbfc9ee1d8d0c802f1ab84267a5ac5fc75f9df55822dd3e72de90',
        'n_fields': 36,
        'n_surrogates_per_field': N_SURR,
        'iaaft_iterations': N_ITER,
        'real_rho_median': float(F.real_rho.median()),
        'iaaft_mean_rho_median': float(F.iaaft_mean_rho.median()),
        'matched_shift_rho_median': float(F.matched_shift_rho.median()),
        'primary_real_minus_iaaft': primary,
        'real_minus_matched_shift': vs_shift,
        'surrogate_quality_pass': quality_ok,
        'surrogate_quality': quality,
        'H': 'dense4 point-random residual locality exceeds IAAFT controls preserving exact one-point values and approximately the power spectrum',
        'T': '36 cached dense4 residual maps; 32 IAAFT surrogates per field; 150 iterations; local-visible/hidden Spearman statistic',
        'D': 'PASS if quality gate passes, n>=30, median real-minus-IAAFT>=0.05 and one-sided sign/Wilcoxon p<0.01; FAIL if median<=0 or either p>=0.10; otherwise UNCERTAIN',
        'C': 'one-point marginal plus two-point/power structure adequately reproduces the current residual locality statistic at tested IAAFT fidelity',
        'U': 'IAAFT power match is approximate; residual maps are nuisance-model products; observational/tracer effects remain; no cosmological inference',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
