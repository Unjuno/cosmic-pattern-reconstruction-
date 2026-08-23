#!/usr/bin/env python3
"""Angular-scale sweep and strict matched-shift null on REAL DR11 RA/Dec.

This script uses only provenance-verified observed DR11 positions.  It compares
true local boundary/interior pairing with two nulls:

1. cell_permutation: destroys all spatial adjacency within each field;
2. matched_shift: preserves every observed patch and its internal clustering,
   but cyclically pairs the visible part of a different patch with the true
   hidden center.  This is a stronger control for one-point distributions and
   field-level selection structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analyze_dr11 import normalize_grid, region_grid, shuffle_grid, verify_and_load
from lofo_validate import exact_signflip_p

GRID = 64
FIELD_ARCMIN = 30.0


def patchify(grid: np.ndarray, patch: int) -> np.ndarray:
    n = GRID // patch
    return np.asarray([
        grid[iy*patch:(iy+1)*patch, ix*patch:(ix+1)*patch].reshape(-1)
        for iy in range(n) for ix in range(n)
    ])


def mask_indices(patch: int) -> tuple[np.ndarray, int, int]:
    start = patch // 4
    end = patch - start
    mask = np.zeros((patch, patch), dtype=bool)
    mask[start:end, start:end] = True
    return np.flatnonzero(mask.reshape(-1)), start, end


def ring_features(X: np.ndarray, patch: int) -> np.ndarray:
    a = X.reshape(-1, patch, patch)
    _, start, end = mask_indices(patch)
    top = a[:, start-1, start-1:end+1]
    bottom = a[:, end, start-1:end+1]
    left = a[:, start:end, start-1]
    right = a[:, start:end, end]
    vals = np.concatenate([top, bottom, left, right], axis=1)
    return np.column_stack([
        vals.mean(1), vals.std(1),
        left.mean(1), right.mean(1), top.mean(1), bottom.mean(1),
        right.mean(1)-left.mean(1), bottom.mean(1)-top.mean(1),
    ])


def shift_patches(X: np.ndarray, patch: int) -> np.ndarray:
    n = GRID // patch
    a = X.reshape(n, n, patch*patch)
    dy = max(1, n // 3)
    dx = max(1, n // 2 - 1)
    return np.roll(a, shift=(dy, dx), axis=(0, 1)).reshape(n*n, patch*patch)


def matched_shift_hybrid(X: np.ndarray, patch: int) -> np.ndarray:
    hidden, _, _ = mask_indices(patch)
    observed = np.setdiff1d(np.arange(patch*patch), hidden)
    shifted = shift_patches(X, patch)
    hybrid = X.copy()
    hybrid[:, observed] = shifted[:, observed]
    return hybrid


def gaussian_prediction(Xtr: np.ndarray, Xte: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    observed = np.setdiff1d(np.arange(Xtr.shape[1]), hidden)
    mu = Xtr.mean(axis=0)
    cov = np.cov(Xtr, rowvar=False)
    coo = cov[np.ix_(observed, observed)] + 0.08*np.eye(len(observed))
    coh = cov[np.ix_(observed, hidden)]
    return mu[hidden] + (Xte[:, observed] - mu[observed]) @ np.linalg.solve(coo, coh)


def reconstruction_score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y-pred)**2))
    corr = float(np.corrcoef(y.ravel(), pred.ravel())[0, 1]) if np.std(y) and np.std(pred) else float('nan')
    return mse, corr


def motif_auc(Xtr: np.ndarray, Xte: np.ndarray, patch: int) -> dict[str, float]:
    hidden, _, _ = mask_indices(patch)
    train_mean = Xtr[:, hidden].mean(1)
    test_mean = Xte[:, hidden].mean(1)
    train_max = Xtr[:, hidden].max(1)
    test_max = Xte[:, hidden].max(1)
    q25, q75 = np.quantile(train_mean, [0.25, 0.75])
    qpeak = np.quantile(train_max, 0.80)
    Ftr, Fte = ring_features(Xtr, patch), ring_features(Xte, patch)
    labels = {
        'void': (train_mean <= q25, test_mean <= q25),
        'overdense': (train_mean >= q75, test_mean >= q75),
        'peak': (train_max >= qpeak, test_max >= qpeak),
    }
    out = {}
    for name, (ytr, yte) in labels.items():
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out[name] = float('nan')
            continue
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Ftr, ytr.astype(int))
        out[name] = float(roc_auc_score(yte.astype(int), clf.predict_proba(Fte)[:, 1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real/dr11/pilot')
    ap.add_argument('--out', default='results/real_dr11/latest')
    args = ap.parse_args()
    datadir, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    prov = json.loads((datadir/'provenance.json').read_text())
    if prov.get('status') != 'REAL_DR11' or prov.get('model_input_columns') != ['ra', 'dec']:
        raise RuntimeError('REAL_DR11 RA/Dec-only provenance required')
    meta = {r['name']: r for r in prov['regions']}
    fields = list(meta)
    grids = {f: region_grid(verify_and_load(m), m) for f, m in meta.items()}

    rows = []
    for patch in [4, 8, 16]:
        real = {f: patchify(normalize_grid(grids[f])[0], patch) for f in fields}
        perm = {f: patchify(normalize_grid(shuffle_grid(grids[f], 20260824+j))[0], patch) for j, f in enumerate(fields)}
        shift = {f: matched_shift_hybrid(real[f], patch) for f in fields}
        for held in fields:
            training = [f for f in fields if f != held]
            for sample, parts in [('real', real), ('cell_permutation', perm), ('matched_shift', shift)]:
                train_raw = np.concatenate([parts[f] for f in training], axis=0)
                test_raw = parts[held]
                scaler = StandardScaler().fit(train_raw)
                Xtr, Xte = scaler.transform(train_raw), scaler.transform(test_raw)
                hidden, _, _ = mask_indices(patch)
                pred = gaussian_prediction(Xtr, Xte, hidden)
                mse, corr = reconstruction_score(Xte[:, hidden], pred)
                motifs = motif_auc(Xtr, Xte, patch)
                rows.append({
                    'patch_cells': patch,
                    'patch_arcmin': FIELD_ARCMIN*patch/GRID,
                    'hole_arcmin': FIELD_ARCMIN*(patch/2)/GRID,
                    'field': held,
                    'sample': sample,
                    'gaussian_mse': mse,
                    'gaussian_corr': corr,
                    'void_auc': motifs['void'],
                    'overdense_auc': motifs['overdense'],
                    'peak_auc': motifs['peak'],
                })

    field_df = pd.DataFrame(rows)
    field_df.to_csv(outdir/'scale_null_field_metrics.csv', index=False)
    comps = []
    for patch in [4, 8, 16]:
        sub = field_df[field_df.patch_cells == patch]
        for null in ['cell_permutation', 'matched_shift']:
            for metric in ['gaussian_corr', 'void_auc', 'overdense_auc', 'peak_auc']:
                w = sub.pivot(index='field', columns='sample', values=metric)
                diff = w['real'] - w[null]
                comps.append({
                    'patch_cells': patch,
                    'patch_arcmin': FIELD_ARCMIN*patch/GRID,
                    'hole_arcmin': FIELD_ARCMIN*(patch/2)/GRID,
                    'null': null,
                    'metric': metric,
                    'real_median': float(np.nanmedian(w['real'])),
                    'null_median': float(np.nanmedian(w[null])),
                    'positive_fields': int((diff > 0).sum()),
                    'mean_paired_advantage': float(np.nanmean(diff)),
                    'exact_one_sided_signflip_p': float(exact_signflip_p(diff.to_numpy())),
                })
            w = sub.pivot(index='field', columns='sample', values='gaussian_mse')
            diff = w[null] - w['real']
            comps.append({
                'patch_cells': patch,
                'patch_arcmin': FIELD_ARCMIN*patch/GRID,
                'hole_arcmin': FIELD_ARCMIN*(patch/2)/GRID,
                'null': null,
                'metric': 'gaussian_mse',
                'real_median': float(np.nanmedian(w['real'])),
                'null_median': float(np.nanmedian(w[null])),
                'positive_fields': int((diff > 0).sum()),
                'mean_paired_advantage': float(np.nanmean(diff)),
                'exact_one_sided_signflip_p': float(exact_signflip_p(diff.to_numpy())),
            })
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(outdir/'scale_null_comparisons.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for metric in ['void_auc', 'overdense_auc', 'peak_auc']:
        vals = []
        xs = []
        for patch in [4, 8, 16]:
            q = comp_df[(comp_df.patch_cells == patch) & (comp_df['null'] == 'matched_shift') & (comp_df.metric == metric)].iloc[0]
            xs.append(float(q.patch_arcmin)); vals.append(float(q.real_median))
        ax.plot(xs, vals, marker='o', label=metric.replace('_auc', ''))
    ax.axhline(0.5, linestyle='--', linewidth=1)
    ax.set_xlabel('patch width [arcmin]')
    ax.set_ylabel('LOFO median AUC')
    ax.set_title('REAL DR11: boundary motif predictability vs angular scale')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/'scale_motif_auc.png', dpi=160)
    plt.close(fig)

    summary = {
        'status': 'REAL_DR11',
        'validation': '12-field LOFO angular-scale sweep',
        'model_input_columns': ['ra', 'dec'],
        'scales_arcmin': [1.875, 3.75, 7.5],
        'nulls': {
            'cell_permutation': 'within-field fine-cell permutation',
            'matched_shift': 'visible cells from a cyclically shifted observed patch paired with the original hidden center',
        },
        'comparisons': comps,
    }
    (outdir/'scale_null_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
