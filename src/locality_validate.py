#!/usr/bin/env python3
"""Measure the angular locality of boundary -> hidden-environment predictability.

Uses only provenance-verified REAL DR11 RA/Dec positions from the deterministic
48-field sample. For each 3.75 arcmin patch the central 1.875 arcmin square is
the hidden target. Predictors are means from increasingly distant observed
regions. A cyclic matched-shift null preserves each field's observed one-point
and patch-level structure while breaking the local target/context pairing.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analyze_dr11 import normalize_grid, region_grid, verify_and_load

GRID = 64
PATCH = 8
CELL_ARCMIN = 30.0 / GRID


def contexts_from_grid(grid: np.ndarray) -> np.ndarray:
    """Return [hidden_mean, local_visible, external bands...] for 64 patches."""
    rows = []
    for iy in range(8):
        for ix in range(8):
            y0, y1 = iy * PATCH, (iy + 1) * PATCH
            x0, x1 = ix * PATCH, (ix + 1) * PATCH
            patch = grid[y0:y1, x0:x1]
            hidden = patch[2:6, 2:6]
            visible = np.ones((PATCH, PATCH), dtype=bool)
            visible[2:6, 2:6] = False
            vals = [float(hidden.mean()), float(patch[visible].mean())]
            for lo, hi in [(1, 4), (5, 8), (9, 16)]:
                ys = slice(max(0, y0 - hi), min(GRID, y1 + hi))
                xs = slice(max(0, x0 - hi), min(GRID, x1 + hi))
                yy, xx = np.mgrid[ys, xs]
                dy = np.where(yy < y0, y0 - yy, np.where(yy >= y1, yy - y1 + 1, 0))
                dx = np.where(xx < x0, x0 - xx, np.where(xx >= x1, xx - x1 + 1, 0))
                dist = np.maximum(dy, dx)
                mask = (dist >= lo) & (dist <= hi)
                vals.append(float(grid[ys, xs][mask].mean()))
            rows.append(vals)
    return np.asarray(rows, dtype=float)


def shifted_feature(values: np.ndarray) -> np.ndarray:
    return np.roll(values.reshape(8, 8), shift=(2, 3), axis=(0, 1)).reshape(-1)


def auc(ytr: np.ndarray, yte: np.ndarray, xtr: np.ndarray, xte: np.ndarray) -> float:
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float('nan')
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(xtr[:, None], ytr.astype(int))
    return float(roc_auc_score(yte.astype(int), clf.predict_proba(xte[:, None])[:, 1]))


def paired_summary(diff: np.ndarray) -> dict:
    x = np.asarray(diff, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return {'n_fields': 0}
    pos = int((x > 0).sum())
    try:
        wp = float(wilcoxon(x, alternative='greater', zero_method='wilcox').pvalue)
    except Exception:
        wp = float('nan')
    return {
        'n_fields': n,
        'positive_fields': pos,
        'sign_test_one_sided_p': float(binomtest(pos, n, .5, alternative='greater').pvalue),
        'wilcoxon_one_sided_p': wp,
        'mean_advantage': float(np.mean(x)),
        'median_advantage': float(np.median(x)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real/dr11/expanded48')
    ap.add_argument('--out', default='results/real_dr11/expanded48')
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = json.loads((data / 'provenance.json').read_text())
    if prov.get('status') != 'REAL_DR11' or prov.get('model_input_columns') != ['ra', 'dec'] or len(prov.get('regions', [])) != 48:
        raise RuntimeError('48-field REAL_DR11 RA/Dec provenance required')

    meta = {r['name']: r for r in prov['regions']}
    fields = list(meta)
    ctx = {}
    for field, m in meta.items():
        grid = normalize_grid(region_grid(verify_and_load(m), m))[0]
        ctx[field] = contexts_from_grid(grid)

    feature_names = ['local_visible', 'external_1_4cells', 'external_5_8cells', 'external_9_16cells']
    distance_labels = {
        'local_visible': 'within same 3.75 arcmin patch, excluding hidden center',
        'external_1_4cells': f'{1*CELL_ARCMIN:.3f}-{4*CELL_ARCMIN:.3f} arcmin beyond patch edge',
        'external_5_8cells': f'{5*CELL_ARCMIN:.3f}-{8*CELL_ARCMIN:.3f} arcmin beyond patch edge',
        'external_9_16cells': f'{9*CELL_ARCMIN:.3f}-{16*CELL_ARCMIN:.3f} arcmin beyond patch edge',
    }

    rows = []
    quintile_rows = []
    for held in fields:
        train_fields = [f for f in fields if f != held]
        train = np.concatenate([ctx[f] for f in train_fields], axis=0)
        test = ctx[held]
        ytrain, ytest = train[:, 0], test[:, 0]
        q25, q75 = np.quantile(ytrain, [.25, .75])
        labels = {
            'void': (ytrain <= q25, ytest <= q25),
            'overdense': (ytrain >= q75, ytest >= q75),
        }

        for j, feature in enumerate(feature_names, start=1):
            xtr, xte = train[:, j], test[:, j]
            xtr_shift = np.concatenate([shifted_feature(ctx[f][:, j]) for f in train_fields])
            xte_shift = shifted_feature(test[:, j])
            for motif, (ytr, yte) in labels.items():
                rows.append({
                    'field': held,
                    'feature': feature,
                    'motif': motif,
                    'real_auc': auc(ytr, yte, xtr, xte),
                    'matched_shift_auc': auc(ytr, yte, xtr_shift, xte_shift),
                })

        # Interpretable train-defined quintiles for the local-visible predictor.
        xtr, xte = train[:, 1], test[:, 1]
        cuts = np.quantile(xtr, [.2, .4, .6, .8])
        bins = np.digitize(xte, cuts)
        xtr_shift = np.concatenate([shifted_feature(ctx[f][:, 1]) for f in train_fields])
        xte_shift = shifted_feature(test[:, 1])
        cuts_shift = np.quantile(xtr_shift, [.2, .4, .6, .8])
        bins_shift = np.digitize(xte_shift, cuts_shift)
        for sample, bb in [('real', bins), ('matched_shift', bins_shift)]:
            for b in range(5):
                mask = bb == b
                if not mask.any():
                    continue
                quintile_rows.append({
                    'field': held,
                    'sample': sample,
                    'quintile': b,
                    'n': int(mask.sum()),
                    'void_rate': float((ytest[mask] <= q25).mean()),
                    'overdense_rate': float((ytest[mask] >= q75).mean()),
                })

    field_df = pd.DataFrame(rows)
    field_df.to_csv(out / 'locality_field_metrics.csv', index=False)
    qdf = pd.DataFrame(quintile_rows)
    qdf.to_csv(out / 'locality_quintiles_by_field.csv', index=False)

    comparisons = []
    for feature in feature_names:
        for motif in ['void', 'overdense']:
            s = field_df[(field_df.feature == feature) & (field_df.motif == motif)]
            diff = s.real_auc.to_numpy() - s.matched_shift_auc.to_numpy()
            comparisons.append({
                'feature': feature,
                'distance_definition': distance_labels[feature],
                'motif': motif,
                'real_median_auc': float(np.nanmedian(s.real_auc)),
                'matched_shift_median_auc': float(np.nanmedian(s.matched_shift_auc)),
                **paired_summary(diff),
            })
    pd.DataFrame(comparisons).to_csv(out / 'locality_comparisons.csv', index=False)

    pooled = []
    for sample in ['real', 'matched_shift']:
        for b in range(5):
            s = qdf[(qdf['sample'] == sample) & (qdf.quintile == b)]
            pooled.append({
                'sample': sample,
                'quintile': b,
                'n': int(s.n.sum()),
                'void_rate': float(np.average(s.void_rate, weights=s.n)),
                'overdense_rate': float(np.average(s.overdense_rate, weights=s.n)),
            })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(out / 'locality_quintiles_pooled.csv', index=False)

    summary = {
        'status': 'REAL_DR11',
        'validation': '48-field LOFO locality decay and train-defined conditional-probability bins',
        'total_rows': int(prov['total_rows']),
        'patch_arcmin': 3.75,
        'hidden_arcmin': 1.875,
        'comparisons': comparisons,
        'quintiles_pooled': pooled,
    }
    (out / 'locality_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
