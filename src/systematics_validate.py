#!/usr/bin/env python3
"""Test whether REAL DR11 boundary predictability survives observing-condition QC.

Observing-condition columns are used only to predict/remove a nuisance component
of the fine-cell count field. Cosmological models still receive RA/Dec only.
Training residuals are cross-fitted by whole sky field to avoid fitting a field
against its own counts.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from analyze_dr11 import (
    mask_indices, motif_metrics, normalize_grid, patchify, reconstruction_predictions,
    region_grid, score_reconstruction, verify_and_load,
)
from lofo_validate import exact_signflip_p
from scale_null_validate import matched_shift_hybrid

GRID = 64
HALF = 0.25
FEATURES = [
    'clean_fraction',
    'nobs_g','nobs_r','nobs_i','nobs_z',
    'log_psfdepth_g','log_psfdepth_r','log_psfdepth_i','log_psfdepth_z',
    'psfsize_g','psfsize_r','psfsize_i','psfsize_z',
    'mw_transmission_g','mw_transmission_r','mw_transmission_i','mw_transmission_z',
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tangent_ra(ra, ra0):
    return ((np.asarray(ra, float)-ra0+180.0)%360.0)-180.0


def load_qc(meta: dict) -> pd.DataFrame:
    path = Path(meta['file']); gz = path.read_bytes(); raw = gzip.decompress(gz)
    if sha256(gz) != meta['stored_gzip_sha256'] or sha256(raw) != meta['canonical_csv_sha256']:
        raise RuntimeError(f'QC provenance hash mismatch: {path}')
    df = pd.read_csv(path)
    if len(df) != int(meta['rows']): raise RuntimeError(f'QC row mismatch: {path}')
    return df


def cell_systematics(df: pd.DataFrame, field_meta: dict) -> np.ndarray:
    ra0, dec0 = float(field_meta['center_ra_deg']), float(field_meta['center_dec_deg'])
    dra = tangent_ra(df.ra, ra0); ddec = df.dec.to_numpy(float)-dec0
    x = np.floor((dra+HALF)/(2*HALF)*GRID).astype(int)
    y = np.floor((ddec+HALF)/(2*HALF)*GRID).astype(int)
    keep = (x>=0)&(x<GRID)&(y>=0)&(y<GRID)
    d = df.loc[keep].copy(); d['cell'] = y[keep]*GRID+x[keep]
    d['clean_fraction'] = (pd.to_numeric(d.maskbits, errors='coerce').fillna(-1)==0).astype(float)
    for band in 'griz':
        d[f'nobs_{band}'] = pd.to_numeric(d[f'nobs_{band}'], errors='coerce')
        d[f'log_psfdepth_{band}'] = np.log1p(np.clip(pd.to_numeric(d[f'psfdepth_{band}'], errors='coerce'), 0, None))
        p = pd.to_numeric(d[f'psfsize_{band}'], errors='coerce')
        d[f'psfsize_{band}'] = p.where(p>0)
        d[f'mw_transmission_{band}'] = pd.to_numeric(d[f'mw_transmission_{band}'], errors='coerce')
    agg = d.groupby('cell')[FEATURES].median()
    out = np.full((GRID*GRID, len(FEATURES)), np.nan, float)
    out[agg.index.to_numpy(int)] = agg.to_numpy(float)
    med = np.nanmedian(out, axis=0)
    med[~np.isfinite(med)] = 0.0
    ii = np.where(~np.isfinite(out)); out[ii] = med[ii[1]]
    return out


def fit_predict(train_fields: list[str], test_field: str, target: dict[str,np.ndarray], cov: dict[str,np.ndarray]) -> tuple[np.ndarray, float]:
    Xtr = np.concatenate([cov[f] for f in train_fields])
    ytr = np.concatenate([target[f].reshape(-1) for f in train_fields])
    Xte = cov[test_field]; yte = target[test_field].reshape(-1)
    scaler = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 100.0])).fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))
    return pred.reshape(GRID, GRID), float(r2_score(yte, pred))


def crossfit_residuals(fields: list[str], target: dict[str,np.ndarray], cov: dict[str,np.ndarray], held: str) -> tuple[dict[str,np.ndarray], float]:
    residual = {}
    training = [f for f in fields if f != held]
    # Cross-fit every training field without itself or the final held-out field.
    for f in training:
        fit_fields = [g for g in training if g != f]
        pred, _ = fit_predict(fit_fields, f, target, cov)
        residual[f] = target[f] - pred
    pred_test, r2 = fit_predict(training, held, target, cov)
    residual[held] = target[held] - pred_test
    return residual, r2


def evaluate_parts(parts: dict[str,np.ndarray], fields: list[str], held: str) -> dict:
    training = [f for f in fields if f != held]
    train_raw = np.concatenate([patchify(parts[f], f)[0] for f in training])
    test_raw = patchify(parts[held], held)[0]
    scaler = StandardScaler().fit(train_raw)
    Xtr, Xte = scaler.transform(train_raw), scaler.transform(test_raw)
    motifs = {m['motif']: m['auc'] for m in motif_metrics(Xtr, Xte)}
    hidden = mask_indices('center25')
    pred = reconstruction_predictions(Xtr, Xte, hidden)['gaussian']
    mse, corr = score_reconstruction(Xte[:, hidden], pred)
    return {'void_auc': motifs['void'], 'overdense_auc': motifs['overdense'], 'peak_auc': motifs['peak'], 'gaussian_corr': corr, 'gaussian_mse': mse}


def evaluate_shift(parts: dict[str,np.ndarray], fields: list[str], held: str) -> dict:
    training = [f for f in fields if f != held]
    shifted = {}
    for f in fields:
        p = patchify(parts[f], f)[0]
        shifted[f] = matched_shift_hybrid(p, 8)
    train_raw = np.concatenate([shifted[f] for f in training])
    test_raw = shifted[held]
    scaler = StandardScaler().fit(train_raw)
    Xtr, Xte = scaler.transform(train_raw), scaler.transform(test_raw)
    motifs = {m['motif']: m['auc'] for m in motif_metrics(Xtr, Xte)}
    hidden = mask_indices('center25')
    pred = reconstruction_predictions(Xtr, Xte, hidden)['gaussian']
    mse, corr = score_reconstruction(Xte[:, hidden], pred)
    return {'void_auc': motifs['void'], 'overdense_auc': motifs['overdense'], 'peak_auc': motifs['peak'], 'gaussian_corr': corr, 'gaussian_mse': mse}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real/dr11/pilot')
    ap.add_argument('--qc', default='data/real/dr11/systematics')
    ap.add_argument('--out', default='results/real_dr11/latest')
    args = ap.parse_args()
    datadir, qcdir, outdir = Path(args.data), Path(args.qc), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prov = json.loads((datadir/'provenance.json').read_text())
    qprov = json.loads((qcdir/'provenance.json').read_text())
    if prov.get('status') != 'REAL_DR11' or qprov.get('status') != 'REAL_DR11_SYSTEMATICS_QC':
        raise RuntimeError('provenance-verified REAL_DR11 inputs required')
    field_meta = {r['name']:r for r in prov['regions']}; qmeta={r['name']:r for r in qprov['regions']}
    fields = list(field_meta)
    if set(fields) != set(qmeta): raise RuntimeError('QC fields differ from RA/Dec fields')

    target, cov = {}, {}
    for f in fields:
        grid = region_grid(verify_and_load(field_meta[f]), field_meta[f])
        target[f] = normalize_grid(grid)[0]
        cov[f] = cell_systematics(load_qc(qmeta[f]), field_meta[f])

    rows=[]
    for held in fields:
        raw = evaluate_parts(target, fields, held)
        residual, selection_r2 = crossfit_residuals(fields, target, cov, held)
        corrected = evaluate_parts(residual, fields, held)
        shift = evaluate_shift(residual, fields, held)
        for sample, metrics in [('raw', raw), ('systematics_residual', corrected), ('residual_matched_shift', shift)]:
            rows.append({'field':held, 'sample':sample, 'selection_model_cell_r2':selection_r2, **metrics})
    df=pd.DataFrame(rows); df.to_csv(outdir/'systematics_field_metrics.csv', index=False)

    comparisons=[]
    wide=df.pivot(index='field', columns='sample')
    for metric in ['void_auc','overdense_auc','peak_auc','gaussian_corr']:
        raw=wide[metric]['raw']; res=wide[metric]['systematics_residual']; null=wide[metric]['residual_matched_shift']
        for label,diff in [('residual_minus_shift',res-null),('residual_minus_raw',res-raw)]:
            comparisons.append({'metric':metric,'comparison':label,'n_fields':12,'positive_fields':int((diff>0).sum()),'median_raw':float(np.nanmedian(raw)),'median_residual':float(np.nanmedian(res)),'median_null':float(np.nanmedian(null)),'mean_difference':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
    raw=wide['gaussian_mse']['raw']; res=wide['gaussian_mse']['systematics_residual']; null=wide['gaussian_mse']['residual_matched_shift']
    for label,diff in [('shift_minus_residual',null-res),('raw_minus_residual',raw-res)]:
        comparisons.append({'metric':'gaussian_mse','comparison':label,'n_fields':12,'positive_fields':int((diff>0).sum()),'median_raw':float(np.nanmedian(raw)),'median_residual':float(np.nanmedian(res)),'median_null':float(np.nanmedian(null)),'mean_difference':float(np.nanmean(diff)),'exact_one_sided_signflip_p':float(exact_signflip_p(diff.to_numpy()))})
    cdf=pd.DataFrame(comparisons); cdf.to_csv(outdir/'systematics_comparisons.csv', index=False)
    summary={'status':'REAL_DR11','validation':'12-field LOFO observing-condition residualization','model_input_columns':['ra','dec'],'qc_covariates':FEATURES,'qc_role':'nuisance residualization only','selection_model_cell_r2_median':float(df[df['sample']=='raw'].selection_model_cell_r2.median()),'comparisons':comparisons}
    (outdir/'systematics_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
