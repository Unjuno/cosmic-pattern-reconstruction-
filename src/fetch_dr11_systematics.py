#!/usr/bin/env python3
"""Fetch DR11 observing-condition columns for the already-provenanced RA/Dec fields.

These columns are QC covariates only. They are never used as cosmological model
features. Retrieval must reproduce exactly the RA/Dec row count of each baseline
field; otherwise the script aborts.
"""
from __future__ import annotations

import argparse, gzip, hashlib, io, json, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dl import queryClient as qc

TABLE = 'ls_dr11.tractor_s'
COLUMNS = [
    'ra','dec','maskbits',
    'nobs_g','nobs_r','nobs_i','nobs_z',
    'psfdepth_g','psfdepth_r','psfdepth_i','psfdepth_z',
    'psfsize_g','psfsize_r','psfsize_i','psfsize_z',
    'mw_transmission_g','mw_transmission_r','mw_transmission_i','mw_transmission_z',
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def q(sql: str, attempts: int = 4) -> str:
    last = None
    for i in range(attempts):
        try:
            out = qc.query(sql=sql, fmt='csv', async_=False)
            if isinstance(out, bytes): out = out.decode('utf-8')
            if not isinstance(out, str) or len(out) < 20: raise RuntimeError(f'unexpected response {type(out)}')
            return out
        except Exception as exc:
            last = exc
            if i+1 < attempts: time.sleep(8*(i+1))
    raise RuntimeError(f'DR11 systematics query failed: {last}')


def tangent_ra(ra, ra0):
    return ((np.asarray(ra, float)-ra0+180.0)%360.0)-180.0


def validate(df: pd.DataFrame, name: str) -> dict:
    if list(df.columns) != COLUMNS:
        raise RuntimeError(f'{name}: unexpected columns {list(df.columns)}')
    mb = pd.to_numeric(df.maskbits, errors='raise')
    if (mb < 0).any(): raise RuntimeError(f'{name}: negative maskbits')
    for c in ['nobs_g','nobs_r','nobs_i','nobs_z']:
        x = pd.to_numeric(df[c], errors='raise')
        if (x < 0).any() or (x > 10000).any(): raise RuntimeError(f'{name}: invalid {c} range')
    for c in ['psfdepth_g','psfdepth_r','psfdepth_i','psfdepth_z']:
        x = pd.to_numeric(df[c], errors='coerce')
        if (x.dropna() < 0).any(): raise RuntimeError(f'{name}: negative {c}')
    for c in ['psfsize_g','psfsize_r','psfsize_i','psfsize_z']:
        x = pd.to_numeric(df[c], errors='coerce')
        good = x[(x > 0) & np.isfinite(x)]
        if len(good) and (good > 20).any(): raise RuntimeError(f'{name}: implausible {c}')
    for c in ['mw_transmission_g','mw_transmission_r','mw_transmission_i','mw_transmission_z']:
        x = pd.to_numeric(df[c], errors='coerce')
        good = x[np.isfinite(x)]
        if len(good) and ((good < 0).any() or (good > 1.0001).any()): raise RuntimeError(f'{name}: invalid {c}')
    return {
        c: {
            'finite_fraction': float(pd.to_numeric(df[c], errors='coerce').notna().mean()),
            'median': float(np.nanmedian(pd.to_numeric(df[c], errors='coerce'))),
        } for c in COLUMNS[2:]
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='data/real/dr11/pilot/provenance.json')
    ap.add_argument('--out', default='data/real/dr11/systematics')
    args = ap.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    if baseline.get('status') != 'REAL_DR11' or baseline.get('model_input_columns') != ['ra','dec']:
        raise RuntimeError('baseline must be provenance-verified REAL_DR11 RA/Dec')
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    records=[]
    for meta in baseline['regions']:
        name, ra0, dec0 = meta['name'], float(meta['center_ra_deg']), float(meta['center_dec_deg'])
        radius = float(meta['query_radius_deg']); half = float(meta['box_width_deg'])/2
        sql = f"SELECT {','.join(COLUMNS)} FROM {TABLE} WHERE brick_primary=1 AND q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},{radius:.8f})"
        print(f'[DR11-QC] {name}', flush=True)
        df = pd.read_csv(io.StringIO(q(sql)))
        df.columns = [str(c).lower() for c in df.columns]
        dra = tangent_ra(df.ra, ra0)
        keep = (np.abs(dra) <= half) & (df.dec.astype(float) >= dec0-half) & (df.dec.astype(float) < dec0+half)
        df = df.loc[keep, COLUMNS].copy().sort_values(['ra','dec'], kind='mergesort').reset_index(drop=True)
        if len(df) != int(meta['rows']):
            raise RuntimeError(f'{name}: baseline rows {meta["rows"]}, QC rows {len(df)}; refusing mismatched selection')
        stats = validate(df, name)
        raw = df.to_csv(index=False, lineterminator='\n').encode('utf-8')
        gz = gzip.compress(raw, compresslevel=9, mtime=0)
        path = outdir/f'{name}.csv.gz'; path.write_bytes(gz)
        records.append({
            'name': name, 'rows': int(len(df)), 'query': sql, 'file': str(path),
            'canonical_csv_sha256': sha256(raw), 'stored_gzip_sha256': sha256(gz),
            'column_stats': stats,
        })
    prov = {
        'status': 'REAL_DR11_SYSTEMATICS_QC',
        'dataset': baseline['dataset'], 'table': TABLE,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'observing-condition QC only; not cosmological model input',
        'columns': COLUMNS, 'regions': records,
    }
    (outdir/'provenance.json').write_text(json.dumps(prov, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': prov['status'], 'regions': len(records), 'rows': sum(r['rows'] for r in records)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
