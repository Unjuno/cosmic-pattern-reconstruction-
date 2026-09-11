#!/usr/bin/env python3
from __future__ import annotations
import hashlib, io, json
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from dl import queryClient as qc

ra0, dec0 = 156.0, 5.0
TRACTOR_BASE = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor'

def norm_text(v):
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        return bytes(v).decode('ascii', errors='ignore').strip().upper()
    if pd.isna(v):
        return ''
    s = str(v).strip().upper()
    return '' if s == 'NAN' else s

def query_df(table: str, radius: float = 0.10):
    sql = f"""
SELECT ls_id_dr11, brickname, ra, dec, type, ref_cat, flux_r, mw_transmission_r
FROM {table}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, {radius:.8f})
""".strip()
    x = qc.query(sql=sql, fmt='csv', async_=False)
    if isinstance(x, bytes):
        x = x.decode('utf-8')
    return pd.read_csv(io.StringIO(x)), sql

def summarize(table: str):
    d, sql = query_df(table)
    t = d['type'].map(norm_text)
    r = d['ref_cat'].map(norm_text)
    flux = pd.to_numeric(d.flux_r, errors='coerce').to_numpy(float)
    mw = pd.to_numeric(d.mw_transmission_r, errors='coerce').to_numpy(float)
    return d, {
      'table': table,
      'rows': int(len(d)),
      'columns': list(d.columns),
      'type_counts': {str(k): int(v) for k,v in t.value_counts(dropna=False).head(20).items()},
      'ref_cat_counts': {str(k): int(v) for k,v in r.value_counts(dropna=False).head(20).items()},
      'finite_flux_r': int(np.isfinite(flux).sum()),
      'positive_flux_r': int((np.isfinite(flux)&(flux>0)).sum()),
      'finite_mw_transmission_r': int(np.isfinite(mw).sum()),
      'positive_mw_transmission_r': int((np.isfinite(mw)&(mw>0)).sum()),
      'sql': sql,
    }

def official_crosscheck(d: pd.DataFrame):
    brick = str(d.brickname.mode().iloc[0]).strip()
    q = d[d.brickname.astype(str).str.strip().eq(brick)].copy()
    url = f'{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits'
    resp = requests.get(url, timeout=240, headers={'User-Agent':'cosmic-pattern-reconstruction-morphology-audit/1.0'})
    resp.raise_for_status()
    b = resp.content
    if not b.startswith(b'SIMPLE'):
        raise RuntimeError(f'not FITS: {b[:40]!r}')
    with fits.open(io.BytesIO(b), memmap=False) as H:
        tab = H[1].data
        names = {str(c).lower(): str(c) for c in tab.names}
        need = ['ls_id_dr11','type','ref_cat']
        missing = [c for c in need if c not in names]
        if missing:
            raise RuntimeError(f'missing official columns {missing}')
        od = pd.DataFrame({
            'ls_id_dr11': np.asarray(tab[names['ls_id_dr11']]).astype(np.int64, copy=False),
            'official_type': [norm_text(x) for x in tab[names['type']]],
            'official_ref_cat': [norm_text(x) for x in tab[names['ref_cat']]],
        })
    q['ls_id_dr11'] = pd.to_numeric(q.ls_id_dr11, errors='coerce').astype('Int64')
    q = q.dropna(subset=['ls_id_dr11']).copy()
    q['ls_id_dr11'] = q.ls_id_dr11.astype(np.int64)
    m = q.merge(od, on='ls_id_dr11', how='inner')
    dl_type = m['type'].map(norm_text).to_numpy()
    dl_ref = m['ref_cat'].map(norm_text).to_numpy()
    ot = m['official_type'].to_numpy()
    orc = m['official_ref_cat'].to_numpy()
    return {
      'brick': brick,
      'official_url': url,
      'official_sha256': hashlib.sha256(b).hexdigest(),
      'official_bytes': int(len(b)),
      'matched_rows': int(len(m)),
      'dl_ref_cat_equals_official_type_fraction': float(np.mean(dl_ref == ot)) if len(m) else None,
      'dl_type_equals_official_ref_cat_fraction': float(np.mean(dl_type == orc)) if len(m) else None,
      'dl_type_equals_official_type_fraction': float(np.mean(dl_type == ot)) if len(m) else None,
      'dl_ref_cat_equals_official_ref_cat_fraction': float(np.mean(dl_ref == orc)) if len(m) else None,
      'examples': [
        {
          'ls_id_dr11': int(row.ls_id_dr11),
          'dl_type': norm_text(row.type),
          'dl_ref_cat': norm_text(row.ref_cat),
          'official_type': norm_text(row.official_type),
          'official_ref_cat': norm_text(row.official_ref_cat),
        }
        for row in m.head(20).itertuples(index=False)
      ],
    }

south, south_summary = summarize('ls_dr11.tractor_s')
joint, joint_summary = summarize('ls_dr11.tractor')
rec = {
  'ls_dr11.tractor_s': south_summary,
  'ls_dr11.tractor': joint_summary,
  'official_crosscheck': official_crosscheck(south),
}
print(json.dumps(rec, indent=2, sort_keys=True))
