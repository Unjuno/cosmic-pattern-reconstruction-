#!/usr/bin/env python3
from __future__ import annotations
import io, json
import numpy as np
import pandas as pd
from dl import queryClient as qc

ra0, dec0 = 156.0, 5.0

def run(table: str):
    sql = f"""
SELECT ra, dec, type, ref_cat, flux_r, mw_transmission_r
FROM {table}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, 0.10000000)
""".strip()
    x = qc.query(sql=sql, fmt='csv', async_=False)
    if isinstance(x, bytes):
        x = x.decode('utf-8')
    d = pd.read_csv(io.StringIO(x))
    t = d['type'].astype(str)
    r = d['ref_cat'].astype(str)
    flux = pd.to_numeric(d.flux_r, errors='coerce').to_numpy(float)
    mw = pd.to_numeric(d.mw_transmission_r, errors='coerce').to_numpy(float)
    return {
      'table': table,
      'rows': int(len(d)),
      'columns': list(d.columns),
      'type_head_repr': [repr(v) for v in t.head(20).tolist()],
      'type_counts': {str(k): int(v) for k,v in t.str.strip().str.upper().value_counts(dropna=False).head(20).items()},
      'ref_cat_counts': {str(k): int(v) for k,v in r.str.strip().str.upper().value_counts(dropna=False).head(20).items()},
      'type_equals_ref_cat_fraction': float(np.mean(t.str.strip().str.upper().to_numpy() == r.str.strip().str.upper().to_numpy())) if len(d) else None,
      'finite_flux_r': int(np.isfinite(flux).sum()),
      'positive_flux_r': int((np.isfinite(flux)&(flux>0)).sum()),
      'finite_mw_transmission_r': int(np.isfinite(mw).sum()),
      'positive_mw_transmission_r': int((np.isfinite(mw)&(mw>0)).sum()),
      'sql': sql,
    }

print(json.dumps({t: run(t) for t in ['ls_dr11.tractor_s','ls_dr11.tractor']}, indent=2, sort_keys=True))
