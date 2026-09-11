#!/usr/bin/env python3
"""Disjoint 2%-4% official DR11 point-random replication.

Uses the same 36 fields, selection features, k=4 IDW smoothing, grouped CV,
residual locality statistic, and decision rule as the accepted 0%-2% prefix
run. Only the sampled rows change: this wrapper reads the next disjoint 2% of
each randomized official point-random file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import point_random_selection_residual_validate as base

START_FRACTION = 0.02


def acquire_points_disjoint(regions: dict[str, dict], fraction: float):
    if START_FRACTION < 0 or START_FRACTION + fraction > 0.05:
        raise RuntimeError('disjoint sample must remain inside first 5% window')
    store = {n: [] for n in regions}
    prov = []
    schema0 = None
    for fi in range(base.N_FILES):
        url = f'{base.BASE}/randoms-1-{fi}.fits'
        hdr, meta = base.inspect_fits(url)
        specs, row_bytes = base.field_specs(hdr)
        missing = sorted(set(base.RANDOM_COLUMNS) - set(specs))
        if missing:
            raise RuntimeError(f'file {fi} missing {missing}')
        schema = {k: specs[k]['tform'] for k in sorted(specs)}
        if schema0 is None:
            schema0 = schema
        elif schema != schema0:
            raise RuntimeError(f'schema mismatch file {fi}')

        ntot = int(meta['rows'])
        row0 = int(math.floor(ntot * START_FRACTION))
        ntake = max(1, int(math.floor(ntot * fraction)))
        if row0 + ntake > ntot:
            raise RuntimeError(f'row window exceeds file {fi}')
        start = int(meta['data_offset']) + row0 * row_bytes
        end = start + ntake * row_bytes - 1
        raw, http = base.fetch_range(url, start, end, 480)
        cols = {c: base.extract(raw, row_bytes, specs[c]) for c in base.RANDOM_COLUMNS}
        ra = cols['RA'].astype(float)
        dec = cols['DEC'].astype(float)
        hits = 0
        for name, r in regions.items():
            ra0 = float(r['center_ra_deg'])
            dec0 = float(r['center_dec_deg'])
            half = float(r.get('box_width_deg', .5)) / 2
            m = (np.abs(base.wrap_deg(ra - ra0)) <= half) & (dec >= dec0 - half) & (dec < dec0 + half)
            ii = np.flatnonzero(m)
            if not len(ii):
                continue
            hits += len(ii)
            d = {}
            for c in base.RANDOM_COLUMNS:
                a = cols[c][ii]
                if c == 'PHOTSYS':
                    d[c] = [bytes(x).decode('ascii', errors='ignore').strip() for x in a]
                else:
                    dtype = np.int64 if c in ['MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_I', 'NOBS_Z'] else float
                    d[c] = a.astype(dtype)
            d['FILE_INDEX'] = np.full(len(ii), fi, dtype=np.int16)
            d['PREFIX_ROW'] = (row0 + ii).astype(np.int64)
            store[name].append(pd.DataFrame(d))

        meta.update({
            'file_index': fi,
            'row_start': row0,
            'row_stop_exclusive': row0 + ntake,
            'row_start_fraction_nominal': START_FRACTION,
            'rows_sampled': ntake,
            'sample_fraction_effective': ntake / ntot,
            'sample_data_bytes': len(raw),
            'sample_data_sha256': base.sha256(raw),
            'data_http': http,
            'target_hits_locked36': int(hits),
        })
        prov.append(meta)
        print(f'[point-selection-disjoint] random {fi+1:02d}/20 rows={row0}:{row0+ntake} hits36={hits}', flush=True)
        del raw, cols

    out = {}
    for n, parts in store.items():
        out[n] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=base.RANDOM_COLUMNS)
    return out, prov


def main() -> int:
    base.acquire_points = acquire_points_disjoint
    rc = base.main()
    # Re-label the generated result so provenance cannot be confused with 0%-2%.
    out = Path('results/real_dr11/point_random_selection_residual36_disjoint')
    if '--out' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--out') + 1])
    p = out / 'summary.json'
    if p.exists():
        import json
        d = json.loads(p.read_text())
        d['status'] = 'REAL_DR11_POINT_RANDOM_SELECTION_RESIDUAL36_DISJOINT_2_4'
        d['row_start_fraction_per_file'] = START_FRACTION
        d['T'] = 'same 36 fields and same pipeline; disjoint rows 2%-4% of each of 20 randomized official point-random files; k=4 IDW selection surfaces; 6-fold whole-field CV'
        d['replication_role'] = 'independent finite-random/kNN-surface replication of the 0%-2% point-random test'
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
