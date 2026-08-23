#!/usr/bin/env python3
"""Discover queryable ls_dr11 tables/columns through Data Lab TAP metadata."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from dl import queryClient as qc


def query(sql: str) -> pd.DataFrame:
    text = qc.query(sql=sql, fmt='csv', async_=False)
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    from io import StringIO
    return pd.read_csv(StringIO(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/real_dr11/latest/schema')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tables_sql = "SELECT schema_name, table_name, description FROM tap_schema.tables WHERE schema_name='ls_dr11' ORDER BY table_name"
    # TAP_SCHEMA.columns is keyed by the fully-qualified table_name and does
    # not expose schema_name on this service.
    cols_sql = "SELECT table_name, column_name, datatype, description FROM tap_schema.columns WHERE table_name IN ('ls_dr11.tractor','ls_dr11.tractor_s','ls_dr11.tractor_n') ORDER BY table_name, column_name"
    tables = query(tables_sql)
    cols = query(cols_sql)
    tables.to_csv(out/'tables.csv', index=False)
    cols.to_csv(out/'tractor_columns.csv', index=False)
    names = [str(x).lower() for x in tables['table_name'].tolist()]
    random_like = [x for x in names if 'random' in x]
    summary = {
        'status': 'REAL_DR11_SCHEMA_DISCOVERY',
        'n_tables': int(len(tables)),
        'random_like_tables': random_like,
        'queries': {'tables': tables_sql, 'columns': cols_sql},
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
