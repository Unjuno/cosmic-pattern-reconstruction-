#!/usr/bin/env python3
"""Fetch bounded, real DESI Legacy Surveys DR11 sky positions.

No simulation fallback exists. The pilot deliberately stores only RA/Dec from
BRICK_PRIMARY sources so the first observational experiment is position-only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dl import queryClient as qc

DATASET = "DESI Legacy Imaging Surveys DR11"
TABLE = "ls_dr11.tractor_s"
BOX_DEG = 0.50
QUERY_RADIUS_DEG = 0.40

REGIONS = [
    ("cosmos", 150.12, 2.21),
    ("xmm_lss", 35.70, -4.75),
    ("cdfs", 53.10, -27.80),
    ("elais_s1", 9.45, -44.00),
    ("south_045m30", 45.00, -30.00),
    ("south_075m45", 75.00, -45.00),
    ("south_105m35", 105.00, -35.00),
    ("south_135m25", 135.00, -25.00),
    ("south_210m35", 210.00, -35.00),
    ("des_315m45", 315.00, -45.00),
    ("south_330m45", 330.00, -45.00),
    ("south_345m30", 345.00, -30.00),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def query_with_retries(sql: str, attempts: int = 4) -> str:
    last = None
    for i in range(attempts):
        try:
            result = qc.query(sql=sql, fmt="csv", async_=False)
            if isinstance(result, bytes):
                result = result.decode("utf-8")
            if not isinstance(result, str) or len(result) < 20:
                raise RuntimeError(f"Unexpected Data Lab response: {type(result)}")
            return result
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(8 * (i + 1))
    raise RuntimeError(f"DR11 query failed after {attempts} attempts: {last}")


def fetch_region(name: str, ra0: float, dec0: float, outdir: Path) -> dict:
    half = BOX_DEG / 2.0
    sql = f"""
SELECT ra, dec
FROM {TABLE}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, {QUERY_RADIUS_DEG:.8f})
""".strip()

    text = query_with_retries(sql)
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    if set(df.columns) != {"ra", "dec"}:
        raise RuntimeError(f"{name}: expected RA/Dec only, got {list(df.columns)}")

    dra = ((df["ra"].astype(float) - ra0 + 180.0) % 360.0) - 180.0
    keep = (
        dra.abs().le(half)
        & (df["dec"].astype(float) >= dec0 - half)
        & (df["dec"].astype(float) < dec0 + half)
    )
    df = df.loc[keep, ["ra", "dec"]].copy()
    df = df.sort_values(["ra", "dec"], kind="mergesort").reset_index(drop=True)

    if len(df) < 5000:
        raise RuntimeError(f"{name}: only {len(df)} rows in real DR11 field; refusing weak/empty input")

    canonical = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    gz = gzip.compress(canonical, compresslevel=9, mtime=0)
    path = outdir / f"{name}.csv.gz"
    path.write_bytes(gz)

    return {
        "name": name,
        "center_ra_deg": ra0,
        "center_dec_deg": dec0,
        "box_width_deg": BOX_DEG,
        "query_radius_deg": QUERY_RADIUS_DEG,
        "table": TABLE,
        "query": sql,
        "rows": int(len(df)),
        "columns": ["ra", "dec"],
        "canonical_csv_sha256": sha256_bytes(canonical),
        "stored_gzip_sha256": sha256_bytes(gz),
        "file": str(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/dr11/pilot")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    region_meta = []
    for name, ra0, dec0 in REGIONS:
        print(f"[DR11] fetching {name} @ ({ra0}, {dec0})", flush=True)
        meta = fetch_region(name, ra0, dec0, outdir)
        region_meta.append(meta)
        print(f"[DR11] {name}: {meta['rows']:,} real rows", flush=True)

    split = {
        "train": [r[0] for r in REGIONS[:6]],
        "validation": [r[0] for r in REGIONS[6:8]],
        "test": [r[0] for r in REGIONS[8:]],
    }
    provenance = {
        "status": "REAL_DR11",
        "dataset": DATASET,
        "table": TABLE,
        "retrieved_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_docs": [
            "https://datalab.noirlab.edu/data/legacy-surveys",
            "https://www.legacysurvey.org/dr11/files/",
        ],
        "selection_at_query": "brick_primary = 1",
        "model_input_columns": ["ra", "dec"],
        "regions": region_meta,
        "field_split": split,
        "total_rows": int(sum(r["rows"] for r in region_meta)),
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"[DR11] total real rows: {provenance['total_rows']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
