#!/usr/bin/env python3
"""Re-materialize the 48 provenance-fixed REAL_DR11 RA/Dec fields.

Uses the exact recorded SQL for each accepted field, reapplies the recorded
0.5-degree box crop, and writes the CSV only after row-count and SHA-256
verification. No field selection or mock fallback occurs here.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
from pathlib import Path

import pandas as pd
from dl import queryClient as qc

PROV = Path("data/real/dr11/expanded48/provenance.json")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def query(sql: str, attempts: int = 5) -> str:
    last = None
    for i in range(attempts):
        try:
            out = qc.query(sql=sql, fmt="csv", async_=False)
            if isinstance(out, bytes): out = out.decode()
            if not isinstance(out, str): raise RuntimeError(type(out))
            return out
        except Exception as e:
            last = e
            if i + 1 < attempts: time.sleep(3 * (i + 1))
    raise RuntimeError(f"query failed: {last}")


def main():
    p = json.loads(PROV.read_text())
    regs = p.get("regions", [])
    if p.get("status") != "REAL_DR11" or len(regs) != 48:
        raise RuntimeError("48-field REAL_DR11 provenance required")
    for i, r in enumerate(regs):
        path = Path(r["file"])
        if path.exists():
            gz = path.read_bytes(); raw = gzip.decompress(gz)
            if sha(gz) == r["stored_gzip_sha256"] and sha(raw) == r["canonical_csv_sha256"]:
                print(f"[materialize-metric48] {i+1}/48 {r['name']} cached+verified", flush=True)
                continue
        print(f"[materialize-metric48] {i+1}/48 {r['name']} query", flush=True)
        d = pd.read_csv(io.StringIO(query(str(r["query"]))))
        d.columns = [str(c).lower() for c in d.columns]
        if list(d.columns) != ["ra", "dec"]:
            raise RuntimeError(f"unexpected columns {list(d.columns)}")
        ra0 = float(r["center_ra_deg"]); dec0 = float(r["center_dec_deg"])
        half = float(r.get("box_width_deg", .5)) / 2
        dra = ((d.ra.astype(float) - ra0 + 180) % 360) - 180
        keep = dra.abs().le(half) & d.dec.astype(float).ge(dec0-half) & d.dec.astype(float).lt(dec0+half)
        d = d.loc[keep, ["ra", "dec"]].sort_values(["ra", "dec"], kind="mergesort").reset_index(drop=True)
        raw = d.to_csv(index=False, lineterminator="\n").encode()
        gz = gzip.compress(raw, compresslevel=9, mtime=0)
        if len(d) != int(r["rows"]): raise RuntimeError(f"row mismatch {r['name']}: {len(d)} != {r['rows']}")
        if sha(raw) != r["canonical_csv_sha256"]: raise RuntimeError(f"canonical SHA mismatch {r['name']}")
        if sha(gz) != r["stored_gzip_sha256"]: raise RuntimeError(f"gzip SHA mismatch {r['name']}")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(gz)
    print(json.dumps({"status":"REAL_DR11","materialized_fields":48,"hash_verified":True}, indent=2))

if __name__ == "__main__":
    main()
