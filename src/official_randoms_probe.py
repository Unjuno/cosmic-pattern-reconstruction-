#!/usr/bin/env python3
"""Probe official DESI Legacy Surveys DR11 random-catalog availability.

This script deliberately performs only a metadata/provenance probe. The official
DR11 random point catalogs are very large (the release documentation describes
20 shuffled files at 2,500 points/deg^2), so a science workflow must not blindly
download all files in CI. The next stage should materialize a bounded extraction
or an indexed derivative before evaluating the REAL_DR11 locality statistic.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/randoms/"
FILES_DOC = "https://www.legacysurvey.org/dr11/files/"
EXPECTED = [f"randoms-1-{i}.fits" for i in range(20)]


def head(url: str) -> dict:
    req = Request(url, method="HEAD", headers={"User-Agent": "cosmic-pattern-reconstruction/official-randoms-probe"})
    try:
        with urlopen(req, timeout=30) as r:
            return {
                "url": url,
                "status": int(getattr(r, "status", 200)),
                "content_length": int(r.headers.get("Content-Length", "0") or 0),
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "accept_ranges": r.headers.get("Accept-Ranges"),
            }
    except Exception as exc:
        return {"url": url, "status": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real_dr11/official_randoms_probe")
    ap.add_argument("--n", type=int, default=2, help="number of numbered files to HEAD-probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = max(1, min(args.n, len(EXPECTED)))
    files = [head(BASE + name) for name in EXPECTED[:n]]
    summary = {
        "status": "DR11_OFFICIAL_RANDOMS_PROBE",
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "files_documentation": FILES_DOC,
        "base_url": BASE,
        "expected_file_pattern": "randoms-1-[0..19].fits",
        "documented_density_per_sqdeg": 2500,
        "documented_file_count": 20,
        "files": files,
        "science_result": None,
        "interpretation": (
            "Availability/provenance probe only. No cosmological or selection-function "
            "claim is made until a bounded official-random extraction is compared with "
            "the provenance-fixed REAL_DR11 fields."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(x.get("status") in {200, 206} for x in files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
