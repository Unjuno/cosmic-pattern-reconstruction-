#!/usr/bin/env python3
"""Run the preregistered bright/faint experiment with official Tractor TYPE values.

Technical acquisition amendment only.  Astro Data Lab DR11 ``tractor_s.type``
was observed in the diagnostic run to return REF_CAT-like values (G3/L4/NULL)
rather than the documented morphology strings.  To avoid guessing an encoding,
this wrapper uses Data Lab only to discover intersecting BRICKNAME values and
reads TYPE/RA/DEC/FLUX_R/MW_TRANSMISSION_R from the official DR11 Tractor FITS.
The fixed fields, tracer definition, magnitude split, random-half control,
point-random selection model, locality statistic, and decision rule are
unchanged.
"""
from __future__ import annotations

import hashlib
import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

import point_random_bright_faint_locality as experiment

TRACTOR_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor"
NEEDED = ["brick_primary", "type", "ra", "dec", "flux_r", "mw_transmission_r"]


def native(a):
    a = np.asarray(a)
    if a.dtype.kind in "iufcb" and a.dtype.byteorder not in ("=", "|"):
        return a.astype(a.dtype.newbyteorder("="), copy=True)
    return a.copy()


def fetch_tractor(brick: str, attempts: int = 5) -> tuple[pd.DataFrame, dict]:
    url = f"{TRACTOR_BASE}/{brick[:3]}/tractor-{brick}.fits"
    last = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                url,
                timeout=240,
                headers={"User-Agent": "cosmic-pattern-reconstruction-bright-faint/1.0"},
            )
            r.raise_for_status()
            b = r.content
            if not b.startswith(b"SIMPLE"):
                raise RuntimeError(f"not FITS: {b[:40]!r}")
            with fits.open(io.BytesIO(b), memmap=False) as hdus:
                tab = hdus[1].data
                names = {str(c).lower(): str(c) for c in tab.names}
                missing = [c for c in NEEDED if c not in names]
                if missing:
                    raise RuntimeError(f"missing Tractor columns {missing}")
                d = pd.DataFrame({c: native(tab[names[c]]) for c in NEEDED})
            d["type"] = d["type"].map(
                lambda x: bytes(x).decode("ascii", errors="ignore").strip().upper()
                if isinstance(x, (bytes, np.bytes_))
                else str(x).strip().upper()
            )
            return d, {
                "url": url,
                "sha256": hashlib.sha256(b).hexdigest(),
                "bytes": int(len(b)),
                "rows": int(len(d)),
            }
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Tractor download failed {url}: {last}") from last


def official_acquire_field_catalog(meta: dict):
    name = str(meta["name"])
    ra0 = float(meta["center_ra_deg"])
    dec0 = float(meta["center_dec_deg"])

    brick_sql = f"""
SELECT brickname
FROM {experiment.TABLE}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, {experiment.QUERY_RADIUS_DEG:.8f})
""".strip()
    text = experiment.query_csv(brick_sql)
    q = pd.read_csv(io.StringIO(text))
    lower = {str(c).lower(): c for c in q.columns}
    if "brickname" not in lower:
        raise RuntimeError(f"{name}: brick discovery returned {list(q.columns)}")
    bricks = sorted({str(x).strip() for x in q[lower["brickname"]].dropna() if str(x).strip()})
    if not bricks:
        raise RuntimeError(f"{name}: no intersecting bricks")

    parts = []
    tractor_prov = []
    for brick in bricks:
        d, p = fetch_tractor(brick)
        tractor_prov.append(p)
        primary = d.brick_primary.astype(bool).to_numpy()
        ra = pd.to_numeric(d.ra, errors="coerce").to_numpy(float)
        dec = pd.to_numeric(d.dec, errors="coerce").to_numpy(float)
        keep = (
            primary
            & np.isfinite(ra)
            & np.isfinite(dec)
            & (np.abs(experiment.wrap_deg(ra - ra0)) <= experiment.FIELD_HALF_DEG)
            & (dec >= dec0 - experiment.FIELD_HALF_DEG)
            & (dec < dec0 + experiment.FIELD_HALF_DEG)
        )
        if np.any(keep):
            parts.append(d.loc[keep, ["ra", "dec", "type", "flux_r", "mw_transmission_r"]].copy())

    if not parts:
        raise RuntimeError(f"{name}: official Tractor bricks contain no primary rows in fixed square")
    d = pd.concat(parts, ignore_index=True)
    # Multiple candidate bricks should not duplicate BRICK_PRIMARY objects, but
    # keep a deterministic de-duplication guard on exact source coordinates.
    d = d.drop_duplicates(subset=["ra", "dec"], keep="first").reset_index(drop=True)

    flux = pd.to_numeric(d.flux_r, errors="coerce").to_numpy(float)
    mw = pd.to_numeric(d.mw_transmission_r, errors="coerce").to_numpy(float)
    ext = (
        d.type.isin(experiment.EXT_TYPES).to_numpy()
        & np.isfinite(flux)
        & np.isfinite(mw)
        & (flux > 0)
        & (mw > 0)
    )
    e = d.loc[ext].copy().reset_index(drop=True)
    e["dered_flux_r"] = flux[ext] / mw[ext]
    if len(e) < 2 * experiment.MIN_EACH:
        raise RuntimeError(
            f"{name}: only {len(e)} official-FITS extended positive-r sources; "
            f"need >= {2 * experiment.MIN_EACH}"
        )

    order = np.argsort(e.dered_flux_r.to_numpy(float), kind="mergesort")
    n_used = 2 * (len(order) // 2)
    order = order[:n_used]
    half = n_used // 2
    faint = e.iloc[order[:half]].copy()
    bright = e.iloc[order[half:]].copy()

    seed = int(hashlib.sha256(f"{name}|extended-random-half-v1".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_used)
    used = e.iloc[order].reset_index(drop=True)
    random_a = used.iloc[perm[:half]].copy()
    random_b = used.iloc[perm[half:]].copy()

    canonical = experiment.canonical_catalog_bytes(d)
    return {
        "bright": bright,
        "faint": faint,
        "random_a": random_a,
        "random_b": random_b,
    }, {
        "field": name,
        "source": "official DR11 South Tractor FITS",
        "brick_discovery_table": experiment.TABLE,
        "brick_discovery_query": brick_sql,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_bricks": bricks,
        "tractor_files": tractor_prov,
        "square_rows": int(len(d)),
        "square_catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "n_extended_positive_r": int(len(e)),
        "n_used_even": int(n_used),
        "n_each": int(half),
        "median_dered_flux_r": float(np.median(used.dered_flux_r.to_numpy(float))),
        "random_half_seed": int(seed),
        "technical_amendment": "TYPE read from official Tractor FITS after Data Lab diagnostic returned REF_CAT-like G3/L4/NULL values for tractor_s.type",
    }


experiment.acquire_field_catalog = official_acquire_field_catalog

if __name__ == "__main__":
    raise SystemExit(experiment.main())
