#!/usr/bin/env python3
"""Run the preregistered bright/faint experiment using the audited Data Lab DR11 column swap.

Technical acquisition amendment only.  Two independent Data Lab probes showed
that both ls_dr11.tractor_s and ls_dr11.tractor return morphology-like values
(PSF/REX/DEV/EXP/SER) in ``ref_cat`` while ``type`` contains REF_CAT-like
values (G3/L4/NULL).  This wrapper therefore queries both columns, validates the
observed mapping field by field, and uses the morphology-like ``ref_cat`` values
as the effective Tractor TYPE.  The fixed fields, tracer definition, magnitude
split, random-half control, point-random sample, nuisance model, locality
statistic, and decision rule are unchanged.
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import point_random_bright_faint_locality as experiment

MORPH_ALLOWED = {"PSF", "REX", "EXP", "DEV", "SER", "DUP"}


def norm_text_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    return x.where(~x.eq("NAN"), "")


def datalab_swap_acquire_field_catalog(meta: dict):
    name = str(meta["name"])
    ra0 = float(meta["center_ra_deg"])
    dec0 = float(meta["center_dec_deg"])
    sql = f"""
SELECT ra, dec, type, ref_cat, flux_r, mw_transmission_r
FROM {experiment.TABLE}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, {experiment.QUERY_RADIUS_DEG:.8f})
""".strip()
    text = experiment.query_csv(sql)
    raw = pd.read_csv(io.StringIO(text))
    required = ["ra", "dec", "type", "ref_cat", "flux_r", "mw_transmission_r"]
    lower = {str(c).lower(): c for c in raw.columns}
    missing = [c for c in required if c not in lower]
    if missing:
        raise RuntimeError(f"{name}: missing Data Lab columns {missing}; got {list(raw.columns)}")
    d = raw.rename(columns={lower[c]: c for c in required})[required].copy()

    ra = pd.to_numeric(d.ra, errors="coerce").to_numpy(float)
    dec = pd.to_numeric(d.dec, errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & (np.abs(experiment.wrap_deg(ra - ra0)) <= experiment.FIELD_HALF_DEG)
        & (dec >= dec0 - experiment.FIELD_HALF_DEG)
        & (dec < dec0 + experiment.FIELD_HALF_DEG)
    )
    d = d.loc[keep].copy().reset_index(drop=True)
    dl_type = norm_text_series(d["type"])
    dl_ref = norm_text_series(d["ref_cat"])

    nonblank_ref = dl_ref[dl_ref.ne("")]
    bad_ref = sorted(set(nonblank_ref) - MORPH_ALLOWED)
    morph_frac = float(nonblank_ref.isin(MORPH_ALLOWED).mean()) if len(nonblank_ref) else 0.0
    if len(nonblank_ref) == 0 or morph_frac < 0.995 or bad_ref:
        raise RuntimeError(
            f"{name}: Data Lab swap audit failed: nonblank ref_cat={len(nonblank_ref)} "
            f"morphology_fraction={morph_frac:.6f} bad={bad_ref[:20]}"
        )
    # Guard against silently returning the originally documented mapping again.
    # If TYPE itself is predominantly morphology, the workaround is no longer
    # appropriate and the run must stop for a fresh schema review.
    nonblank_type = dl_type[dl_type.ne("")]
    type_morph_frac = float(nonblank_type.isin(MORPH_ALLOWED).mean()) if len(nonblank_type) else 0.0
    if type_morph_frac > 0.50:
        raise RuntimeError(
            f"{name}: Data Lab mapping appears repaired/changed; type morphology fraction="
            f"{type_morph_frac:.6f}. Do not apply swap workaround."
        )

    # Effective morphology, preserving the preregistered TYPE definition.
    d["datalab_type_raw"] = dl_type
    d["datalab_ref_cat_raw"] = dl_ref
    d["type"] = dl_ref
    d = d[["ra", "dec", "type", "flux_r", "mw_transmission_r", "datalab_type_raw", "datalab_ref_cat_raw"]]

    flux = pd.to_numeric(d.flux_r, errors="coerce").to_numpy(float)
    mw = pd.to_numeric(d.mw_transmission_r, errors="coerce").to_numpy(float)
    ext = d.type.isin(experiment.EXT_TYPES).to_numpy() & np.isfinite(flux) & np.isfinite(mw) & (flux > 0) & (mw > 0)
    e = d.loc[ext].copy().reset_index(drop=True)
    e["dered_flux_r"] = flux[ext] / mw[ext]
    if len(e) < 2 * experiment.MIN_EACH:
        raise RuntimeError(f"{name}: only {len(e)} extended positive-r sources; need >= {2*experiment.MIN_EACH}")

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
        "table": experiment.TABLE,
        "query": sql,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "radial_rows_returned": int(len(raw)),
        "square_rows": int(len(d)),
        "square_catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "n_extended_positive_r": int(len(e)),
        "n_used_even": int(n_used),
        "n_each": int(half),
        "median_dered_flux_r": float(np.median(used.dered_flux_r.to_numpy(float))),
        "random_half_seed": int(seed),
        "datalab_column_swap_audit": {
            "effective_morphology_column": "ref_cat",
            "raw_type_nonblank": int(dl_type.ne("").sum()),
            "raw_type_morphology_fraction": type_morph_frac,
            "raw_ref_cat_nonblank": int(dl_ref.ne("").sum()),
            "raw_ref_cat_morphology_fraction": morph_frac,
            "raw_type_top_counts": {str(k): int(v) for k, v in dl_type.value_counts().head(12).items()},
            "raw_ref_cat_top_counts": {str(k): int(v) for k, v in dl_ref.value_counts().head(12).items()},
        },
        "technical_amendment": "Use Data Lab ref_cat as effective morphology TYPE only after strict per-field audit of the observed DR11 type/ref_cat swap; no science threshold changed.",
    }


experiment.acquire_field_catalog = datalab_swap_acquire_field_catalog

if __name__ == "__main__":
    raise SystemExit(experiment.main())
