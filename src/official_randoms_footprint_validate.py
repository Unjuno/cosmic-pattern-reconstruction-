#!/usr/bin/env python3
"""REAL_DR11 official-random brick-footprint selection stress test.

This is a bounded precursor to a full point-level official-random test.  It uses
``survey-bricks-dr11-randoms-5.1.0.fits`` from the official DR11 randoms
products.  That file is derived for interpreting the official random catalogs
and contains the random-catalog PHOTSYS assignment plus brick geometry/area and
the DR11 primary-coverage fraction inherited from ``survey-bricks.fits``.

Primary question
----------------
After correcting the observed source-count field by the official brick-level
DR11-S primary coverage, does local visible-ring -> hidden-center continuity
remain stronger than a within-field matched-shift control?

This does NOT replace a point-level random-catalog test: sub-brick masks, depth,
PSF, deblending, foregrounds and other selection structure can remain.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import binomtest, spearmanr, wilcoxon

GRID = 64
WIDTH_DEG = 0.28
PATCH = 16
STRIDE = 8
COVERAGE_FLOOR = 0.05
MIN_VALID_FIELDS = 36
EFFECT_FLOOR = 0.10

HIDDEN = np.zeros((PATCH, PATCH), bool)
HIDDEN[4:12, 4:12] = True
RING = np.zeros((PATCH, PATCH), bool)
RING[3, 3:13] = True
RING[12, 3:13] = True
RING[4:12, 3] = True
RING[4:12, 12] = True


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wrap_deg(x: np.ndarray | float) -> np.ndarray:
    return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0


def robust_z(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    v = np.asarray(a[valid], float)
    med = np.median(v)
    scale = np.median(np.abs(v - med)) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(v)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    out[valid] = (v - med) / scale
    return out


def rho(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic
    return float(r) if np.isfinite(r) else float("nan")


def load_real(meta: dict) -> pd.DataFrame:
    path = Path(meta["file"])
    gz = path.read_bytes()
    raw = gzip.decompress(gz)
    if sha256(gz) != meta["stored_gzip_sha256"]:
        raise RuntimeError(f"gzip SHA mismatch: {path}")
    if sha256(raw) != meta["canonical_csv_sha256"]:
        raise RuntimeError(f"canonical SHA mismatch: {path}")
    d = pd.read_csv(path)
    if list(d.columns) != ["ra", "dec"] or len(d) != int(meta["rows"]):
        raise RuntimeError(f"invalid REAL_DR11 field: {path}")
    return d


def source_grid(d: pd.DataFrame, meta: dict) -> np.ndarray:
    ra0 = float(meta["center_ra_deg"])
    dec0 = float(meta["center_dec_deg"])
    dra = wrap_deg(d.ra.to_numpy(float) - ra0)
    x = dra * np.cos(np.deg2rad(dec0))
    y = d.dec.to_numpy(float) - dec0
    h = WIDTH_DEG / 2.0
    g, _, _ = np.histogram2d(y, x, bins=GRID, range=[[-h, h], [-h, h]])
    return g.astype(float)


def cell_centers(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    ra0 = float(meta["center_ra_deg"])
    dec0 = float(meta["center_dec_deg"])
    step = WIDTH_DEG / GRID
    axis = -WIDTH_DEG / 2.0 + step * (np.arange(GRID) + 0.5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    dec = dec0 + yy
    ra = (ra0 + xx / np.cos(np.deg2rad(dec0))) % 360.0
    return ra, dec


def read_random_bricks(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = path.read_bytes()
    with fits.open(path, memmap=True) as hdul:
        tab = hdul[1].data
        names = [str(n).lower() for n in tab.names]
        d = pd.DataFrame({n: np.asarray(tab[orig]).byteswap().view(np.asarray(tab[orig]).dtype.newbyteorder("="))
                          for n, orig in zip(names, tab.names)})
    for c in d.columns:
        if d[c].dtype.kind == "S":
            d[c] = d[c].str.decode("ascii", errors="ignore")
    required = {"ra", "dec", "ra1", "ra2", "dec1", "dec2", "photsys", "area_per_brick"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"official random brick summary missing columns: {missing}; available={list(d.columns)}")
    cov_candidates = [c for c in d.columns if "fallprimge1" in c and "dr11" in c and c.endswith("s")]
    if len(cov_candidates) != 1:
        raise RuntimeError(
            "expected exactly one DR11-S primary-coverage column in official random brick summary; "
            f"candidates={cov_candidates}; available={list(d.columns)}"
        )
    covcol = cov_candidates[0]
    return d, {
        "sha256": sha256(raw),
        "bytes": len(raw),
        "rows": int(len(d)),
        "columns": list(d.columns),
        "coverage_column": covcol,
    }


def brick_coverage_grid(bricks: pd.DataFrame, meta: dict, covcol: str) -> tuple[np.ndarray, dict]:
    ra, dec = cell_centers(meta)
    ra0 = float(meta["center_ra_deg"])
    half = WIDTH_DEG / 2.0 + 0.5
    cand = bricks[
        (bricks.dec2.astype(float) > dec.min() - 0.02)
        & (bricks.dec1.astype(float) <= dec.max() + 0.02)
        & (np.abs(wrap_deg(bricks.ra.astype(float).to_numpy() - ra0)) < half)
    ].copy()
    cov = np.full((GRID, GRID), np.nan, float)
    assigned = np.zeros((GRID, GRID), bool)
    used = []
    for r in cand.itertuples(index=False):
        br = float(getattr(r, "ra"))
        w = float((float(getattr(r, "ra2")) - float(getattr(r, "ra1"))) % 360.0)
        inside = (
            (np.abs(wrap_deg(ra - br)) <= w / 2.0 + 1e-10)
            & (dec >= float(getattr(r, "dec1")) - 1e-10)
            & (dec < float(getattr(r, "dec2")) + 1e-10)
        )
        if not np.any(inside):
            continue
        ph = str(getattr(r, "photsys")).strip()
        cv = float(getattr(r, covcol)) if ph == "S" else 0.0
        cov[inside] = cv
        assigned[inside] = True
        used.append(str(getattr(r, "brickname")) if hasattr(r, "brickname") else f"brick:{getattr(r, 'brickid', '?')}")
    return cov, {
        "candidate_bricks": int(len(cand)),
        "used_bricks": sorted(set(used)),
        "assigned_fraction": float(assigned.mean()),
        "median_coverage": float(np.nanmedian(cov)) if np.any(np.isfinite(cov)) else float("nan"),
    }


def patch_locality(z: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    ring, hidden = [], []
    for y in range(0, GRID - PATCH + 1, STRIDE):
        for x in range(0, GRID - PATCH + 1, STRIDE):
            p = z[y:y+PATCH, x:x+PATCH]
            v = valid[y:y+PATCH, x:x+PATCH]
            if not np.all(v[RING]) or not np.all(v[HIDDEN]):
                continue
            ring.append(float(np.mean(p[RING])))
            hidden.append(float(np.mean(p[HIDDEN])))
    ring = np.asarray(ring, float)
    hidden = np.asarray(hidden, float)
    if len(ring) < 10:
        return float("nan"), float("nan"), int(len(ring))
    real = rho(ring, hidden)
    shift = max(1, len(hidden) // 3)
    null = rho(ring, np.roll(hidden, shift))
    return real, null, int(len(ring))


def paired_stats(delta: np.ndarray) -> dict:
    a = np.asarray(delta, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    nz = a[a != 0]
    pos = int((a > 0).sum())
    sign_p = float(binomtest(pos, len(a), 0.5, alternative="greater").pvalue)
    try:
        wx = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) else 1.0
    except Exception:
        wx = float("nan")
    return {
        "n": int(len(a)),
        "positive": pos,
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "sign_p_one_sided": sign_p,
        "wilcoxon_p_one_sided": wx,
    }


def decision(primary: dict) -> str:
    if primary.get("n", 0) < MIN_VALID_FIELDS:
        return "UNCERTAIN"
    med = float(primary["median"])
    sp = float(primary["sign_p_one_sided"])
    wp = float(primary["wilcoxon_p_one_sided"])
    if med >= EFFECT_FLOOR and sp < 0.01 and wp < 0.01:
        return "PASS"
    if med <= 0.0 or sp >= 0.05 or wp >= 0.05:
        return "FAIL"
    return "UNCERTAIN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/dr11/expanded48")
    ap.add_argument("--random-bricks", required=True)
    ap.add_argument("--out", default="results/real_dr11/official_random_brick48")
    args = ap.parse_args()
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = json.loads((root / "provenance.json").read_text())
    regs = prov.get("regions", [])
    if prov.get("status") != "REAL_DR11" or len(regs) != 48:
        raise RuntimeError("48-field REAL_DR11 provenance required")

    bricks, bmeta = read_random_bricks(Path(args.random_bricks))
    covcol = bmeta["coverage_column"]
    rows = []
    qrows = []
    for i, meta in enumerate(regs):
        counts = source_grid(load_real(meta), meta)
        cov, qm = brick_coverage_grid(bricks, meta, covcol)
        valid = np.isfinite(cov) & (cov >= COVERAGE_FLOOR)
        if valid.mean() < 0.85:
            print(f"[official-random-brick] {meta['name']} low valid fraction={valid.mean():.3f}", flush=True)
        raw = robust_z(np.log1p(counts), valid)
        adjusted = robust_z(np.log1p(counts / np.clip(cov, COVERAGE_FLOOR, 1.0)), valid)
        rr, rn, nr = patch_locality(raw, valid)
        ar, an, na = patch_locality(adjusted, valid)
        coverage_count_rho = rho(cov[valid], counts[valid]) if np.unique(cov[valid]).size > 1 else float("nan")
        rows.append({
            "field": meta["name"],
            "raw_real_rho": rr,
            "raw_shift_rho": rn,
            "adjusted_real_rho": ar,
            "adjusted_shift_rho": an,
            "adjusted_advantage": ar - an if np.isfinite(ar) and np.isfinite(an) else float("nan"),
            "coverage_count_rho": coverage_count_rho,
            "valid_cell_fraction": float(valid.mean()),
            "n_raw_patches": nr,
            "n_adjusted_patches": na,
        })
        qrows.append({"field": meta["name"], **qm})
        print(f"[official-random-brick] {i+1}/48 {meta['name']} adjusted={ar:.3f} shift={an:.3f}", flush=True)

    df = pd.DataFrame(rows)
    qf = pd.DataFrame(qrows)
    df.to_csv(out / "field_metrics.csv", index=False)
    qf.to_csv(out / "coverage_mapping_qc.csv", index=False)
    primary = paired_stats(df.adjusted_advantage.to_numpy())
    result = decision(primary)
    summary = {
        "status": "REAL_DR11_OFFICIAL_RANDOM_BRICK_FOOTPRINT_TEST",
        "decision": result,
        "science_scope": "official-random brick-level footprint/primary-coverage stress test; not point-level randoms",
        "n_fields_requested": 48,
        "input_total_rows": int(prov["total_rows"]),
        "projection": "tangent plane x=dRA*cos(dec0), y=dDec",
        "grid": GRID,
        "physical_square_width_deg": WIDTH_DEG,
        "coverage_floor": COVERAGE_FLOOR,
        "official_random_brick_file": bmeta,
        "primary_hypothesis": {
            "H": "brick-coverage-adjusted ring-hidden locality remains above matched shift",
            "T": f"48 provenance-fixed REAL_DR11 fields; n_min={MIN_VALID_FIELDS}; one primary paired field statistic",
            "D": f"PASS iff median adjusted advantage >= {EFFECT_FLOOR:.2f} and one-sided sign/Wilcoxon p<0.01; FAIL if median<=0 or either p>=0.05; otherwise UNCERTAIN",
            "C": "official brick-level footprint/coverage explains the observed local continuity",
            "U": "sub-brick masks/depth/PSF, deblending, Galactic foregrounds, tracer population, brick-summary approximation",
        },
        "primary": primary,
        "raw_median_real_rho": float(np.nanmedian(df.raw_real_rho)),
        "raw_median_shift_rho": float(np.nanmedian(df.raw_shift_rho)),
        "adjusted_median_real_rho": float(np.nanmedian(df.adjusted_real_rho)),
        "adjusted_median_shift_rho": float(np.nanmedian(df.adjusted_shift_rho)),
        "coverage_count_rho_median": float(np.nanmedian(df.coverage_count_rho)),
        "median_valid_cell_fraction": float(np.nanmedian(df.valid_cell_fraction)),
        "interpretation_guardrail": (
            "PASS would reject only the hypothesis that the official DR11 brick-level footprint/primary-coverage summary "
            "explains the locality signal. It would not establish a cosmological origin and would not replace a "
            "point-level official-random selection-function test."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
