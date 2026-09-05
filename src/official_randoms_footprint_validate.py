#!/usr/bin/env python3
"""REAL_DR11 official-random brick-footprint selection stress test.

This is a bounded precursor to a full point-level official-random test. It uses
``survey-bricks-dr11-randoms-5.1.0.fits`` from the official DR11 randoms
products. That file carries the official PHOTSYS assignment together with the
brick geometry, DRVERSION, primary-coverage fractions, and area metadata used
to interpret the point random catalogs.

Primary question
----------------
After correcting the observed source-count field by the official resolved
brick-level primary coverage, does local visible-ring -> hidden-center
continuity remain stronger than a within-field matched-shift control?

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
    if len(v) == 0:
        return out
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


def load_random_bricks(path: Path) -> tuple[np.recarray, dict[str, str], dict]:
    raw = path.read_bytes()
    tab = fits.getdata(path, 1, memmap=True)
    names = {str(n).lower(): str(n) for n in tab.names}
    required = {
        "brickname", "brickid", "ra", "dec", "ra1", "ra2", "dec1", "dec2",
        "fallprimge1dr9s", "fallprimge1dr11", "drversion", "photsys", "area_per_brick",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(
            f"official random brick summary missing columns: {missing}; available={sorted(names)}"
        )
    return tab, names, {
        "sha256": sha256(raw),
        "bytes": len(raw),
        "rows": int(len(tab)),
        "columns": sorted(names),
        "coverage_rule": (
            "PHOTSYS=S and DRVERSION=11 -> FALLPRIMGE1DR11; "
            "PHOTSYS=S and DRVERSION=9 -> FALLPRIMGE1DR9S; otherwise 0"
        ),
    }


def col(tab: np.recarray, names: dict[str, str], key: str) -> np.ndarray:
    return np.asarray(tab[names[key]])


def text_value(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("ascii", errors="ignore").strip()
    return str(x).strip()


def brick_coverage_grid(
    tab: np.recarray, names: dict[str, str], meta: dict
) -> tuple[np.ndarray, dict]:
    ra, dec = cell_centers(meta)
    ra0 = float(meta["center_ra_deg"])
    half = WIDTH_DEG / 2.0 + 0.5

    bra = col(tab, names, "ra").astype(float)
    bdec1 = col(tab, names, "dec1").astype(float)
    bdec2 = col(tab, names, "dec2").astype(float)
    cand = np.flatnonzero(
        (bdec2 > dec.min() - 0.02)
        & (bdec1 <= dec.max() + 0.02)
        & (np.abs(wrap_deg(bra - ra0)) < half)
    )

    bra1 = col(tab, names, "ra1").astype(float)
    bra2 = col(tab, names, "ra2").astype(float)
    drversion = col(tab, names, "drversion").astype(int)
    cov9s = col(tab, names, "fallprimge1dr9s").astype(float)
    cov11 = col(tab, names, "fallprimge1dr11").astype(float)
    photsys = col(tab, names, "photsys")
    bricknames = col(tab, names, "brickname")

    cov = np.full((GRID, GRID), np.nan, float)
    assigned = np.zeros((GRID, GRID), bool)
    used: list[str] = []
    version_counts = {"9": 0, "11": 0, "other": 0}

    for j in cand:
        width = float((bra2[j] - bra1[j]) % 360.0)
        inside = (
            (np.abs(wrap_deg(ra - bra[j])) <= width / 2.0 + 1e-10)
            & (dec >= bdec1[j] - 1e-10)
            & (dec < bdec2[j] + 1e-10)
        )
        if not np.any(inside):
            continue
        ph = text_value(photsys[j])
        drv = int(drversion[j])
        if ph == "S" and drv == 11:
            cv = float(cov11[j])
            version_counts["11"] += 1
        elif ph == "S" and drv == 9:
            cv = float(cov9s[j])
            version_counts["9"] += 1
        else:
            cv = 0.0
            version_counts["other"] += 1
        cov[inside] = cv
        assigned[inside] = True
        used.append(text_value(bricknames[j]))

    finite = np.isfinite(cov)
    return cov, {
        "candidate_bricks": int(len(cand)),
        "used_bricks": sorted(set(used)),
        "used_drversion_counts": version_counts,
        "assigned_fraction": float(assigned.mean()),
        "median_coverage": float(np.nanmedian(cov)) if np.any(finite) else float("nan"),
        "min_coverage": float(np.nanmin(cov)) if np.any(finite) else float("nan"),
        "max_coverage": float(np.nanmax(cov)) if np.any(finite) else float("nan"),
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
    null = rho(ring, np.roll(hidden, max(1, len(hidden) // 3)))
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

    tab, names, random_meta = load_random_bricks(Path(args.random_bricks))
    rows, qrows = [], []
    for i, meta in enumerate(regs):
        counts = source_grid(load_real(meta), meta)
        cov, qm = brick_coverage_grid(tab, names, meta)
        valid = np.isfinite(cov) & (cov >= COVERAGE_FLOOR)
        raw = robust_z(np.log1p(counts), valid)
        adjusted = robust_z(np.log1p(counts / np.clip(cov, COVERAGE_FLOOR, 1.0)), valid)
        rr, rn, nr = patch_locality(raw, valid)
        ar, an, na = patch_locality(adjusted, valid)
        cr = rho(cov[valid], counts[valid]) if np.unique(cov[valid]).size > 1 else float("nan")
        rows.append({
            "field": meta["name"],
            "raw_real_rho": rr,
            "raw_shift_rho": rn,
            "adjusted_real_rho": ar,
            "adjusted_shift_rho": an,
            "adjusted_advantage": ar - an if np.isfinite(ar) and np.isfinite(an) else float("nan"),
            "coverage_count_rho": cr,
            "valid_cell_fraction": float(valid.mean()),
            "n_raw_patches": nr,
            "n_adjusted_patches": na,
        })
        qrows.append({"field": meta["name"], **qm})
        print(
            f"[official-random-brick] {i+1}/48 {meta['name']} "
            f"valid={valid.mean():.3f} adjusted={ar:.3f} shift={an:.3f}",
            flush=True,
        )

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
        "official_random_brick_file": random_meta,
        "primary_hypothesis": {
            "H": "official brick-coverage-adjusted ring-hidden locality remains above matched shift",
            "T": f"48 provenance-fixed REAL_DR11 fields; n_min={MIN_VALID_FIELDS}; one primary paired field statistic",
            "D": f"PASS iff median adjusted advantage >= {EFFECT_FLOOR:.2f} and one-sided sign/Wilcoxon p<0.01; FAIL if median<=0 or either p>=0.05; otherwise UNCERTAIN",
            "C": "official resolved brick-level footprint/coverage explains the observed local continuity",
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
            "PASS rejects only the hypothesis that the official DR11 resolved brick-level footprint/primary-coverage "
            "summary explains the locality signal. It does not establish a cosmological origin and does not replace "
            "a point-level official-random selection-function test."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
