#!/usr/bin/env python3
"""REAL_DR11 metric-corrected locality/isotropy validation on 48 fields.

Reprojects each provenance-fixed field to a tangent-plane square using
x = dRA*cos(dec0), y = dDec.  The same 0.28-deg physical square and 64x64 grid
are used for every field, so x/y cells have the same angular size.
No simulated catalog is used.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

PROV = Path("data/real/dr11/expanded48/provenance.json")
OUT = Path("results/real_dr11/metric_isotropy48")
GRID = 64
WIDTH_DEG = 0.28
LAGS = [1, 2, 4, 8]
PATCH = 16
STRIDE = 8
HIDDEN = np.zeros((PATCH, PATCH), bool); HIDDEN[4:12, 4:12] = True
RING = np.zeros((PATCH, PATCH), bool)
RING[3, 3:13] = True; RING[12, 3:13] = True
RING[4:12, 3] = True; RING[4:12, 12] = True


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_region(meta: dict) -> pd.DataFrame:
    path = Path(meta["file"])
    gz = path.read_bytes(); raw = gzip.decompress(gz)
    if sha(gz) != meta["stored_gzip_sha256"] or sha(raw) != meta["canonical_csv_sha256"]:
        raise RuntimeError(f"provenance hash mismatch: {path}")
    d = pd.read_csv(path)
    if list(d.columns) != ["ra", "dec"] or len(d) != int(meta["rows"]):
        raise RuntimeError(f"bad REAL_DR11 source file: {path}")
    return d


def metric_grid(d: pd.DataFrame, meta: dict) -> np.ndarray:
    ra0 = float(meta["center_ra_deg"]); dec0 = float(meta["center_dec_deg"])
    dra = ((d.ra.to_numpy(float) - ra0 + 180.0) % 360.0) - 180.0
    x = dra * np.cos(np.deg2rad(dec0))
    y = d.dec.to_numpy(float) - dec0
    h = WIDTH_DEG / 2.0
    g, _, _ = np.histogram2d(y, x, bins=GRID, range=[[-h, h], [-h, h]])
    z = np.log1p(g)
    med = np.median(z); sc = np.median(np.abs(z - med)) * 1.4826
    if not np.isfinite(sc) or sc < 1e-6: sc = np.std(z)
    if not np.isfinite(sc) or sc < 1e-6: sc = 1.0
    return (z - med) / sc


def rho(a, b) -> float:
    r = float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).statistic)
    return r if np.isfinite(r) else 0.0


def directional(z: np.ndarray, lag: int) -> dict:
    return {
        "x": rho(z[:, :-lag], z[:, lag:]),
        "y": rho(z[:-lag, :], z[lag:, :]),
        "diag_ne": rho(z[:-lag, :-lag], z[lag:, lag:]),
        "diag_nw": rho(z[:-lag, lag:], z[lag:, :-lag]),
    }


def locality_patch(z: np.ndarray) -> tuple[float, float, int]:
    ring = []; hidden = []
    for y in range(0, GRID - PATCH + 1, STRIDE):
        for x in range(0, GRID - PATCH + 1, STRIDE):
            p = z[y:y+PATCH, x:x+PATCH]
            ring.append(float(p[RING].mean())); hidden.append(float(p[HIDDEN].mean()))
    ring = np.asarray(ring); hidden = np.asarray(hidden)
    real = rho(ring, hidden)
    shifted = rho(ring, np.roll(hidden, max(1, len(hidden)//3)))
    return real, shifted, len(ring)


def paired_stats(d, alternative="two-sided") -> dict:
    a = np.asarray(d, float); a = a[np.isfinite(a) & (a != 0)]
    if len(a) == 0: return {"n": 0}
    pos = int((a > 0).sum())
    b = binomtest(pos, len(a), .5, alternative=alternative).pvalue
    try: w = wilcoxon(a, alternative=alternative).pvalue
    except Exception: w = np.nan
    return {"n": int(len(a)), "positive": pos, "median": float(np.median(a)), "mean": float(np.mean(a)),
            "sign_p": float(b), "wilcoxon_p": float(w)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = json.loads(PROV.read_text())
    regs = p.get("regions", [])
    if p.get("status") != "REAL_DR11" or len(regs) != 48:
        raise RuntimeError("48-field REAL_DR11 provenance required")
    rows = []; locrows = []
    for i, meta in enumerate(regs):
        z = metric_grid(load_region(meta), meta)
        rng = np.random.default_rng(20260829 + i)
        zn = rng.permutation(z.ravel()).reshape(z.shape)
        for lag in LAGS:
            r = directional(z, lag); n = directional(zn, lag)
            rows.append({"field": meta["name"], "dec": float(meta["center_dec_deg"]), "lag": lag,
                         "sep_arcmin": lag * WIDTH_DEG / GRID * 60.0,
                         **{f"rho_{k}": v for k, v in r.items()},
                         **{f"null_{k}": v for k, v in n.items()}})
        lr, ln, npatch = locality_patch(z)
        locrows.append({"field": meta["name"], "dec": float(meta["center_dec_deg"]),
                        "ring_hidden_rho": lr, "matched_shift_rho": ln, "n_patches": npatch})
        print(f"[metric-isotropy] {i+1}/48 {meta['name']} locality={lr:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "directional_field_metrics.csv", index=False)
    ldf = pd.DataFrame(locrows); ldf.to_csv(OUT / "locality_field_metrics.csv", index=False)

    lag_summary = []
    for lag, g in df.groupby("lag"):
        card = (g.rho_x.to_numpy() + g.rho_y.to_numpy()) / 2
        null_card = (g.null_x.to_numpy() + g.null_y.to_numpy()) / 2
        lag_summary.append({
            "lag": int(lag), "sep_arcmin": float(g.sep_arcmin.iloc[0]),
            "median_rho_x": float(np.median(g.rho_x)), "median_rho_y": float(np.median(g.rho_y)),
            "median_cardinal": float(np.median(card)), "median_null_cardinal": float(np.median(null_card)),
            "x_minus_y": paired_stats(g.rho_x - g.rho_y, "two-sided"),
            "diag_ne_minus_nw": paired_stats(g.rho_diag_ne - g.rho_diag_nw, "two-sided"),
            "cardinal_minus_shuffle": paired_stats(card - null_card, "greater"),
        })
    summary = {
        "status": "REAL_DR11_METRIC_ISOTROPY",
        "n_fields": 48,
        "projection": "tangent plane x=dRA*cos(dec0), y=dDec",
        "physical_square_width_deg": WIDTH_DEG,
        "cell_arcmin": WIDTH_DEG / GRID * 60.0,
        "lags": lag_summary,
        "locality": {
            "median_ring_hidden_rho": float(np.median(ldf.ring_hidden_rho)),
            "median_matched_shift_rho": float(np.median(ldf.matched_shift_rho)),
            "real_minus_shift": paired_stats(ldf.ring_hidden_rho - ldf.matched_shift_rho, "greater"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
