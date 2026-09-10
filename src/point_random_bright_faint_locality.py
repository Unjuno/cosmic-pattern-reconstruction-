#!/usr/bin/env python3
"""REAL_DR11 bright/faint extended cross-locality after point-random selection residualization.

The science target is observed DR11 source positions.  Catalog morphology and
r-band flux are used only to define two disjoint extended-source populations.
Official DR11 point-random metadata are used to model survey selection.  The
primary statistic is continuous same-patch bright<->faint residual locality
relative to a within-field matched-shift null.  A deterministic random half/half
split of the identical extended source set is a required positive control.

No simulated cosmology is used and PASS does not establish a cosmological
origin.  It rejects only the tested magnitude-population-specific / sampled
point-random selection explanation of the local continuity signal.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dl import queryClient as qc
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import analyze_dr11 as adr
import bispectrum_phase_validate as bp
import locality_validate as loc
import point_random_selection_residual_validate as prs

TABLE = "ls_dr11.tractor_s"
EXT_TYPES = {"REX", "EXP", "DEV", "SER"}
GRID = 64
FIELD_HALF_DEG = 0.25
QUERY_RADIUS_DEG = 0.40
POINT_RANDOM_FRACTION = 0.04
MIN_EACH = 1000
MIN_VALID_FRACTION = 0.85
N_FOLDS = 6

LOCKED_FIELDS = list(prs.LOCKED_FIELDS)

# Transport-only retries.  This changes no scientific input or decision rule.
_ORIGINAL_FETCH_RANGE = prs.fetch_range


def retry_fetch_range(url: str, start: int, end: int, timeout: int = 360):
    delays = (0, 5, 15, 30)
    last = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            return _ORIGINAL_FETCH_RANGE(url, start, end, timeout)
        except Exception as exc:
            last = exc
            print(
                f"[bright-faint-transport] Range attempt {attempt}/{len(delays)} failed "
                f"bytes={start}-{end} url={url}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    raise RuntimeError(
        f"HTTP Range failed after {len(delays)} attempts bytes={start}-{end} url={url}: {last}"
    ) from last


prs.fetch_range = retry_fetch_range


def query_csv(sql: str, attempts: int = 5) -> str:
    last = None
    for attempt in range(attempts):
        try:
            x = qc.query(sql=sql, fmt="csv", async_=False)
            if isinstance(x, bytes):
                x = x.decode("utf-8")
            if not isinstance(x, str) or not x.strip():
                raise RuntimeError(f"invalid Data Lab response {type(x)}")
            if x.lstrip().lower().startswith(("error", "<!doctype", "<html")):
                raise RuntimeError(x[:500])
            return x
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Data Lab query failed after {attempts} attempts: {last}") from last


def wrap_deg(x: np.ndarray | float) -> np.ndarray:
    return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0


def canonical_catalog_bytes(d: pd.DataFrame) -> bytes:
    cols = ["ra", "dec", "type", "flux_r", "mw_transmission_r"]
    x = d[cols].copy()
    x = x.sort_values(cols, kind="mergesort").reset_index(drop=True)
    return x.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def acquire_field_catalog(meta: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    name = str(meta["name"])
    ra0 = float(meta["center_ra_deg"])
    dec0 = float(meta["center_dec_deg"])
    sql = f"""
SELECT ra, dec, type, flux_r, mw_transmission_r
FROM {TABLE}
WHERE brick_primary = 1
  AND q3c_radial_query(ra, dec, {ra0:.8f}, {dec0:.8f}, {QUERY_RADIUS_DEG:.8f})
""".strip()
    text = query_csv(sql)
    d = pd.read_csv(io.StringIO(text))
    required = ["ra", "dec", "type", "flux_r", "mw_transmission_r"]
    if list(d.columns) != required:
        # Be strict about schema drift but allow harmless case differences.
        lower = {str(c).lower(): c for c in d.columns}
        missing = [c for c in required if c not in lower]
        if missing:
            raise RuntimeError(f"{name}: missing Data Lab columns {missing}; got {list(d.columns)}")
        d = d.rename(columns={lower[c]: c for c in required})[required]

    ra = pd.to_numeric(d.ra, errors="coerce").to_numpy(float)
    dec = pd.to_numeric(d.dec, errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & (np.abs(wrap_deg(ra - ra0)) <= FIELD_HALF_DEG)
        & (dec >= dec0 - FIELD_HALF_DEG)
        & (dec < dec0 + FIELD_HALF_DEG)
    )
    d = d.loc[keep].copy().reset_index(drop=True)
    d["type"] = d["type"].astype(str).str.strip().str.upper()
    flux = pd.to_numeric(d.flux_r, errors="coerce").to_numpy(float)
    mw = pd.to_numeric(d.mw_transmission_r, errors="coerce").to_numpy(float)
    ext = d.type.isin(EXT_TYPES).to_numpy() & np.isfinite(flux) & np.isfinite(mw) & (flux > 0) & (mw > 0)
    e = d.loc[ext].copy().reset_index(drop=True)
    e["dered_flux_r"] = flux[ext] / mw[ext]
    if len(e) < 2 * MIN_EACH:
        raise RuntimeError(f"{name}: only {len(e)} extended positive-r sources; need >= {2*MIN_EACH}")

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

    canonical = canonical_catalog_bytes(d)
    return {
        "bright": bright,
        "faint": faint,
        "random_a": random_a,
        "random_b": random_b,
    }, {
        "field": name,
        "table": TABLE,
        "query": sql,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "radial_rows_returned": int(len(pd.read_csv(io.StringIO(text)))),
        "square_rows": int(len(d)),
        "square_catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "n_extended_positive_r": int(len(e)),
        "n_used_even": int(n_used),
        "n_each": int(half),
        "median_dered_flux_r": float(np.median(used.dered_flux_r.to_numpy(float))),
        "random_half_seed": int(seed),
    }


def source_grid(d: pd.DataFrame, meta: dict) -> np.ndarray:
    return adr.region_grid(d[["ra", "dec"]], meta).astype(float)


def rho(a, b) -> float:
    r = float(spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic)
    return r if np.isfinite(r) else float("nan")


def fit_selection_model(data: dict, train: list[str], tracer: str, seed: int):
    X, y = [], []
    for name in train:
        d = data[name]
        v = d["valid"]
        counts = d[tracer]
        mean = float(np.mean(counts[v])) + 1e-6
        X.append(d["sel"][v])
        y.append(counts[v] / mean)
    X = np.concatenate(X)
    y = np.concatenate(y)
    if len(y) > 120000:
        rng = np.random.default_rng(seed)
        ii = rng.choice(len(y), 120000, replace=False)
        X, y = X[ii], y[ii]
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=70,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=70,
        l2_regularization=1.5,
        random_state=seed,
    )
    model.fit(X, np.clip(y, 1e-4, None))
    return model


def pair_locality(src_grid: np.ndarray, tgt_grid: np.ndarray) -> tuple[float, float, float]:
    src = loc.contexts_from_grid(src_grid)
    tgt = loc.contexts_from_grid(tgt_grid)
    visible = src[:, 1]
    hidden = tgt[:, 0]
    real = rho(visible, hidden)
    shifted = rho(loc.shifted_feature(visible), hidden)
    return real, shifted, real - shifted


def summarize(values: np.ndarray) -> dict:
    return bp.paired(np.asarray(values, float))


def primary_decision(cross: dict, random_control: dict, n_valid: int) -> str:
    if n_valid < 30:
        return "UNCERTAIN_COVERAGE"
    control_ok = (
        float(random_control["median"]) >= 0.10
        and float(random_control["sign_p_one_sided"]) < 0.01
        and float(random_control["wilcoxon_p_one_sided"]) < 0.01
    )
    if not control_ok:
        return "UNCERTAIN_POSITIVE_CONTROL"
    if (
        float(cross["median"]) >= 0.05
        and float(cross["sign_p_one_sided"]) < 0.01
        and float(cross["wilcoxon_p_one_sided"]) < 0.01
    ):
        return "PASS_SHARED_BRIGHT_FAINT_LOCALITY"
    if (
        float(cross["median"]) <= 0
        or float(cross["sign_p_one_sided"]) >= 0.10
        or float(cross["wilcoxon_p_one_sided"]) >= 0.10
    ):
        return "FAIL_SHARED_BRIGHT_FAINT_LOCALITY"
    return "UNCERTAIN_SHARED_BRIGHT_FAINT_LOCALITY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", default="data/real/dr11/expanded48/provenance.json")
    ap.add_argument("--fraction", type=float, default=POINT_RANDOM_FRACTION)
    ap.add_argument("--out", default="results/real_dr11/point_random_bright_faint_locality36")
    args = ap.parse_args()
    if not (0 < args.fraction <= 0.05):
        raise RuntimeError("point-random fraction must be in (0, 0.05]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = json.loads(Path(args.provenance).read_text())
    if prov.get("status") != "REAL_DR11" or len(prov.get("regions", [])) != 48:
        raise RuntimeError("48-field REAL_DR11 provenance required")
    all_meta = {r["name"]: r for r in prov["regions"]}
    if any(name not in all_meta for name in LOCKED_FIELDS):
        raise RuntimeError("locked 36-field set missing from expanded48 provenance")
    regions = {name: all_meta[name] for name in LOCKED_FIELDS}

    # Acquire the source properties first.  A schema / availability failure is
    # detected before the much larger point-random Range scan begins.
    subsets: dict[str, dict[str, pd.DataFrame]] = {}
    source_prov = []
    for i, name in enumerate(LOCKED_FIELDS):
        ss, sp = acquire_field_catalog(regions[name])
        subsets[name] = ss
        source_prov.append(sp)
        print(
            f"[bright-faint] source {i+1:02d}/36 {name} each={sp['n_each']} "
            f"square={sp['square_rows']}",
            flush=True,
        )
    (out / "source_provenance.json").write_text(
        json.dumps({"status": "REAL_DR11_EXTENDED_BRIGHT_FAINT_SOURCE", "regions": source_prov}, indent=2, sort_keys=True) + "\n"
    )

    points, remote_prov = prs.acquire_points(regions, args.fraction)
    pd.DataFrame([
        {
            "file_index": p["file_index"],
            "url": p["url"],
            "rows": p["rows"],
            "rows_sampled": p["rows_sampled"],
            "sample_fraction_effective": p["sample_fraction_effective"],
            "sample_data_bytes": p["sample_data_bytes"],
            "sample_data_sha256": p["sample_data_sha256"],
            "etag": p["data_http"]["etag"],
            "content_range": p["data_http"]["content_range"],
            "header_prefix_sha256": p["header_prefix_sha256"],
        }
        for p in remote_prov
    ]).to_csv(out / "remote_file_provenance.csv", index=False)

    data = {}
    qc_rows = []
    feature_names0 = None
    for i, name in enumerate(LOCKED_FIELDS):
        m = regions[name]
        sel, valid, q, feature_names = prs.selection_grid(points[name], m)
        if feature_names0 is None:
            feature_names0 = feature_names
        elif feature_names != feature_names0:
            raise RuntimeError("point-random selection feature mismatch")
        rec = dict(q)
        rec.update({"field": name, "n_each": int(len(subsets[name]["bright"]))})
        qc_rows.append(rec)
        if q["valid_cell_fraction"] < MIN_VALID_FRACTION:
            print(f"[bright-faint] reject {name}: valid={q['valid_cell_fraction']:.3f}", flush=True)
            continue
        d = {"sel": sel, "valid": valid}
        for tracer in ["bright", "faint", "random_a", "random_b"]:
            d[tracer] = source_grid(subsets[name][tracer], m)
        data[name] = d
        print(
            f"[bright-faint] grid {i+1:02d}/36 {name} randoms={len(points[name])} "
            f"valid={valid.mean():.3f}",
            flush=True,
        )

    Q = pd.DataFrame(qc_rows)
    Q.to_csv(out / "point_random_qc.csv", index=False)
    valid_names = [n for n in LOCKED_FIELDS if n in data]

    # Cache selection surfaces for future tracer experiments so they can avoid
    # rescanning the remote 20-file point-random product.
    if valid_names:
        np.savez_compressed(
            out / "selection_surfaces.npz",
            fields=np.asarray(valid_names),
            feature_names=np.asarray(feature_names0),
            selection=np.stack([data[n]["sel"].astype(np.float32) for n in valid_names]),
            valid=np.stack([data[n]["valid"] for n in valid_names]),
        )

    fold_id = {n: i % N_FOLDS for i, n in enumerate(valid_names)}
    rows = []
    model_rows = []
    residual_cache = {k: [] for k in ["bright", "faint", "random_a", "random_b"]}
    residual_cache_fields = []

    for fold in range(N_FOLDS):
        test = [n for n in valid_names if fold_id[n] == fold]
        train = [n for n in valid_names if fold_id[n] != fold]
        if not test or len(train) < 20:
            continue
        models = {
            tracer: fit_selection_model(data, train, tracer, 44000 + 1000 * fold + ti)
            for ti, tracer in enumerate(["bright", "faint", "random_a", "random_b"])
        }
        for name in test:
            d = data[name]
            v = d["valid"]
            maps = {"raw": {}, "residual": {}}
            metrics = {"field": name, "fold": fold}
            idx = LOCKED_FIELDS.index(name)
            for ti, tracer in enumerate(["bright", "faint", "random_a", "random_b"]):
                counts = d[tracer]
                mean = float(np.mean(counts[v])) + 1e-6
                pred_rel = np.full((GRID, GRID), np.nan, float)
                pred_rel[v] = np.clip(models[tracer].predict(d["sel"][v]), 0.03, 20)
                true_rel = counts[v] / mean
                metrics[f"{tracer}_selection_r2"] = float(r2_score(true_rel, pred_rel[v]))
                metrics[f"{tracer}_selection_spearman"] = rho(true_rel, pred_rel[v])
                expected = pred_rel * mean
                raw, residual = prs.normalized_maps(counts, expected, v, 880000 + idx * 20 + ti * 2)
                maps["raw"][tracer] = raw
                maps["residual"][tracer] = residual
            model_rows.append(metrics)

            pairs = [
                ("bright", "bright", "bright_self"),
                ("faint", "faint", "faint_self"),
                ("bright", "faint", "bright_to_faint"),
                ("faint", "bright", "faint_to_bright"),
                ("random_a", "random_b", "random_cross_ab"),
                ("random_b", "random_a", "random_cross_ba"),
            ]
            for sample in ["raw", "residual"]:
                for src, tgt, label in pairs:
                    real, shift, adv = pair_locality(maps[sample][src], maps[sample][tgt])
                    rows.append(
                        {
                            "field": name,
                            "fold": fold,
                            "sample": sample,
                            "pair": label,
                            "rho": real,
                            "matched_shift_rho": shift,
                            "advantage": adv,
                        }
                    )
            residual_cache_fields.append(name)
            for tracer in residual_cache:
                residual_cache[tracer].append(maps["residual"][tracer].astype(np.float32))
        print(f"[bright-faint] fold {fold+1}/{N_FOLDS} train={len(train)} test={len(test)}", flush=True)

    D = pd.DataFrame(rows)
    M = pd.DataFrame(model_rows)
    D.to_csv(out / "field_metrics.csv", index=False)
    M.to_csv(out / "selection_model_metrics.csv", index=False)
    if residual_cache_fields:
        order = np.argsort(np.asarray(residual_cache_fields))
        np.savez_compressed(
            out / "residual_maps.npz",
            fields=np.asarray(residual_cache_fields)[order],
            **{k: np.stack(v)[order] for k, v in residual_cache.items()},
        )

    pair_summary = []
    for (sample, pair), g in D.groupby(["sample", "pair"]):
        pair_summary.append(
            {
                "sample": sample,
                "pair": pair,
                "rho_median": float(np.nanmedian(g.rho)),
                "matched_shift_median": float(np.nanmedian(g.matched_shift_rho)),
                "real_minus_shift": summarize(g.advantage.to_numpy(float)),
            }
        )

    sym_rows = []
    for sample in ["raw", "residual"]:
        g = D[D["sample"] == sample].pivot(index="field", columns="pair", values="advantage")
        for field in g.index:
            cross = 0.5 * (g.loc[field, "bright_to_faint"] + g.loc[field, "faint_to_bright"])
            random_cross = 0.5 * (g.loc[field, "random_cross_ab"] + g.loc[field, "random_cross_ba"])
            self_avg = 0.5 * (g.loc[field, "bright_self"] + g.loc[field, "faint_self"])
            sym_rows.append(
                {
                    "field": field,
                    "sample": sample,
                    "bright_faint_cross_advantage": float(cross),
                    "random_half_cross_advantage": float(random_cross),
                    "bright_faint_self_advantage": float(self_avg),
                    "random_minus_bright_faint": float(random_cross - cross),
                    "self_minus_cross": float(self_avg - cross),
                }
            )
    S = pd.DataFrame(sym_rows)
    S.to_csv(out / "symmetric_field_metrics.csv", index=False)
    sr = S[S["sample"] == "residual"].set_index("field")
    cross_primary = summarize(sr.bright_faint_cross_advantage.to_numpy(float))
    random_control = summarize(sr.random_half_cross_advantage.to_numpy(float))
    random_minus_cross = summarize(sr.random_minus_bright_faint.to_numpy(float))
    self_minus_cross = summarize(sr.self_minus_cross.to_numpy(float))
    direction = D[(D["sample"] == "residual")].pivot(index="field", columns="pair", values="advantage")
    faint_minus_bright_direction = summarize(
        (direction.faint_to_bright - direction.bright_to_faint).to_numpy(float)
    )
    decision = primary_decision(cross_primary, random_control, len(valid_names))

    summary = {
        "status": "REAL_DR11_POINT_RANDOM_BRIGHT_FAINT_CROSS_LOCALITY36",
        "decision": decision,
        "n_locked_fields": 36,
        "n_valid_fields": int(len(valid_names)),
        "field_geometry": "same 0.5 deg sky-centered RA/Dec grid as point-random selection residual experiment",
        "source_definition": "brick_primary=1; TYPE in REX/EXP/DEV/SER; finite positive r flux and MW_TRANSMISSION_R; exact equal halves by within-field dereddened r flux",
        "random_control": "deterministic disjoint equal-count random half/half split of the identical extended source set in each field",
        "point_random_fraction_per_file": float(args.fraction),
        "point_random_files": 20,
        "selection_features": feature_names0,
        "selection_surface": "k=4 inverse-distance interpolation; fourth-nearest radius <=0.08 deg; >=85% valid cells",
        "cross_validation": "6-fold grouped by whole field; independent Poisson HGB selection model for bright/faint/random_a/random_b",
        "primary": "symmetric mean of bright->faint and faint->bright residual local-visible minus matched-shift Spearman advantages",
        "positive_control_primary": "symmetric random_a<->random_b residual cross-locality advantage",
        "predeclared_decision": {
            "coverage": "at least 30 valid fields",
            "positive_control": "random-half median advantage >=0.10 and one-sided sign/Wilcoxon p<0.01",
            "pass": "bright/faint median cross advantage >=0.05 and one-sided sign/Wilcoxon p<0.01, conditional on positive-control pass",
            "fail": "conditional on positive-control pass: median<=0 or either p>=0.10",
            "otherwise": "UNCERTAIN",
        },
        "primary_bright_faint_cross": cross_primary,
        "positive_control_random_half_cross": random_control,
        "random_minus_bright_faint": random_minus_cross,
        "self_minus_cross": self_minus_cross,
        "directional_faint_to_bright_minus_bright_to_faint": faint_minus_bright_direction,
        "pair_summaries": pair_summary,
        "selection_model_r2_medians": {
            c: float(np.nanmedian(M[f"{c}_selection_r2"])) if len(M) else float("nan")
            for c in ["bright", "faint", "random_a", "random_b"]
        },
        "selection_model_spearman_medians": {
            c: float(np.nanmedian(M[f"{c}_selection_spearman"])) if len(M) else float("nan")
            for c in ["bright", "faint", "random_a", "random_b"]
        },
        "interpretation_guardrail": "PASS establishes shared residual angular locality across disjoint brightness populations under the tested sampled point-random nuisance model. It does not establish cosmological origin, matter density, gravity, or higher-order structure.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
