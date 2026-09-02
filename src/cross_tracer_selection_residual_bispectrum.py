#!/usr/bin/env python3
"""REAL DR11 selection-residualized PSF-vs-extended cross-bispectrum.

This directly combines two accepted stress programs:
1) shared PSF/extended cross-tracer phase coupling; and
2) g/r/i/z pixel-level survey-selection residualization.

The first 48 pre-registered expanded48 centers are scanned in fixed order.
The first 36 candidates passing only objective acquisition, tracer-availability,
and survey-coverage gates are accepted. Per accepted brick, PSF and extended
(REX/EXP/DEV/SER) populations are count-equalized with fixed RNG. After the
morphology split, science inputs are positions/count maps only.

For grouped whole-brick CV, separate selection count models are trained for
PSF-like and extended maps using official DR11 g/r/i/z depth, NEXP, PSF-size,
and MASKBITS/BRICK_PRIMARY maps from other bricks. Raw, selection-expected,
and selection-residualized PSF/extended maps are tested with the same symmetric
mixed closure products and exact-Fourier-amplitude phase-null used by the
accepted cross-tracer bispectrum experiment.

No simulated cosmology or mock catalog is used.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import bispectrum_phase_validate as bp
import cross_tracer_bispectrum_validate as cb
import selection_null_multiband as mb
import selection_null_multiband36 as mb36
import tracer_split_validate as tr

core = mb.core
BANDS = mb.BANDS
GRID = 64
TARGET_FIELDS = 36
MAX_CANDIDATES = 48
MIN_PER_TRACER = 1000
MIN_PATCHES = 8


def build_selection_features(brick: str):
    urls = mb.product_urls_all(brick)
    jobs = {"maskbits": (urls["maskbits"], "MASKBITS")}
    for b in BANDS:
        jobs[f"depth_{b}"] = (urls[f"depth_{b}"], f"DEPTH_{b.upper()}")
        jobs[f"nexp_{b}"] = (urls[f"nexp_{b}"], None)
        jobs[f"psfsize_{b}"] = (urls[f"psfsize_{b}"], None)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {k: ex.submit(core.read_image, u, w) for k, (u, w) in jobs.items()}
        got = {k: v.result() for k, v in fut.items()}

    mask, _, pmask = got["maskbits"]
    shape = mask.shape
    feat, names = [], []
    product_prov = {"maskbits": pmask}
    for b in BANDS:
        depth, _, pdepth = got[f"depth_{b}"]
        nexp, _, pnexp = got[f"nexp_{b}"]
        psf, _, ppsf = got[f"psfsize_{b}"]
        if not (depth.shape == nexp.shape == psf.shape == shape):
            raise RuntimeError(f"shape_mismatch:{brick}:{b}")
        for arr, nms in [
            (core.continuous_features(depth, True, True),
             [f"log_depth_{b}_mean", f"log_depth_{b}_std", f"depth_{b}_positive_frac"]),
            (core.continuous_features(nexp, False, True),
             [f"nexp_{b}_mean", f"nexp_{b}_std", f"nexp_{b}_positive_frac"]),
            (core.continuous_features(psf, False, True),
             [f"psfsize_{b}_mean", f"psfsize_{b}_std", f"psfsize_{b}_positive_frac"]),
        ]:
            feat += arr
            names += nms
        product_prov.update({f"depth_{b}": pdepth, f"nexp_{b}": pnexp, f"psfsize_{b}": ppsf})

    feat += core.mask_features(mask)
    names += ["primary_frac", "clean_frac"] + [f"maskbit_{k}_frac" for k in range(core.BIT_COUNT)]
    sel = np.stack(feat, axis=-1)
    primary = sel[:, :, names.index("primary_frac")]
    valid = (
        (primary >= 0.9375)
        & (sel[:, :, names.index("depth_r_positive_frac")] >= 0.9375)
        & (sel[:, :, names.index("nexp_r_positive_frac")] >= 0.9375)
    )
    dummy = np.zeros((GRID, GRID), float)
    patches, _, _ = mb36.make_patches(dummy, sel, valid)
    if len(patches) < MIN_PATCHES:
        raise RuntimeError(f"coverage:{len(patches)}")
    return sel, valid, names, product_prov, int(len(patches))


def acquire(centers: dict):
    regs = centers.get("regions", [])[:MAX_CANDIDATES]
    data, provenance, rejected = {}, [], []
    used = set()
    rng = np.random.default_rng(20260903)

    for j, rec in enumerate(regs):
        if len(data) >= TARGET_FIELDS:
            break
        name = rec["name"]
        ra = float(rec["center_ra_deg"])
        dec = float(rec["center_dec_deg"])
        try:
            brick, brick_query = tr.choose_brick(ra, dec)
        except Exception as e:
            rejected.append({"field": name, "reason": "brick_resolution", "error": str(e)})
            continue
        if brick in used:
            rejected.append({"field": name, "brick": brick, "reason": "duplicate"})
            continue
        # Reserve brick before downstream availability tests, matching the accepted fixed-order protocol.
        used.add(brick)
        print(f"[cross-selection] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}", flush=True)
        try:
            tractor, tractor_prov = tr.get_tractor(brick)
        except Exception as e:
            rejected.append({"field": name, "brick": brick, "reason": "tractor_acquisition", "error": str(e)})
            continue
        psf = tractor[tractor.type.eq("PSF")].copy()
        ext = tractor[tractor.type.isin(tr.EXT_TYPES)].copy()
        n = min(len(psf), len(ext))
        if n < MIN_PER_TRACER:
            rejected.append({"field": name, "brick": brick, "reason": "tracer_availability",
                             "n_psf": int(len(psf)), "n_extended": int(len(ext))})
            continue
        try:
            sel, valid, feature_names, product_prov, n_patches = build_selection_features(brick)
        except Exception as e:
            rejected.append({"field": name, "brick": brick, "reason": "selection_acquisition_or_coverage", "error": str(e)})
            continue

        psf_eq = psf.iloc[rng.choice(len(psf), n, replace=False)]
        ext_eq = ext.iloc[rng.choice(len(ext), n, replace=False)]
        gp = tr.grid_from(psf_eq).astype(float)
        ge = tr.grid_from(ext_eq).astype(float)
        data[name] = {
            "brick": brick, "psf": gp, "ext": ge, "sel": sel,
            "valid": valid, "feature_names": feature_names,
        }
        provenance.append({
            "field": name, "brick": brick, "brick_choice_query": brick_query,
            "tractor": tractor_prov, "n_psf": int(len(psf)), "n_extended": int(len(ext)),
            "equalized_n": int(n), "valid_cell_fraction": float(valid.mean()),
            "n_coverage_patches": n_patches, "selection_products": product_prov,
        })
        print(f"[cross-selection] accept {len(data)}/{TARGET_FIELDS}: eq={n} valid={valid.mean():.3f}", flush=True)

    return data, provenance, rejected


def fit_selection_model(data, train_names, tracer: str, seed: int):
    X, y = [], []
    for name in train_names:
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
        loss="poisson", max_iter=70, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=70, l2_regularization=1.5, random_state=seed,
    )
    model.fit(X, np.clip(y, 1e-4, None))
    return model


def fill_invalid(a, valid, seed):
    z = np.asarray(a, float).copy()
    vals = z[valid]
    if not len(vals):
        raise RuntimeError("no valid cells")
    rng = np.random.default_rng(seed)
    z[~valid] = rng.choice(vals, size=int((~valid).sum()), replace=True)
    return z


def cross_stats_with_null(a, b, tri, seed_a, seed_b):
    real = cb.mixed_stats(a, b, tri)
    ap = bp.exact_phase(a, seed_a)
    bpmap = bp.exact_phase(b, seed_b)
    n1 = cb.mixed_stats(a, bpmap, tri)
    n2 = cb.mixed_stats(ap, b, tri)
    null = {}
    for fam in tri:
        null[fam] = {
            "phase_lock": float((n1[fam]["phase_lock"] + n2[fam]["phase_lock"]) / 2),
            "bicoherence": float((n1[fam]["bicoherence"] + n2[fam]["bicoherence"]) / 2),
            "signed_bicoherence": float((n1[fam]["signed_bicoherence"] + n2[fam]["signed_bicoherence"]) / 2),
            "n_triangles": real[fam]["n_triangles"],
        }
    return real, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", default="data/real/dr11/expanded48/provenance.json")
    ap.add_argument("--out", default="results/real_dr11/cross_tracer_selection_residual36")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    centers = json.loads(Path(args.centers).read_text())
    if centers.get("status") != "REAL_DR11":
        raise RuntimeError("provenance-verified REAL_DR11 centers required")

    data, provenance, rejected = acquire(centers)
    if len(data) != TARGET_FIELDS:
        (out / "availability_rejections.json").write_text(json.dumps(rejected, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"only {len(data)} accepted from fixed {MAX_CANDIDATES}")
    names = list(data)
    if len({tuple(data[n]["feature_names"]) for n in names}) != 1:
        raise RuntimeError("selection feature mismatch")

    nfold = max(2, min(args.folds, len(names)))
    fold_id = {name: i % nfold for i, name in enumerate(names)}
    tri = bp.build_triangles(seed=20260901, max_pairs=16000)
    rows, model_rows = [], []

    for fold in range(nfold):
        test_names = [n for n in names if fold_id[n] == fold]
        train_names = [n for n in names if fold_id[n] != fold]
        models = {
            "psf": fit_selection_model(data, train_names, "psf", 7100 + fold),
            "ext": fit_selection_model(data, train_names, "ext", 8100 + fold),
        }
        for name in test_names:
            d = data[name]
            v = d["valid"]
            maps = {}
            metrics = {"field": name, "fold": fold}
            for ti, tracer in enumerate(["psf", "ext"]):
                counts = d[tracer]
                mean = float(np.mean(counts[v])) + 1e-6
                pred_rel = np.full((GRID, GRID), np.nan, float)
                pred_rel[v] = np.clip(models[tracer].predict(d["sel"][v]), 0.03, 20)
                expected = pred_rel * mean
                residual = np.zeros((GRID, GRID), float)
                residual[v] = counts[v] - expected[v]
                true_rel = counts[v] / mean
                metrics[f"{tracer}_selection_r2"] = float(r2_score(true_rel, pred_rel[v]))
                metrics[f"{tracer}_selection_spearman"] = float(spearmanr(true_rel, pred_rel[v]).statistic)
                idx = names.index(name)
                maps[(tracer, "raw")] = bp.gaussianize(fill_invalid(counts, v, 110000 + idx*20 + ti))
                maps[(tracer, "selection")] = bp.gaussianize(fill_invalid(expected, v, 120000 + idx*20 + ti))
                maps[(tracer, "residual")] = bp.gaussianize(fill_invalid(residual, v, 130000 + idx*20 + ti))
            model_rows.append(metrics)

            idx = names.index(name)
            for si, sample in enumerate(["raw", "selection", "residual"]):
                real, null = cross_stats_with_null(
                    maps[("psf", sample)], maps[("ext", sample)], tri,
                    200000 + idx*30 + si*2, 200001 + idx*30 + si*2,
                )
                for fam in tri:
                    rows.append({"field": name, "fold": fold, "family": fam, "sample": sample, "control": "real", **real[fam]})
                    rows.append({"field": name, "fold": fold, "family": fam, "sample": sample, "control": "phase_null", **null[fam]})
        print(f"[cross-selection] fold {fold+1}/{nfold}: train={len(train_names)} test={len(test_names)}", flush=True)

    D = pd.DataFrame(rows)
    M = pd.DataFrame(model_rows)
    D.to_csv(out / "field_metrics.csv", index=False)
    M.to_csv(out / "selection_model_metrics.csv", index=False)

    summary = {
        "status": "REAL_DR11_SELECTION_RESIDUAL_CROSS_TRACER_BISPECTRUM",
        "n_fields": TARGET_FIELDS,
        "candidate_protocol": "first 36 passing objective combined tracer/coadd/coverage gates from fixed first-48 expanded48 order",
        "subsets": "count-equalized official Tractor PSF vs REX/EXP/DEV/SER; positions/count maps after split",
        "cross_validation": f"{nfold}-fold grouped by whole brick",
        "selection_features": "official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY support",
        "primary_family": "all",
        "primary_metric": "residual cross-tracer bicoherence minus exact-amplitude phase-null",
        "predeclared_pass": "median difference > 0 AND one-sided sign p < .05 AND one-sided Wilcoxon p < .05 for all-family residual bicoherence",
        "selection_model": {
            "psf_r2_median": float(np.nanmedian(M.psf_selection_r2)),
            "psf_spearman_median": float(np.nanmedian(M.psf_selection_spearman)),
            "extended_r2_median": float(np.nanmedian(M.ext_selection_r2)),
            "extended_spearman_median": float(np.nanmedian(M.ext_selection_spearman)),
        },
        "availability_rejections": rejected,
        "families": {},
    }

    for fam, g in D.groupby("family"):
        fs = {"n_triangles": int(g.n_triangles.iloc[0])}
        for metric in ["phase_lock", "bicoherence", "signed_bicoherence"]:
            tab = g.pivot(index="field", columns=["sample", "control"], values=metric)
            rec = {}
            for sample in ["raw", "selection", "residual"]:
                real = tab[(sample, "real")]
                null = tab[(sample, "phase_null")]
                rec[f"{sample}_real_median"] = float(np.nanmedian(real))
                rec[f"{sample}_phase_null_median"] = float(np.nanmedian(null))
                rec[f"{sample}_minus_phase"] = bp.paired(real - null)
            # Directly quantify what selection residualization does to the shared coupling.
            rec["residual_minus_raw"] = bp.paired(tab[("residual", "real")] - tab[("raw", "real")])
            fs[metric] = rec
        summary["families"][fam] = fs

    p = summary["families"]["all"]["bicoherence"]["residual_minus_phase"]
    summary["primary_decision"] = (
        "PASS_SELECTION_RESISTANT_SHARED_CROSS_TRACER_PHASE_COUPLING"
        if p["median"] > 0 and p["wilcoxon_p_one_sided"] < 0.05 and p["sign_p_one_sided"] < 0.05
        else "FAIL_OR_UNCERTAIN_SELECTION_RESISTANT_SHARED_CROSS_TRACER_PHASE_COUPLING"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "status": summary["status"], "source_centers": str(args.centers),
        "regions": provenance, "rejections": rejected,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
