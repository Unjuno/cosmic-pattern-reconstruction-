#!/usr/bin/env python3
"""Post-hoc covariance-capacity audit for the REAL DR11 residual hole task.

This follow-up reuses the already-observed 18/9/9 split from the accepted
covariance reconstruction. It is therefore mechanistic/exploratory, not a new
confirmatory blind test.

Question: did the isotropic model win because radial symmetry is specifically
useful, or mostly because it regularizes a noisy 64x64 empirical covariance?

Models:
  - full: unrestricted empirical covariance (accepted baseline);
  - stationary: translation-invariant but direction-dependent covariance;
  - isotropic: radial covariance (accepted winner);
  - ledoit_wolf: generic linear covariance shrinkage;
  - oas: Oracle Approximating Shrinkage covariance.

All models share the same selection residualization, train/validation/test
field split, patch extraction, validation-only diagonal regularization, and
test metrics. No simulated cosmology or mock target is used.

Because the test fields have already been inspected in the earlier experiment,
the decision labels below are diagnostic only. A future independent field set
is required for confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

import selection_residual_covariance_reconstruct as base
import selection_residual_locality_decay as sl
import cross_tracer_selection_residual_bispectrum as cs


def psd_project(C: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = (np.asarray(C, float) + np.asarray(C, float).T) / 2.0
    w, V = np.linalg.eigh(out)
    floor = max(1e-8, 1e-6 * float(np.median(np.diag(reference))))
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def stationary_cov(C: np.ndarray) -> tuple[np.ndarray, int]:
    """Average covariance by signed displacement, identifying h with -h."""
    coords = np.array([(y, x) for y in range(base.PATCH) for x in range(base.PATCH)], int)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, (yi, xi) in enumerate(coords):
        for j, (yj, xj) in enumerate(coords):
            h = (int(yj - yi), int(xj - xi))
            hm = (-h[0], -h[1])
            key = min(h, hm)
            groups.setdefault(key, []).append((i, j))
    out = np.zeros_like(C, dtype=float)
    for pairs in groups.values():
        value = float(np.mean([C[i, j] for i, j in pairs]))
        for i, j in pairs:
            out[i, j] = value
    return psd_project(out, C), len(groups)


def radial_class_count() -> int:
    coords = np.array([(y, x) for y in range(base.PATCH) for x in range(base.PATCH)], int)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    return int(len(np.unique(d2)))


def strong_pairwise(result: dict) -> bool:
    return (
        int(result["n"]) == 9
        and int(result["positive"]) >= 8
        and float(result["sign_p_one_sided"]) < 0.05
        and float(result["wilcoxon_p_one_sided"]) < 0.05
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real_dr11/selection_residual_covariance_capacity36")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = sl.acquire()
    names = list(data)
    if len(names) != 36:
        raise RuntimeError(f"locked field count changed: {len(names)}")

    train = [n for i, n in enumerate(names) if i % 4 in (0, 1)]
    validation = [n for i, n in enumerate(names) if i % 4 == 2]
    test = [n for i, n in enumerate(names) if i % 4 == 3]
    if (len(train), len(validation), len(test)) != (18, 9, 9):
        raise RuntimeError("split invariant failed")

    train_maps, selection_metrics = base.crossfit_train_residuals(data, train)
    final_selection_model = cs.fit_selection_model(data, train, "counts", 18000)
    maps = dict(train_maps)
    for role, subset in [("validation", validation), ("test", test)]:
        for name in subset:
            z, v, r = base.residual_map(data, name, final_selection_model)
            maps[name] = (z, v)
            selection_metrics.append({"field": name, "role": role, "selection_count_corr": r})

    X: dict[str, np.ndarray] = {}
    meta: dict[str, list[dict]] = {}
    for role, subset in [("train", train), ("validation", validation), ("test", test)]:
        arrays = []
        rows = []
        for name in subset:
            z, v = maps[name]
            x, r = base.extract_full_valid_patches(z, v, name)
            arrays.append(x)
            rows += r
        X[role] = np.concatenate(arrays) if arrays else np.empty((0, 64))
        meta[role] = rows
    if min(len(X["train"]), len(X["validation"]), len(X["test"])) < 30:
        raise RuntimeError({k: len(v) for k, v in X.items()})

    mu, Cfull = base.empirical_cov(X["train"])
    Cstationary, n_stationary_classes = stationary_cov(Cfull)
    lw = LedoitWolf(assume_centered=False).fit(X["train"])
    oas = OAS(assume_centered=False).fit(X["train"])
    covariances = {
        "full": Cfull,
        "stationary": Cstationary,
        "isotropic": base.isotropize(Cfull),
        "ledoit_wolf": lw.covariance_,
        "oas": oas.covariance_,
    }

    tuning = []
    chosen = {}
    for model, C in covariances.items():
        best = None
        for reg in base.LAMBDAS:
            A, var = base.cond_operator(mu, C, reg)
            pred = base.predict(X["validation"], mu, A)
            m = base.metrics(X["validation"][:, base.HIDX], pred)
            rec = {"model": model, "reg": reg, **m}
            tuning.append(rec)
            if best is None or m["mse"] < best[0]:
                best = (m["mse"], reg, A, var)
        chosen[model] = {"reg": float(best[1]), "A": best[2], "var": best[3], "validation_mse": float(best[0])}

    rows = []
    mean_pred = np.broadcast_to(mu[base.HIDX], (len(X["test"]), len(base.HIDX))).copy()
    test_meta = pd.DataFrame(meta["test"])
    for field in test:
        ii = np.flatnonzero(test_meta.field.to_numpy() == field)
        y = X["test"][ii][:, base.HIDX]
        rows.append({"field": field, "model": "mean", "reg": np.nan, **base.metrics(y, mean_pred[ii])})
        for model, c in chosen.items():
            pred = base.predict(X["test"][ii], mu, c["A"])
            rows.append({"field": field, "model": model, "reg": c["reg"], **base.metrics(y, pred)})

    results = pd.DataFrame(rows)
    results.to_csv(out / "test_field_metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "validation_tuning.csv", index=False)
    pd.DataFrame(selection_metrics).to_csv(out / "selection_model_metrics.csv", index=False)

    wide = results.pivot(index="field", columns="model", values="mse")
    pairwise = {
        f"mean_minus_{model}_mse": base.paired_positive((wide["mean"] - wide[model]).to_numpy())
        for model in covariances
    }
    pairwise.update({
        "oas_minus_isotropic_mse": base.paired_positive((wide["oas"] - wide["isotropic"]).to_numpy()),
        "ledoit_wolf_minus_isotropic_mse": base.paired_positive((wide["ledoit_wolf"] - wide["isotropic"]).to_numpy()),
        "stationary_minus_isotropic_mse": base.paired_positive((wide["stationary"] - wide["isotropic"]).to_numpy()),
        "isotropic_minus_stationary_mse": base.paired_positive((wide["isotropic"] - wide["stationary"]).to_numpy()),
    })

    iso_vs_oas = pairwise["oas_minus_isotropic_mse"]
    iso_vs_lw = pairwise["ledoit_wolf_minus_isotropic_mse"]
    if strong_pairwise(iso_vs_oas) and strong_pairwise(iso_vs_lw):
        regularization_decision = "SUPPORT_ISOTROPY_BEYOND_GENERIC_SHRINKAGE_POSTHOC"
    else:
        regularization_decision = "NO_POSTHOC_SUPPORT_ISOTROPY_BEYOND_GENERIC_SHRINKAGE"

    iso_vs_stationary = pairwise["stationary_minus_isotropic_mse"]
    stationary_vs_iso = pairwise["isotropic_minus_stationary_mse"]
    if strong_pairwise(iso_vs_stationary):
        stationarity_decision = "SUPPORT_RADIAL_OVER_DIRECTIONAL_STATIONARY_POSTHOC"
    elif strong_pairwise(stationary_vs_iso):
        stationarity_decision = "SUPPORT_DIRECTIONAL_STATIONARY_OVER_RADIAL_POSTHOC"
    else:
        stationarity_decision = "NO_STRONG_POSTHOC_DIFFERENCE_RADIAL_VS_DIRECTIONAL_STATIONARY"

    summary = {
        "status": "REAL_DR11_SELECTION_RESIDUAL_COVARIANCE_CAPACITY_AUDIT",
        "inference_scope": "post-hoc mechanistic audit on the previously observed 18/9/9 split; not a new confirmatory blind test",
        "field_split": {"train": train, "validation": validation, "test": test},
        "n_patches": {k: int(len(v)) for k, v in X.items()},
        "patch_arcmin": 3.75,
        "hidden_arcmin": 1.875,
        "covariance_capacity": {
            "full_symmetric_entries": int(base.PATCH**2 * (base.PATCH**2 + 1) // 2),
            "stationary_displacement_classes": int(n_stationary_classes),
            "isotropic_distance_classes": radial_class_count(),
            "ledoit_wolf_shrinkage": float(lw.shrinkage_),
            "oas_shrinkage": float(oas.shrinkage_),
        },
        "models": {model: {"chosen_reg": float(c["reg"]), "validation_mse": float(c["validation_mse"])} for model, c in chosen.items()},
        "test_medians": {},
        "paired_tests": pairwise,
        "diagnostic_rule": "specific isotropy support requires isotropic MSE lower than both OAS and Ledoit-Wolf in >=8/9 test fields with one-sided sign and Wilcoxon p<.05 for both comparisons; labels remain post-hoc",
        "regularization_decision": regularization_decision,
        "stationarity_decision": stationarity_decision,
    }
    for model, g in results.groupby("model"):
        summary["test_medians"][model] = {c: float(np.nanmedian(g[c])) for c in ["mse", "corr", "hidden_mean_corr"] if np.isfinite(g[c]).any()}

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "provenance.json").write_text(json.dumps({"status": summary["status"], "source_regions": provenance}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
