#!/usr/bin/env python3
"""Fresh-field confirmation of structured covariance reconstruction in REAL DR11.

This is a confirmatory follow-up to the accepted 36-field covariance result and
the explicitly post-hoc covariance-capacity audit.  All 36 historical
selection-qualified bricks are training data only.  Test candidates come from
the same deterministic sky grid/shuffle as expanded48, starting at candidate
index 54 (after the historical 48-field sample was fixed), and are accepted in
fixed order using acquisition/coverage gates only.

Primary question: does translation-invariant directional covariance reduce
hidden-cell MSE relative to BOTH the historical train-mean predictor and OAS
shrinkage on 20 fresh fields?

No new-field target-density outcome, reconstruction score, or effect size is
used for model selection, regularization tuning, field ordering, or stopping.
No simulated cosmology or mock target is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

import cross_tracer_selection_residual_bispectrum as cs
import fetch_dr11_expanded as exp
import selection_residual_covariance_capacity as cap
import selection_residual_covariance_reconstruct as base
import selection_residual_locality_decay as sl
import tracer_split_validate as tr

TARGET_FIELDS = 20
CANDIDATE_START = 54
CANDIDATE_STOP = 174  # exclusive; frozen before execution
MIN_SEP_DEG = 6.0
FROZEN_REG = {
    "stationary": 0.1,
    "isotropic": 0.01,
    "oas": 1.0,
    "ledoit_wolf": 0.3,
}


def field_name(candidate_index: int, ra: float, dec: float) -> str:
    sign = "p" if dec >= 0 else "m"
    return f"c{candidate_index:03d}_ra{int(round(ra))%360:03d}_{sign}{abs(int(round(dec))):02d}"


def historical_centers(path: Path) -> list[tuple[float, float]]:
    d = json.loads(path.read_text())
    if d.get("status") != "REAL_DR11":
        raise RuntimeError("historical expanded48 provenance is not REAL_DR11")
    regs = d.get("regions", [])
    if len(regs) != 48:
        raise RuntimeError(f"expected 48 historical expanded fields, got {len(regs)}")
    return [(float(r["center_ra_deg"]), float(r["center_dec_deg"])) for r in regs]


def acquire_confirmation_fields(old_data: dict, expanded_provenance: Path):
    old_centers = historical_centers(expanded_provenance)
    accepted_centers: list[tuple[float, float]] = []
    data: dict[str, dict] = {}
    provenance: list[dict] = []
    rejected: list[dict] = []
    used_bricks = {str(d["brick"]) for d in old_data.values()}
    candidates = exp.candidate_centers()

    if len(candidates) < CANDIDATE_STOP:
        raise RuntimeError("candidate grid unexpectedly shortened")

    for candidate_index in range(CANDIDATE_START, CANDIDATE_STOP):
        if len(data) >= TARGET_FIELDS:
            break
        ra, dec = candidates[candidate_index]
        name = field_name(candidate_index, ra, dec)
        prior = old_centers + accepted_centers
        nearest = min(exp.angular_sep(ra, dec, r, d) for r, d in prior)
        if nearest < MIN_SEP_DEG:
            rejected.append({
                "candidate_index": candidate_index, "field": name,
                "ra": ra, "dec": dec, "reason": "prequery_separation_reject",
                "nearest_prior_center_deg": float(nearest),
            })
            continue

        try:
            brick, brick_query = tr.choose_brick(ra, dec)
        except Exception as exc:
            rejected.append({
                "candidate_index": candidate_index, "field": name,
                "ra": ra, "dec": dec, "reason": "brick_resolution",
                "error": str(exc),
            })
            continue
        if brick in used_bricks:
            rejected.append({
                "candidate_index": candidate_index, "field": name,
                "ra": ra, "dec": dec, "brick": brick, "reason": "duplicate_brick",
            })
            continue

        print(f"[confirm20] candidate {candidate_index}: {name}->{brick}", flush=True)
        try:
            tractor, tractor_prov = tr.get_tractor(brick)
            sel, valid, feature_names, selection_prov, n_patches = cs.build_selection_features(brick)
        except Exception as exc:
            rejected.append({
                "candidate_index": candidate_index, "field": name,
                "ra": ra, "dec": dec, "brick": brick,
                "reason": "official_product_acquisition_or_coverage", "error": str(exc),
            })
            continue

        counts = tr.grid_from(tractor).astype(float)
        used_bricks.add(brick)
        accepted_centers.append((ra, dec))
        data[name] = {
            "brick": brick,
            "counts": counts,
            "sel": sel,
            "valid": valid,
            "feature_names": feature_names,
        }
        provenance.append({
            "candidate_index": candidate_index,
            "field": name,
            "center_ra_deg": ra,
            "center_dec_deg": dec,
            "nearest_prior_center_deg": float(nearest),
            "brick": brick,
            "brick_choice_query": brick_query,
            "tractor": tractor_prov,
            "source_rows": int(len(tractor)),
            "valid_cell_fraction": float(valid.mean()),
            "n_coverage_patches": int(n_patches),
            "selection_products": selection_prov,
        })
        print(
            f"[confirm20] accept {len(data)}/{TARGET_FIELDS}: "
            f"rows={len(tractor)} valid={valid.mean():.3f}",
            flush=True,
        )

    if len(data) != TARGET_FIELDS:
        raise RuntimeError(
            f"only {len(data)} fresh fields accepted from frozen candidate "
            f"indices {CANDIDATE_START}:{CANDIDATE_STOP}"
        )
    return data, provenance, rejected


def collect_patches(maps: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str]):
    arrays = []
    rows = []
    for name in names:
        z, valid = maps[name]
        x, recs = base.extract_full_valid_patches(z, valid, name)
        if not len(x):
            raise RuntimeError(f"no fully valid reconstruction patches for {name}")
        arrays.append(x)
        rows.extend(recs)
    return np.concatenate(arrays), pd.DataFrame(rows)


def strong(result: dict) -> bool:
    return (
        int(result["n"]) == TARGET_FIELDS
        and int(result["positive"]) >= 16
        and float(result["sign_p_one_sided"]) < 0.01
        and float(result["wilcoxon_p_one_sided"]) < 0.01
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded-provenance", default="data/real/dr11/expanded48/provenance.json")
    ap.add_argument("--out", default="results/real_dr11/selection_residual_covariance_confirm20")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Historical data are training-only for this confirmation.
    old_data, old_provenance = sl.acquire()
    old_names = list(old_data)
    if len(old_names) != 36:
        raise RuntimeError(f"historical training-field invariant failed: {len(old_names)}")

    new_data, new_provenance, rejected = acquire_confirmation_fields(
        old_data, Path(args.expanded_provenance)
    )
    new_names = list(new_data)
    if len({tuple(old_data[n]["feature_names"]) for n in old_names}) != 1:
        raise RuntimeError("historical selection-feature mismatch")
    if len({tuple(new_data[n]["feature_names"]) for n in new_names}) != 1:
        raise RuntimeError("new-field selection-feature mismatch")
    if tuple(old_data[old_names[0]]["feature_names"]) != tuple(new_data[new_names[0]]["feature_names"]):
        raise RuntimeError("historical/new selection-feature mismatch")

    # Historical residuals are out-of-fold; new-field nuisance predictions use
    # one final model fitted only on historical fields.
    old_maps, selection_metrics = base.crossfit_train_residuals(old_data, old_names)
    final_selection_model = cs.fit_selection_model(old_data, old_names, "counts", 26000)
    new_maps = {}
    for name in new_names:
        z, valid, corr = base.residual_map(new_data, name, final_selection_model)
        new_maps[name] = (z, valid)
        selection_metrics.append({
            "field": name, "role": "fresh_test",
            "selection_count_corr": corr,
        })

    Xtrain, train_meta = collect_patches(old_maps, old_names)
    Xtest, test_meta = collect_patches(new_maps, new_names)
    if len(Xtrain) < 300 or len(Xtest) < 80:
        raise RuntimeError({"train_patches": len(Xtrain), "test_patches": len(Xtest)})

    mu, Cfull = base.empirical_cov(Xtrain)
    Cstationary, stationary_classes = cap.stationary_cov(Cfull)
    Ciso = base.isotropize(Cfull)
    oas = OAS(assume_centered=False).fit(Xtrain)
    lw = LedoitWolf(assume_centered=False).fit(Xtrain)
    covariance = {
        "stationary": Cstationary,
        "isotropic": Ciso,
        "oas": oas.covariance_,
        "ledoit_wolf": lw.covariance_,
    }
    operators = {
        model: base.cond_operator(mu, C, FROZEN_REG[model])[0]
        for model, C in covariance.items()
    }

    rows = []
    field_col = test_meta.field.to_numpy()
    for field in new_names:
        ii = np.flatnonzero(field_col == field)
        y = Xtest[ii][:, base.HIDX]
        mean_pred = np.broadcast_to(mu[base.HIDX], y.shape)
        rows.append({
            "field": field, "model": "mean", "reg": np.nan,
            **base.metrics(y, mean_pred),
        })
        for model, A in operators.items():
            pred = base.predict(Xtest[ii], mu, A)
            rows.append({
                "field": field, "model": model, "reg": FROZEN_REG[model],
                **base.metrics(y, pred),
            })

    R = pd.DataFrame(rows)
    R.to_csv(out / "test_field_metrics.csv", index=False)
    pd.DataFrame(selection_metrics).to_csv(out / "selection_model_metrics.csv", index=False)
    test_meta.to_csv(out / "test_patch_index.csv", index=False)

    wide = R.pivot(index="field", columns="model", values="mse")
    comparisons = {
        "mean_minus_stationary_mse": base.paired_positive((wide["mean"] - wide["stationary"]).to_numpy()),
        "oas_minus_stationary_mse": base.paired_positive((wide["oas"] - wide["stationary"]).to_numpy()),
        "ledoit_wolf_minus_stationary_mse": base.paired_positive((wide["ledoit_wolf"] - wide["stationary"]).to_numpy()),
        "mean_minus_isotropic_mse": base.paired_positive((wide["mean"] - wide["isotropic"]).to_numpy()),
        "oas_minus_isotropic_mse": base.paired_positive((wide["oas"] - wide["isotropic"]).to_numpy()),
        "isotropic_minus_stationary_mse": base.paired_positive((wide["isotropic"] - wide["stationary"]).to_numpy()),
        "stationary_minus_isotropic_mse": base.paired_positive((wide["stationary"] - wide["isotropic"]).to_numpy()),
    }

    a = comparisons["mean_minus_stationary_mse"]
    b = comparisons["oas_minus_stationary_mse"]
    if strong(a) and strong(b):
        decision = "PASS_STRUCTURED_STATIONARY_CONFIRMATION"
    elif float(a["median"]) <= 0 or float(b["median"]) <= 0:
        decision = "FAIL_STRUCTURED_STATIONARY_CONFIRMATION"
    else:
        decision = "UNCERTAIN_STRUCTURED_STATIONARY_CONFIRMATION"

    summary = {
        "status": "REAL_DR11_SELECTION_RESIDUAL_COVARIANCE_INDEPENDENT_CONFIRMATION",
        "decision": decision,
        "inference_scope": "fresh-field confirmatory test; 36 historical fields training-only; 20 fixed-order post-index-53 candidates used for test",
        "candidate_protocol": {
            "seed": 20260824,
            "candidate_start_inclusive": CANDIDATE_START,
            "candidate_stop_exclusive": CANDIDATE_STOP,
            "target_fields": TARGET_FIELDS,
            "min_center_separation_deg": MIN_SEP_DEG,
            "gates": [
                "separation from historical/new confirmation centers",
                "unique official Tractor brick resolution",
                "official Tractor and selection-product acquisition",
                "existing >=8 complete selection-coverage patches",
            ],
            "explicitly_forbidden_gates": [
                "target-density threshold", "reconstruction score", "covariance score", "effect size"
            ],
        },
        "fresh_fields": new_names,
        "n_rejected_candidates_before_completion": int(len(rejected)),
        "n_patches": {"historical_train": int(len(Xtrain)), "fresh_test": int(len(Xtest))},
        "patch_arcmin": 3.75,
        "hidden_arcmin": 1.875,
        "frozen_regularization": FROZEN_REG,
        "covariance_capacity": {
            "stationary_displacement_classes": int(stationary_classes),
            "isotropic_distance_classes": int(cap.radial_class_count()),
            "oas_shrinkage": float(oas.shrinkage_),
            "ledoit_wolf_shrinkage": float(lw.shrinkage_),
        },
        "test_medians": {},
        "paired_tests": comparisons,
        "primary_rule": "PASS iff stationary beats both mean and OAS in >=16/20 fresh fields and both one-sided sign and Wilcoxon p<.01 for both comparisons; FAIL if either median advantage is non-positive; otherwise UNCERTAIN",
        "radial_vs_directional_status": "secondary only; no radial-isotropy claim is licensed by the primary decision",
    }
    for model, g in R.groupby("model"):
        summary["test_medians"][model] = {
            c: float(np.nanmedian(g[c]))
            for c in ["mse", "corr", "hidden_mean_corr"]
            if np.isfinite(g[c]).any()
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "status": summary["status"],
        "historical_training_regions": old_provenance,
        "fresh_test_regions": new_provenance,
        "candidate_rejections": rejected,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
