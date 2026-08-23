#!/usr/bin/env python3
"""Leave-one-field-out validation on real DR11 position fields.

Each of the 12 observed sky fields is held out in turn. The paired spatial-null
uses the same observed cell counts randomly permuted within each field.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from analyze_dr11 import (
    mask_indices,
    motif_metrics,
    normalize_grid,
    patchify,
    reconstruction_predictions,
    region_grid,
    score_reconstruction,
    shuffle_grid,
    verify_and_load,
)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    n = len(values)
    ge = 0
    total = 1 << n
    for bits in range(total):
        signs = np.fromiter((1.0 if (bits >> i) & 1 else -1.0 for i in range(n)), dtype=float, count=n)
        if float(np.mean(values * signs)) >= observed - 1e-15:
            ge += 1
    return ge / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/dr11/pilot")
    ap.add_argument("--out", default="results/real_dr11/latest")
    args = ap.parse_args()
    datadir, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((datadir / "provenance.json").read_text())
    if provenance.get("status") != "REAL_DR11" or provenance.get("model_input_columns") != ["ra", "dec"]:
        raise RuntimeError("LOFO requires provenance-verified REAL_DR11 RA/Dec-only input")

    meta_by_name = {r["name"]: r for r in provenance["regions"]}
    fields = list(meta_by_name)
    real_parts, null_parts = {}, {}
    for j, (field, meta) in enumerate(meta_by_name.items()):
        grid = region_grid(verify_and_load(meta), meta)
        norm_grid, _ = normalize_grid(grid)
        real_parts[field], _ = patchify(norm_grid, field)
        null_grid = shuffle_grid(grid, 20260824 + j)
        norm_null, _ = normalize_grid(null_grid)
        null_parts[field], _ = patchify(norm_null, field)

    rows = []
    for held_out in fields:
        training_fields = [f for f in fields if f != held_out]
        for sample, parts in [("real", real_parts), ("spatial_permutation_null", null_parts)]:
            train_raw = np.concatenate([parts[f] for f in training_fields], axis=0)
            test_raw = parts[held_out]
            scaler = StandardScaler().fit(train_raw)
            train = scaler.transform(train_raw)
            test = scaler.transform(test_raw)

            hidden = mask_indices("center25")
            gaussian = reconstruction_predictions(train, test, hidden)["gaussian"]
            mse, corr = score_reconstruction(test[:, hidden], gaussian)
            motifs = {m["motif"]: m for m in motif_metrics(train, test)}
            rows.append({
                "field": held_out,
                "sample": sample,
                "gaussian_center25_mse": mse,
                "gaussian_center25_corr": corr,
                "void_auc": motifs["void"]["auc"],
                "overdense_auc": motifs["overdense"]["auc"],
                "peak_auc": motifs["peak"]["auc"],
                "void_test_rate": motifs["void"]["test_rate"],
                "overdense_test_rate": motifs["overdense"]["test_rate"],
                "peak_test_rate": motifs["peak"]["test_rate"],
            })

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "lofo_field_metrics.csv", index=False)
    wide = result.pivot(index="field", columns="sample")

    comparisons = []
    definitions = {
        "gaussian_center25_corr": "real - null; positive favors real spatial predictability",
        "gaussian_center25_mse": "null - real; positive favors real reconstruction",
        "void_auc": "real - null; positive favors real boundary predictability",
        "overdense_auc": "real - null; positive favors real boundary predictability",
        "peak_auc": "real - null; positive favors real boundary predictability",
    }
    for metric, definition in definitions.items():
        real = wide[metric]["real"].astype(float)
        null = wide[metric]["spatial_permutation_null"].astype(float)
        diff = null - real if metric.endswith("_mse") else real - null
        finite = np.isfinite(diff.to_numpy())
        comparisons.append({
            "metric": metric,
            "definition": definition,
            "n_fields": int(finite.sum()),
            "positive_fields": int((diff[finite] > 0).sum()),
            "real_median": float(np.nanmedian(real)),
            "null_median": float(np.nanmedian(null)),
            "mean_paired_advantage": float(np.nanmean(diff)),
            "exact_one_sided_signflip_p": float(exact_signflip_p(diff.to_numpy())),
        })

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(outdir / "lofo_comparisons.csv", index=False)
    summary = {
        "status": "REAL_DR11",
        "validation": "12-field leave-one-field-out",
        "n_fields": len(fields),
        "model_input_columns": ["ra", "dec"],
        "null": "within-field permutation of the observed 64x64 count cells",
        "comparisons": comparisons,
    }
    (outdir / "lofo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
