#!/usr/bin/env python3
"""Analyze real DR11 sky positions with artificial held-out holes.

Targets are observed DR11 counts before masking. No simulated cosmological
field is generated. A within-field cell permutation is used only as a spatial
null, preserving the observed one-point count distribution.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

GRID = 64
PATCH = 8
HALF = 0.25


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_and_load(meta: dict) -> pd.DataFrame:
    path = Path(meta["file"])
    gz = path.read_bytes()
    raw = gzip.decompress(gz)
    if sha256_bytes(gz) != meta["stored_gzip_sha256"]:
        raise RuntimeError(f"hash mismatch: {path}")
    if sha256_bytes(raw) != meta["canonical_csv_sha256"]:
        raise RuntimeError(f"canonical hash mismatch: {path}")
    df = pd.read_csv(path)
    if len(df) != int(meta["rows"]):
        raise RuntimeError(f"row-count mismatch: {path}")
    if list(df.columns) != ["ra", "dec"]:
        raise RuntimeError(f"position-only input expected, got {list(df.columns)}")
    return df


def tangent_ra(ra: np.ndarray, ra0: float) -> np.ndarray:
    return ((np.asarray(ra, float) - ra0 + 180.0) % 360.0) - 180.0


def region_grid(df: pd.DataFrame, meta: dict) -> np.ndarray:
    dra = tangent_ra(df["ra"].to_numpy(), float(meta["center_ra_deg"]))
    ddec = df["dec"].to_numpy(float) - float(meta["center_dec_deg"])
    keep = (np.abs(dra) < HALF) & (np.abs(ddec) < HALF)
    h, _, _ = np.histogram2d(
        ddec[keep], dra[keep], bins=GRID,
        range=[[-HALF, HALF], [-HALF, HALF]],
    )
    return h.astype(np.float64)


def normalize_grid(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.log1p(grid)
    med = np.median(x)
    scale = np.median(np.abs(x - med)) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (x - med) / scale, {
        "log1p_median": float(med),
        "robust_scale": float(scale),
        "mean_count": float(grid.mean()),
    }


def patchify(norm_grid: np.ndarray, field: str) -> tuple[np.ndarray, list[dict]]:
    xs, rows = [], []
    n = GRID // PATCH
    for iy in range(n):
        for ix in range(n):
            p = norm_grid[iy*PATCH:(iy+1)*PATCH, ix*PATCH:(ix+1)*PATCH]
            xs.append(p.reshape(-1))
            rows.append({"field": field, "patch_y": iy, "patch_x": ix})
    return np.asarray(xs), rows


def shuffle_grid(grid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = grid.reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(grid.shape)


def mask_indices(kind: str) -> np.ndarray:
    m = np.zeros((PATCH, PATCH), dtype=bool)
    if kind == "center25":
        m[2:6, 2:6] = True
    elif kind == "corner25":
        m[0:4, 0:4] = True
    elif kind == "stripe25":
        m[:, 3:5] = True
    elif kind == "random25":
        idx = np.random.default_rng(1701).choice(PATCH*PATCH, 16, replace=False)
        m.reshape(-1)[idx] = True
    else:
        raise ValueError(kind)
    return np.flatnonzero(m.reshape(-1))


def reconstruction_predictions(Xtr: np.ndarray, Xte: np.ndarray, hidden: np.ndarray) -> dict[str, np.ndarray]:
    obs = np.setdiff1d(np.arange(Xtr.shape[1]), hidden)
    mu = Xtr.mean(axis=0)
    pred = {"mean": np.broadcast_to(mu[hidden], (len(Xte), len(hidden))).copy()}

    cov = np.cov(Xtr, rowvar=False)
    coo = cov[np.ix_(obs, obs)] + 0.08 * np.eye(len(obs))
    coh = cov[np.ix_(obs, hidden)]
    pred["gaussian"] = mu[hidden] + (Xte[:, obs] - mu[obs]) @ np.linalg.solve(coo, coh)

    k = min(20, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(Xtr)
    W = pca.components_.T
    Wo, Wh = W[obs], W[hidden]
    z = (Xte[:, obs] - pca.mean_[obs]) @ Wo @ np.linalg.inv(Wo.T @ Wo + 0.08 * np.eye(k))
    pred["pca"] = pca.mean_[hidden] + z @ Wh.T

    nn = NearestNeighbors(n_neighbors=min(20, len(Xtr))).fit(Xtr[:, obs])
    inds = nn.kneighbors(Xte[:, obs], return_distance=False)
    pred["knn20"] = np.stack([Xtr[ii][:, hidden].mean(axis=0) for ii in inds])
    return pred


def score_reconstruction(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y - pred) ** 2))
    if np.std(y) == 0 or np.std(pred) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y.ravel(), pred.ravel())[0, 1])
    return mse, corr


def ring_features(X: np.ndarray) -> np.ndarray:
    a = X.reshape(-1, PATCH, PATCH)
    ring = np.zeros((PATCH, PATCH), bool)
    ring[1, 1:7] = True
    ring[6, 1:7] = True
    ring[2:6, 1] = True
    ring[2:6, 6] = True
    vals = a[:, ring]
    left = a[:, 2:6, 1].mean(1)
    right = a[:, 2:6, 6].mean(1)
    top = a[:, 1, 2:6].mean(1)
    bottom = a[:, 6, 2:6].mean(1)
    return np.column_stack([
        vals.mean(1), vals.std(1), left, right, top, bottom,
        right-left, bottom-top,
    ])


def motif_metrics(Xtr: np.ndarray, Xte: np.ndarray) -> list[dict]:
    hid = mask_indices("center25")
    tr_mean = Xtr[:, hid].mean(1)
    te_mean = Xte[:, hid].mean(1)
    q25, q75 = np.quantile(tr_mean, [0.25, 0.75])
    tr_max = Xtr[:, hid].max(1)
    te_max = Xte[:, hid].max(1)
    qpeak = np.quantile(tr_max, 0.80)
    Ftr, Fte = ring_features(Xtr), ring_features(Xte)

    labels = {
        "void": (tr_mean <= q25, te_mean <= q25),
        "overdense": (tr_mean >= q75, te_mean >= q75),
        "peak": (tr_max >= qpeak, te_max >= qpeak),
    }
    output = []
    for name, (ytr, yte) in labels.items():
        auc = float("nan")
        if len(np.unique(ytr)) == 2 and len(np.unique(yte)) == 2:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ftr, ytr.astype(int))
            auc = float(roc_auc_score(yte.astype(int), clf.predict_proba(Fte)[:, 1]))
        output.append({
            "motif": name,
            "auc": auc,
            "train_rate": float(ytr.mean()),
            "test_rate": float(yte.mean()),
        })
    return output


def fit_surface(Xtr: np.ndarray, Xte: np.ndarray, test_rows: list[dict], outdir: Path) -> dict:
    pca = PCA(n_components=2, random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    tw = float(trustworthiness(Xte, Zte, n_neighbors=min(10, len(Xte)-1)))

    ref = GaussianMixture(3, covariance_type="full", random_state=0).fit(Ztr).predict(Zte)
    aris = []
    rng = np.random.default_rng(33)
    for b in range(30):
        ii = rng.integers(0, len(Ztr), len(Ztr))
        try:
            labels = GaussianMixture(3, covariance_type="full", random_state=b+1).fit(Ztr[ii]).predict(Zte)
            aris.append(adjusted_rand_score(ref, labels))
        except Exception:
            pass

    surf = pd.DataFrame(test_rows)
    surf["z1"] = Zte[:, 0]
    surf["z2"] = Zte[:, 1]
    surf["gmm3"] = ref
    surf.to_csv(outdir / "pattern_surface.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(Zte[:, 0], Zte[:, 1], s=20, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("REAL DR11 position-pattern surface")
    fig.tight_layout()
    fig.savefig(outdir / "pattern_surface.png", dpi=160)
    plt.close(fig)

    return {
        "pca2_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "test_trustworthiness": tw,
        "gmm3_bootstrap_ari_median": float(np.median(aris)) if aris else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/dr11/pilot")
    ap.add_argument("--out", default="results/real_dr11/latest")
    args = ap.parse_args()
    datadir, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((datadir / "provenance.json").read_text())
    if provenance.get("status") != "REAL_DR11":
        raise RuntimeError("Input provenance is not REAL_DR11")
    if provenance.get("model_input_columns") != ["ra", "dec"]:
        raise RuntimeError("Pilot requires position-only RA/Dec input")

    split = provenance["field_split"]
    meta_by_name = {r["name"]: r for r in provenance["regions"]}
    real_parts, null_parts, rows = {}, {}, {}
    field_norm = []

    for j, (field, meta) in enumerate(meta_by_name.items()):
        grid = region_grid(verify_and_load(meta), meta)
        norm_grid, norm_meta = normalize_grid(grid)
        real_parts[field], rows[field] = patchify(norm_grid, field)
        field_norm.append({"field": field, **norm_meta})

        null_grid = shuffle_grid(grid, 20260824 + j)
        norm_null_grid, _ = normalize_grid(null_grid)
        null_parts[field], _ = patchify(norm_null_grid, field)

    def cat(parts: dict, names: list[str]) -> np.ndarray:
        return np.concatenate([parts[n] for n in names], axis=0)

    def cat_rows(names: list[str]) -> list[dict]:
        return [r for n in names for r in rows[n]]

    Xtr0, Xte0 = cat(real_parts, split["train"]), cat(real_parts, split["test"])
    Ntr0, Nte0 = cat(null_parts, split["train"]), cat(null_parts, split["test"])
    scaler = StandardScaler().fit(Xtr0)
    Xtr, Xte = scaler.transform(Xtr0), scaler.transform(Xte0)
    null_scaler = StandardScaler().fit(Ntr0)
    Ntr, Nte = null_scaler.transform(Ntr0), null_scaler.transform(Nte0)

    summary = {
        "status": "REAL_DR11",
        "dataset": provenance["dataset"],
        "input_total_rows": int(provenance["total_rows"]),
        "model_input_columns": ["ra", "dec"],
        "field_split": split,
        "surface": fit_surface(Xtr, Xte, cat_rows(split["test"]), outdir),
    }

    reconstruction_rows = []
    for sample, train, test in [
        ("real", Xtr, Xte),
        ("spatial_permutation_null", Ntr, Nte),
    ]:
        for mask in ["center25", "random25", "corner25", "stripe25"]:
            hidden = mask_indices(mask)
            y = test[:, hidden]
            for method, pred in reconstruction_predictions(train, test, hidden).items():
                mse, corr = score_reconstruction(y, pred)
                reconstruction_rows.append({
                    "sample": sample,
                    "mask": mask,
                    "method": method,
                    "hidden_fraction": float(len(hidden)/(PATCH*PATCH)),
                    "mse": mse,
                    "corr": corr,
                    "n_test_patches": int(len(test)),
                })
    rec = pd.DataFrame(reconstruction_rows)
    rec.to_csv(outdir / "reconstruction_metrics.csv", index=False)

    motif_rows = []
    for sample, train, test in [
        ("real", Xtr, Xte),
        ("spatial_permutation_null", Ntr, Nte),
    ]:
        for metric in motif_metrics(train, test):
            motif_rows.append({"sample": sample, **metric, "n_test_patches": int(len(test))})
    motifs = pd.DataFrame(motif_rows)
    motifs.to_csv(outdir / "motif_metrics.csv", index=False)
    pd.DataFrame(field_norm).to_csv(outdir / "field_normalization.csv", index=False)

    center = rec[(rec["sample"] == "real") & (rec["mask"] == "center25")].sort_values("mse")
    best = center.iloc[0]
    summary["reconstruction_best_center25"] = {
        "method": str(best["method"]),
        "mse": float(best["mse"]),
        "corr": float(best["corr"]),
    }
    summary["motif_auc"] = {
        f"{r.sample}:{r.motif}": float(r.auc) for r in motifs.itertuples()
    }
    summary["null_control"] = (
        "Within each field, the observed 64x64 count cells are randomly permuted, "
        "preserving the one-point count distribution while destroying spatial adjacency."
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    c = rec[rec["mask"] == "center25"]
    labels, values = [], []
    for sample in ["real", "spatial_permutation_null"]:
        for method in ["mean", "gaussian", "pca", "knn20"]:
            labels.append(f"{sample}\n{method}")
            values.append(float(c[(c["sample"] == sample) & (c["method"] == method)]["mse"].iloc[0]))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("hidden standardized MSE")
    ax.set_title("REAL DR11 vs spatial-permutation null — central 25% hole")
    fig.tight_layout()
    fig.savefig(outdir / "center25_reconstruction.png", dpi=160)
    plt.close(fig)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
