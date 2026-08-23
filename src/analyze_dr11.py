#!/usr/bin/env python3
"""Analyze real DR11 sky positions with artificial held-out holes.

The hidden targets are observed DR11 counts before masking. No simulated target
field is generated anywhere in this program.
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
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
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
RNG = np.random.default_rng(20260824)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_and_load(meta: dict) -> pd.DataFrame:
    path = Path(meta["file"])
    gz = path.read_bytes()
    if sha256_bytes(gz) != meta["stored_gzip_sha256"]:
        raise RuntimeError(f"hash mismatch: {path}")
    raw = gzip.decompress(gz)
    if sha256_bytes(raw) != meta["canonical_csv_sha256"]:
        raise RuntimeError(f"canonical hash mismatch: {path}")
    df = pd.read_csv(path)
    if len(df) != int(meta["rows"]):
        raise RuntimeError(f"row-count mismatch: {path}")
    return df


def tangent_ra(ra: np.ndarray, ra0: float) -> np.ndarray:
    return ((np.asarray(ra, float) - ra0 + 180.0) % 360.0) - 180.0


def region_grid(df: pd.DataFrame, meta: dict, tracer: str) -> np.ndarray:
    if tracer == "extended_clean":
        t = df["type"].astype(str).str.strip().str.upper()
        mb = pd.to_numeric(df["maskbits"], errors="coerce").fillna(-1).astype(np.int64)
        df = df.loc[(mb == 0) & (t != "PSF")].copy()
    elif tracer != "all_primary":
        raise ValueError(tracer)

    dra = tangent_ra(df["ra"].to_numpy(), float(meta["center_ra_deg"]))
    ddec = df["dec"].to_numpy(float) - float(meta["center_dec_deg"])
    keep = (np.abs(dra) < HALF) & (np.abs(ddec) < HALF)
    h, _, _ = np.histogram2d(
        ddec[keep], dra[keep], bins=GRID,
        range=[[-HALF, HALF], [-HALF, HALF]],
    )
    return h.astype(np.float64)


def patchify(grid: np.ndarray, field: str, tracer: str) -> tuple[np.ndarray, list[dict]]:
    xs, rows = [], []
    n = GRID // PATCH
    for iy in range(n):
        for ix in range(n):
            p = grid[iy*PATCH:(iy+1)*PATCH, ix*PATCH:(ix+1)*PATCH]
            xs.append(np.log1p(p).reshape(-1))
            rows.append({"field": field, "tracer": tracer, "patch_y": iy, "patch_x": ix,
                         "mean_count": float(p.mean()), "sum_count": float(p.sum())})
    return np.asarray(xs), rows


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
    all_idx = np.arange(Xtr.shape[1])
    obs = np.setdiff1d(all_idx, hidden)
    mu = Xtr.mean(axis=0)

    pred = {"mean": np.broadcast_to(mu[hidden], (len(Xte), len(hidden))).copy()}

    cov = np.cov(Xtr, rowvar=False)
    coo = cov[np.ix_(obs, obs)] + 0.08 * np.eye(len(obs))
    coh = cov[np.ix_(obs, hidden)]
    A = np.linalg.solve(coo, coh)
    pred["gaussian"] = mu[hidden] + (Xte[:, obs] - mu[obs]) @ A

    k = min(20, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(Xtr)
    W = pca.components_.T
    Wo, Wh = W[obs], W[hidden]
    inv = np.linalg.inv(Wo.T @ Wo + 0.08 * np.eye(k))
    z = (Xte[:, obs] - pca.mean_[obs]) @ Wo @ inv
    pred["pca"] = pca.mean_[hidden] + z @ Wh.T

    nn = NearestNeighbors(n_neighbors=min(20, len(Xtr))).fit(Xtr[:, obs])
    inds = nn.kneighbors(Xte[:, obs], return_distance=False)
    pred["knn20"] = np.stack([Xtr[ii][:, hidden].mean(axis=0) for ii in inds])
    return pred


def score_reconstruction(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y - p) ** 2))
    if np.std(y) == 0 or np.std(p) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y.ravel(), p.ravel())[0, 1])
    return mse, corr


def ring_features(X: np.ndarray) -> np.ndarray:
    a = X.reshape(-1, PATCH, PATCH)
    ring = np.zeros((PATCH, PATCH), bool)
    ring[1, 1:7] = True
    ring[6, 1:7] = True
    ring[2:6, 1] = True
    ring[2:6, 6] = True
    vals = a[:, ring]
    left = a[:, 2:6, 1].mean((1, 2))
    right = a[:, 2:6, 6].mean((1, 2))
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
    outputs = []
    labels = {
        "void": (tr_mean <= q25, te_mean <= q25),
        "overdense": (tr_mean >= q75, te_mean >= q75),
        "peak": (tr_max >= qpeak, te_max >= qpeak),
    }
    for name, (ytr, yte) in labels.items():
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            auc = float("nan")
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ftr, ytr.astype(int))
            prob = clf.predict_proba(Fte)[:, 1]
            auc = float(roc_auc_score(yte.astype(int), prob))
        outputs.append({"motif": name, "auc": auc, "train_rate": float(ytr.mean()), "test_rate": float(yte.mean())})
    return outputs


def fit_surface(Xtr: np.ndarray, Xte: np.ndarray, rowste: list[dict], tracer: str, outdir: Path) -> dict:
    pca = PCA(n_components=2, random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    tw = float(trustworthiness(Xte, Zte, n_neighbors=min(10, len(Xte)-1)))

    ref = GaussianMixture(3, covariance_type="full", random_state=0).fit(Ztr).predict(Zte)
    aris = []
    rng = np.random.default_rng(33)
    for b in range(30):
        ii = rng.integers(0, len(Ztr), len(Ztr))
        try:
            lab = GaussianMixture(3, covariance_type="full", random_state=b+1).fit(Ztr[ii]).predict(Zte)
            aris.append(adjusted_rand_score(ref, lab))
        except Exception:
            pass

    surf = pd.DataFrame(rowste)
    surf["z1"] = Zte[:, 0]
    surf["z2"] = Zte[:, 1]
    surf["gmm3"] = ref
    surf.to_csv(outdir / f"pattern_surface_{tracer}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(Zte[:, 0], Zte[:, 1], c=surf["mean_count"], s=20, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"REAL DR11 pattern surface — {tracer}")
    fig.colorbar(sc, ax=ax, label="observed mean sources / fine cell")
    fig.tight_layout()
    fig.savefig(outdir / f"pattern_surface_{tracer}.png", dpi=160)
    plt.close(fig)

    return {
        "pca2_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "test_trustworthiness": tw,
        "gmm3_bootstrap_ari_median": float(np.median(aris)) if aris else float("nan"),
        "test_patches": int(len(Xte)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/dr11/pilot")
    ap.add_argument("--out", default="results/real_dr11/latest")
    args = ap.parse_args()
    datadir, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    prov = json.loads((datadir / "provenance.json").read_text())
    if prov.get("status") != "REAL_DR11":
        raise RuntimeError("Input provenance is not REAL_DR11")
    split = prov["field_split"]
    meta_by_name = {r["name"]: r for r in prov["regions"]}

    reconstruction_rows, motif_rows, summary = [], [], {"status": "REAL_DR11", "dataset": prov["dataset"], "tracers": {}}

    for tracer in ["all_primary", "extended_clean"]:
        Xparts, rows = {}, {}
        for field, meta in meta_by_name.items():
            df = verify_and_load(meta)
            grid = region_grid(df, meta, tracer)
            xp, rp = patchify(grid, field, tracer)
            Xparts[field], rows[field] = xp, rp

        def cat(names):
            return np.concatenate([Xparts[n] for n in names], axis=0)
        def catrows(names):
            return [r for n in names for r in rows[n]]

        Xtr_raw = cat(split["train"])
        Xva_raw = cat(split["validation"])
        Xte_raw = cat(split["test"])
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xva, Xte = scaler.transform(Xtr_raw), scaler.transform(Xva_raw), scaler.transform(Xte_raw)

        surf = fit_surface(Xtr, Xte, catrows(split["test"]), tracer, outdir)

        for mask in ["center25", "random25", "corner25", "stripe25"]:
            hid = mask_indices(mask)
            preds = reconstruction_predictions(Xtr, Xte, hid)
            y = Xte[:, hid]
            for method, pred in preds.items():
                mse, corr = score_reconstruction(y, pred)
                reconstruction_rows.append({
                    "tracer": tracer, "mask": mask, "method": method,
                    "hidden_fraction": float(len(hid)/(PATCH*PATCH)), "mse": mse, "corr": corr,
                    "n_test_patches": int(len(Xte)),
                })

        for m in motif_metrics(Xtr, Xte):
            motif_rows.append({"tracer": tracer, **m, "n_test_patches": int(len(Xte))})

        summary["tracers"][tracer] = {
            **surf,
            "train_fields": split["train"], "validation_fields": split["validation"], "test_fields": split["test"],
            "train_patches": int(len(Xtr)), "validation_patches": int(len(Xva)), "test_patches": int(len(Xte)),
        }

    rec = pd.DataFrame(reconstruction_rows)
    mot = pd.DataFrame(motif_rows)
    rec.to_csv(outdir / "reconstruction_metrics.csv", index=False)
    mot.to_csv(outdir / "motif_metrics.csv", index=False)

    # Compact comparison figure.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    center = rec[rec["mask"] == "center25"].copy()
    keys = [(t, m) for t in ["all_primary", "extended_clean"] for m in ["mean", "gaussian", "pca", "knn20"]]
    vals = [float(center[(center.tracer == t) & (center.method == m)].mse.iloc[0]) for t, m in keys]
    labels = [f"{t}\n{m}" for t, m in keys]
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels, rotation=45, ha="right")
    ax.set_ylabel("hidden standardized MSE")
    ax.set_title("REAL DR11: central 25% hole reconstruction")
    fig.tight_layout()
    fig.savefig(outdir / "center25_reconstruction.png", dpi=160)
    plt.close(fig)

    summary["reconstruction_best_center25"] = {}
    for tracer in ["all_primary", "extended_clean"]:
        s = center[center.tracer == tracer].sort_values("mse").iloc[0]
        summary["reconstruction_best_center25"][tracer] = {"method": str(s.method), "mse": float(s.mse), "corr": float(s.corr)}
    summary["motif_auc"] = {f"{r.tracer}:{r.motif}": float(r.auc) for r in mot.itertuples()}
    summary["input_total_rows"] = int(prov["total_rows"])
    summary["provenance_file"] = str(datadir / "provenance.json")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
