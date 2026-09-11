#!/usr/bin/env python3
"""REAL DR11 x DESI DR1 matched-extended redshift-locality gate.

This is a stricter follow-up to the negative/non-replicated all-source redshift
view.  DESI DR1 primary galaxy spectra are one-to-one angular matched to the
observed DR11 extended-source population within 1.5 arcsec.  The DR11 map is
therefore built only from the same broad tracer population represented in the
spectroscopic sample.  The null permutes observed redshifts within exact
(survey, program) groups while leaving every angular position fixed.

No simulated cosmology is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import binomtest, wilcoxon

from redshift_view_validate import (
    HALF, QUERY_RADIUS, ZEDGES, query, square_clip, count_grid_from_offsets,
    local_map, broad_map, corr,
)
from redshift_view_replication import (
    candidates as replication_candidates, sep, MIN_SEP_DEG,
    MIN_DR11_RADIAL, MIN_DESI_RADIAL,
)

TABLE_IMG = "ls_dr11.tractor_s"
TABLE_Z = "desi_dr1.zpix"
EXT_TYPES = {"REX", "EXP", "DEV", "SER"}
MORPH_ALLOWED = {"PSF", "REX", "EXP", "DEV", "SER", "DUP"}
MATCH_ARCSEC = 1.5
MIN_MATCHED = 60
MIN_BIN = 12
N_PERM = 300
SEED = 20260911
TARGET_REPLICATION = 24


def norm_text(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    return x.where(~x.eq("NAN"), "")


def qdf(sql: str) -> pd.DataFrame:
    x = query(sql)
    d = pd.read_csv(io.StringIO(x))
    d.columns = [str(c).lower() for c in d.columns]
    return d


def one_value(sql: str) -> int:
    return int(qdf(sql).iloc[0, 0])


def coverage_counts(ra: float, dec: float) -> tuple[int, int]:
    iz = (
        f"SELECT COUNT(*) AS n FROM {TABLE_Z} WHERE zwarn=0 AND zcat_primary=TRUE "
        "AND spectype='GALAXY' "
        f"AND z>={ZEDGES[0]:.4f} AND z<{ZEDGES[-1]:.4f} AND "
        f"q3c_radial_query(mean_fiber_ra,mean_fiber_dec,{ra:.8f},{dec:.8f},{QUERY_RADIUS:.8f})"
    )
    ii = (
        f"SELECT COUNT(*) AS n FROM {TABLE_IMG} WHERE brick_primary=1 AND "
        f"q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{QUERY_RADIUS:.8f})"
    )
    return one_value(iz), one_value(ii)


def select_replication(original: list[dict]) -> tuple[list[dict], list[dict]]:
    old = [(float(r["center_ra_deg"]), float(r["center_dec_deg"])) for r in original]
    accepted, trials = [], []
    for idx, (ra, dec) in enumerate(replication_candidates()):
        if len(accepted) >= TARGET_REPLICATION:
            break
        previous = old + [(r["ra"], r["dec"]) for r in accepted]
        if any(sep(ra, dec, a, b) < MIN_SEP_DEG for a, b in previous):
            trials.append({"candidate_index": idx, "ra": ra, "dec": dec, "status": "separation_reject"})
            continue
        nz, ni = coverage_counts(ra, dec)
        if nz < MIN_DESI_RADIAL:
            trials.append({"candidate_index": idx, "ra": ra, "dec": dec, "status": "desi_coverage_reject", "n_desi_radial": nz})
            continue
        if ni < MIN_DR11_RADIAL:
            trials.append({"candidate_index": idx, "ra": ra, "dec": dec, "status": "dr11_coverage_reject", "n_desi_radial": nz, "n_dr11_radial": ni})
            continue
        name = f"r{len(accepted):02d}_ra{int(ra):03d}_{'p' if dec >= 0 else 'm'}{abs(int(dec)):02d}"
        rec = {"name": name, "ra": float(ra), "dec": float(dec), "n_desi_radial": nz, "n_dr11_radial": ni}
        accepted.append(rec)
        trials.append({"candidate_index": idx, **rec, "status": "accepted"})
        print(f"[matched-z-select] accept {len(accepted):02d}/{TARGET_REPLICATION} {name} DESI={nz} DR11={ni}", flush=True)
    if len(accepted) != TARGET_REPLICATION:
        raise RuntimeError(f"only {len(accepted)} independent replication centers selected")
    return accepted, trials


def acquire_matched(ra0: float, dec0: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    isql = (
        "SELECT ra, dec, type, ref_cat FROM ls_dr11.tractor_s WHERE brick_primary=1 AND "
        f"q3c_radial_query(ra,dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS:.8f})"
    )
    im = qdf(isql)
    im = square_clip(im, ra0, dec0, "ra", "dec")
    dl_type = norm_text(im["type"])
    dl_ref = norm_text(im["ref_cat"])
    nonblank_ref = dl_ref[dl_ref.ne("")]
    bad = sorted(set(nonblank_ref) - MORPH_ALLOWED)
    ref_morph_frac = float(nonblank_ref.isin(MORPH_ALLOWED).mean()) if len(nonblank_ref) else 0.0
    nonblank_type = dl_type[dl_type.ne("")]
    type_morph_frac = float(nonblank_type.isin(MORPH_ALLOWED).mean()) if len(nonblank_type) else 0.0
    if len(nonblank_ref) == 0 or ref_morph_frac < 0.995 or bad:
        raise RuntimeError(f"Data Lab morphology-swap audit failed at {ra0},{dec0}: ref fraction={ref_morph_frac} bad={bad[:10]}")
    if type_morph_frac > 0.50:
        raise RuntimeError(f"Data Lab mapping changed at {ra0},{dec0}; type morphology fraction={type_morph_frac}")
    im["morph"] = dl_ref
    im = im[im.morph.isin(EXT_TYPES)].copy().reset_index(drop=True)

    zsql = (
        "SELECT mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, survey, program "
        f"FROM {TABLE_Z} WHERE zwarn=0 AND zcat_primary=TRUE AND spectype='GALAXY' "
        f"AND z>={ZEDGES[0]:.4f} AND z<{ZEDGES[-1]:.4f} AND "
        f"q3c_radial_query(mean_fiber_ra,mean_fiber_dec,{ra0:.8f},{dec0:.8f},{QUERY_RADIUS:.8f})"
    )
    dz = qdf(zsql)
    dz = square_clip(dz, ra0, dec0, "ra", "dec")
    dz["z"] = pd.to_numeric(dz.z, errors="coerce")
    dz = dz[np.isfinite(dz.z)].reset_index(drop=True)

    if len(im) == 0 or len(dz) == 0:
        return im.iloc[:0].copy(), dz.iloc[:0].copy(), {"imaging_sql": isql, "desi_sql": zsql, "n_extended": len(im), "n_desi": len(dz)}

    c = np.cos(np.deg2rad(dec0))
    ix = np.column_stack([im["_dra"].to_numpy(float) * c * 3600.0, im["_ddec"].to_numpy(float) * 3600.0])
    zx = np.column_stack([dz["_dra"].to_numpy(float) * c * 3600.0, dz["_ddec"].to_numpy(float) * 3600.0])
    tree = cKDTree(ix)
    dist, ind = tree.query(zx, k=1)
    cand = pd.DataFrame({"desi_index": np.arange(len(dz)), "img_index": ind, "sep_arcsec": dist})
    cand = cand[cand.sep_arcsec <= MATCH_ARCSEC].sort_values(["sep_arcsec", "desi_index"], kind="mergesort")
    cand = cand.drop_duplicates("img_index", keep="first").sort_values("desi_index", kind="mergesort").reset_index(drop=True)
    mi = im.iloc[cand.img_index.to_numpy()].copy().reset_index(drop=True)
    mz = dz.iloc[cand.desi_index.to_numpy()].copy().reset_index(drop=True)
    mz["match_sep_arcsec"] = cand.sep_arcsec.to_numpy(float)
    key = mz[["ra", "dec", "z", "survey", "program", "match_sep_arcsec"]].to_csv(index=False, lineterminator="\n").encode()
    prov = {
        "imaging_sql": isql,
        "desi_sql": zsql,
        "n_extended_square": int(len(im)),
        "n_desi_square": int(len(dz)),
        "n_matched": int(len(mz)),
        "desi_match_fraction": float(len(mz) / len(dz)) if len(dz) else 0.0,
        "median_match_sep_arcsec": float(np.median(mz.match_sep_arcsec)) if len(mz) else None,
        "canonical_matched_sha256": hashlib.sha256(key).hexdigest(),
        "datalab_swap_audit": {
            "raw_ref_cat_morphology_fraction": ref_morph_frac,
            "raw_type_morphology_fraction": type_morph_frac,
            "raw_ref_cat_top_counts": {str(k): int(v) for k, v in dl_ref.value_counts().head(10).items()},
            "raw_type_top_counts": {str(k): int(v) for k, v in dl_type.value_counts().head(10).items()},
        },
    }
    return mi, mz, prov


def stat(img_local, img_broad, dra, ddec, z):
    rows = []
    for k in range(len(ZEDGES) - 1):
        lo, hi = ZEDGES[k], ZEDGES[k + 1]
        m = (z >= lo) & (z < hi)
        n = int(m.sum())
        if n < MIN_BIN:
            continue
        g = count_grid_from_offsets(dra[m], ddec[m])
        rows.append((n, corr(img_local, local_map(g)), corr(img_broad, broad_map(g))))
    if len(rows) < 2:
        return None
    w = np.asarray([r[0] for r in rows], float)
    lc = np.asarray([r[1] for r in rows], float)
    bc = np.asarray([r[2] for r in rows], float)
    gl, gb = np.isfinite(lc), np.isfinite(bc)
    if gl.sum() < 2:
        return None
    return {
        "local": float(np.average(lc[gl], weights=w[gl])),
        "broad": float(np.average(bc[gb], weights=w[gb])) if gb.any() else float("nan"),
        "max": float(np.nanmax(lc)),
        "n_bins": int(len(rows)),
    }


def shuffle_stratified(z, survey, program, rng):
    out = np.asarray(z, float).copy()
    key = np.char.add(np.char.add(np.asarray(survey, str).astype("U32"), "|"), np.asarray(program, str).astype("U32"))
    for k in np.unique(key):
        ii = np.flatnonzero(key == k)
        if len(ii) > 1:
            out[ii] = rng.permutation(out[ii])
    return out


def paired(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return {"n_fields": 0}
    pos = int((x > 0).sum())
    try:
        wp = float(wilcoxon(x, alternative="greater").pvalue)
    except Exception:
        wp = float("nan")
    return {
        "n_fields": n,
        "positive_fields": pos,
        "sign_test_one_sided_p": float(binomtest(pos, n, 0.5, alternative="greater").pvalue),
        "wilcoxon_one_sided_p": wp,
        "mean_difference": float(np.mean(x)),
        "median_difference": float(np.median(x)),
    }


def eval_set(centers: list[dict], label: str, out: Path):
    rng = np.random.default_rng(SEED + (0 if label == "primary" else 10000))
    rows, provenance, nulls = [], [], {}
    for j, c in enumerate(centers):
        name, ra0, dec0 = c["name"], float(c["ra"]), float(c["dec"])
        mi, mz, prov = acquire_matched(ra0, dec0)
        prov.update({"set": label, "field": name, "center_ra_deg": ra0, "center_dec_deg": dec0})
        provenance.append(prov)
        if len(mz) < MIN_MATCHED:
            rows.append({"set": label, "field": name, "status": "low_match_n", "n_matched": int(len(mz))})
            continue
        ic = count_grid_from_offsets(mi["_dra"].to_numpy(float), mi["_ddec"].to_numpy(float))
        il, ib = local_map(ic), broad_map(ic)
        dra, ddec, z = mz["_dra"].to_numpy(float), mz["_ddec"].to_numpy(float), mz.z.to_numpy(float)
        survey = mz.survey.fillna("NA").astype(str).to_numpy()
        program = mz.program.fillna("NA").astype(str).to_numpy()
        actual = stat(il, ib, dra, ddec, z)
        if actual is None:
            rows.append({"set": label, "field": name, "status": "insufficient_bins", "n_matched": int(len(mz))})
            continue
        nl, nb, nm = [], [], []
        for _ in range(N_PERM):
            st = stat(il, ib, dra, ddec, shuffle_stratified(z, survey, program, rng))
            if st is not None:
                nl.append(st["local"]); nb.append(st["broad"]); nm.append(st["max"])
        if len(nl) < N_PERM // 2:
            rows.append({"set": label, "field": name, "status": "null_failure", "n_matched": int(len(mz))})
            continue
        nl, nb, nm = np.asarray(nl), np.asarray(nb), np.asarray(nm)
        nulls[name] = nl
        rows.append({
            "set": label, "field": name, "status": "accepted", "n_matched": int(len(mz)),
            "n_groups": int(len(set(zip(survey, program)))), "n_bins": actual["n_bins"],
            "actual_local": actual["local"], "null_local": float(nl.mean()), "delta_local": actual["local"] - float(nl.mean()),
            "actual_broad": actual["broad"], "null_broad": float(nb.mean()), "delta_broad": actual["broad"] - float(nb.mean()),
            "actual_max": actual["max"], "null_max": float(nm.mean()), "delta_max": actual["max"] - float(nm.mean()),
            "field_perm_p": float((1 + (nl >= actual["local"]).sum()) / (len(nl) + 1)),
            "match_fraction": prov["desi_match_fraction"], "median_sep_arcsec": prov["median_match_sep_arcsec"],
        })
        print(f"[matched-z] {label} {j+1:02d}/{len(centers)} {name} matched={len(mz)} delta={rows[-1]['delta_local']:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{label}_field_metrics.csv", index=False)
    (out / f"{label}_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    a = df[df.status.eq("accepted")].copy()
    B = min((len(nulls[f]) for f in a.field), default=0)
    if B and len(a):
        gnull = np.asarray([np.mean([nulls[f][b] for f in a.field]) for b in range(B)])
        ga = float(a.actual_local.mean())
        gp = float((1 + (gnull >= ga).sum()) / (B + 1))
    else:
        ga = gp = float("nan")
    return {
        "set": label,
        "accepted_fields": int(len(a)),
        "local_delta": paired(a.delta_local.to_numpy(float)),
        "broad_delta": paired(a.delta_broad.to_numpy(float)),
        "max_delta": paired(a.delta_max.to_numpy(float)),
        "actual_local_median": float(a.actual_local.median()) if len(a) else float("nan"),
        "null_local_median": float(a.null_local.median()) if len(a) else float("nan"),
        "median_match_fraction": float(a.match_fraction.median()) if len(a) else float("nan"),
        "global_actual_mean": ga,
        "global_perm_p": gp,
    }, a


def decision(primary, replication, combined, combined_global_p):
    if combined["local_delta"].get("n_fields", 0) < 20 or replication["accepted_fields"] < 8:
        return "UNCERTAIN_MATCHED_REDSHIFT_COVERAGE"
    pm = float(primary["local_delta"]["median_difference"])
    rm = float(replication["local_delta"]["median_difference"])
    c = combined["local_delta"]
    if (
        pm > 0 and rm > 0 and float(c["median_difference"]) >= 0.01
        and float(c["sign_test_one_sided_p"]) < 0.01
        and float(c["wilcoxon_one_sided_p"]) < 0.01
        and float(combined_global_p) < 0.01
    ):
        return "PASS_MATCHED_TRACER_REDSHIFT_LOCALIZATION"
    if (
        rm <= 0 or float(c["median_difference"]) <= 0
        or float(c["sign_test_one_sided_p"]) >= 0.10
        or float(c["wilcoxon_one_sided_p"]) >= 0.10
        or float(combined_global_p) >= 0.10
    ):
        return "FAIL_MATCHED_TRACER_REDSHIFT_LOCALIZATION"
    return "UNCERTAIN_MATCHED_TRACER_REDSHIFT_LOCALIZATION"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", default="data/real/dr11/expanded48/provenance.json")
    ap.add_argument("--out", default="results/real_dr11/matched_extended_redshift")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = json.loads(Path(args.centers).read_text())
    if p.get("status") != "REAL_DR11" or len(p.get("regions", [])) != 48:
        raise RuntimeError("fixed REAL_DR11 expanded48 centers required")
    primary_centers = [{"name": r["name"], "ra": r["center_ra_deg"], "dec": r["center_dec_deg"]} for r in p["regions"]]
    repl_centers, selection_trials = select_replication(p["regions"])
    (out / "replication_selection.json").write_text(json.dumps(selection_trials, indent=2, sort_keys=True) + "\n")

    ps, pdf = eval_set(primary_centers, "primary", out)
    rs, rdf = eval_set(repl_centers, "replication", out)
    comb = pd.concat([pdf, rdf], ignore_index=True)
    cd = paired(comb.delta_local.to_numpy(float))
    cbd = paired(comb.delta_broad.to_numpy(float))
    cmd = paired(comb.delta_max.to_numpy(float))
    # Combine set-level global permutation p-values conservatively by requiring
    # both directions and using the larger p as the decision quantity.
    combined_global_p = float(max(ps["global_perm_p"], rs["global_perm_p"]))
    combined = {"local_delta": cd, "broad_delta": cbd, "max_delta": cmd, "n_fields": int(len(comb))}
    dec = decision(ps, rs, combined, combined_global_p)
    summary = {
        "status": "REAL_DR11_DESI_DR1_MATCHED_EXTENDED_REDSHIFT",
        "decision": dec,
        "match_radius_arcsec": MATCH_ARCSEC,
        "tracer": "DESI DR1 primary ZWARN=0 SPECTYPE=GALAXY matched one-to-one to DR11 extended REX/EXP/DEV/SER",
        "datalab_morphology_mapping": "per-field audited observed DR11 Data Lab mapping: ref_cat contains morphology values while type does not",
        "z_edges": ZEDGES.tolist(),
        "null": "within-field redshift permutation separately within exact (survey,program); all angular positions fixed",
        "predeclared_decision": {
            "coverage": ">=20 combined accepted fields and >=8 independent replication fields",
            "pass": "primary and replication median delta >0; combined median delta >=0.01; combined one-sided sign and Wilcoxon p<0.01; both set-level global permutation p<0.01",
            "fail": "replication median<=0, or combined median<=0, or combined sign/Wilcoxon p>=0.10, or either set-level global permutation p>=0.10",
            "otherwise": "UNCERTAIN",
        },
        "primary": ps,
        "replication": rs,
        "combined": combined,
        "decision_global_p_max": combined_global_p,
        "interpretation_guardrail": "PASS would establish redshift-localized clustering in a matched observed galaxy tracer under the tested null; it would not by itself identify matter density, gravity, or higher-order cosmic grammar.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
