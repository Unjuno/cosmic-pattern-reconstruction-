#!/usr/bin/env python3
"""Deterministic point-level DR11 random-catalog prefix-sampling pilot.

The official DR11 documentation states that rows inside each ``randoms-1-X.fits``
file are randomly ordered, so reading only the first N rows retains randomness.
This pilot exploits that property to test a bounded acquisition path before any
large point-level selection-function analysis.

No cosmological inference is made here. The primary question is operational:
can a small, deterministic prefix from all 20 official point-random catalogs
recover point-level survey metadata in essentially all 48 provenance-fixed
REAL_DR11 fields without downloading the full ~801 GB random directory?

The implementation never writes the remote FITS files. It reads the FITS headers
with HTTP Range requests, determines the binary-table row layout, then downloads
only a fixed prefix of table rows from each file. The sampled rows are filtered
to the 48 fixed 0.5 deg x 0.5 deg RA/Dec boxes and a compact provenance artifact
is written locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/randoms"
N_FILES = 20
DOCUMENTED_DENSITY_PER_SQDEG = 2500.0
HEADER_PROBE_BYTES = 262_144
BLOCK = 2880
CARD = 80
CORE_COLUMNS = [
    "RA", "DEC", "MASKBITS", "NOBS_G", "NOBS_R", "NOBS_Z",
    "PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_Z", "EBV",
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wrap_deg(x: np.ndarray | float) -> np.ndarray:
    return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0


def _fetch_range(url: str, start: int, end: int, timeout: int = 120) -> tuple[bytes, dict]:
    if end < start:
        raise ValueError((start, end))
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "cosmic-pattern-reconstruction-point-random-pilot/1.0",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = int(getattr(r, "status", 0) or 0)
        cr = r.headers.get("Content-Range")
        cl = r.headers.get("Content-Length")
        if status != 206 or not cr:
            raise RuntimeError(
                f"server did not honor bounded Range request for {url}: "
                f"status={status}, Content-Range={cr}, Content-Length={cl}"
            )
        expected = end - start + 1
        if cl is not None and int(cl) != expected:
            raise RuntimeError(f"unexpected Content-Length {cl} != {expected} for {url}")
        data = r.read(expected + 1)
        if len(data) != expected:
            raise RuntimeError(f"range length mismatch {len(data)} != {expected} for {url}")
        meta = {
            "status": status,
            "content_range": cr,
            "content_length": int(cl) if cl is not None else None,
            "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
            "accept_ranges": r.headers.get("Accept-Ranges"),
        }
    return data, meta


def _split_value_comment(s: str) -> str:
    s = s.rstrip()
    if not s:
        return ""
    if s.lstrip().startswith("'"):
        q = s.find("'")
        q = s.find("'", q + 1)
        while q >= 0 and q + 1 < len(s) and s[q + 1] == "'":
            q = s.find("'", q + 2)
        if q >= 0:
            return s[: q + 1].strip()
    return s.split("/", 1)[0].strip()


def _parse_value(v: str):
    v = v.strip()
    if not v:
        return None
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1].replace("''", "'").strip()
    if v == "T":
        return True
    if v == "F":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v.replace("D", "E"))
    except ValueError:
        return v


def _parse_header(blob: bytes, offset: int) -> tuple[dict, int]:
    hdr: dict[str, object] = {}
    pos = offset
    end_card_end = None
    while pos + CARD <= len(blob):
        card = blob[pos : pos + CARD].decode("ascii", errors="strict")
        key = card[:8].strip()
        if key == "END":
            end_card_end = pos + CARD
            break
        if card[8:10] == "= " and key:
            hdr[key] = _parse_value(_split_value_comment(card[10:]))
        pos += CARD
    if end_card_end is None:
        raise RuntimeError(f"FITS END card not found from byte offset {offset}")
    header_bytes = int(math.ceil((end_card_end - offset) / BLOCK) * BLOCK)
    return hdr, offset + header_bytes


def _hdu_data_bytes(h: dict) -> int:
    naxis = int(h.get("NAXIS", 0) or 0)
    if str(h.get("XTENSION", "")).upper() == "BINTABLE":
        return int(h["NAXIS1"]) * int(h["NAXIS2"]) + int(h.get("PCOUNT", 0) or 0)
    if naxis <= 0:
        return 0
    bitpix = abs(int(h.get("BITPIX", 8) or 8))
    n = 1
    for i in range(1, naxis + 1):
        n *= int(h[f"NAXIS{i}"])
    n *= max(1, int(h.get("GCOUNT", 1) or 1))
    n += int(h.get("PCOUNT", 0) or 0)
    return n * bitpix // 8


def _next_hdu_offset(header_end: int, data_bytes: int) -> int:
    return header_end + int(math.ceil(data_bytes / BLOCK) * BLOCK) if data_bytes else header_end


def _inspect_remote_fits(url: str) -> tuple[dict, dict]:
    probe, http = _fetch_range(url, 0, HEADER_PROBE_BYTES - 1)
    primary, p_end = _parse_header(probe, 0)
    ext_off = _next_hdu_offset(p_end, _hdu_data_bytes(primary))
    ext, data_off = _parse_header(probe, ext_off)
    if str(ext.get("XTENSION", "")).upper() != "BINTABLE":
        raise RuntimeError(f"first extension is not BINTABLE for {url}: {ext.get('XTENSION')}")
    if data_off > len(probe):
        raise RuntimeError(f"header probe too small for {url}: data_off={data_off}")
    info = {
        "url": url,
        "data_offset": int(data_off),
        "row_bytes": int(ext["NAXIS1"]),
        "rows": int(ext["NAXIS2"]),
        "tfields": int(ext["TFIELDS"]),
        "header_prefix_sha256": sha256(probe[:data_off]),
        "http_header_probe": http,
    }
    return ext, info


def _tform_width(tform: str) -> tuple[int, int, str]:
    s = str(tform).strip().upper()
    m = re.match(r"^(\d*)([LXBIJKAEDCMPQ])(?:\([^)]*\))?$", s)
    if not m:
        raise RuntimeError(f"unsupported FITS TFORM={tform!r}")
    repeat = int(m.group(1) or 1)
    code = m.group(2)
    unit = {
        "L": 1, "X": 0, "B": 1, "I": 2, "J": 4, "K": 8,
        "A": 1, "E": 4, "D": 8, "C": 8, "M": 16, "P": 8, "Q": 16,
    }[code]
    width = int(math.ceil(repeat / 8)) if code == "X" else repeat * unit
    return width, repeat, code


def _field_specs(h: dict) -> tuple[dict[str, dict], int]:
    specs: dict[str, dict] = {}
    offset = 0
    for i in range(1, int(h["TFIELDS"]) + 1):
        name = str(h[f"TTYPE{i}"]).strip().upper()
        tform = str(h[f"TFORM{i}"]).strip().upper()
        width, repeat, code = _tform_width(tform)
        specs[name] = {
            "offset": offset,
            "width": width,
            "repeat": repeat,
            "code": code,
            "tform": tform,
        }
        offset += width
    if offset != int(h["NAXIS1"]):
        raise RuntimeError(f"TFORM widths sum to {offset}, NAXIS1={h['NAXIS1']}")
    return specs, offset


def _extract_scalar(data: bytes, row_bytes: int, spec: dict) -> np.ndarray:
    code = spec["code"]
    repeat = int(spec["repeat"])
    if repeat != 1 and code != "A":
        raise RuntimeError(f"pilot only extracts scalar numeric fields, got {spec}")
    n = len(data) // row_bytes
    off = int(spec["offset"])
    if code == "A":
        dt = np.dtype(f"S{repeat}")
    else:
        dt = {
            "L": np.dtype("S1"),
            "B": np.dtype("u1"),
            "I": np.dtype(">i2"),
            "J": np.dtype(">i4"),
            "K": np.dtype(">i8"),
            "E": np.dtype(">f4"),
            "D": np.dtype(">f8"),
        }.get(code)
        if dt is None:
            raise RuntimeError(f"unsupported extracted FITS code {code}")
    return np.ndarray((n,), dtype=dt, buffer=data, offset=off, strides=(row_bytes,)).copy()


def _load_regions(path: Path) -> list[dict]:
    p = json.loads(path.read_text())
    regs = p.get("regions", [])
    if p.get("status") != "REAL_DR11" or len(regs) != 48:
        raise RuntimeError("48-field REAL_DR11 provenance required")
    out = []
    for r in regs:
        out.append({
            "field": r["name"],
            "ra0": float(r["center_ra_deg"]),
            "dec0": float(r["center_dec_deg"]),
            "width_deg": float(r.get("box_width_deg", 0.5)),
            "source_rows": int(r["rows"]),
        })
    return out


def _in_box(ra: np.ndarray, dec: np.ndarray, r: dict) -> np.ndarray:
    half = r["width_deg"] / 2.0
    return (np.abs(wrap_deg(ra - r["ra0"])) <= half) & (dec >= r["dec0"] - half) & (dec < r["dec0"] + half)


def _sky_box_area_sqdeg(r: dict) -> float:
    # Small-angle area of an RA/Dec rectangle. Degree is an angular unit; the
    # cos(dec) factor converts RA coordinate width to physical angular width.
    return float(r["width_deg"] ** 2 * math.cos(math.radians(r["dec0"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", default="data/real/dr11/expanded48/provenance.json")
    ap.add_argument("--fraction", type=float, default=0.002)
    ap.add_argument("--files", type=int, default=N_FILES)
    ap.add_argument("--out", default="results/real_dr11/point_random_prefix_pilot")
    args = ap.parse_args()
    if not (0 < args.fraction <= 0.02):
        raise RuntimeError("pilot fraction must be in (0, 0.02]")
    if not (1 <= args.files <= N_FILES):
        raise RuntimeError("files must be 1..20")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions = _load_regions(Path(args.provenance))

    target_rows: list[dict] = []
    file_meta: list[dict] = []
    reference_schema = None
    effective_density = 0.0
    total_remote_bytes_sampled = 0

    for fi in range(args.files):
        url = f"{BASE}/randoms-1-{fi}.fits"
        hdr, info = _inspect_remote_fits(url)
        specs, row_bytes = _field_specs(hdr)
        missing = sorted(set(CORE_COLUMNS) - set(specs))
        if missing:
            raise RuntimeError(f"file {fi} missing required point-random columns: {missing}; available={sorted(specs)}")
        schema = {k: specs[k]["tform"] for k in sorted(specs)}
        if reference_schema is None:
            reference_schema = schema
        elif schema != reference_schema:
            raise RuntimeError(f"point-random schema differs in file {fi}")

        n_total = int(info["rows"])
        n_take = max(1, int(math.floor(n_total * args.fraction)))
        effective_fraction = n_take / n_total
        effective_density += DOCUMENTED_DENSITY_PER_SQDEG * effective_fraction
        start = int(info["data_offset"])
        end = start + n_take * row_bytes - 1
        raw, http_data = _fetch_range(url, start, end, timeout=240)
        total_remote_bytes_sampled += len(raw)
        if len(raw) % row_bytes:
            raise RuntimeError(f"sample bytes not divisible by row size for file {fi}")

        cols = {c: _extract_scalar(raw, row_bytes, specs[c]) for c in CORE_COLUMNS}
        ra = cols["RA"].astype(float)
        dec = cols["DEC"].astype(float)
        if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)):
            raise RuntimeError(f"non-finite RA/DEC in file {fi} prefix")
        if np.any((ra < 0) | (ra >= 360) | (dec < -90) | (dec > 90)):
            raise RuntimeError(f"invalid RA/DEC range in file {fi} prefix")

        hit_total = 0
        row_index = np.arange(n_take, dtype=np.int64)
        for r in regions:
            m = _in_box(ra, dec, r)
            idx = np.flatnonzero(m)
            hit_total += len(idx)
            for j in idx:
                target_rows.append({
                    "field": r["field"],
                    "file_index": fi,
                    "prefix_row_index": int(row_index[j]),
                    "ra": float(ra[j]),
                    "dec": float(dec[j]),
                    "maskbits": int(cols["MASKBITS"][j]),
                    "nobs_g": int(cols["NOBS_G"][j]),
                    "nobs_r": int(cols["NOBS_R"][j]),
                    "nobs_z": int(cols["NOBS_Z"][j]),
                    "psfdepth_g": float(cols["PSFDEPTH_G"][j]),
                    "psfdepth_r": float(cols["PSFDEPTH_R"][j]),
                    "psfdepth_z": float(cols["PSFDEPTH_Z"][j]),
                    "ebv": float(cols["EBV"][j]),
                })
        info.update({
            "file_index": fi,
            "schema": schema,
            "rows_sampled": n_take,
            "sample_fraction_effective": effective_fraction,
            "sample_data_bytes": len(raw),
            "sample_data_sha256": sha256(raw),
            "http_data": http_data,
            "target_hits": int(hit_total),
        })
        file_meta.append(info)
        print(
            f"[point-random-prefix] file={fi:02d} rows={n_take}/{n_total} "
            f"bytes={len(raw)} target_hits={hit_total}", flush=True
        )

    points = pd.DataFrame(target_rows)
    if len(points):
        points = points.sort_values(["field", "file_index", "prefix_row_index"], kind="mergesort").reset_index(drop=True)
    points.to_csv(out / "target_points.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})

    rows = []
    for r in regions:
        d = points[points.field == r["field"]] if len(points) else pd.DataFrame()
        area = _sky_box_area_sqdeg(r)
        expected = effective_density * area
        n = int(len(d))
        rows.append({
            "field": r["field"],
            "center_ra_deg": r["ra0"],
            "center_dec_deg": r["dec0"],
            "box_width_deg": r["width_deg"],
            "sky_area_sqdeg_approx": area,
            "source_rows": r["source_rows"],
            "random_points": n,
            "expected_random_points_nominal": expected,
            "observed_over_expected": n / expected if expected > 0 else np.nan,
            "maskbits_zero_fraction": float((d.maskbits == 0).mean()) if n else np.nan,
            "all_grz_nobs_positive_fraction": float(((d.nobs_g > 0) & (d.nobs_r > 0) & (d.nobs_z > 0)).mean()) if n else np.nan,
            "median_psfdepth_r": float(d.psfdepth_r.median()) if n else np.nan,
            "median_ebv": float(d.ebv.median()) if n else np.nan,
        })
    fields = pd.DataFrame(rows)
    fields.to_csv(out / "field_counts.csv", index=False)

    with_points = int((fields.random_points > 0).sum())
    ge5 = int((fields.random_points >= 5).sum())
    med_n = float(fields.random_points.median())
    med_ratio = float(fields.observed_over_expected.median())
    # Acquisition feasibility only. The thresholds are fixed before the pilot
    # result is inspected and are intentionally not a science detection rule.
    if with_points >= 46 and ge5 >= 40 and med_n >= 8 and 0.45 <= med_ratio <= 1.55:
        decision = "PASS"
    elif with_points < 40 or med_n < 4 or not (0.20 <= med_ratio <= 2.50):
        decision = "FAIL"
    else:
        decision = "UNCERTAIN"

    summary = {
        "status": "REAL_DR11_POINT_RANDOM_PREFIX_PILOT",
        "decision": decision,
        "science_result": None,
        "purpose": "bounded acquisition/provenance feasibility for point-level official random selection-function work",
        "documented_random_order_property": "rows within each official randoms-1-X file are randomly ordered; first N rows retain randomness",
        "files_sampled": int(args.files),
        "requested_fraction_per_file": float(args.fraction),
        "effective_density_per_sqdeg_nominal": float(effective_density),
        "remote_sample_bytes": int(total_remote_bytes_sampled),
        "target_point_rows": int(len(points)),
        "fields_with_points": with_points,
        "fields_with_at_least_5_points": ge5,
        "median_points_per_field": med_n,
        "median_observed_over_nominal_expected": med_ratio,
        "H": "deterministic prefixes from all official point-random files recover usable point-level survey metadata across nearly all 48 fixed REAL_DR11 fields",
        "T": f"first {args.fraction:.4%} of rows from each of {args.files} randomly ordered official random files; 48 fixed 0.5-degree RA/Dec boxes",
        "D": "PASS iff >=46 fields have >=1 point, >=40 have >=5, median points/field>=8, and median observed/nominal expected is in [0.45,1.55]; FAIL if <40 fields have points, median<4, or median ratio outside [0.20,2.50]; otherwise UNCERTAIN",
        "C": "prefix sampling is too sparse or operationally biased to support a bounded point-level selection pilot",
        "U": "finite prefix sampling noise, footprint edges/PHOTSYS resolution, RA/Dec box area approximation, remote HTTP behavior; no cosmological inference",
        "core_columns": CORE_COLUMNS,
        "schema": reference_schema,
        "files": file_meta,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files" and k != "schema"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
