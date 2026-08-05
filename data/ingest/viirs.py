"""Download FIRMS VIIRS hotspots + generate k-of-N persistence labels.

Each label at anchor date t answers: "will >=k hotspots appear in the 7-day
window (t+1 .. t+window]?"  The Dec-31 boundary uses Jan hotspots from the
following year, handled by **--years START END**.

Usage:
  uv run --python 3.12 python data/ingest/viirs.py --years 2019 2023
  uv run --python 3.12 python data/ingest/viirs.py --year 2019 --keep-raw
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("viirs")

URL = "https://firms.modaps.eosdis.nasa.gov/data/country/zips/viirs-snpp_{year}_all_countries.zip"
COLUMNS = [
    "latitude", "longitude", "bright_ti4", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument", "confidence",
    "version", "bright_ti5", "frp", "daynight", "type",
]
KEEP = ["lon", "lat", "date", "confidence", "frp"]

# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #


def _download(url, dest, timeout=600, retries=3):
    if dest.exists():
        logger.info("skip download: %s exists", dest.name)
        return dest
    for attempt in range(1, retries + 1):
        try:
            logger.info("downloading %s (attempt %d)", dest.name, attempt)
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                dl = 0
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
                    dl += len(chunk)
                    if total:
                        logger.info("  %d / %d MB (%.0f%%)", dl // 2**20, total // 2**20, 100 * dl / total)
            logger.info("download complete: %s (%d MB)", dest.name, dl // 2**20)
            return dest
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning("attempt %d failed: %s", attempt, exc)
            time.sleep(5 * attempt)

# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def _spatial_join(detections, grid):
    logger.info("spatial join: %d detections -> grid", len(detections))
    riau_bbox = (
        grid["lon"].min(), grid["lat"].min(),
        grid["lon"].max(), grid["lat"].max(),
    )
    xmin, ymin, xmax, ymax = riau_bbox
    det = detections[
        (detections["lon"] >= xmin) & (detections["lon"] <= xmax) &
        (detections["lat"] >= ymin) & (detections["lat"] <= ymax)
    ].copy()
    logger.info("  bbox filter: %d -> %d", len(detections), len(det))

    lats, lons = det["lat"].to_numpy(), det["lon"].to_numpy()
    n_det = len(det)
    cell_idx = np.full(n_det, -1, dtype=np.int64)
    min_lat, max_lat = grid["lat"].min(), grid["lat"].max()
    min_lon, max_lon = grid["lon"].min(), grid["lon"].max()
    lat_ok = (lats >= min_lat) & (lats <= max_lat)
    lon_ok = (lons >= min_lon) & (lons <= max_lon)
    ok = lat_ok & lon_ok
    logger.info("  in-bbox: %d / %d (dropping %d out-of-bbox)", ok.sum(), n_det, (~ok).sum())

    cells = grid[["cell_idx", "lon", "lat", "row", "col"]].to_numpy()
    for i in np.where(ok)[0]:
        dist = (cells[:, 1] - lons[i]) ** 2 + (cells[:, 2] - lats[i]) ** 2
        cell_idx[i] = int(cells[dist.argmin(), 0])

    det["cell_idx"] = cell_idx
    det = det[det["cell_idx"] >= 0]
    hits = det.groupby(["cell_idx", "date"]).size().reset_index(name="count")
    riau = grid[grid["is_riau"]].reset_index(drop=True)
    hits = hits[hits["cell_idx"].isin(riau["cell_idx"])]
    return hits


def _build_labels(hits, all_dates, riau_cells, k=2, window=7):
    """Binary label at each anchor date t: 1 iff >=k hits in (t+1 .. t+window].

    ``hits`` may span into the following year (for Dec-boundary windows).
    """
    logger.info("building labels (k=%d, window=%d days)", k, window)
    hits = hits.copy()
    hits["date"] = pd.to_datetime(hits["date"])
    all_dates = pd.to_datetime(all_dates)

    # Build the complete cell × date matrix for a padded range so the
    # (t+1 .. t+window] filter includes early-January hits from the next year.
    pad_end = all_dates[-1] + pd.Timedelta(days=window)
    padded_dates = pd.date_range(all_dates[0], pad_end)
    anchor = pd.DataFrame([
        (c, str(t.date()))
        for c in riau_cells for t in padded_dates
    ], columns=["cell_idx", "date"])
    anchor["date"] = pd.to_datetime(anchor["date"])
    merged = anchor.merge(hits, on=["cell_idx", "date"], how="left")
    merged["count"] = merged["count"].fillna(0).astype(int)

    labels = []
    for t in all_dates:
        fut = merged[
            (merged["date"] > str(t.date())) &
            (merged["date"] <= str((t + pd.Timedelta(days=window)).date()))
        ]
        fut_hits = fut.groupby("cell_idx")["count"].sum().reset_index()
        fut_hits["label"] = (fut_hits["count"] >= k).astype(int)
        fut_hits["date"] = str(t.date())
        labels.append(fut_hits[["cell_idx", "date", "label"]])
    return pd.concat(labels, ignore_index=True)


def _stats(hits, labels, year):
    det = hits["count"].sum()
    cells = hits["cell_idx"].nunique()
    pos = labels["fire_label"].sum()
    neg = len(labels) - pos
    logger.info("--- VIIRS %d stats ---", year)
    logger.info("  hotspot detections: %d", det)
    logger.info("  affected cells: %d", cells)
    logger.info("  label samples: %d (%d positive, %.2f%%)", len(labels), pos, 100 * pos / len(labels))

# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #


def _load_year(year, cache_dir, grid, keep_raw=False):
    """Download + parse FIRMS CSV for one year, return (hits_df, raw_df)."""
    url = URL.format(year=year)
    zip_path = cache_dir / f"viirs-snpp_{year}_all_countries.zip"
    _download(url, zip_path)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            indonesia = [n for n in names if "indonesia" in n.lower()]
            if not indonesia:
                logger.error("Indonesia file not found in zip. Contents: %s", names[:20])
                sys.exit(1)
            target = indonesia[0]
            logger.info("extracting: %s", target)
            raw = zf.read(target)
    except zipfile.BadZipFile:
        logger.warning("corrupt zip: %s — removing and exiting (re-run to re-download)", zip_path.name)
        zip_path.unlink()
        sys.exit(1)

    if not keep_raw:
        zip_path.unlink()
        logger.info("removed raw zip")

    df = pd.read_csv(io.BytesIO(raw), names=COLUMNS, skiprows=1, low_memory=False)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "acq_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    conf = df["confidence"].astype(str).str.strip().str.lower()
    keep = conf.isin({"n", "nominal", "h", "high"})
    df = df[keep][KEEP].copy()
    logger.info("  confidence filter: kept %d / %d detections", len(df), len(df) + (~keep).sum())

    frp_valid = df["frp"].dropna()
    if len(frp_valid):
        q = frp_valid.quantile([0, 0.25, 0.5, 0.75, 0.90, 0.95, 1.0])
        logger.info("  FRP (MW): min=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f p95=%.1f max=%.1f",
                    q.iloc[0], q.iloc[1], q.iloc[2], q.iloc[3], q.iloc[4], q.iloc[5], q.iloc[6])

    hits = _spatial_join(df, grid)
    return hits, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="single year (legacy; prefer --years)")
    parser.add_argument("--years", type=int, nargs=2, default=None,
                        help="start and end year (inclusive)")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/viirs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/output/viirs"))
    parser.add_argument("--grid-csv", type=Path, default=Path("data/output/grid/grid_cells.csv"))
    args = parser.parse_args()

    if args.year is not None:
        years = range(args.year, args.year + 1)
    elif args.years is not None:
        years = range(args.years[0], args.years[1] + 1)
    else:
        parser.error("either --year or --years is required")
    years = list(years)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(args.grid_csv)
    riau_cells = grid[grid["is_riau"]]["cell_idx"].tolist()
    riau_lookup = grid[["cell_idx", "row", "col"]].drop_duplicates()

    # Load all years: {year: hits_df}
    hits_by_year: dict[int, pd.DataFrame] = {}
    for yr in years:
        logger.info("--- VIIRS %d ---", yr)
        hits_by_year[yr], _ = _load_year(yr, args.cache_dir, grid, keep_raw=args.keep_raw)

    # Build labels per year using Y + Y+1 hotspots for boundary windows.
    for i, yr in enumerate(years):
        yr_hits = hits_by_year[yr].copy()
        if yr + 1 in hits_by_year:
            yr_hits = pd.concat([yr_hits, hits_by_year[yr + 1]], ignore_index=True)
        all_dates = pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
        labels = _build_labels(yr_hits, all_dates, riau_cells, args.k, args.window)

        out_path = args.out_dir / f"labels_{yr}.csv"
        labels = labels.merge(riau_lookup, on="cell_idx", how="left")
        labels = labels[["cell_idx", "row", "col", "date", "label"]].rename(columns={"label": "fire_label"})
        labels.sort_values(["cell_idx", "date"]).to_csv(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path, len(labels))

        _stats(yr_hits, labels, yr)


if __name__ == "__main__":
    main()
