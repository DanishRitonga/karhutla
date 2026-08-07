"""GEE ingestion for Sentinel-1 GRD backscatter and CHIRPS v3.0 SAT daily rain.

Pulls feature data for every cell of the fixed Riau grid into per-cell,
per-date long-format tables. Each source keeps its own table; ``cell_idx``
is the universal join key (same as grid/output/grid_cells.csv).

Design rules (design log sections 3 and 10):

  * Every bounding-box cell gets features (6,970 cells) even though only
    ``is_riau`` cells receive labels -- the 15 x 15 patches need neighbour
    context (weather, radar) beyond the administrative edge.
  * Sentinel-1 is aggregated with a **median** reducer at ~100 m sampling
    scale. The median doubles as a crude speckle-suppression filter, which
    GEE cannot do natively (Refined Lee is not available server-side). The
    per-cell median is the design-log aggregation step.
  * CHIRPS v3 ``sat`` variant is consumed as the native ``precipitation``
    band (mm/day). Cells are ~5 km and the CHIRPS pixel is ~5.6 km, so the
    mean reducer at native scale is effectively nearest-value.
  * Source independence (L10): CHIRPS-sat is disaggregated from IMERG, not
    ERA5, so it stays independent of the ERA5-Land weather stream.

Outputs (one CSV per month, long format):
  * chirps_v3sat_YYYYMM.csv   -> cell_idx, date, precip_mm
  * s1_ASCENDING_YYYYMM.csv   -> cell_idx, date, vv_db, vh_db, angle_deg
  * s1_DESCENDING_YYYYMM.csv  -> cell_idx, date, vv_db, vh_db, angle_deg
  * vh_vv_db (VH - VV in dB) is derived in pandas after the pull.

Requires an authenticated Earth Engine account:

    earthengine authenticate          # interactive browser flow, once
    earthengine set-project <PROJECT> # or pass --project on every run

Run (from the datathon project root):

    uv run --python 3.12 python grid/scripts/gee_ingest.py \
        --source all --start 2019-01-01 --end 2019-01-31 --project ee-yours
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

# Indonesia Equal Area Conic (Albers) -- must match grid_definition.py.
ALBERS_ID_AEAC_PROJ4 = (
    "+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gee_ingest")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeeConfig:
    """Static configuration for one ingestion run."""

    project: str | None = None
    grid_csv: Path = Path("grid/output/grid_cells.csv")
    grid_meta: Path = Path("grid/output/grid_meta.json")
    output_dir: Path = Path("grid/output/gee")
    # Native-ish sampling scales for the two sources (metres).
    chirps_scale: int = 5566
    s1_scale: int = 100
    # Revisit gap tolerances: CHIRPS chunks by calendar month; S1 chunks by
    # orbit pass over a sliding window.
    chirps_chunk_days: int = 31
    s1_chunk_days: int = 45
    # GEE aborts collection queries that accumulate > 5000 elements, so the
    # 6970-cell grid is reduced in slices of this many features per call.
    feature_chunk_size: int = 2000


# --------------------------------------------------------------------------- #
# Grid -> GEE feature collection
# --------------------------------------------------------------------------- #


class RiauGridCells:
    """Loads the fixed grid and exposes it as an ee FeatureCollection."""

    def __init__(self, config: GeeConfig) -> None:
        import ee  # imported lazily so the module imports without auth

        self.ee = ee
        self.config = config
        self.cells = pd.read_csv(config.grid_csv)
        self.meta = json.loads(config.grid_meta.read_text())
        self._fc: ee.FeatureCollection | None = None
        self._features: list[dict] | None = None

    def _cell_polygons_wgs(self) -> list[dict]:
        """Build one square 5 km polygon per cell, corners transformed to WGS84."""
        transformer = pyproj.Transformer.from_crs(
            ALBERS_ID_AEAC_PROJ4, "EPSG:4326", always_xy=True
        )
        cs = self.meta["cell_size_m"]
        x0 = self.meta["x0"]
        y0 = self.meta["y0"]

        features = []
        for row in self.cells.itertuples(index=False):
            x = x0 + row.col * cs
            y = y0 + row.row * cs
            # Cell (row, col) spans [x, x+cs) x [y, y+cs); corners ordered
            # counter-clockwise starting bottom-left.
            xs = [x, x + cs, x + cs, x]
            ys = [y, y, y + cs, y + cs]
            lon, lat = transformer.transform(xs, ys)
            lon = np.asarray(lon)
            lat = np.asarray(lat)
            ring = [list(p) for p in zip(lon.tolist(), lat.tolist())]
            ring.append(ring[0])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "cell_idx": int(row.cell_idx),
                        "row": int(row.row),
                        "col": int(row.col),
                    },
                }
            )
        return features

    def feature_collection(self):
        """Return (and cache) the ee.FeatureCollection of 5 km cells."""
        if self._fc is None:
            self._fc = self.ee.FeatureCollection(self._features_wgs())
            logger.info("Built GEE FeatureCollection with %d cell polygons", len(self._features_wgs()))
        return self._fc

    def _features_wgs(self) -> list[dict]:
        """Lazily built list of GeoJSON feature dicts (one per cell)."""
        if self._features is None:
            self._features = self._cell_polygons_wgs()
        return self._features

    def feature_chunks(self, chunk_size: int):
        """Yield ``ee.FeatureCollection`` slices of <= ``chunk_size`` cells.

        GEE aborts collection queries that accumulate over 5000 elements, so
        the 6970-cell grid must be reduced in slices.
        """
        feats = self._features_wgs()
        for i in range(0, len(feats), chunk_size):
            yield self.ee.FeatureCollection(feats[i : i + chunk_size])

    def bounds_geometry(self):
        """WGS84 bounding geometry of the grid (for filterBounds)."""
        lon_min, lon_max = float(self.cells["lon"].min()), float(self.cells["lon"].max())
        lat_min, lat_max = float(self.cells["lat"].min()), float(self.cells["lat"].max())
        return self.ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)


# --------------------------------------------------------------------------- #
# GEE client / auth
# --------------------------------------------------------------------------- #


class GeeClient:
    """Handles authentication and provides a bound reduction helper."""

    def __init__(self, config: GeeConfig) -> None:
        import ee

        self.ee = ee
        self.config = config

    def initialize(self) -> None:
        ee = self.ee
        try:
            if self.config.project:
                ee.Initialize(project=self.config.project)
            else:
                ee.Initialize()
            logger.info("Earth Engine initialized (project=%s)", self.config.project or "default")
        except Exception as exc:  # noqa: BLE001 - auth failure is user-facing
            logger.error(
                "Earth Engine authentication failed: %s\n"
                "Please run the one-time setup, then retry:\n"
                "    earthengine authenticate\n"
                "    earthengine set-project <PROJECT-ID>\n"
                "or re-run with: --project <PROJECT-ID>",
                exc,
            )
            sys.exit(2)

    def reduce_regions(self, image, collection, reducer, scale: int):
        """Run reduceRegions with a small backoff/retry for transient errors."""
        ee = self.ee
        last_err = None
        for attempt in range(3):
            try:
                fc = image.reduceRegions(
                    collection=collection,
                    reducer=reducer,
                    scale=scale,
                )
                info = fc.getInfo()
                return info
            except Exception as exc:  # noqa: BLE001 - transient EE errors
                last_err = exc
                logger.warning("reduceRegions attempt %d failed: %s", attempt + 1, exc)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"reduceRegions failed after 3 attempts: {last_err}")


# --------------------------------------------------------------------------- #
# CHIRPS v3.0 SAT
# --------------------------------------------------------------------------- #


class ChirpsIngester:
    """CHIRPS v3.0 SAT daily precipitation -> per-cell daily table."""

    COLLECTION = "UCSB-CHC/CHIRPS/V3/DAILY_SAT"
    BAND = "precipitation"

    def __init__(self, client: GeeClient, cells: RiauGridCells, config: GeeConfig) -> None:
        self.client = client
        self.cells = cells
        self.config = config
        self.ee = client.ee

    def _stack_month(self, month_start: date, month_end: date):
        """Return a single stacked image with one band per day (YYYYMMDD)."""
        ee = self.ee
        bbox = self.cells.bounds_geometry()
        ic = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(month_start.isoformat(), (month_end + timedelta(days=1)).isoformat())
        )

        def rename_day(img):
            day = img.date().format("yyyyMMdd")
            return img.select([self.BAND]).rename([day])

        return ic.map(rename_day).toBands()

    def _pull(self, month_start: date, month_end: date) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("CHIRPS: pulling %s -> %s", month_start, month_end)
        stacked = self._stack_month(month_start, month_end)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked,
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=self.config.chirps_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("CHIRPS: no data for %s -> %s", month_start, month_end)
            return None

        id_cols = ["cell_idx", "row", "col"]
        for c in id_cols:
            if c not in wide.columns:
                wide[c] = np.nan

        # Bands look like 0_20190101 (toBands() prefix + date).
        band_cols = [
            c
            for c in wide.columns
            if "_" in c and c.split("_")[-1].isdigit() and len(c.split("_")[-1]) == 8
        ]
        if not band_cols:
            logger.warning("CHIRPS: no day bands in %s -> %s", month_start, month_end)
            return None

        long = wide.melt(
            id_vars=id_cols,
            value_vars=band_cols,
            var_name="band",
            value_name="precip_mm",
        )
        long["date"] = long["band"].str.split("_").str[-1]
        long["date"] = pd.to_datetime(long["date"], format="%Y%m%d").dt.date
        long = long.dropna(subset=["precip_mm"])
        return long[id_cols + ["date", "precip_mm"]]

    def ingest(self, start: date, end: date) -> list[Path]:
        written = []
        for month_start, month_end in _iter_months(start, end):
            df = self._pull(month_start, month_end)
            if df is None or df.empty:
                continue
            path = self.config.output_dir / f"chirps_v3sat_{month_start:%Y%m}.csv"
            df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
            written.append(path)
            logger.info("CHIRPS: wrote %s (%d rows)", path, len(df))
        return written


# --------------------------------------------------------------------------- #
# Sentinel-1 GRD
# --------------------------------------------------------------------------- #


class Sentinel1Ingester:
    """Sentinel-1 GRD VV/VH backscatter -> per-cell, per-acquisition table."""

    COLLECTION = "COPERNICUS/S1_GRD"
    BANDS = ["VV", "VH", "angle"]

    def __init__(self, client: GeeClient, cells: RiauGridCells, config: GeeConfig) -> None:
        self.client = client
        self.cells = cells
        self.config = config
        self.ee = client.ee

    def _stack_window(self, start: date, end: date, orbit: str):
        """Stack S1 images in [start, end] for one orbit pass, one band/date."""
        ee = self.ee
        bbox = self.cells.bounds_geometry()
        ic = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )

        def rename_image(img):
            d = img.date().format("yyyyMMdd")
            return img.select(self.BANDS).rename([d.cat("_vv"), d.cat("_vh"), d.cat("_angle")])

        return ic.map(rename_image).toBands()

    def _pull(self, start: date, end: date, orbit: str) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("S1 %s: pulling %s -> %s", orbit, start, end)
        stacked = self._stack_window(start, end, orbit)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked,
                collection=fc,
                reducer=ee.Reducer.median(),
                scale=self.config.s1_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("S1 %s: no acquisitions in %s -> %s", orbit, start, end)
            return None

        id_cols = ["cell_idx", "row", "col"]
        for c in id_cols:
            if c not in wide.columns:
                wide[c] = np.nan

        # Bands look like <system:index>_20190115_vv (toBands prefixes with the
        # S1 product id, which itself contains underscores). The date and field
        # are always the last two underscore tokens.
        band_cols = [
            c
            for c in wide.columns
            if c.split("_")[-1] in {"vv", "vh", "angle"}
            and c.split("_")[-2].isdigit()
            and len(c.split("_")[-2]) == 8
        ]
        if not band_cols:
            logger.warning("S1 %s: no acquisition bands in %s -> %s", orbit, start, end)
            return None

        long = wide.melt(id_vars=id_cols, value_vars=band_cols, var_name="band")
        long["date"] = long["band"].str.split("_").str[-2]
        long["field"] = long["band"].str.split("_").str[-1]
        long["date"] = pd.to_datetime(long["date"], format="%Y%m%d").dt.date
        long = long.pivot_table(
            index=id_cols + ["date"], columns="field", values="value"
        ).reset_index()
        long = long.dropna(subset=["vv", "vh"])
        long["vh_vv_db"] = long["vh"] - long["vv"]
        long = long.rename(columns={"vv": "vv_db", "vh": "vh_db", "angle": "angle_deg"})
        return long[id_cols + ["date", "vv_db", "vh_db", "angle_deg", "vh_vv_db"]]

    def ingest(self, start: date, end: date, orbit: str | None = None) -> list[Path]:
        orbits = [orbit] if orbit else ["ASCENDING", "DESCENDING"]
        written = []
        for orb in orbits:
            for win_start, win_end in _iter_windows(start, end, self.config.s1_chunk_days):
                df = self._pull(win_start, win_end, orb)
                if df is None or df.empty:
                    continue
                path = self.config.output_dir / f"s1_{orb}_{win_start:%Y%m}.csv"
                if path.exists():
                    # Multiple windows can fall in one calendar month; append.
                    existing = pd.read_csv(path, dtype={"cell_idx": int})
                    df = pd.concat([existing, df]).drop_duplicates(
                        subset=["cell_idx", "date"], keep="last"
                    )
                df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
                written.append(path)
                logger.info("S1 %s: wrote %s (%d rows)", orb, path, len(df))
        return written


# --------------------------------------------------------------------------- #
# Iteration helpers
# --------------------------------------------------------------------------- #


def _iter_months(start: date, end: date):
    """Yield (first_day, last_day) pairs for each month in [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        yield cur, chunk_end
        cur = nxt


def _iter_windows(start: date, end: date, chunk_days: int):
    """Yield sliding [start, end] windows of ``chunk_days`` over the range."""
    cur = start
    while cur <= end:
        nxt = cur + timedelta(days=chunk_days - 1)
        yield cur, min(nxt, end)
        cur = nxt + timedelta(days=1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="GEE ingestion for Riau karhutla features")
    parser.add_argument("--source", choices=["chirps", "sentinel1", "all"], default="all")
    parser.add_argument("--orbit", choices=["ASCENDING", "DESCENDING"], default=None,
                        help="Restrict Sentinel-1 to one orbit pass (default: both)")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--project", default=None, help="Earth Engine cloud project ID")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default grid/output/gee)")
    args = parser.parse_args()

    config = GeeConfig(
        project=args.project,
        output_dir=args.out or Path("grid/output/gee"),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    client = GeeClient(config)
    client.initialize()
    cells = RiauGridCells(config)

    written: list[Path] = []
    if args.source in ("chirps", "all"):
        written += ChirpsIngester(client, cells, config).ingest(args.start, args.end)
    if args.source in ("sentinel1", "all"):
        written += Sentinel1Ingester(client, cells, config).ingest(
            args.start, args.end, args.orbit
        )

    logger.info("Done. %d output file(s) written to %s", len(written), config.output_dir)
    for p in written:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
