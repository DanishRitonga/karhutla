"""Shared GEE infrastructure for Riau karhutla feature ingestion.

Private module — imported by ``era5land.py`` and ``dynamic_world.py``.

Provides:
    * ``GeeConfig``         — dataclass with grid paths, project, sampling scales.
    * ``RiauGridCells``     — loads the fixed 5 km equal-area grid as a GEE
                                                         FeatureCollection.
    * ``GeeClient``         — Earth Engine auth, initialisation, reduceRegions.
    * ``ALBERS_ID_AEAC_PROJ4`` — proj4 string shared with grid_definition.py.
    * ``_iter_months``, ``_iter_windows`` — calendar iteration helpers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

ALBERS_ID_AEAC_PROJ4 = (
    "+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("_gee")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeeConfig:
    """Static configuration shared across all GEE ingestion scripts."""

    project: str | None = None
    grid_csv: Path = Path("data/output/grid/grid_cells.csv")
    grid_meta: Path = Path("data/output/grid/grid_meta.json")
    era5land_dir: Path = Path("data/output/era5land")
    dynamic_world_dir: Path = Path("data/output/dynamic_world")
    era5land_scale: int = 9000
    dynamic_world_scale: int = 100
    era5land_chunk_days: int = 31
    dynamic_world_chunk_days: int = 31
    feature_chunk_size: int = 2000


# --------------------------------------------------------------------------- #
# Grid -> GEE FeatureCollection
# --------------------------------------------------------------------------- #


class RiauGridCells:
    """Loads the fixed grid and exposes it as an ee FeatureCollection."""

    def __init__(self, config: GeeConfig) -> None:
        import ee  # lazy import — module can be imported without auth

        self.ee = ee
        self.config = config
        self.cells = pd.read_csv(config.grid_csv)
        self.meta = json.loads(config.grid_meta.read_text())
        self._fc: ee.FeatureCollection | None = None
        self._features: list[dict] | None = None

    def _cell_polygons_wgs(self) -> list[dict]:
        transformer = pyproj.Transformer.from_crs(
            ALBERS_ID_AEAC_PROJ4, "EPSG:4326", always_xy=True
        )
        cs = self.meta["cell_size_m"]
        x0 = self.meta["x0"]
        y0 = self.meta["y0"]

        features = []
        for row in self.cells.itertuples(index=False):
            x, y = x0 + row.col * cs, y0 + row.row * cs
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
        if self._fc is None:
            feats = self._features_wgs()
            self._fc = self.ee.FeatureCollection(feats)
            logger.info("Built GEE FeatureCollection with %d cell polygons", len(feats))
        return self._fc

    def _features_wgs(self) -> list[dict]:
        if self._features is None:
            self._features = self._cell_polygons_wgs()
        return self._features

    def feature_chunks(self, chunk_size: int):
        feats = self._features_wgs()
        for i in range(0, len(feats), chunk_size):
            yield self.ee.FeatureCollection(feats[i : i + chunk_size])

    def bounds_geometry(self):
        lon_min = float(self.cells["lon"].min())
        lon_max = float(self.cells["lon"].max())
        lat_min = float(self.cells["lat"].min())
        lat_max = float(self.cells["lat"].max())
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
        except Exception as exc:
            logger.error(
                "Earth Engine authentication failed: %s\n"
                "One-time setup needed:\n"
                "    earthengine authenticate\n"
                "    earthengine set-project <PROJECT-ID>\n"
                "or re-run with: --project <PROJECT-ID>",
                exc,
            )
            sys.exit(2)

    def reduce_regions(self, image, collection, reducer, scale: int):
        ee = self.ee
        last_err = None
        for attempt in range(3):
            try:
                fc = image.reduceRegions(
                    collection=collection, reducer=reducer, scale=scale
                )
                info = fc.getInfo()
                return info
            except Exception as exc:
                last_err = exc
                logger.warning("reduceRegions attempt %d failed: %s", attempt + 1, exc)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"reduceRegions failed after 3 attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Iteration helpers
# --------------------------------------------------------------------------- #


def _iter_months(start: date, end: date):
    """Yield (first_day, last_day) for each calendar month in [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def _iter_windows(start: date, end: date, chunk_days: int):
    """Yield non-overlapping [start, end] windows of ``chunk_days`` days."""
    cur = start
    while cur <= end:
        nxt = cur + timedelta(days=chunk_days - 1)
        yield cur, min(nxt, end)
        cur = nxt + timedelta(days=1)
