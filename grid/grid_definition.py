"""Riau 5 km equal-area grid definition (Albers Indonesia Equal Area Conic).

Builds the fixed spatial grid used by the karhutla hotspot prediction
pipeline. The grid is a single source of truth: ``cell_idx`` is the
universal join key shared by every data source (FIRMS, ERA5-Land,
CHIRPS, Sentinel-1, Dynamic World, peat map).

Design rules (locked in the design log, section 4):

  * The grid is built in an equal-area projection (Indonesia Equal Area
    Conic, Albers) NOT in degrees, so every 5 km cell covers the same
    ground area. Riau spans UTM zones 47N/48N, so a single UTM zone
    would distort one side of the province. Note EPSG:9470 (SRGI2013)
    is a GEOGRAPHIC CRS, not projected; the equal-area conic is passed
    via its proj4 string.
  * The origin is aligned to multiples of 5 km so future provinces can
    share the same grid.
  * ``is_riau`` marks cells whose centre lies inside the Riau
    boundary. Only these cells receive labels and are evaluated; all
    cells in the bounding box still receive features because the
    15 x 15 patches need neighbour context (weather, radar) beyond the
    administrative edge.
  * Out-of-bounding-box points are DROPPED, never clipped to an edge
    cell. Clipping silently fabricates false-positive labels at the
    border (the original degree-based draft did exactly this with
    ``np.clip``).

Row convention: row 0 is the southernmost row; cell (row, col) spans
[x0 + col*5 km, x0 + (col+1)*5 km) x [y0 + row*5 km, y0 + (row+1)*5 km).
Cell centres are offset by +0.5 cell in both axes.

Run from the ``grid/`` project root:

    uv run --python 3.12 python scripts/grid_definition.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from pyproj import CRS
from shapely.geometry import Point

# Indonesia Equal Area Conic (Albers), the BPS/statistical standard used for
# equal-area analysis of Indonesia. EPSG:9470 (SRGI2013) is a GEOGRAPHIC CRS,
# not projected, so it cannot define an equal-area grid.
ALBERS_ID_AEAC_PROJ4 = (
    "+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grid_definition")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GridConfig:
    """Static configuration for the Riau grid."""

    cell_size_m: int = 5000
    target_crs: CRS = field(
        default_factory=lambda: CRS.from_proj4(ALBERS_ID_AEAC_PROJ4)
    )
    wgs_epsg: int = 4326
    province: str = "Riau"
    province_field: str = "WADMPR"
    admin_url: str = (
        "https://kspservices.big.go.id/satupeta/rest/services/"
        "RBI/Administrasi_AR_KabKota_50K/MapServer/0/query"
    )
    user_agent: str = "Mozilla/5.0 (karhutla datathon grid)"
    request_timeout_s: int = 60
    batch_size: int = 1000
    # Optional offline fallback: if set and boundary fetch fails, load from disk.
    boundary_fallback: str | None = None
    # Boundary polygon parts smaller than this (in m^2) are dropped so that
    # tiny offshore islets do not inflate the bounding box. Default 1 km^2.
    min_part_area_m2: float = 1e6


# --------------------------------------------------------------------------- #
# Boundary fetching
# --------------------------------------------------------------------------- #


class ArcGISQuery:
    """Paginated query helper for an ArcGIS REST FeatureServer layer."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config

    def query_all(self, where: str) -> gpd.GeoDataFrame:
        """Return all features matching ``where`` as a WGS84 GeoDataFrame."""
        features = []
        offset = 0
        while True:
            params = {
                "where": where,
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "resultRecordCount": self.config.batch_size,
                "resultOffset": offset,
            }
            resp = requests.get(
                self.config.admin_url,
                params=params,
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.request_timeout_s,
            )
            resp.raise_for_status()
            geojson = resp.json()
            batch = geojson.get("features", [])
            if not batch:
                break
            features.extend(batch)
            if len(batch) < self.config.batch_size:
                break
            offset += self.config.batch_size

        if not features:
            raise RuntimeError(f"ArcGIS query returned no features for: {where}")

        gdf = gpd.GeoDataFrame.from_features(features)
        gdf.set_crs(self.config.wgs_epsg, inplace=True)
        return gdf


class RiauBoundary:
    """Fetches (or loads) the Riau administrative boundary."""

    def __init__(self, config: GridConfig, query: ArcGISQuery | None = None) -> None:
        self.config = config
        self.query = query or ArcGISQuery(config)

    def load(self) -> gpd.GeoDataFrame:
        """Return the Riau kabupaten/kota boundary in WGS84."""
        try:
            gdf = self.query.query_all(
                where=f"{self.config.province_field} = '{self.config.province}'"
            )
            logger.info(
                "Fetched %d polygons for %s from BIG ArcGIS REST",
                len(gdf),
                self.config.province,
            )
        except Exception as exc:  # noqa: BLE001 - network / server fallback
            if self.config.boundary_fallback is not None:
                logger.warning("BIG fetch failed (%s); loading fallback %s", exc, self.config.boundary_fallback)
                gdf = gpd.read_file(self.config.boundary_fallback)
            else:
                raise

        provinces = gdf[self.config.province_field].unique() if self.config.province_field in gdf.columns else []
        logger.info("Boundary provinces present: %s", list(provinces))
        return gdf


# --------------------------------------------------------------------------- #
# Grid core
# --------------------------------------------------------------------------- #


class EqualAreaGrid:
    """The fixed 5 km equal-area grid over the Riau bounding box."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self.boundary_wgs: gpd.GeoDataFrame | None = None
        self.boundary_proj: gpd.GeoDataFrame | None = None
        self.x0: float | None = None
        self.y0: float | None = None
        self.cols: int | None = None
        self.rows: int | None = None
        self.cells: pd.DataFrame | None = None

    # -- construction ------------------------------------------------------ #

    def build(self, boundary_wgs: gpd.GeoDataFrame) -> "EqualAreaGrid":
        """Construct the grid from the boundary; returns self."""
        self.boundary_wgs = boundary_wgs
        self.boundary_proj = boundary_wgs.to_crs(self.config.target_crs)
        self.boundary_proj = self._filter_tiny_parts()

        xmin, ymin, xmax, ymax = self.boundary_proj.total_bounds
        self.x0 = self.config.cell_size_m * int(np.floor(xmin / self.config.cell_size_m))
        self.y0 = self.config.cell_size_m * int(np.floor(ymin / self.config.cell_size_m))
        self.cols = int(np.ceil((xmax - self.x0) / self.config.cell_size_m))
        self.rows = int(np.ceil((ymax - self.y0) / self.config.cell_size_m))

        self._build_cell_table()
        self._compute_is_riau()
        logger.info(
            "Grid: %d cols x %d rows = %d bbox cells, %d Riau cells",
            self.cols,
            self.rows,
            self.cols * self.rows,
            int(self.cells["is_riau"].sum()),
        )
        return self

    def _filter_tiny_parts(self) -> gpd.GeoDataFrame:
        """Drop boundary polygon parts smaller than ``min_part_area_m2``.

        The projected boundary is exploded into individual polygon parts,
        parts below the area threshold (e.g. tiny Malacca-strait islets) are
        removed, and the survivors are dissolved into a single union. When no
        part survives, the original boundary is returned unchanged.
        """
        min_area = self.config.min_part_area_m2
        if min_area <= 0:
            return self.boundary_proj

        exploded = self.boundary_proj.explode(index_parts=True)
        exploded = exploded[exploded.geometry.area >= min_area]
        if exploded.empty:
            logger.warning(
                "All boundary parts below min area %.0f m^2; using unfiltered boundary",
                min_area,
            )
            return self.boundary_proj

        dissolved = gpd.GeoDataFrame(
            {"geometry": [exploded.union_all()]},
            crs=self.config.target_crs,
        )
        n_kept = len(exploded)
        logger.info(
            "Boundary filter: kept %d part(s) >= %.1f km^2, dropped tiny parts",
            n_kept,
            min_area / 1e6,
        )
        return dissolved

    def _build_cell_table(self) -> None:
        col_idx, row_idx = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        col = col_idx.ravel()
        row = row_idx.ravel()
        x = self.x0 + (col + 0.5) * self.config.cell_size_m
        y = self.y0 + (row + 0.5) * self.config.cell_size_m

        cells = pd.DataFrame(
            {
                "row": row,
                "col": col,
                "cell_idx": row * self.cols + col,
                "x_center_m": x,
                "y_center_m": y,
            }
        )

        pts_proj = gpd.points_from_xy(x, y, crs=self.config.target_crs)
        pts_wgs = gpd.GeoSeries(pts_proj, crs=self.config.target_crs).to_crs(self.config.wgs_epsg)
        cells["lon"] = pts_wgs.x.values
        cells["lat"] = pts_wgs.y.values
        cells["is_riau"] = False
        self.cells = cells

    def _compute_is_riau(self) -> None:
        union = self.boundary_proj.union_all()
        centers = gpd.points_from_xy(
            self.cells["x_center_m"], self.cells["y_center_m"], crs=self.config.target_crs
        )
        self.cells["is_riau"] = gpd.GeoSeries(centers, crs=self.config.target_crs).within(union).values

    # -- coordinate mapping ------------------------------------------------ #

    def _to_projected(self, lon: float | np.ndarray, lat: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = gpd.points_from_xy(lon, lat, crs=self.config.wgs_epsg).to_crs(self.config.target_crs)
        return np.asarray([p.x for p in pts]), np.asarray([p.y for p in pts])

    def _to_wgs(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = gpd.points_from_xy(x, y, crs=self.config.target_crs).to_crs(self.config.wgs_epsg)
        return np.asarray([p.x for p in pts]), np.asarray([p.y for p in pts])

    def assign_cell_idx(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Vectorised cell assignment.

        Returns an int64 array of cell_idx aligned to ``lon``/``lat``.
        Points outside the bounding box are assigned -1 (DROPPED, never
        clipped to an edge cell).
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y = self._to_projected(lon, lat)
        col = np.floor((x - self.x0) / self.config.cell_size_m).astype(np.int64)
        row = np.floor((y - self.y0) / self.config.cell_size_m).astype(np.int64)
        valid = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        idx = np.full(len(lon), -1, dtype=np.int64)
        idx[valid] = row[valid] * self.cols + col[valid]
        return idx

    def latlon_to_cell(self, lon: float, lat: float) -> tuple[int, int] | None:
        """Return (row, col) for a single point, or None if outside the box."""
        idx = self.assign_cell_idx(np.array([lon]), np.array([lat]))[0]
        if idx < 0:
            return None
        return int(idx // self.cols), int(idx % self.cols)

    def cell_to_lonlat(self, row: int, col: int) -> tuple[float, float]:
        """Inverse mapping: WGS84 centre of cell (row, col)."""
        x = self.x0 + (col + 0.5) * self.config.cell_size_m
        y = self.y0 + (row + 0.5) * self.config.cell_size_m
        lon, lat = self._to_wgs(np.array([x]), np.array([y]))
        return float(lon[0]), float(lat[0])

    # -- reporting --------------------------------------------------------- #

    def summary(self) -> dict:
        n_total = self.cols * self.rows
        n_riau = int(self.cells["is_riau"].sum())
        xmin, ymin, xmax, ymax = self.boundary_proj.total_bounds
        return {
            "crs": self.config.target_crs.to_string(),
            "cell_size_m": self.config.cell_size_m,
            "x0": self.x0,
            "y0": self.y0,
            "cols": self.cols,
            "rows": self.rows,
            "bbox_cells": n_total,
            "riau_cells": n_riau,
            "fill_rate_pct": round(100.0 * n_riau / n_total, 1),
            "boundary_bounds_projected": [round(v, 1) for v in (xmin, ymin, xmax, ymax)],
        }

    # -- persistence ------------------------------------------------------- #

    def save(self, output_dir: Path) -> list[Path]:
        """Write grid cells CSV, metadata JSON, and the projected boundary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "grid_cells.csv"
        self.cells.to_csv(csv_path, index=False)

        meta_path = output_dir / "grid_meta.json"
        meta_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

        boundary_path = output_dir / "riau_boundary_aea.gpkg"
        self.boundary_proj.to_file(boundary_path, driver="GPKG")

        logger.info("Saved grid_cells.csv, grid_meta.json, riau_boundary_aea.gpkg to %s", output_dir)
        return [csv_path, meta_path, boundary_path]


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


class GridPlotter:
    """Renders the grid: boundary outline + Riau cells + full bbox grid."""

    def __init__(self, grid: EqualAreaGrid) -> None:
        self.grid = grid

    def render(self, output_path: Path, dpi: int = 150) -> None:
        g = self.grid
        cs = g.config.cell_size_m
        fig, ax = plt.subplots(figsize=(11, 11), dpi=dpi)

        # 1. Riau cells (orange) -- centres inside the boundary
        riau = g.cells[g.cells["is_riau"]]
        polys = [
            MplPolygon(
                [
                    (x - cs / 2, y - cs / 2),
                    (x + cs / 2, y - cs / 2),
                    (x + cs / 2, y + cs / 2),
                    (x - cs / 2, y + cs / 2),
                ],
                closed=True,
                facecolor="#f9b234",
                edgecolor="none",
                alpha=0.6,
            )
            for x, y in zip(riau["x_center_m"], riau["y_center_m"])
        ]
        for poly in polys:
            ax.add_patch(poly)

        # 2. Full bounding-box grid lines (light)
        xs = np.arange(g.x0, g.x0 + (g.cols + 1) * cs + 1, cs)
        ys = np.arange(g.y0, g.y0 + (g.rows + 1) * cs + 1, cs)
        segs = list(
            [((v, y0), (v, y1)) for v in xs for y0, y1 in ((ys[0], ys[-1]),)]
        ) + list(
            [((x0, h), (x1, h)) for h in ys for x0, x1 in ((xs[0], xs[-1]),)]
        )
        ax.add_collection(
            LineCollection(segs, colors="0.55", linewidths=0.3, alpha=0.7)
        )

        # 3. Boundary outline on top
        g.boundary_proj.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2)

        ax.set_aspect("equal")
        ax.set_title(
            f"Riau 5 km equal-area grid (Albers Indonesia Equal Area Conic)\n"
            f"{g.cols} x {g.rows} = {g.cols * g.rows} bbox cells, "
            f"{int(riau.shape[0])} Riau cells"
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.margins(0.02)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        logger.info("Rendered %s", output_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    config = GridConfig()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"

    boundary = RiauBoundary(config).load()
    grid = EqualAreaGrid(config).build(boundary)

    summary = grid.summary()
    print(json.dumps(summary, indent=2))

    saved = grid.save(output_dir)
    png = output_dir / "riau_grid.png"
    GridPlotter(grid).render(png)
    saved.append(png)

    # Sanity: round-trip a known centre
    lon, lat = grid.cell_to_lonlat(0, 0)
    rc = grid.latlon_to_cell(lon, lat)
    logger.info("Round-trip (0,0) -> lon/lat (%.4f, %.4f) -> %s", lon, lat, rc)


if __name__ == "__main__":
    main()
