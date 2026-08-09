"""Fetch the peat ecosystem map (FEG 1:50.000) and rasterize it onto the grid.

Source: BIG / Satupeta ArcGIS REST, layer 48 of
``SUMBER_DAYA_ALAM_DAN_LINGKUNGAN``
(``Peta Fungsi Ekosistem Gambut Skala 1:50.000``, PP 57/2016).

Attributes of interest:

  * ``peat_thick``  peat-depth class as an Indonesian string, e.g.
                    ``'5,0 - 6,0 meter'`` (comma decimal, dash range).
  * ``tnh_gambut``  ``'Tanah Gambut'`` vs ``'Non Gambut (Mineral)'``.
  * ``feg_peat``    ``'Fungsi Lindung (Gambut > 3 m)'`` vs
                    ``'Fungsi Budidaya (Gambut < 3 m)'``.
  * ``kode_khg``    hydrological-unit id (no depth).

Per 5 km cell the script emits two static ``peat`` channels:

  * ``peat_frac``      fraction of the cell covered by ``Tanah Gambut``
                       polygons (0..1, area-weighted).
  * ``peat_depth_m``   area-weighted mean of the ``peat_thick`` midpoint
                       over the peat-covered part of the cell.

Both are broadcast across time by the downstream tensor assembly (the
design log treats peat as the *static spatial* channel).

Run (from the datathon project root):

    uv run --python 3.12 python grid/scripts/fetch_peat.py
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_peat")

# Same equal-area CRS as grid_definition.py (Albers Indonesia Equal Area Conic).
ALBERS_ID_AEAC_PROJ4 = (
    "+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)

RIAN_BBOX_WGS84 = (100.05, -1.13, 103.81, 2.92)  # xmin, ymin, xmax, ymax
PEAT_URL = (
    "https://kspservices.big.go.id/satupeta/rest/services/PUBLIK/"
    "SUMBER_DAYA_ALAM_DAN_LINGKUNGAN/MapServer/48/query"
)
THICK_RE = re.compile(r"([\d]+(?:,[\d]+)?)\s*[-–]\s*([\d]+(?:,[\d]+)?)")


@dataclass(frozen=True)
class PeatConfig:
    """Static configuration for the peat fetch + rasterize step."""

    url: str = PEAT_URL
    bbox: tuple[float, float, float, float] = RIAN_BBOX_WGS84
    grid_csv: Path = Path("data/output/grid/grid_cells.csv")
    boundary_gpkg: Path = Path("data/output/grid/riau_boundary_aea.gpkg")
    out_dir: Path = Path("data/output/peat")
    maps_dir: Path = Path("data/output/maps")
    batch_size: int = 1000
    request_timeout_s: int = 90
    retry: int = 3
    user_agent: str = "datathon-poc/1.0"
    dpi: int = 150

    @property
    def target_crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_proj4(ALBERS_ID_AEAC_PROJ4)

    @property
    def cell_size_m(self) -> int:
        return 5000


def _parse_thickness(raw: object) -> float | None:
    """Parse ``'5,0 - 6,0 meter'`` -> 5.5 (midpoint in metres)."""
    if raw is None or not isinstance(raw, str):
        return None
    m = THICK_RE.search(raw)
    if not m:
        return None
    lo = float(m.group(1).replace(",", "."))
    hi = float(m.group(2).replace(",", "."))
    return (lo + hi) / 2.0


class PeatFetcher:
    """Paginated download of the FEG layer within the Riau bbox."""

    def __init__(self, config: PeatConfig) -> None:
        self.config = config

    def fetch(self) -> gpd.GeoDataFrame:
        cfg = self.config
        features: list[dict] = []
        offset = 0
        xmin, ymin, xmax, ymax = cfg.bbox
        while True:
            params = {
                "where": "1=1",
                "geometry": f"{xmin},{ymin},{xmax},{ymax}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "kode_khg,peat_thick,tnh_gambut,feg_peat",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
                "resultRecordCount": cfg.batch_size,
                "resultOffset": offset,
            }
            batch = self._query_with_retry(params)
            if not batch:
                break
            features.extend(batch)
            logger.info("fetched %d/%d features", len(features), len(features) + 0)
            if len(batch) < cfg.batch_size:
                break
            offset += cfg.batch_size

        if not features:
            raise RuntimeError("Peat layer query returned no features")
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf.set_crs(4326, inplace=True)
        logger.info("total peat features in Riau bbox: %d", len(gdf))
        return gdf

    def _query_with_retry(self, params: dict) -> list[dict]:
        cfg = self.config
        last: Exception | None = None
        for attempt in range(cfg.retry):
            try:
                resp = requests.get(
                    cfg.url,
                    params=params,
                    headers={"User-Agent": cfg.user_agent},
                    timeout=cfg.request_timeout_s,
                )
                resp.raise_for_status()
                return resp.json().get("features", [])
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"peat query failed after {cfg.retry} tries") from last


def _cell_squares(cells: pd.DataFrame, cfg: PeatConfig) -> gpd.GeoDataFrame:
    """Build the 5 km cell squares directly in the projected grid CRS."""
    cs = cfg.cell_size_m
    geom = [
        shapely.box(
            x - cs / 2, y - cs / 2, x + cs / 2, y + cs / 2
        )
        for x, y in zip(cells["x_center_m"], cells["y_center_m"])
    ]
    return gpd.GeoDataFrame(cells[["cell_idx"]], geometry=geom, crs=cfg.target_crs)


class PeatRasterizer:
    """Area-weighted rasterization of peat depth onto the 5 km grid."""

    def __init__(self, config: PeatConfig) -> None:
        self.config = config

    def run(self, peat: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config
        cells = pd.read_csv(cfg.grid_csv)
        cell_gdf = _cell_squares(cells, cfg)

        peat = peat.to_crs(cfg.target_crs).copy()
        peat["peat_mid"] = peat["peat_thick"].map(_parse_thickness)
        peat_ok = peat[peat["tnh_gambut"] == "Tanah Gambut"].copy()
        logger.info("peat polygons classified as Tanah Gambut: %d", len(peat_ok))

        joined = gpd.overlay(
            cell_gdf[["cell_idx", "geometry"]],
            peat_ok[["geometry", "peat_mid"]],
            how="intersection",
            keep_geom_type=False,
        )
        joined["inter_area"] = joined.geometry.area
        joined["weighted"] = joined["inter_area"] * joined["peat_mid"]

        agg = joined.groupby("cell_idx").agg(
            peat_area=("inter_area", "sum"),
            wsum=("weighted", "sum"),
        )
        agg["peat_depth_m"] = agg["wsum"] / agg["peat_area"]
        cell_area = cfg.cell_size_m**2
        agg["peat_frac"] = (agg["peat_area"] / cell_area).clip(upper=1.0)

        out = cells[["cell_idx", "row", "col", "is_riau"]].copy()
        out = out.merge(
            agg[["peat_frac", "peat_depth_m"]],
            on="cell_idx",
            how="left",
        )
        out["peat_frac"] = out["peat_frac"].fillna(0.0)
        out["peat_depth_m"] = out["peat_depth_m"].fillna(0.0)
        return out


class PeatPlotter:
    """Render a peat field as a choropleth on the grid (riau_grid style)."""

    def __init__(self, config: PeatConfig) -> None:
        self.config = config
        self.boundary = gpd.read_file(config.boundary_gpkg)
        self.cells = pd.read_csv(config.grid_csv)[
            ["cell_idx", "x_center_m", "y_center_m"]
        ].set_index("cell_idx")

    def render(self, data: pd.DataFrame, field: str, cmap: str, label: str) -> Path:
        cfg = self.config
        cells = data.merge(self.cells, left_on="cell_idx", right_index=True)
        cells = cells[cells[field].notna()].copy()
        cs = cfg.cell_size_m
        vals = cells[field].to_numpy()
        if field == "peat_frac":
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98))
            norm = Normalize(vmin=lo, vmax=hi)
        cm = plt.get_cmap(cmap)

        patches, facecolors = [], []
        for _, row in cells.iterrows():
            x, y = row["x_center_m"], row["y_center_m"]
            patches.append(Rectangle((x - cs / 2, y - cs / 2), cs, cs))
            facecolors.append(cm(norm(row[field])))

        fig, ax = plt.subplots(figsize=(11, 11), dpi=cfg.dpi)
        ax.add_collection(
            PatchCollection(patches, facecolors=facecolors, edgecolors="none")
        )
        self.boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(label)

        ax.set_title(
            f"Peat — {field}\n"
            "Riau, 5 km equal-area grid (Albers Indonesia Equal Area Conic)",
            fontsize=12,
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.margins(0.02)
        fig.tight_layout()
        out = cfg.maps_dir / f"peat_{field}.png"
        fig.savefig(out, dpi=cfg.dpi)
        plt.close(fig)
        logger.info("Rendered %s", out)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + rasterize Riau peat map")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = PeatConfig()
    if args.out_dir is not None:
        config = PeatConfig(out_dir=args.out_dir, maps_dir=config.maps_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.maps_dir.mkdir(parents=True, exist_ok=True)

    fetcher = PeatFetcher(config)
    peat = fetcher.fetch()

    data = PeatRasterizer(config).run(peat)
    csv_path = config.out_dir / "peat_cell.csv"
    data.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(data))

    if not args.no_plots:
        plotter = PeatPlotter(config)
        plotter.render(data, "peat_frac", "YlGn", "peat area fraction")
        plotter.render(data, "peat_depth_m", "YlGnBu", "peat depth (m, area-weighted midpoint)")


if __name__ == "__main__":
    main()
