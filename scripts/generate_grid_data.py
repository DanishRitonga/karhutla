"""Regenerate ``backend/data/grid_data.json`` from the real Riau grid.

Replaces the prototype 108x81 grid (wrong CRS) with the actual 85x82 Albers
equal-area grid built by ``data/grid/grid_definition.py``. The backend's
``app/grid.decode_cells()`` reads this file and produces cell ids of the form
``RIAU_{r}_{c}``, so predictions.parquet must be keyed the same way.

Requirements (already in the repo):
  * data/output/grid/grid_cells.csv   (row, col, cell_idx, x_center_m, ...)
  * data/output/grid/riau_boundary_aea.gpkg
  * BIG ArcGIS admin service (reachable from Indonesia / this sandbox)

Usage::

    uv run --python 3.12 python scripts/generate_grid_data.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import CRS, Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_grid_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_CSV = PROJECT_ROOT / "data/output/grid/grid_cells.csv"
BOUNDARY_GPKG = PROJECT_ROOT / "data/output/grid/riau_boundary_aea.gpkg"
OUT = PROJECT_ROOT / "backend/data/grid_data.json"

ADMIN_URL = (
    "https://kspservices.big.go.id/satupeta/rest/services/RBI/"
    "Administrasi_AR_KabKota_50K/MapServer/0/query"
)

ALBERS = "+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"


def fetch_riau_kabupaten() -> gpd.GeoDataFrame:
    """Return WGS84 polygons for Riau's kabupaten/kota with WADMKK names."""
    geoms, names = [], []
    offset, batch = 0, 1000
    while True:
        params = {
            "where": "WADMPR = 'Riau'",
            "outFields": "WADMPR,WADMKK",
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": batch,
        }
        r = requests.get(ADMIN_URL, params=params, timeout=60)
        r.raise_for_status()
        fc = r.json()
        feats = fc.get("features", [])
        for f in feats:
            geom = f["geometry"]
            name = f["properties"].get("WADMKK")
            if geom and name:
                geoms.append(geom)
                names.append(name)
        offset += batch
        if len(feats) < batch:
            break
    gdf = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": g, "properties": {"WADMKK": n}}
         for g, n in zip(geoms, names)],
        crs="EPSG:4326",
    )
    gdf = gdf.drop_duplicates("WADMKK")
    return gdf


def albers_centroids(gdf: gpd.GeoDataFrame) -> dict[str, list[float]]:
    """Map kabupaten name -> (x, y) centroid in the Albers grid CRS."""
    proj = gdf.to_crs(ALBERS)
    out = {}
    for _, row in proj.iterrows():
        p = row.geometry.representative_point()
        out[row["WADMKK"]] = [round(p.x, 1), round(p.y, 1)]
    return out


def rows_rle(cells: pd.DataFrame, cols: int) -> list:
    """Encode is_riau cells as [[r, [[c0, c1], ...]], ...] column runs."""
    riau = cells[cells["is_riau"]].sort_values(["row", "col"])
    runs: list = []
    cur_row, run_start, run_end = None, None, None
    for _, c in riau.iterrows():
        r = int(c["row"])
        col = int(c["col"])
        if r != cur_row:
            if cur_row is not None:
                runs.append([cur_row, [[run_start, run_end]]])
            cur_row = r
            run_start = run_end = col
            continue
        if col == run_end + 1:
            run_end = col
        else:
            runs[-1][1].append([run_start, run_end])
            run_start = run_end = col
    if cur_row is not None:
        runs.append([cur_row, [[run_start, run_end]]])
    return runs


def outline_from_boundary(path: Path) -> list:
    """Return [ring][ [x, y], ... ] for the Albers province boundary."""
    b = gpd.read_file(path)
    rings = []
    for geom in b.geometry:
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            rings.append([[round(c[0], 1), round(c[1], 1)] for c in poly.exterior.coords])
    return rings


def main() -> None:
    cells = pd.read_csv(GRID_CSV)
    cols = int(cells["col"].max()) + 1
    rows = int(cells["row"].max()) + 1
    cell = 5000
    minx = int(cells["x_center_m"].min() - cell / 2)
    miny = int(cells["y_center_m"].min() - cell / 2)

    logger.info("fetching Riau kabupaten centroids from BIG ...")
    kab = fetch_riau_kabupaten()
    regions = albers_centroids(kab)
    logger.info("regions (%d): %s", len(regions), sorted(regions))

    grid = {
        "minx": minx,
        "miny": miny,
        "cell": float(cell),
        "cols": cols,
        "rows": rows,
        "rowsRLE": rows_rle(cells, cols),
        "regions": regions,
        "outline": outline_from_boundary(BOUNDARY_GPKG),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(grid), encoding="utf-8")
    logger.info("wrote %s (%d bbox cells, %d Riau cells)",
                OUT, cols * rows, int(cells["is_riau"].sum()))


if __name__ == "__main__":
    main()
