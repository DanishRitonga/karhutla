"""
real_grid.py
============
Loads the REAL, already-built Riau grid + peat map handed off by the
teammate working on data ingestion (`real_grid_data/`), instead of
rebuilding the grid from a boundary at runtime.

Why this exists
----------------
`real_data.py`'s `_get_grid()` normally does one of two things:
  1. fetch the official BIG boundary live (kspservices.big.go.id) and
     build the grid from it, or
  2. fall back to `riau_boundary_fallback.geojson` (a GitHub-mirrored,
     NOT-authoritative boundary) when that service isn't reachable --
     which is always the case in this sandbox (network allowlist blocks
     it), so every run so far has silently used the fallback.

The teammate's environment DID reach the real BIG service. They ran
`grid_definition.py` there and handed over its output:
  - grid_cells.csv   -- one row per bbox cell: cell_idx, row, col,
                         x_center_m, y_center_m, lon, lat, is_riau
  - grid_meta.json    -- x0, y0, cell_size_m, cols, rows, crs
  - riau_boundary_aea.gpkg -- the projected boundary polygon itself
                         (kept for plotting, not needed for the tensor
                         pipeline)
  - peat_cell.csv    -- real peat_frac / peat_depth_m per cell, from
                         BIG's Peta Fungsi Ekosistem Gambut layer
                         (replaces data.py's synthetic peat channel)

`PrebuiltGrid` below reproduces the two methods real_data.py actually
calls on a grid object (`assign_cell_idx`, plus `.rows`/`.cols`/`.cells`)
using ONLY grid_meta.json + grid_cells.csv -- no boundary geometry, no
network call, no geopandas overlay. This is safe because
`EqualAreaGrid.assign_cell_idx` is a pure affine transform (project
lon/lat -> Albers -> floor-divide by cell_size); the boundary is only
needed to *build* the grid (decide is_riau), not to *use* it, and
is_riau is already baked into grid_cells.csv.

cols=85, rows=82 here match the fallback grid's dimensions exactly (same
bbox, same 5 km cells -- only the is_riau mask differs: 3,598 real cells
vs 3,356 fallback cells, since the real BIG boundary is more precise).
So swapping grid sources is a clean drop-in: no PATCH-margin or
downstream shape changes needed anywhere else in the pipeline.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyproj


class PrebuiltGrid:
    """Drop-in replacement for grid_definition.EqualAreaGrid, built from
    a previously-saved grid_cells.csv + grid_meta.json instead of a live
    boundary fetch. Implements only what real_data.py actually uses."""

    def __init__(self, cells_csv: str, meta_json: str):
        self.cells = pd.read_csv(cells_csv).sort_values("cell_idx").reset_index(drop=True)
        meta = json.loads(open(meta_json).read())
        self.x0 = meta["x0"]
        self.y0 = meta["y0"]
        self.cell_size_m = meta["cell_size_m"]
        self.cols = meta["cols"]
        self.rows = meta["rows"]
        self.meta = meta

        expected = self.cols * self.rows
        if len(self.cells) != expected:
            raise ValueError(
                f"grid_cells.csv has {len(self.cells)} rows, expected "
                f"cols*rows={expected} from grid_meta.json -- files out of sync"
            )

        self._transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", meta["crs"], always_xy=True
        )

    def assign_cell_idx(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Same contract as EqualAreaGrid.assign_cell_idx: int64 cell_idx
        per point, -1 for points outside the bbox (dropped, not clipped)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y = self._transformer.transform(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        col = np.floor((x - self.x0) / self.cell_size_m).astype(np.int64)
        row = np.floor((y - self.y0) / self.cell_size_m).astype(np.int64)
        valid = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        idx = np.full(len(lon), -1, dtype=np.int64)
        idx[valid] = row[valid] * self.cols + col[valid]
        return idx


def try_load_prebuilt_grid(data_dir: str) -> "PrebuiltGrid | None":
    """Returns a PrebuiltGrid if grid_cells.csv + grid_meta.json exist in
    data_dir, else None (caller should fall back to the old boundary path)."""
    cells_csv = os.path.join(data_dir, "grid_cells.csv")
    meta_json = os.path.join(data_dir, "grid_meta.json")
    if os.path.exists(cells_csv) and os.path.exists(meta_json):
        return PrebuiltGrid(cells_csv, meta_json)
    return None


def load_peat_depth_grid(peat_csv: str, grid_rows: int, grid_cols: int) -> tuple[np.ndarray, float]:
    """Loads real peat_depth_m from peat_cell.csv into a [grid_h, grid_w]
    array aligned to (row, col), for broadcasting into the static peat
    channel. Returns (array, coverage_fraction) where coverage_fraction
    is the fraction of Riau cells that have peat_frac > 0 (informational,
    not a data-quality flag -- most of Riau genuinely isn't peatland).
    """
    df = pd.read_csv(peat_csv)
    expected = grid_rows * grid_cols
    if len(df) != expected:
        raise ValueError(
            f"peat_cell.csv has {len(df)} rows, expected rows*cols={expected} "
            f"-- does this peat file match the grid you're using?"
        )
    depth = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    depth[df["row"].values, df["col"].values] = df["peat_depth_m"].values.astype(np.float32)

    riau = df[df["is_riau"]]
    coverage = float((riau["peat_frac"] > 0).mean()) if len(riau) else 0.0
    return depth, coverage
