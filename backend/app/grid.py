"""
Loader grid + helper geometri, port 1:1 dari logika di App.jsx
(decodeRLE, cellCenter, nearestRegion) supaya hasil backend identik dengan
yang sudah dirender di prototype.

Begitu grid_cells.geojson asli (hasil grid_cells.zip) sudah dipakai, ganti
`load_grid()` untuk baca file itu; struktur GridCell di bawah cukup diisi
dari kolom geojson (cell_idx, geometry) alih-alih r/c.
"""
import json
from dataclasses import dataclass
from functools import lru_cache

import config


@dataclass(frozen=True)
class GridCell:
    id: str          # "RIAU_{r}_{c}"
    r: int
    c: int
    x: float          # koordinat proyeksi (meter), pusat cell
    y: float
    region: str


@lru_cache
def load_grid_raw() -> dict:
    with open(config.GRID_PATH, encoding="utf-8") as f:
        return json.load(f)


def cell_center(grid: dict, r: int, c: int):
    x = grid["minx"] + (c + 0.5) * grid["cell"]
    y = grid["miny"] + (r + 0.5) * grid["cell"]
    return x, y


def nearest_region(grid: dict, x: float, y: float) -> str:
    best_name, best_d = None, float("inf")
    for name, (rx, ry) in grid["regions"].items():
        d = (rx - x) ** 2 + (ry - y) ** 2
        if d < best_d:
            best_d, best_name = d, name
    return best_name


@lru_cache
def decode_cells() -> tuple[GridCell, ...]:
    """Decode rowsRLE -> daftar GridCell, hasilnya di-cache karena grid statis."""
    grid = load_grid_raw()
    cells = []
    for r, ranges in grid["rowsRLE"]:
        for a, b in ranges:
            for c in range(a, b + 1):
                x, y = cell_center(grid, r, c)
                region = nearest_region(grid, x, y)
                cells.append(GridCell(id=f"RIAU_{r}_{c}", r=r, c=c, x=x, y=y, region=region))
    return tuple(cells)


def cell_bbox(grid: dict, r: int, c: int):
    """Kotak persegi cell dalam koordinat proyeksi, dipakai untuk geometry di /api/grid."""
    cell = grid["cell"]
    x0 = grid["minx"] + c * cell
    y0 = grid["miny"] + r * cell
    return [
        [x0, y0], [x0 + cell, y0], [x0 + cell, y0 + cell], [x0, y0 + cell], [x0, y0],
    ]
