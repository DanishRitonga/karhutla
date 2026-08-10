"""
Loader grid + helper geometri, port 1:1 dari logika di App.jsx
(decodeRLE, cellCenter, nearestRegion) supaya hasil backend identik dengan
yang sudah dirender di prototype.

Begitu grid_cells.geojson asli (hasil grid_cells.zip) sudah dipakai, ganti
`load_grid()` untuk baca file itu; struktur GridCell di bawah cukup diisi
dari kolom geojson (cell_idx, geometry) alih-alih r/c.
"""
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import config

logger = logging.getLogger("karhutla.grid")


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
    """Kabupaten dengan centroid terdekat.

    Ini pendekatan Voronoi, bukan batas administratif: wilayahnya cenderung
    sama besar padahal luas kabupaten di Riau timpang jauh, sehingga kota kecil
    menyerap ratusan cell milik tetangganya. Dipertahankan sebagai jalur mundur;
    penentuan yang benar dibaca dari regionOf (lihat decode_cells).
    """
    regions = grid.get("regions") or {}
    if not regions:
        # Grid yang sangat lama bisa tidak punya "regions" sama sekali.
        # Kembalikan penanda eksplisit alih-alih KeyError yang membuat seluruh
        # backend gagal start.
        return "Tidak diketahui"
    best_name, best_d = None, float("inf")
    for name, (rx, ry) in regions.items():
        d = (rx - x) ** 2 + (ry - y) ** 2
        if d < best_d:
            best_d, best_name = d, name
    return best_name


@lru_cache
def decode_cells() -> tuple[GridCell, ...]:
    """Decode rowsRLE -> daftar GridCell, hasilnya di-cache karena grid statis.

    Dua pengaman di sini:

    1. `seen` menolak (r, c) ganda. rowsRLE yang cacat bisa memuat rentang yang
       saling tumpang tindih dalam satu baris, dan tanpa ini /api/predictions
       mengirim cell_idx yang sama dua kali -- jumlah cell tetap terlihat benar
       sementara sebagian cell justru hilang. Grid saat ini sudah bersih; ini
       jaring pengaman supaya regresi serupa tidak lolos diam-diam.

    2. Kabupaten diambil dari tabel point-in-polygon (regionNames + regionOf)
       kalau generator sudah menyediakannya, dan baru jatuh ke nearest_region()
       kalau belum.
    """
    grid = load_grid_raw()

    names = grid.get("regionNames")
    region_of = grid.get("regionOf")
    use_lookup = bool(names and region_of)
    if not use_lookup:
        logger.warning(
            "grid_data.json tanpa regionOf; kabupaten ditebak dari centroid "
            "terdekat (Voronoi, tidak akurat di batas wilayah)."
        )

    cells: list[GridCell] = []
    seen: set[tuple[int, int]] = set()
    duplicates = 0

    for r, ranges in grid["rowsRLE"]:
        for a, b in ranges:
            for c in range(a, b + 1):
                if (r, c) in seen:
                    duplicates += 1
                    continue
                seen.add((r, c))

                x, y = cell_center(grid, r, c)
                if use_lookup:
                    i = region_of.get(f"{r}_{c}")
                    region = names[i] if i is not None else nearest_region(grid, x, y)
                else:
                    region = nearest_region(grid, x, y)

                cells.append(GridCell(id=f"RIAU_{r}_{c}", r=r, c=c, x=x, y=y, region=region))

    if duplicates:
        logger.warning(
            "rowsRLE memuat %d cell duplikat (rentang tumpang tindih di %s); "
            "diabaikan, tapi grid-nya perlu diregenerate.",
            duplicates, config.GRID_PATH,
        )

    return tuple(cells)


def _rdp(points: list, eps: float) -> list:
    """Ramer-Douglas-Peucker, versi iteratif.

    Iteratif, bukan rekursif: satu ring outline berisi 78 ribu titik, dan versi
    rekursif akan menabrak batas rekursi Python.
    """
    n = len(points)
    if n < 3:
        return list(points)

    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx, dy = x2 - x1, y2 - y1
        denom = (dx * dx + dy * dy) ** 0.5

        best_d, best_k = -1.0, -1
        for k in range(i + 1, j):
            x0, y0 = points[k]
            if denom:
                d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
            else:
                d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            if d > best_d:
                best_d, best_k = d, k

        if best_d > eps:
            keep[best_k] = True
            stack.append((i, best_k))
            stack.append((best_k, j))

    return [p for p, k in zip(points, keep) if k]


# Toleransi penyederhanaan outline, dalam meter. Peta dirender ke viewBox
# selebar 640 px untuk provinsi selebar ~425 km, jadi 1 piksel = ~664 m.
# 150 m berarti pergeseran maksimum seperempat piksel -- tidak terlihat mata,
# tapi memangkas 163.369 titik menjadi ~1.900 dan payload /api/grid/meta dari
# 3,98 MB jadi 0,05 MB.
OUTLINE_SIMPLIFY_M = float(os.getenv("OUTLINE_SIMPLIFY_M", "150"))


@lru_cache
def simplified_outline() -> tuple:
    """Outline provinsi yang sudah disederhanakan, dihitung sekali lalu di-cache."""
    grid = load_grid_raw()
    rings = grid.get("outline") or []
    if OUTLINE_SIMPLIFY_M <= 0:
        return tuple(tuple(tuple(p) for p in ring) for ring in rings)

    before = sum(len(r) for r in rings)
    simplified = [_rdp(ring, OUTLINE_SIMPLIFY_M) for ring in rings]
    after = sum(len(r) for r in simplified)
    logger.info(
        "outline disederhanakan: %d -> %d titik (toleransi %.0f m)",
        before, after, OUTLINE_SIMPLIFY_M,
    )
    return tuple(tuple(tuple(p) for p in ring) for ring in simplified)


def cell_bbox(grid: dict, r: int, c: int):
    """Kotak persegi cell dalam koordinat proyeksi, dipakai untuk geometry di /api/grid."""
    cell = grid["cell"]
    x0 = grid["minx"] + c * cell
    y0 = grid["miny"] + r * cell
    return [
        [x0, y0], [x0 + cell, y0], [x0 + cell, y0 + cell], [x0, y0 + cell], [x0, y0],
    ]
