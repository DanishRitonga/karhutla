from fastapi import APIRouter
from app.grid import decode_cells, load_grid_raw, cell_bbox, simplified_outline
from app.schemas import GridCellOut

router = APIRouter(tags=["grid"])


@router.get("/api/grid/meta")
def get_grid_meta():
    """
    Metadata grid untuk keperluan render peta di frontend: batas koordinat,
    ukuran cell, dimensi grid, outline provinsi (Riau), dan titik centroid
    tiap kabupaten. Dipanggil sekali saat dashboard dimuat (nilainya statis).
    """
    grid = load_grid_raw()
    return {
        "minx": grid["minx"],
        "miny": grid["miny"],
        "cell": grid["cell"],
        "cols": grid["cols"],
        "rows": grid["rows"],
        # Outline disederhanakan (lihat app.grid.simplified_outline): outline
        # mentah berisi 163.369 titik = 3,98 MB, diunduh ulang tiap kali
        # dashboard dibuka, untuk peta selebar 640 px.
        "outline": simplified_outline(),
        "regions": grid["regions"],
    }


@router.get("/api/grid", response_model=list[GridCellOut])
def get_grid():
    """
    Grid spasial Riau (resolusi 5km). Sumber saat ini: grid yang sudah
    tertanam di prototype front-end. Setelah grid_cells.geojson asli dari
    pipeline pengolahan data tersedia, ganti isi fungsi ini untuk membaca
    file itu langsung (geometry-nya jadi polygon asli, bukan bbox kotak).
    """
    grid = load_grid_raw()
    cells = decode_cells()
    out = []
    for cell in cells:
        coords = cell_bbox(grid, cell.r, cell.c)
        out.append({
            "cell_idx": cell.id,
            "r": cell.r,
            "c": cell.c,
            "region": cell.region,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return out
