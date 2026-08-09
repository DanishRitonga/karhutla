"""
Mode simulasi risiko, PORT LANGSUNG dari App.jsx (hash, riskForCell,
summarize, nearestInsight) supaya angka yang keluar dari backend sama
persis dengan yang selama ini tampil di prototype front-end.

Ini dipakai sebagai FALLBACK selama model & dataset asli belum di-deploy
ke HuggingFace (lihat config.USE_REAL_MODEL). Setelah model asli siap,
fungsi predict_day() di predictor.py akan menggantikan risk_for_cell() di
sini tanpa mengubah kontrak response API.
"""
from app.grid import GridCell, load_grid_raw

RISK_LEVELS = {
    "low": {"label": "Rendah"},
    "mid": {"label": "Sedang"},
    "high": {"label": "Tinggi"},
    "vhigh": {"label": "Sangat tinggi"},
}

HOT_ANCHORS = [
    {"name": "Bengkalis", "w": 1.0},
    {"name": "Siak", "w": 0.75},
]


def _hash(a: int, b: int, c: int) -> float:
    """Port dari fungsi hash() di App.jsx (harus pakai aritmetika 32-bit unsigned)."""
    MASK = 0xFFFFFFFF
    h = (a * 374761393 + b * 668265263 + c * 2246822519) & MASK
    h = ((h ^ (h >> 13)) * 1274126177) & MASK
    h = h ^ (h >> 16)
    return (h % 10000) / 10000


def risk_for_cell(cell: GridCell, day: int) -> tuple[str, float]:
    grid = load_grid_raw()
    heat = 0.0
    for anchor in HOT_ANCHORS:
        ax, ay = grid["regions"][anchor["name"]]
        dist = ((ax - cell.x) ** 2 + (ay - cell.y) ** 2) ** 0.5 / 1000
        falloff = pow(2.718281828, -dist / 55)
        heat += anchor["w"] * falloff

    noise = _hash(cell.r, cell.c, day * 97 + 11)
    day_boost = (day - 1) * 0.015
    score = min(0.97, heat * 0.75 + noise * 0.28 + day_boost)
    score = round(score * 100) / 100

    if score > 0.72:
        level = "vhigh"
    elif score > 0.5:
        level = "high"
    elif score > 0.3:
        level = "mid"
    else:
        level = "low"
    return level, score


def summarize(rows: list[dict]) -> dict:
    """rows: list of {region, level, score}. Port dari summarize() di App.jsx."""
    total = len(rows)
    high = sum(1 for r in rows if r["level"] in ("high", "vhigh"))
    by_region: dict[str, dict] = {}
    for r in rows:
        b = by_region.setdefault(r["region"], {"high": 0, "total": 0, "sum": 0.0})
        b["total"] += 1
        b["sum"] += r["score"]
        if r["level"] in ("high", "vhigh"):
            b["high"] += 1
    ranking = sorted(
        (
            {"name": name, "avg": v["sum"] / v["total"], "high": v["high"], "total": v["total"]}
            for name, v in by_region.items()
        ),
        key=lambda x: x["avg"],
        reverse=True,
    )
    return {"total": total, "high": high, "ranking": ranking}
