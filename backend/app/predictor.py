"""
Titik tunggal yang dipanggil router untuk dapat prediksi risiko per hari.
Router TIDAK PERNAH tahu apakah datanya simulasi atau model asli -- ini
yang membuat swap ke model asli nanti tidak perlu ubah kode router sama
sekali, cukup set env var HF_MODEL_REPO / HF_DATASET_REPO lalu isi
`_load_real_predictions()`.
"""
import logging
import os
import time

import config
import pandas as pd
from app.grid import decode_cells
from app.simulate import risk_for_cell

logger = logging.getLogger("karhutla.predictor")

# Berapa lama predictions.parquet dari HF di-cache di memori sebelum
# dicek ulang. Model/dataset dikelola pihak lain dan bisa di-refresh
# kapan saja di HF Hub tanpa restart backend -- TTL ini yang membuat
# backend otomatis mengambil versi terbaru tanpa perlu deploy ulang.
_CACHE_TTL_SECONDS = int(os.getenv("PREDICTIONS_CACHE_TTL", "600"))  # default 10 menit

_REQUIRED_PREDICTION_COLUMNS = {"cell_idx", "day", "probability"}

_cache: dict = {"df": None, "loaded_at": 0.0}

# Alasan percobaan load TERAKHIR gagal (None = percobaan terakhir sukses,
# atau belum pernah dicoba). Dipakai prediction_source_status() untuk
# /api/health -- terpisah dari config.USE_REAL_MODEL yang cuma menandakan
# NIAT (HF_MODEL_REPO sudah diisi), bukan apakah data itu benar-benar
# berhasil dimuat saat ini.
_last_load_error: str | None = None

# Cache hasil predict_day() PER HARI. Endpoint seperti region-summary detail
# manggil predict_day(1), predict_day(7), dan predict_day(day) sekaligus
# untuk satu request -- tanpa cache ini, tiap panggilan mengulang seluruh
# loop di semua cell dari nol. Di-invalidate otomatis setiap kali
# predictions.parquet reload (lihat _real_predictions_df) atau lewat
# clear_cache() (dipanggil dari POST /api/admin/refresh-predictions).
_day_cache: dict[int, list[dict]] = {}


def clear_cache() -> None:
    """Kosongkan semua cache prediksi (dipanggil dari endpoint admin)."""
    _cache["df"] = None
    _cache["loaded_at"] = 0.0
    _day_cache.clear()


def _real_predictions_df() -> "pd.DataFrame | None":
    """Load (dan cache dengan TTL) tabel prediksi asli (cell_idx/day/probability).

    Sumber dibaca dari dua tempat, preferensi lokal dulu:
      1. config.LOCAL_PREDICTIONS_PATH — parquet yang dibuat prepare.sh di
         container start (self-sufficient, tanpa unduhan HF saat runtime).
      2. Fallback: download predictions.parquet dari HF model repo.
    """
    global _last_load_error
    if not config.USE_REAL_MODEL:
        return None

    now = time.time()
    if _cache["df"] is None or (now - _cache["loaded_at"]) > _CACHE_TTL_SECONDS:
        try:
            if config.LOCAL_PREDICTIONS_PATH.is_file():
                df = pd.read_parquet(config.LOCAL_PREDICTIONS_PATH)
            else:
                from app import hf_loader
                # Sesuaikan nama file dengan yang di-upload ke HF model repo
                # (lihat model_training/03_generate_predictions.py di pipeline training).
                path = hf_loader.download_model_file("predictions.parquet")
                df = pd.read_parquet(path)
        except Exception as exc:
            # File lokal rusak / network/auth/file-not-found dari HF --
            # fallback ke simulasi dengan alasan yang sama seperti schema rusak
            # di bawah, jangan biarkan endpoint prediksi 500 karena HF sedang
            # bermasalah.
            logger.error(
                "Gagal mengambil/membaca predictions.parquet: %s. "
                "Fallback ke mode simulasi.", exc,
            )
            _cache["df"] = None
            _cache["loaded_at"] = now
            _day_cache.clear()
            _last_load_error = f"gagal baca prediksi ({type(exc).__name__}): {exc}"
            return None

        # Validasi schema: kalau pipeline training berubah nama kolom (mis.
        # "prob" alih-alih "probability"), lebih baik fallback ke simulasi
        # dengan log jelas daripada bikin SEMUA endpoint prediksi 500 karena
        # KeyError di tengah request orang lain. TTL tetap di-update supaya
        # backend tidak mencoba download ulang tiap request selama TTL,
        # tapi tetap otomatis coba lagi setelah TTL habis (kalau-kalau
        # pipeline sudah diperbaiki & re-upload).
        missing = _REQUIRED_PREDICTION_COLUMNS - set(df.columns)
        if missing:
            logger.error(
                "predictions.parquet dari HF tidak punya kolom wajib %s (kolom yang ada: %s). "
                "Fallback ke mode simulasi sampai schema diperbaiki.",
                sorted(missing), list(df.columns),
            )
            _cache["df"] = None
            _cache["loaded_at"] = now
            _day_cache.clear()
            _last_load_error = f"kolom wajib hilang: {sorted(missing)}"
            return None

        # Validasi: kalau pipeline training punya bug dan menghasilkan baris
        # duplikat untuk (cell_idx, day) yang sama, `.set_index("cell_idx")`
        # di bawah akan mengembalikan Series alih-alih scalar saat di-lookup
        # -> crash di float(...). Lebih aman drop duplikat & log peringatan
        # daripada bikin seluruh endpoint prediksi down karena satu baris kotor.
        dup_mask = df.duplicated(subset=["cell_idx", "day"], keep=False)
        if dup_mask.any():
            logger.warning(
                "predictions.parquet punya %d baris duplikat (cell_idx+day) -- "
                "mengambil baris pertama saja per pasangan",
                int(dup_mask.sum()),
            )
            df = df.drop_duplicates(subset=["cell_idx", "day"], keep="first")

        _cache["df"] = df
        _cache["loaded_at"] = now
        _day_cache.clear()  # data sumber berubah -> cache per-hari basi
        _last_load_error = None  # load sukses, hapus alasan gagal sebelumnya

    return _cache["df"]


def prediction_source_status() -> str:
    """
    Status sumber prediksi yang SEDANG dipakai, dipanggil dari /api/health.

    Beda dengan config.USE_REAL_MODEL (cuma menandakan NIAT -- HF_MODEL_REPO
    sudah diisi) -- fungsi ini benar-benar mencoba (lewat cache TTL yang sama
    dengan predict_day(), jadi tidak menambah beban fetch) dan melaporkan
    apakah data model asli ACTUALLY berhasil dimuat saat ini. Ini yang
    membedakan "real" (terpasang & terpakai) dari "simulation_fallback"
    (terpasang tapi diam-diam jatuh ke simulasi karena schema rusak / HF
    tidak bisa diakses) -- dua kondisi yang sebelumnya sama-sama terbaca
    "real" di /api/health karena cuma mengecek config.
    """
    if not config.USE_REAL_MODEL:
        return "disabled"  # simulasi memang mode yang dipilih, bukan fallback
    if _real_predictions_df() is not None:
        return "real"
    return f"simulation_fallback ({_last_load_error})" if _last_load_error else "simulation_fallback"


def predict_day(day: int) -> list[dict]:
    """Return list of {id, r, c, x, y, region, level, score} untuk 1 horizon hari."""
    if day in _day_cache:
        return _day_cache[day]

    real_df = _real_predictions_df()
    cells = decode_cells()

    if real_df is not None:
        # ── Mode model asli ──────────────────────────────────────────
        sub = real_df[real_df["day"] == day].set_index("cell_idx")
        rows = []
        for cell in cells:
            if cell.id not in sub.index:
                continue
            score = float(sub.loc[cell.id, "probability"])
            level = _level_from_score(score)
            rows.append({
                "id": cell.id, "r": cell.r, "c": cell.c,
                "x": cell.x, "y": cell.y, "region": cell.region,
                "level": level, "score": round(score, 2),
            })
        _day_cache[day] = rows
        return rows

    # ── Mode simulasi (default, sampai model asli tersedia) ─────────
    rows = []
    for cell in cells:
        level, score = risk_for_cell(cell, day)
        rows.append({
            "id": cell.id, "r": cell.r, "c": cell.c,
            "x": cell.x, "y": cell.y, "region": cell.region,
            "level": level, "score": score,
        })
    _day_cache[day] = rows
    return rows


def _level_from_score(score: float) -> str:
    if score > 0.72:
        return "vhigh"
    if score > 0.5:
        return "high"
    if score > 0.3:
        return "mid"
    return "low"
