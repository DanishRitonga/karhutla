"""
Konteks cuaca (prakiraan 7 hari per-kabupaten) untuk dimasukkan ke prompt LLM.

Menghasilkan teks ringkas berbahasa Indonesia dari output model prakiraan
cuaca per-sel (weather_forecast.parquet + weather_wind_dir.parquet di HF model
repo), diagregasi ke rata-rata per kabupaten agar mudah dibaca agent.

Kelas ini TIDAK mengubah perilaku endpoint -- hanya menambah konteks ke prompt
LLM di app/ai_summary.py. Semua kegagalan (file hilang / schema berubah /
network) dikembalikan sebagai string kosong sehingga fitur cuaca tidak pernah
mematikan RAG.
"""
from __future__ import annotations

import logging
import os
import time

import config
import pandas as pd
from app.grid import decode_cells

logger = logging.getLogger("karhutla.weather")

_TTL_SECONDS = int(os.getenv("WEATHER_CACHE_TTL", "600"))

_REQUIRED_COLUMNS = {"cell_idx", "day", "channel", "value"}

# Nama channel yang dibaca dari parquet + unit untuk ditampilkan ke LLM.
_CHANNEL_LABELS = {
    "temp_c": "suhu",
    "rh_pct": "kelembaban",
    "wind_ms": "angin",
    "precip_mm": "hujan",
    "soil_moisture_pct": "kelembaban tanah",
    "solar_wm2": "radiasi matahari",
}

_cache: dict = {"forecast": None, "wind_dir": None, "loaded_at": 0.0}


def clear_cache() -> None:
    _cache["forecast"] = None
    _cache["wind_dir"] = None
    _cache["loaded_at"] = 0.0


def _weather_forecast_df() -> "pd.DataFrame | None":
    """Load (dengan TTL) weather_forecast.parquet dari HF model repo."""
    if not config.USE_REAL_MODEL:
        return None
    now = time.time()
    if _cache["forecast"] is None or (now - _cache["loaded_at"]) > _TTL_SECONDS:
        from app import hf_loader
        try:
            path = hf_loader.download_model_file("weather_forecast.parquet")
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Gagal memuat weather_forecast.parquet: %s", exc)
            _cache["forecast"] = None
            _cache["loaded_at"] = now
            return None
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning("weather_forecast.parquet tidak punya kolom wajib %s", sorted(missing))
            _cache["forecast"] = None
            _cache["loaded_at"] = now
            return None
        _cache["forecast"] = df
        _cache["loaded_at"] = now
    return _cache["forecast"]


def _wind_dir_df() -> "pd.DataFrame | None":
    """Load (dengan TTL) weather_wind_dir.parquet (opsional)."""
    if not config.USE_REAL_MODEL:
        return None
    now = time.time()
    if _cache["wind_dir"] is None or (now - _cache["loaded_at"]) > _TTL_SECONDS:
        from app import hf_loader
        try:
            path = hf_loader.download_model_file("weather_wind_dir.parquet")
            df = pd.read_parquet(path)
        except Exception:
            _cache["wind_dir"] = None
            _cache["loaded_at"] = now
            return None
        if not {"cell_idx", "wind_dir"}.issubset(set(df.columns)):
            _cache["wind_dir"] = None
            _cache["loaded_at"] = now
            return None
        _cache["wind_dir"] = df
        _cache["loaded_at"] = now
    return _cache["wind_dir"]


def build_weather_context() -> str:
    """Konteks cuaca 7 hari per-kabupaten untuk prompt LLM (string kosong jika gagal)."""
    df = _weather_forecast_df()
    if df is None or df.empty:
        return ""

    # cell.id ("RIAU_r_c") -> region
    cell_to_region = {cell.id: cell.region for cell in decode_cells()}
    df = df[df["cell_idx"].isin(cell_to_region)]
    if df.empty:
        return ""

    df["region"] = df["cell_idx"].map(cell_to_region)
    # Rata-rata 7 hari x semua sel per (region, channel)
    agg = df.groupby(["region", "channel"])["value"].mean().unstack("channel")
    agg = agg[[c for c in _CHANNEL_LABELS if c in agg.columns]]

    # Arah angin dominan per region (mode), opsional
    wd = _wind_dir_df()
    wind_mode: dict[str, str] = {}
    if wd is not None and not wd.empty:
        wd = wd[wd["cell_idx"].isin(cell_to_region)]
        if not wd.empty:
            wd = wd.copy()
            wd["region"] = wd["cell_idx"].map(cell_to_region)
            wind_mode = wd.groupby("region")["wind_dir"].agg(lambda s: s.mode().iloc[0]).to_dict()

    lines = ["Prakiraan cuaca per kabupaten (rata-rata 7 hari ke depan):"]
    for region in sorted(agg.index):
        parts = []
        for ch in agg.columns:
            label = _CHANNEL_LABELS.get(ch, ch)
            parts.append(f"{label} {agg.loc[region, ch]:.1f}")
        if region in wind_mode:
            parts.append(f"arah angin dominan {wind_mode[region]}")
        lines.append(f"- {region}: {', '.join(parts)}")
    return "\n".join(lines)
