"""
Entrypoint backend Karhutla Dashboard.

Jalankan lokal:
    uvicorn main:app --reload --port 8000

Endpoint tersedia di http://localhost:8000/docs (Swagger UI otomatis).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from app.routers import grid, hotspots, predictions, region_summary, explainability, ai

app = FastAPI(
    title="Karhutla Early Warning API",
    description="Backend untuk dashboard prediksi hotspot karhutla Provinsi Riau.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(grid.router)
app.include_router(hotspots.router)
app.include_router(predictions.router)
app.include_router(region_summary.router)
app.include_router(explainability.router)
app.include_router(ai.router)


@app.get("/")
def root():
    return {
        "service": "Karhutla Early Warning API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    from app.predictor import prediction_source_status

    return {
        "status": "ok",
        "mode_data": "real (HuggingFace)" if config.USE_REAL_DATA else "simulated",
        "mode_model": "real (HuggingFace)" if config.USE_REAL_MODEL else "simulated",
        # Beda dari mode_model: itu status niat/konfigurasi (HF_MODEL_REPO
        # diset atau tidak), ini status ACTUAL saat request health ini
        # diproses -- supaya "real" yang diam-diam fallback ke simulasi
        # (schema predictions.parquet rusak, atau HF tidak bisa diakses)
        # kelihatan jelas alih-alih terbaca sehat padahal tidak.
        "prediction_source": prediction_source_status(),
    }


@app.post("/api/admin/refresh-predictions")
def refresh_predictions(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    """
    Paksa backend ambil ulang predictions.parquet dari HF sekarang juga,
    tanpa menunggu TTL cache (lihat PREDICTIONS_CACHE_TTL di predictor.py).
    Panggil ini lewat webhook/cron setelah selesai upload batch inference baru.

    Kalau env var ADMIN_API_KEY diset, request harus menyertakan header
    `X-Admin-Key` yang cocok -- tanpa itu, endpoint ini bisa dipanggil siapa
    saja yang tahu URL-nya (oke untuk demo, TIDAK oke untuk deployment publik).
    """
    if config.ADMIN_API_KEY and x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(401, "X-Admin-Key tidak valid atau tidak disertakan")

    from app.predictor import clear_cache
    clear_cache()
    return {"status": "cache dikosongkan, prediksi berikutnya akan diambil ulang dari HF"}
