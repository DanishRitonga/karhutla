from fastapi import APIRouter, HTTPException, Query
from app.predictor import predict_day

router = APIRouter(tags=["predictions"])


@router.get("/api/predictions")
def get_predictions(day: int = Query(1, ge=1, le=7, description="Horizon prediksi (t+1 .. t+7)")):
    """
    Layer utama dashboard: probabilitas risiko kebakaran per cell untuk
    horizon +N hari. Sumbernya otomatis beralih ke model asli begitu
    HF_MODEL_REPO diset (lihat app/predictor.py) -- kontrak response di
    endpoint ini TIDAK berubah, jadi frontend tidak perlu diubah.
    """
    rows = predict_day(day)
    return [
        {
            "cell_idx": r["id"],
            "region": r["region"],
            "day": day,
            "probability": r["score"],
            "risk_level": r["level"],
            # posisi disertakan supaya frontend bisa render peta dari satu
            # fetch ini saja, tanpa perlu gabung dengan /api/grid terpisah
            "r": r["r"],
            "c": r["c"],
            "x": r["x"],
            "y": r["y"],
        }
        for r in rows
    ]


@router.get("/api/predictions/{cell_idx}")
def get_prediction_for_cell(cell_idx: str, day: int = Query(1, ge=1, le=7)):
    rows = predict_day(day)
    for r in rows:
        if r["id"] == cell_idx:
            return {
                "cell_idx": r["id"], "region": r["region"], "day": day,
                "probability": r["score"], "risk_level": r["level"],
                "r": r["r"], "c": r["c"], "x": r["x"], "y": r["y"],
            }
    raise HTTPException(404, f"cell_idx '{cell_idx}' tidak ditemukan")
