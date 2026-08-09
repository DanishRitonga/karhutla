"""
Test otomatis untuk semua endpoint, jalan dalam mode simulasi (default,
tidak butuh HF_DATASET_REPO/HF_MODEL_REPO/network).

Jalankan:
    pytest -v
"""
import pytest
from fastapi.testclient import TestClient

import config
from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode_model"] == "simulated"


def test_grid_meta():
    r = client.get("/api/grid/meta")
    assert r.status_code == 200
    body = r.json()
    for key in ("minx", "miny", "cell", "cols", "rows", "outline", "regions"):
        assert key in body
    assert "Bengkalis" in body["regions"]


def test_grid_list():
    r = client.get("/api/grid")
    assert r.status_code == 200
    cells = r.json()
    assert len(cells) > 0
    first = cells[0]
    assert set(first.keys()) == {"cell_idx", "r", "c", "region", "geometry"}
    assert first["geometry"]["type"] == "Polygon"


@pytest.mark.parametrize("day", [1, 4, 7])
def test_predictions(day):
    r = client.get("/api/predictions", params={"day": day})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    row = rows[0]
    assert row["day"] == day
    assert 0.0 <= row["probability"] <= 1.0
    assert row["risk_level"] in {"low", "mid", "high", "vhigh"}
    assert {"r", "c", "x", "y"}.issubset(row.keys())


def test_predictions_invalid_day():
    r = client.get("/api/predictions", params={"day": 99})
    assert r.status_code == 422  # di luar rentang 1..7


def test_prediction_single_cell():
    cell_idx = client.get("/api/predictions", params={"day": 1}).json()[0]["cell_idx"]
    r = client.get(f"/api/predictions/{cell_idx}", params={"day": 1})
    assert r.status_code == 200
    assert r.json()["cell_idx"] == cell_idx


def test_prediction_single_cell_not_found():
    r = client.get("/api/predictions/TIDAK_ADA", params={"day": 1})
    assert r.status_code == 404


def test_region_summary():
    r = client.get("/api/region-summary", params={"day": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total_cells"] > 0
    assert body["high_risk_cells"] <= body["total_cells"]
    assert len(body["ranking"]) > 0
    assert "ai_summary" in body and len(body["ai_summary"]) > 0


def test_region_summary_detail():
    r = client.get("/api/region-summary/Bengkalis", params={"day": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["region"] == "Bengkalis"
    assert 0.0 <= body["risk_score"] <= 1.0


def test_explainability():
    cell_idx = client.get("/api/predictions", params={"day": 1}).json()[0]["cell_idx"]
    r = client.get(f"/api/explainability/{cell_idx}", params={"day": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["cell_idx"] == cell_idx
    assert body["source"] == "placeholder"
    for key in ("rainfall_anomaly_pct", "soil_moisture_pct", "peat_fraction"):
        assert key in body["factors"]


def test_explainability_not_found():
    r = client.get("/api/explainability/TIDAK_ADA", params={"day": 1})
    assert r.status_code == 404


def test_hotspots():
    r = client.get("/api/hotspots", params={"start_date": "2023-01-01", "end_date": "2023-01-03"})
    assert r.status_code == 200
    rows = r.json()
    dates = {row["date"] for row in rows}
    assert dates.issubset({"2023-01-01", "2023-01-02", "2023-01-03"})


def test_hotspots_invalid_range():
    r = client.get("/api/hotspots", params={"start_date": "2023-01-05", "end_date": "2023-01-01"})
    assert r.status_code == 400


def test_hotspots_range_too_long():
    r = client.get("/api/hotspots", params={"start_date": "2023-01-01", "end_date": "2023-06-01"})
    assert r.status_code == 400


def test_refresh_predictions_cache():
    r = client.post("/api/admin/refresh-predictions")
    assert r.status_code == 200


def test_weekly_insight():
    r = client.get("/api/weekly-insight", params={"day": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["day_range"] == "1-7"
    assert len(body["summary"]) > 0
    assert body["source"] == "template"  # tanpa OPENAI_API_KEY di env test


def test_region_summary_ai_summary_matches_weekly_insight_source():
    # ai_summary di region-summary sekarang dipasok dari layer yang sama
    r = client.get("/api/region-summary", params={"day": 2})
    assert r.status_code == 200
    assert len(r.json()["ai_summary"]) > 0


def test_ask():
    r = client.post("/api/ask", json={"question": "Kabupaten mana yang paling berisiko?", "day": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["day_range"] == "1-7"
    assert len(body["answer"]) > 0
    assert body["source"] == "template"


def test_ask_invalid_body():
    r = client.post("/api/ask", json={"question": "", "day": 3})
    assert r.status_code == 422  # question kosong ditolak validasi


def test_ask_intent_teraman():
    r = client.post("/api/ask", json={"question": "Kabupaten mana yang paling aman?", "day": 3})
    body = r.json()
    assert "risiko terendah" in body["answer"].lower()


def test_ask_intent_jumlah():
    r = client.post("/api/ask", json={"question": "Ada berapa grid risiko tinggi?", "day": 3})
    body = r.json()
    assert "grid" in body["answer"].lower()


def test_admin_refresh_open_without_key():
    # Tanpa ADMIN_API_KEY di-set di environment, endpoint tetap terbuka.
    r = client.post("/api/admin/refresh-predictions")
    assert r.status_code == 200


def test_admin_refresh_requires_key_when_configured(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "rahasia123")
    try:
        r = client.post("/api/admin/refresh-predictions")
        assert r.status_code == 401

        r = client.post("/api/admin/refresh-predictions", headers={"X-Admin-Key": "salah"})
        assert r.status_code == 401

        r = client.post("/api/admin/refresh-predictions", headers={"X-Admin-Key": "rahasia123"})
        assert r.status_code == 200
    finally:
        monkeypatch.setattr(config, "ADMIN_API_KEY", "")


def test_predict_day_cache_speeds_up_repeat_calls():
    from app.predictor import _day_cache, clear_cache
    clear_cache()
    assert 2 not in _day_cache
    client.get("/api/predictions", params={"day": 2})
    assert 2 in _day_cache  # sudah di-cache setelah dipanggil sekali
    clear_cache()
    assert _day_cache == {}


def test_real_predictions_bad_schema_falls_back_to_simulation(monkeypatch, tmp_path):
    """
    predictions.parquet dari HF dengan kolom salah (mis. 'prob' alih-alih
    'probability') tidak boleh bikin endpoint 500 -- backend harus fallback
    diam-diam ke mode simulasi.
    """
    import pandas as pd
    from app import predictor as predictor_module

    bad_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"cell_idx": ["RIAU_0_0"], "day": [1], "prob": [0.9]}).to_parquet(bad_path)

    monkeypatch.setattr(config, "USE_REAL_MODEL", True)
    monkeypatch.setattr("app.hf_loader.download_model_file", lambda filename: str(bad_path))
    predictor_module.clear_cache()

    try:
        rows = predictor_module.predict_day(1)
        assert len(rows) > 0  # tetap dapat data (mode simulasi), bukan crash
        assert predictor_module._real_predictions_df() is None
    finally:
        predictor_module.clear_cache()
        monkeypatch.setattr(config, "USE_REAL_MODEL", False)
