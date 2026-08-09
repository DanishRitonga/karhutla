from fastapi import APIRouter, HTTPException, Query
import config
from app.grid import decode_cells
from app.simulate import _hash, risk_for_cell

router = APIRouter(tags=["explainability"])


@router.get("/api/explainability/{cell_idx}")
def explainability(cell_idx: str, day: int = Query(1, ge=1, le=7)):
    """
    Faktor pendorong risiko untuk satu cell (dipakai panel 'Evidence' /
    'Mengapa wilayah ini berisiko?' di Explorer).

    Mode saat ini: PLACEHOLDER -- angka diturunkan deterministik dari
    cell_idx (konsisten tiap request, TAPI tidak dihitung dari hubungan
    sebab-akibat apa pun). Response selalu menyertakan `"source":
    "placeholder"` supaya frontend/penguji tidak salah kira ini hasil
    model asli. Setelah model asli tersedia, ganti _simulated_factors()
    dengan pemanggilan SHAP/feature-importance dari model yang di-load
    lewat app/hf_loader.download_model_file(...), dan source jadi "model".
    """
    cells = {c.id: c for c in decode_cells()}
    if cell_idx not in cells:
        raise HTTPException(404, f"cell_idx '{cell_idx}' tidak ditemukan")
    cell = cells[cell_idx]

    if config.USE_REAL_MODEL:
        factors, narrative, source = _real_factors(cell, day)
    else:
        factors, narrative, source = _simulated_factors(cell, day)

    return {
        "cell_idx": cell_idx,
        "region": cell.region,
        "factors": factors,
        "narrative": narrative,
        "source": source,
    }


def _simulated_factors(cell, day: int):
    level, score = risk_for_cell(cell, day)
    # Variasi kecil per-cell dari hash supaya tiap cell tidak identik
    rain_anom = -round((0.3 + _hash(cell.r, cell.c, 1) * 0.5) * 100)          # ~ -30% .. -80%
    soil_pct = round(10 + _hash(cell.r, cell.c, 2) * 25)                      # ~ 10 .. 35
    peat_frac = round(0.2 + _hash(cell.r, cell.c, 3) * 0.6, 2)                # ~ 0.2 .. 0.8

    narrative = (
        f"Risiko { {'low':'rendah','mid':'sedang','high':'tinggi','vhigh':'sangat tinggi'}[level] } "
        f"di {cell.region} dipengaruhi oleh curah hujan rendah selama sepuluh hari terakhir "
        f"dan dominasi lahan gambut di area ini."
    )
    factors = {
        "rainfall_anomaly_pct": rain_anom,
        "soil_moisture_pct": soil_pct,
        "peat_fraction": peat_frac,
    }
    # "placeholder", BUKAN "simulated" -- beda arti penting: angka prediksi
    # risiko (predict_day) memang didesain sebagai simulasi yang konsisten
    # dengan hash deterministik, tapi angka di sini TIDAK dihitung dari
    # hubungan sebab-akibat apa pun -- murni angka acak-terkontrol supaya UI
    # ada isinya. Frontend/penguji tidak boleh mengira ini hasil SHAP asli.
    return factors, narrative, "placeholder"


def _real_factors(cell, day: int):
    """
    Placeholder untuk SHAP/feature-importance dari model asli. Contoh alur:

        from app import hf_loader
        import joblib
        model_path = hf_loader.download_model_file("model.pkl")
        model = joblib.load(model_path)
        shap_values = explainer(model, features_for(cell, day))
        return shap_values, narrative_from(shap_values), "model"

    Sampai model asli siap, fungsi ini fallback ke simulasi supaya endpoint
    tidak pernah error.
    """
    return _simulated_factors(cell, day)
