# Karhutla Riau — Multimodal Peatland Fire Early-Warning System

Sistem peringatan dini kebakaran gambut (karhutla) untuk Provinsi Riau: memadukan **5 sensor/data source** (ERA5-Land cuaca, CHIRPS curah hujan, Sentinel-1 SAR, Dynamic World tutupan lahan, peta kedalaman gambut) menjadi grid 5 km, lalu memprediksi risiko kebakaran **7 hari ke depan** per sel grid — dengan model LightGBM (operasional, PR-AUC 0.712) serta pembanding ConvLSTM / Temporal Transformer / Random Forest / XGBoost / ensemble.

Pipeline end-to-end: **ingest → tensor assembly → training → inference → dashboard** (frontend React + backend FastAPI + RAG regulasi + prediksi cuaca 7-hari). Data, model, dan app di-hosting di HuggingFace; kode di GitHub.

| Komponen | Repo | Status |
|---|---|---|
| Kode | `github.com/DanishRitonga/karhutla` | branch `dnsh` |
| Data mentah + tensor | HF dataset `danishritonga/karhutla` | `raw/` (8 sumber) + `tensors/` |
| Model + artefak | HF model `danishritonga/karhutla` | checkpoint + predictions/weather parquet |
| Aplikasi live | HF Space `danishritonga/karhutla` | https://danishritonga-karhutla.hf.space/ |

---

## 0. Quickstart (pakai artefak jadi — direkomendasikan)

Cara tercepat untuk menjalankan model tanpa mengulang ingest puluhan jam:

```bash
# 1. Clone + setup
git clone https://github.com/DanishRitonga/karhutla.git
cd karhutla && git checkout dnsh
uv sync --python 3.12
uv run --python 3.12 python -c "import model.data as d; d._ensure_tensors_local(d.Path('data/output/tensors'))"
```

Tensor (data.npy 1.1 GB, labels.npy, meta.json) otomatis diunduh dari HF dataset repo
kalau belum ada di `data/output/tensors/`.

**Cek model sudah bisa dipakai:**

```bash
# Risk map PNG untuk satu tanggal (checkpoint LightGBM operasional)
uv run --python 3.12 python model/risk_map.py --date 2023-09-25
# → data/output/maps/risk_2023-09-25.png

# Reproduksi tabel perbandingan model (semua model, regime operasional)
uv run --python 3.12 python model/train.py --regime operational --no-dl \
  --n-train 20000 --n-val 5000 --n-test 10000 --out-dir outputs_tabonly
```

Hasil eksperimen lengkap & analisis ada di [`docs/report/`](docs/report/) (test1–test5,
model_selection, caveats).

---

## 1. Struktur Repositori

```
.
├── data/
│   ├── grid/                # grid_definition.py (grid 5 km Albers), plot_grid_feature.py
│   ├── ingest/              # _gee.py + 6 script ingest per sumber
│   ├── loader/              # tensor_assembly.py (CSV → data.npy [D,H,W,23])
│   └── README.md            # deskripsi data + skema kolom tiap sumber
├── model/                   # train.py, train_weather.py, risk_map.py, models.py, data.py
├── scripts/                 # generate_predictions.py, generate_weather.py,
│                            # generate_grid_data.py, deploy_space.sh
├── backend/                 # FastAPI + RAG (mirror HF Space) — self-contained
├── web/                     # Frontend React + Vite + Tailwind
├── docs/
│   ├── paper/               # Concept paper (NIPS 2015 template), literatur review
│   └── report/              # Laporan eksperimen (test1–test5), model_selection, caveats
├── TRAINING_AGENT.md        # Instruksi replikasi training di device terpisah
├── design_log_karhutla_riau*.md  # Design log (perancangan + leakage audit L1–L10)
└── pyproject.toml           # uv project, Python ≥3.12
```

---

## 2. Setup & Dependensi

**Persyaratan:** Python 3.12, [uv](https://docs.astral.sh/uv/), akun Google Earth Engine
(khusus full ingest).

```bash
uv sync --python 3.12
```

Semua dependensi (earthengine-api, geopandas, lightgbm, xgboost, torch, shap, pyarrow,
scikit-learn, dll.) dideklarasikan di `pyproject.toml`.

**Auth GEE** (hanya dibutuhkan untuk bagian ingest §3):

```bash
uv run earthengine authenticate
# cloud project ID: practical-day-489508-u5
```

**Auth HuggingFace** (untuk unduh tensor/artefak otomatis):

```bash
uv run huggingface-cli login
```

---

## 3. Data Ingestion (full pipeline — opsional)

> Bagian ini untuk **rekonstruksi dari nol**. Untuk reproduksi cepat, lompat ke §0/§4
> (tensor & model sudah disediakan). Estimasi waktu total: beberapa jam GEE pull +
> ~1.8 GB download FIRMS.

<details>
<summary>Klik untuk membuka panduan full ingest</summary>

Semua script berjalan dari root proyek. Grid 5 km (6,970 sel bbox, 3,598 sel Riau,
EPSG:9470 Albers) sudah tetap di `data/output/grid/`; seluruh sumber di-join lewat
`cell_idx`.

### 3.1 Grid (sudah disediakan, opsional regenerate)

```bash
uv run --python 3.12 python data/grid/grid_definition.py
# → data/output/grid/{grid_cells.csv, grid_meta.json, riau_boundary_aea.gpkg, riau_grid.png}
```

### 3.2 CHIRPS v3 SAT — curah hujan harian (GEE)

```bash
uv run --python 3.12 python data/ingest/chirpsv3.py \
  --start 2019-01-01 --end 2023-12-31 --project practical-day-489508-u5
# → data/output/chirpsv3/chirps_v3sat_YYYYMM.csv (60 file)
```

### 3.3 Sentinel-1 — backscatter SAR (GEE, download + gap-fill)

```bash
uv run --python 3.12 python data/ingest/sentinel1.py \
  --start 2019-01-01 --end 2023-12-31 --project practical-day-489508-u5
# → data/output/sentinel1/s1_{ORBIT}_YYYYMM.csv + sentinel1_filled/ (120 file)
# --download saja / --fill saja untuk tahap terpisah; --max-gap 14 default
```

### 3.4 ERA5-Land — cuaca & soil moisture (GEE, hourly → daily)

```bash
uv run --python 3.12 python data/ingest/era5land.py \
  --start 2019-01-01 --end 2023-12-31 --project practical-day-489508-u5
# → data/output/era5land/era5land_YYYYMM.csv (60 file)
```

> Catatan: band flux (`total_precipitation`, `surface_solar_radiation_downwards`) di
> GEE HOURLY bersifat kumulatif; script menjumlahkan 24 jam dengan benar (lihat
> `FLUX_BANDS` di `era5land.py`).

### 3.5 Dynamic World — tutupan lahan (GEE)

```bash
uv run --python 3.12 python data/ingest/dynamic_world.py \
  --start 2019-01-01 --end 2023-12-31 --project practical-day-489508-u5
# → data/output/dynamic_world/dynamic_world_YYYYMM.csv (60 file)
```

### 3.6 Peta gambut (BIG ArcGIS, satu kali)

```bash
uv run --python 3.12 python data/ingest/peat.py
# → data/output/peat/peat_cell.csv (kedalaman gambut per sel, rasterisasi area-weighted)
```

### 3.7 Label VIIRS FIRMS (download langsung NASA, bukan GEE)

```bash
uv run --python 3.12 python data/ingest/viirs.py \
  --years 2019 2023 --keep-raw --raw-csv-dir real_data/viirs-snpp
# → data/output/viirs/labels_YYYY.csv (k=2, window [t+1, t+7], 6,506 sel darat)
```

Label = hotspot VIIRS 375 m (confidence nominal/high) yang persist ≥2 deteksi dalam
jendela 7 hari (`k=2`). GEE tidak dipakai karena koleksi VIIRS NRT-nya hanya mulai
2023-09; label diambil dari arsip FIRMS CSV langsung.

</details>

---

## 4. Tensor Assembly

Gabungkan semua CSV per-sumber menjadi satu tensor padat `[hari, baris, kolom, 23 channel]`:

```bash
uv run --python 3.12 python data/loader/tensor_assembly.py \
  --start 2019-01-01 --end 2023-12-31
# → data/output/tensors/{data.npy, labels.npy, meta.json}
```

Layout 23 channel (lihat `data/loader/tensor_assembly.py`):

| Ch | Sumber | Keterangan |
|---|---|---|
| 0–7 | ERA5-Land | t2m, d2m, u10, v10, swvl1, swvl2, ssr, tp |
| 8 | CHIRPS | precip mm/hari |
| 9–11 | Sentinel-1 | vv_db, vh_db, sar_available (mask) |
| 12–19 | Dynamic World | 8 kelas probabilitas (0+mask untuk sel tanpa data) |
| 20 | — | dw_available (mask) |
| 21 | Gambut | kedalaman (m) statik |
| 22 | — | hotspot_count_lag (riwayat 14 hari, dari deteksi mentah) |

Label: `-1` = bukan target (di luar Riau/margin), `0/1` = tidak/ada kebakaran
(window k=2). Split: train 2019–2021, val 2022, test 2023.

---

## 5. Model Experiments

### 5.1 Fire risk — semua model & regime

```bash
# Full suite (tabular + DL). DL butuh GPU + RAM besar; tensornya ~13 GB di RAM
# saat N=50k. Untuk device kecil pakai --no-dl.
uv run --python 3.12 python model/train.py \
  --regime operational --epochs 15 --n-train 20000 --n-val 5000 --n-test 20000 \
  --out-dir outputs
```

Model yang dievaluasi (`model/train.py`):
- **Tabular:** Persistence, Meteorological LR (ERA5-only), Logistic Regression,
  Random Forest, **LightGBM**, XGBoost, Ensemble (soft-voting RF+LGBM+XGB)
- **Spatiotemporal DL:** ConvLSTM (hidden 12,12), Temporal Transformer (ResNet-18)
- 2 regime: `env` (21 ch, tanpa riwayat kebakaran) vs `operational` (22 ch, + fire history)
- `--balance seasonal`: negative matching 1:1 musiman (Sinato & Rivas 2026), diagnostik

Output: `comparison_table_{regime}.csv`, `shap_importance_{regime}.png`,
`attention_heatmap_{regime}.png`, checkpoint `checkpoint_{regime}.json` +
`model_{key}_{regime}.joblib` per model.

**Hasil utama (test 2023, 50k sample):** LightGBM **operational** PR-AUC **0.712**,
F1 0.625, Recall 0.583, ROC-AUC 0.879, threshold 0.420. DL (ConvLSTM/Transformer)
tidak mengungguli GBDT pada data rare-event ini — bukti eksperimen di
`docs/report/test1–test5.md` & `model_selection.md`.

### 5.2 Weather forecast 7-hari (konteks LLM)

```bash
uv run --python 3.12 python model/train_weather.py \
  --n-train 20000 --n-val 5000 --n-test 10000 --out-dir outputs_weather
# → 9 LGBMRegressor (channel cuaca 0–8), multi-output 7 hari, vs baseline persistence
# → checkpoint_weather.json + model_weather_{channel}.joblib + comparison_weather.csv
```

### 5.3 Risk map & artefak serving

```bash
# Peta risiko satu tanggal (untuk semua 6,970 sel)
uv run --python 3.12 python model/risk_map.py --date 2023-09-25

# Generate predictions.parquet {cell_idx, day 1..7, probability} untuk backend
uv run --python 3.12 python scripts/generate_predictions.py \
  --date 2023-09-25 --checkpoint-dir outputs_tabonly --tensor-dir data/output/tensors

# Generate weather_forecast.parquet (5 fitur turunan untuk konteks LLM)
uv run --python 3.12 python scripts/generate_weather.py \
  --date 2023-09-25 --checkpoint-dir outputs_weather --tensor-dir data/output/tensors
```

> Probabilitas yang sama ditulis untuk day=1..7: model memprediksi satu risiko
> untuk jendela [t+1, t+7] (bukan per-hari).

---

## 6. Backend (FastAPI + RAG)

Backend menyajikan prediksi + ringkasan wilayah + AI (RAG regulasi + konteks cuaca)
ke frontend. **Self-contained**: saat container start, `prepare.sh` menarik tensor +
checkpoint dari HF, menjalankan generate_predictions/weather, lalu menyajikan parquet
lokal.

```bash
# Jalan lokal (mode simulasi tanpa env)
cd backend
uv run --python 3.12 uvicorn main:app --port 8000
```

Env opsional (`backend/.env`, lihat `.env.example`):
`HF_MODEL_REPO=danishritonga/karhutla`, `HF_DATASET_REPO=danishritonga/karhutla`,
`OPENAI_API_KEY=...` (untuk mode LLM), `ADMIN_API_KEY=...`, `ALLOWED_ORIGINS=...`.

Endpoint utama: `/api/health`, `/api/grid`, `/api/predictions?day=N`,
`/api/region-summary?day=N`, `/api/explainability/{cell_idx}?day=N`,
`/api/weekly-insight`, `/api/ask` (tanya LLM), `/api/admin/refresh-predictions`.

### Deploy ke HuggingFace Space

```bash
source .env           # berisi HF_TOKEN=...
bash scripts/deploy_space.sh
# rsync backend/ → HF Space danishritonga/karhutla, lalu HF rebuild otomatis
```

Live: https://danishritonga-karhutla.hf.space/ (mode_model real, mode_data simulated).

---

## 7. Frontend (React + Vite + Tailwind)

Fully API-driven — tidak ada logika bisnis di sisi klien; semua data dari backend.

```bash
cd web
npm install
cp .env.example .env   # VITE_API_BASE=http://localhost:8000 (atau URL Space)
npm run dev            # development, http://localhost:5173
npm run build          # → dist/, siap static hosting
```

`web/README.md` berisi dokumentasi lebih detail.

---

## 8. Reproducibility Notes

- **Data mentah** (8 sumber, CSV per-bulan) & **tensor** di HF dataset repo
  `danishritonga/karhutla` (`raw/`, `tensors/`); auto-download saat train/generate.
- **Model** (LightGBM fire + 9 weather) & artefak serving di HF model repo
  `danishritonga/karhutla`.
- **Leakage control:** train-only normalisasi (z-score, statistik dari train saja),
  split temporal ketat (train ≤2021, val 2022, test 2023), fire-history dihitung dari
  deteksi mentah (bukan label k=2), audit L1–L10 di design log.
- **Known caveats** (data sensor, label FIRMS, cloud sparsity DW, ERA5 flux bug):
  lihat `docs/report/caveats.md`.
- **Hasil eksperimen:** `docs/report/test1.md`–`test5.md`, `model_selection.md`.
- **Concept paper & literatur:** `docs/paper/`.

---

## Lisensi & Kontak

Proyek Datathon 2026 (tim karhutla). Data sensor: Copernicus (CC-BY), NASA FIRMS
(public domain), CHIRPS (UCSB), Dynamic World (CC-BY 4.0, Brown et al. 2022),
ERA5-Land (CC-BY 4.0), Peta FEG/BIG (open data pemerintah).
