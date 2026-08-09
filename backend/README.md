---
title: Karhutla Backend
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Karhutla Early Warning — Backend

FastAPI backend untuk dashboard karhutla. Sekarang jalan dalam **mode
simulasi** (menghasilkan angka yang identik dengan yang selama ini tampil
di prototype front-end React), dan sudah disiapkan supaya nanti tinggal
"dicolok" ke model + dataset asli yang di-deploy ke HuggingFace Hub, tanpa
perlu mengubah kontrak API atau kode frontend.

## Jalankan lokal

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Buka `http://localhost:8000/docs` untuk Swagger UI interaktif.

## Endpoint

| Endpoint | Fungsi |
|---|---|
| `GET /api/grid` | Grid spasial Riau (cell_idx, region, geometry) |
| `GET /api/hotspots?start_date=&end_date=` | Hotspot historis per cell per hari |
| `GET /api/predictions?day=1..7` | Probabilitas risiko per cell untuk horizon +N hari |
| `GET /api/predictions/{cell_idx}?day=` | Prediksi satu cell |
| `GET /api/region-summary?day=` | Statistik agregat + ranking kabupaten |
| `GET /api/region-summary/{region_name}?day=` | Detail satu kabupaten (dipakai saat klik nama region) |
| `GET /api/explainability/{cell_idx}?day=` | Faktor pendorong risiko (rainfall, soil moisture, peat fraction) |
| `GET /api/weekly-insight?day=` | AI Weekly Insight — ringkasan naratif, dari template atau LLM |
| `POST /api/ask` | Ask AI — jawab pertanyaan bebas berdasarkan region-summary hari yang sama |
| `GET /api/health` | Cek status + mode aktif (simulasi/asli) |
| `POST /api/admin/refresh-predictions` | Paksa ambil ulang `predictions.parquet` dari HF sekarang (tanpa nunggu TTL cache) |

## Keamanan endpoint admin

`POST /api/admin/refresh-predictions` terbuka tanpa autentikasi secara
default (nyaman untuk demo). **Sebelum deploy publik**, set `ADMIN_API_KEY`
di environment, lalu sertakan header ini tiap panggil endpoint tersebut:
```
X-Admin-Key: <nilai ADMIN_API_KEY Anda>
```
Tanpa key yang cocok, endpoint balas `401`.

## AI Weekly Insight & Ask AI

`GET /api/weekly-insight` dan `POST /api/ask` (lihat `app/ai_summary.py`)
punya alur data:

```
Predictions -> Region Summary -> [Template / LLM] -> JSON -> Frontend
```

Frontend tidak pernah menghitung ringkasan sendiri, cuma menampilkan field
`summary` / `answer`. Dua mode, otomatis switch:

- **Tanpa `OPENAI_API_KEY`** (default): teks disusun dari template yang
  dibangun dari angka region-summary asli — deterministik, selalu akurat
  terhadap data, tidak butuh network/API key.
- **Dengan `OPENAI_API_KEY`**: teks natural dari OpenAI, dengan
  region-summary sebagai konteks dan RAG aktif untuk dokumen di
  `./rag/context`.
  Kalau panggilan API gagal (rate limit, network), otomatis fallback ke
  template — endpoint tidak pernah error ke frontend karena ini.

Set `OPENAI_API_KEY` di `.env` untuk mengaktifkan mode LLM / RAG.

## Testing

```bash
pytest -v
```

18 test mencakup semua endpoint dalam mode simulasi (tidak butuh koneksi
ke HuggingFace). Jalankan ini setiap kali sebelum deploy.

## Struktur

```
backend/
  main.py              # FastAPI app + router registration
  config.py             # env vars (path data, repo HF, CORS)
  app/
    grid.py             # load & decode grid (port dari App.jsx)
    simulate.py          # simulasi risiko (port dari riskForCell dkk di App.jsx)
    predictor.py          # titik switch: simulasi <-> model asli
    hf_loader.py           # download file dari HuggingFace Hub
    schemas.py              # response models (Pydantic)
    routers/                 # satu file per endpoint
  data/
    grid_data.json            # grid asli hasil ekstraksi dari prototype
```

## Kenapa masih "simulasi"?

Grid, hash noise, dan kurva jarak-ke-anchor di `app/simulate.py` adalah
**port langsung** dari `decodeRLE`/`riskForCell`/`summarize` di `App.jsx`
prototype Anda — bukan angka acak baru. Tujuannya supaya begitu backend ini
dipasang di belakang frontend yang sudah ada, tampilannya tidak berubah
sama sekali (regresi nol), sambil struktur endpoint sudah final dan siap
menerima data/model asli.

## Menyambungkan ke model & dataset asli di HuggingFace

Begitu pipeline pengolahan data & training model selesai:

1. **Dataset repo** — upload `grid_cells.geojson`, `viirs_daily.parquet`,
   dan tabel lain yang perlu diakses backend runtime:
   ```bash
   huggingface-cli upload <username>/karhutla-dataset ./data/processed --repo-type=dataset
   ```
2. **Model repo** — upload hasil training (`predictions.parquet` hasil
   batch inference, atau file model + kode inference-nya):
   ```bash
   huggingface-cli upload <username>/karhutla-model ./model_artifacts --repo-type=model
   ```
3. Set environment variable di tempat backend di-deploy:
   ```
   HF_DATASET_REPO=<username>/karhutla-dataset
   HF_MODEL_REPO=<username>/karhutla-model
   HF_TOKEN=hf_xxx        # kalau repo private
   ```
4. Restart backend. `config.USE_REAL_DATA` / `USE_REAL_MODEL` otomatis
   jadi `True`, dan `app/predictor.py` + router `hotspots`/`explainability`
   otomatis membaca dari HuggingFace alih-alih simulasi — **tidak ada kode
   lain yang perlu diubah**.

Titik yang masih perlu Anda isi manual saat itu tiba:
- `app/predictor.py` → `_real_predictions_df()`: sesuaikan nama file &
  skema kolom `predictions.parquet` Anda. Kolom wajib: `cell_idx`, `day`,
  `probability` (lihat `_REQUIRED_PREDICTION_COLUMNS`) — kalau kolom
  hilang atau namanya beda, backend otomatis fallback ke mode simulasi
  (bukan crash) dan log error yang menyebutkan kolom mana yang hilang.
- `app/routers/explainability.py` → `_real_factors()`: ganti dengan
  pemanggilan SHAP / feature importance dari model asli.
- Cache prediksi asli punya TTL 10 menit (`PREDICTIONS_CACHE_TTL` env var,
  dalam detik) supaya backend otomatis ambil versi terbaru dari HF tanpa
  restart. Untuk refresh instan setelah upload batch baru, panggil
  `POST /api/admin/refresh-predictions`.

## Deploy

### HuggingFace Spaces (Docker SDK)
`Dockerfile` sudah disiapkan (listen di `$PORT`, default 7860 sesuai
konvensi HF Spaces). Push folder `backend/` sebagai Space baru dengan SDK
"Docker", lalu isi env var di Settings → Repository secrets.

### Platform lain (Railway/Render/VPS)
```bash
docker build -t karhutla-backend .
docker run -p 8000:8000 -e PORT=8000 --env-file .env karhutla-backend
```

## Hubungkan ke frontend prototype

Di `src/App.jsx`, ganti bagian yang memakai `GRID`, `getDay(day)`,
`summarize(cells)` dengan `fetch` ke endpoint di atas (mis.
`fetch('/api/predictions?day=' + day)`). Struktur field response sudah
dibuat semirip mungkin dengan objek `cell` yang dipakai komponen
`RiauMap` supaya perubahannya minimal.
