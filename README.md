# Karhutla Early Warning — Dashboard

```
project/
  web/       # Frontend (React + Vite)
  backend/   # Backend (FastAPI)
```

Model & dataset (training, upload ke HuggingFace) dikelola terpisah.

## Jalankan lokal

```bash
# terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# terminal 2 — frontend
cd web
npm install
cp .env.example .env      # VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

## Endpoint backend

```
GET  /api/grid
GET  /api/grid/meta
GET  /api/predictions?day=1..7
GET  /api/predictions/{cell_idx}?day=
GET  /api/region-summary?day=
GET  /api/region-summary/{region_name}?day=
GET  /api/explainability/{cell_idx}?day=
GET  /api/weekly-insight?day=          ← AI weekly insight (template/LLM)
POST /api/ask                           ← Ask AI (template/LLM)
GET  /api/health
POST /api/admin/refresh-predictions
```

Arsitektur:
```
Frontend (React)
  ↓
FastAPI Backend
  ↓
Predictor Layer  ──── simulasi ↔ model asli (switch lewat env var HF_*)
  ↓
Region Summary  ──── agregasi ranking kabupaten
  ↓
AI Summary Layer ──── template ↔ LLM (switch lewat ANTHROPIC_API_KEY)
```

Frontend tidak menghitung apa pun sendiri — semua angka & narasi datang
dari backend, sudah termasuk AI Weekly Insight dan Ask AI.

## Deploy

- **backend/** — siap jadi HuggingFace Space (Docker SDK, lihat
  frontmatter di `backend/README.md`) atau Railway/Render/VPS lewat
  `backend/Dockerfile`.
- **web/** — `npm run build` (isi `.env` dengan URL backend production
  dulu) → `web/dist/`, siap upload ke static hosting apa pun. `dist/`
  yang sudah ada di zip ini masih hasil build dengan URL backend lokal.

Detail lengkap ada di README masing-masing folder.
