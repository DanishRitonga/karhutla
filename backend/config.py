"""
Konfigurasi backend. Semua yang bisa berubah saat deploy (path data, repo
HuggingFace untuk model & dataset asli) diatur lewat environment variable,
supaya container yang sama bisa dipakai di localhost maupun di deployment
tanpa ubah kode.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Grid statis (hasil ekstraksi dari prototype). Nanti bisa diganti file
# grid_cells.geojson asli dari pipeline pengolahan data.
GRID_PATH = DATA_DIR / "grid_data.json"

# ── Sumber data/model asli di HuggingFace (dipakai belakangan) ──────────
# Set env var ini setelah model & dataset di-upload ke HF Hub. Selama env
# var ini kosong, backend otomatis pakai mode simulasi (persis seperti
# prototype) supaya dashboard tetap bisa jalan tanpa model asli.
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")     # contoh: "username/karhutla-dataset"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")          # contoh: "username/karhutla-risk-model"
HF_TOKEN = os.getenv("HF_TOKEN", "")                     # perlu kalau repo private

USE_REAL_DATA = bool(HF_DATASET_REPO)
USE_REAL_MODEL = bool(HF_MODEL_REPO)

# ── LLM (opsional) untuk AI Weekly Insight & Ask AI ──────────────────────
# Kosong = pakai ringkasan berbasis template (tetap akurat, deterministik,
# tanpa perlu API key). Isi ANTHROPIC_API_KEY untuk ringkasan/jawaban dalam
# bahasa natural yang dihasilkan LLM, dengan fallback otomatis ke template
# kalau panggilan API gagal (rate limit, network, dst).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
USE_LLM_SUMMARY = bool(ANTHROPIC_API_KEY)

# Kalau diset, endpoint admin butuh header X-Admin-Key yang cocok.
# Kosong = endpoint tetap terbuka (nyaman untuk demo/dev), TAPI harus diisi
# sebelum deploy publik -- lihat README bagian "Keamanan endpoint admin".
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# CORS: origin frontend yang boleh akses backend ini
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
