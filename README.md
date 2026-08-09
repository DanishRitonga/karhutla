# Karhutla Dashboard — Frontend (Vite + React)

```bash
npm install
cp .env.example .env   # isi VITE_API_BASE_URL
npm run dev             # development, http://localhost:5173
npm run build            # -> dist/, siap deploy ke static hosting
```

Semua data (grid, prediksi, region-summary, explainability, AI weekly
insight, Ask AI) diambil dari backend lewat `VITE_API_BASE_URL` — tidak
ada logika bisnis yang dihitung di frontend.
