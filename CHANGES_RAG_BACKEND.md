# RAG + Backend Integration Changes

Dokumen ini merangkum perubahan yang saya buat pada implementasi RAG di `rag/` dan integrasinya ke backend FastAPI di `backend/`.

## 1. Implementasi RAG di `rag/`

Saya membuat pipeline RAG lengkap yang membaca dokumen PDF dari `rag/context/` dan menjawab pertanyaan dengan OpenAI API.

### File yang ditambahkan

- `rag/__init__.py`
  - Menjadikan folder `rag/` sebagai package Python.
  - Mengekspor fungsi inti RAG.
- `rag/pdf_loader.py`
  - Membaca seluruh PDF di `rag/context/`.
  - Mengekstrak teks per halaman dan mengubahnya jadi objek dokumen.
- `rag/chunker.py`
  - Memecah teks menjadi chunk overlap untuk retrieval.
- `rag/openai_client.py`
  - Wrapper HTTP ke OpenAI API untuk embeddings dan chat completion.
  - Dipakai tanpa dependency `requests`, hanya standard library.
- `rag/rag_engine.py`
  - Core RAG: build index, load index, cosine similarity retrieval, dan answer generation.
- `rag/main.py`
  - CLI entrypoint dengan subcommand `build`, `retrieve`, `ask`, dan `batch`.
- `rag/batch_runner.py`
  - Menjalankan Q&A massal dari `.txt` atau `.csv`.
  - Menulis hasil ke CSV.
- `rag/README.md`
  - Dokumentasi cara pakai RAG.
- `rag/requirements.txt`
  - Dependency minimum untuk folder `rag/`.
- `rag/.gitignore`
  - Mengabaikan index hasil embedding dan cache Python.
- `rag/questions_example.txt`
  - Contoh file input untuk batch question answering.

### Perilaku RAG yang dibuat

- Mengambil konteks dari PDF di `rag/context/`.
- Membuat vector index lokal di `rag/index/rag_index.json`.
- Melakukan retrieval top-k berdasarkan cosine similarity embedding.
- Menjawab pertanyaan dengan prompt berbasis konteks yang terambil.
- Mendukung batch processing pertanyaan dari file input.

## 2. Integrasi ke backend FastAPI

Saya mengubah layer AI backend agar memakai implementasi RAG ini, tetapi bentuk response JSON ke frontend tetap dipertahankan.

### File yang diubah

- `backend/app/ai_summary.py`
  - Diganti dari implementasi Anthropic/templated summary ke layer OpenAI/RAG.
  - `weekly_insight()` dan `answer_question()` sekarang bisa memakai RAG dari `rag/`.
  - Tetap fallback ke template jika `OPENAI_API_KEY` kosong atau API gagal.
- `backend/config.py`
  - Mengganti konfigurasi AI dari `ANTHROPIC_API_KEY` ke `OPENAI_API_KEY` dan `OPENAI_BASE_URL`.
  - `USE_LLM_SUMMARY` sekarang aktif jika `OPENAI_API_KEY` tersedia.
- `backend/main.py`
  - Menambahkan bootstrap path supaya backend bisa mengimpor package sibling `rag/`.
  - Ini perlu agar import `rag.openai_client` dan `rag.rag_engine` aman saat backend dijalankan.
- `backend/.env.example`
  - Contoh environment variable diperbarui ke OpenAI.
- `backend/README.md`
  - Dokumentasi AI backend diperbarui supaya menjelaskan mode template vs OpenAI/RAG.
- `backend/requirements.txt`
  - Catatan dependency AI disesuaikan dengan integrasi baru.

### Perilaku backend yang dipertahankan

- Response shape frontend tidak diubah.
- `GET /api/weekly-insight` tetap mengembalikan:
  - `day`
  - `summary`
  - `source`
- `POST /api/ask` tetap mengembalikan:
  - `question`
  - `day`
  - `answer`
  - `source`
- `GET /api/region-summary` tetap mengirim field `ai_summary`.

## 3. Hal teknis penting

- Saya menggunakan OpenAI API dari backend untuk embed dan chat completion.
- RAG context diambil dari `rag/context/` yang berisi PDF hukum/dokumen sumber.
- Index vector disimpan lokal, bukan di frontend.
- Backend tetap punya fallback template supaya endpoint tidak mati kalau API key tidak ada atau network gagal.
- Import lintas-folder antara `backend/` dan `rag/` dibuat aman dengan bootstrap path repo root.

## 4. Cara pakai ringkas

### RAG standalone

```bash
python -m rag.main build
python -m rag.main ask "Pertanyaan Anda"
python -m rag.main batch --input-file rag/questions_example.txt
```

### Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Set environment variable:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 5. Kontrak data frontend yang tidak diubah

Saya sengaja tidak mengubah struktur data yang dikirim ke frontend agar integrasi UI tetap minim.

- `region-summary` masih dipakai untuk kartu ringkasan.
- `weekly-insight` masih menyediakan `summary`.
- `ask` masih menyediakan `answer`.

## 6. Sample data dan command untuk tes endpoint FastAPI

### Prasyarat

Jalankan backend dari folder `backend/`.

```bash
cd /home/adhyaksawp/Documents/Projects/karhutla/backend
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8000
```

Kalau ingin tes mode template tanpa OpenAI, cukup kosongkan `OPENAI_API_KEY`.

### Tes health

```bash
curl http://localhost:8000/api/health
```

Contoh respons:

```json
{
  "status": "ok",
  "mode_data": "simulated",
  "mode_model": "simulated",
  "prediction_source": "simulated"
}
```

### Tes weekly insight

```bash
curl "http://localhost:8000/api/weekly-insight?day=3"
```

Contoh respons:

```json
{
  "day": 3,
  "summary": "...",
  "source": "template"
}
```

### Tes ask endpoint

Sample payload:

```json
{
  "question": "Kabupaten mana yang paling berisiko pada hari ke-3?",
  "day": 3
}
```

Command:

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Kabupaten mana yang paling berisiko pada hari ke-3?","day":3}'
```

Contoh respons:

```json
{
  "question": "Kabupaten mana yang paling berisiko pada hari ke-3?",
  "day": 3,
  "answer": "...",
  "source": "template"
}
```

### Tes region summary yang dipakai frontend

```bash
curl "http://localhost:8000/api/region-summary?day=3"
```

Field yang perlu dicek:

- `day`
- `total_cells`
- `high_risk_cells`
- `predicted_hotspots`
- `ranking`
- `ai_summary`

### Tes refresh cache admin

Kalau `ADMIN_API_KEY` belum diisi, endpoint ini bisa dipanggil langsung:

```bash
curl -X POST "http://localhost:8000/api/admin/refresh-predictions"
```

Kalau `ADMIN_API_KEY` diaktifkan, kirim header:

```bash
curl -X POST "http://localhost:8000/api/admin/refresh-predictions" \
  -H "X-Admin-Key: rahasia123"
```

### Tes mode RAG/OpenAI

Untuk memaksa backend memakai RAG, isi `OPENAI_API_KEY`, lalu ulangi request di atas.
Jika index belum ada, backend akan membangunnya dari PDF di `rag/context/`.

Contoh pertanyaan yang bagus untuk tes fungsi RAG:

- "Apa kewajiban pelaku usaha dalam pencegahan kebakaran lahan?"
- "Apa sanksi untuk pembakaran hutan menurut dokumen yang tersedia?"
- "Bagaimana tanggung jawab pemerintah dalam perlindungan lingkungan hidup?"
