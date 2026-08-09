# RAG OpenAI untuk dokumen di backend/rag/context

Implementasi ini membuat pipeline RAG sederhana berbasis OpenAI API:
- ekstraksi teks PDF dari `backend/rag/context`
- chunking teks
- embedding ke OpenAI
- retrieval top-k dengan cosine similarity
- jawaban akhir menggunakan model chat OpenAI

## Prasyarat

- Python 3.12+
- Environment variable `OPENAI_API_KEY` sudah diset
- Paket Python:

```bash
pip install pypdf
```

## Menjalankan

Dari root project:

```bash
python -m backend.rag.main build
```

Tanya ke sistem RAG:

```bash
python -m backend.rag.main ask "Apa kewajiban perusahaan sawit terkait pencegahan karhutla?"
```

Lihat chunk retrieval saja:

```bash
python -m backend.rag.main retrieve "Apa sanksi pidana pembakaran hutan?"
```

Jalankan banyak pertanyaan dari file `.txt` (satu pertanyaan per baris):

```bash
python -m backend.rag.main batch --input-file backend/rag/questions.txt
```

Contoh siap pakai tersedia di `backend/rag/questions_example.txt`.

Atau dari file `.csv` (dengan kolom `question`):

```bash
python -m backend.rag.main batch --input-file backend/rag/questions.csv --output-file backend/rag/output/hasil.csv
```

## Opsi penting

- `--index-file`: lokasi file index JSON (default: `backend/rag/index/rag_index.json`)
- `--embedding-model`: default `text-embedding-3-small`
- `--generation-model`: default `gpt-4.1-mini`
- `--top-k`: jumlah chunk yang diambil (default 5)
- `--rebuild`: paksa build index ulang saat `ask`
- `batch --input-file`: proses pertanyaan massal dari `.txt`/`.csv`
- `batch --question-column`: nama kolom pertanyaan untuk input CSV
