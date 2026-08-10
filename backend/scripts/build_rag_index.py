"""Bangun index RAG sekali, di luar jalur request.

Index disimpan ke rag/index/rag_index.json. Commit hasilnya ke repo supaya
container Space yang baru langsung punya index tanpa perlu membangun ulang.

Kenapa tidak dibangun saat request saja: parsing 5 PDF (1395 halaman) makan
~10 detik, lalu 2.215 chunk dikirim sebagai 35 panggilan embedding berurutan.
Kalau itu terjadi di dalam satu HTTP request, request pertama sesudah container
baru akan menggantung bermenit-menit -- dan container HuggingFace Space selalu
baru setiap kali Space bangun dari tidur.

Pakai::

    export OPENAI_API_KEY=sk-...
    python scripts/build_rag_index.py

Lalu::

    git add rag/index/rag_index.json
    git commit -m "rag: prebuild index"
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_rag_index")


def main() -> None:
    import config
    from app.ai_summary import build_rag_index_now, _RAG_INDEX_FILE

    if not config.OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY belum diisi. Set dulu env var-nya, atau isi di .env"
        )

    if _RAG_INDEX_FILE.exists():
        logger.warning("index sudah ada di %s -- akan ditimpa", _RAG_INDEX_FILE)

    build_rag_index_now()

    size_mb = _RAG_INDEX_FILE.stat().st_size / 1_048_576
    logger.info("selesai: %s (%.1f MB)", _RAG_INDEX_FILE, size_mb)
    if size_mb > 10:
        logger.warning(
            "ukurannya di atas 10 MB -- lacak lewat Git LFS seperti PDF di "
            "rag/context, jangan sebagai file biasa."
        )


if __name__ == "__main__":
    main()
