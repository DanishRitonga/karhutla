#!/usr/bin/env sh
set -e

if [ ! -f "rag/index/rag_index.json" ]; then
  echo "==> Building RAG index before start..."
  python -m rag.main build
else
  echo "==> RAG index already exists, skipping build."
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"