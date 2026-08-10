#!/usr/bin/env sh
set -e

# Tarik data 2023 + checkpoint model lalu proses jadi parquet lokal
# (predictions.parquet, weather_forecast.parquet) sebelum server start.
echo "==> prepare.sh: menyiapkan artefak prakiraan..."
/app/prepare.sh

if [ ! -f "rag/index/rag_index.json" ]; then
  echo "==> Building RAG index before start..."
  python -m rag.main build
else
  echo "==> RAG index already exists, skipping build."
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"