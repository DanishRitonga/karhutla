#!/usr/bin/env bash
set -e

# Menyiapkan artefak prakiraan (predictions.parquet, weather_forecast.parquet,
# weather_wind_dir.parquet) langsung di dalam container saat start.
#
# Alur: pull checkpoint model dari HF model repo → generate_predictions.py &
# generate_weather.py (yang otomatis menarik tensor dari HF dataset repo via
# model.data._ensure_tensors_local) → parquet lokal di DATA_DIR. Backend lalu
# menyajikan file lokal itu (app/predictor.py & app/weather.py membaca
# DATA_DIR dulu, HF hanya jadi fallback).
#
# Idempotent: kalau kedua parquet sudah ada, dilewati (boot cepat).
#
# Env:
#   DATA_DIR       (default /app/data)
#   FORECAST_DATE  anchor tanggal prakiraan (default 2023-09-25, puncak musim)
#   HF_MODEL_REPO  model repo (default danishritonga/karhutla)
#   HF_TOKEN       opsional, untuk repo private

DATA_DIR="${DATA_DIR:-/app/data}"
CHECKPOINT_DIR="${DATA_DIR}/checkpoints"
TENSOR_DIR="${TENSOR_DIR:-${DATA_DIR}/tensors}"
FORECAST_DATE="${FORECAST_DATE:-2023-09-25}"
MODEL_REPO="${HF_MODEL_REPO:-danishritonga/karhutla}"
HF_TOKEN="${HF_TOKEN:-}"
# Interpreter; override PYTHON untuk lingkungan uv (mis. `uv run --python 3.12 python`).
PYTHON="${PYTHON:-python}"
run_py() { eval "${PYTHON}" "$@"; }

# Direktori tempat prepare.sh berada (backend/ di monorepo, /app di container).
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPTS_DIR:-${SELF_DIR}/scripts}"

PREDICTIONS="${DATA_DIR}/predictions.parquet"
WEATHER="${DATA_DIR}/weather_forecast.parquet"

if [ -f "${PREDICTIONS}" ] && [ -f "${WEATHER}" ]; then
  echo "==> Artefak prakiraan sudah ada, lewati prepare."
  exit 0
fi

echo "==> Menyiapkan artefak prakiraan (date=${FORECAST_DATE})"
mkdir -p "${CHECKPOINT_DIR}" "${TENSOR_DIR}"

echo "==> Download checkpoint dari HF model repo ${MODEL_REPO}"
export MODEL_REPO CHECKPOINT_DIR HF_TOKEN
run_py - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.getenv("MODEL_REPO", "danishritonga/karhutla")
ckpt_dir = Path(os.getenv("CHECKPOINT_DIR", "data/checkpoints"))
token = os.getenv("HF_TOKEN") or None

_WEATHER = ["chirps_precip", "d2m", "ssr", "swvl1", "swvl2", "t2m", "tp", "u10", "v10"]
files = [
    "checkpoint_operational.json",
    "model_lgbm_operational.joblib",
    "checkpoint_weather.json",
] + [f"model_weather_{n}.joblib" for n in _WEATHER]

for f in files:
    dest = ckpt_dir / f
    if dest.exists():
        continue
    print(f"  download {f}")
    path = hf_hub_download(repo_id=repo, filename=f, repo_type="model", token=token)
    shutil.copy(path, dest)
PY

echo "==> Generate predictions.parquet"
run_py "${SCRIPTS_DIR}/generate_predictions.py" \
  --date "${FORECAST_DATE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensor-dir "${TENSOR_DIR}" \
  --out "${PREDICTIONS}"

echo "==> Generate weather_forecast.parquet"
run_py "${SCRIPTS_DIR}/generate_weather.py" \
  --date "${FORECAST_DATE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensor-dir "${TENSOR_DIR}" \
  --out "${WEATHER}"

echo "==> Artefak prakiraan siap."
