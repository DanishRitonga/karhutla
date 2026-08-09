"""
Jembatan ke HuggingFace Hub. Dipanggil hanya kalau config.USE_REAL_DATA /
USE_REAL_MODEL bernilai True (artinya env var HF_DATASET_REPO / HF_MODEL_REPO
sudah diisi setelah Anda upload dataset & model ke HF).

Alur nanti:
    1. Upload training_table.parquet + hasil prediksi model ke dataset repo
       HF (huggingface-cli upload / Dataset.push_to_hub).
    2. Upload model (misal LightGBM .txt / .pkl, atau ConvLSTM .pt) ke model
       repo HF.
    3. Set env var HF_DATASET_REPO dan HF_MODEL_REPO saat deploy backend.
    4. Backend otomatis download & cache file-file itu lewat fungsi di
       bawah, lalu predictor.py memakainya alih-alih simulate.py.
"""
from functools import lru_cache

import config


@lru_cache
def _hub():
    from huggingface_hub import hf_hub_download
    return hf_hub_download


def download_dataset_file(filename: str) -> str:
    """Download satu file (misal 'grid_cells.geojson') dari HF dataset repo, return path lokal."""
    if not config.USE_REAL_DATA:
        raise RuntimeError("HF_DATASET_REPO belum diset di environment")
    hf_hub_download = _hub()
    return hf_hub_download(
        repo_id=config.HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=config.HF_TOKEN or None,
    )


def download_model_file(filename: str) -> str:
    """Download satu file model dari HF model repo, return path lokal (sudah di-cache)."""
    if not config.USE_REAL_MODEL:
        raise RuntimeError("HF_MODEL_REPO belum diset di environment")
    hf_hub_download = _hub()
    return hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=filename,
        token=config.HF_TOKEN or None,
    )
