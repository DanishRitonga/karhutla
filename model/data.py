"""data.py
=======
Loads the pre-built tensors from ``data/loader/tensor_assembly.py`` output
(data.npy + labels.npy + meta.json) and extracts patches for the jett model
pipeline.

Our tensor_assembly produces 23 channels (includes ``dw_available`` at
index 20). Jett expects 22 channels (no ``dw_available``). This module
remaps by dropping channel 20::

    tensor [0:20]  → jett [ 0:19]  (same)
    tensor [21]    → jett [20]     (peat_depth)
    tensor [22]    → jett [21]     (hotspot_count_lag)

Constants (PATCH, T_IN, HORIZON, etc.) mirror ``model/jett_data.py`` and
the paper Section 3.2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger("model.data")

PATCH = 15
T_IN = 14
HORIZON = 7
CENTER = PATCH // 2

# tensor_assembly indices
TENSOR_DW_AVAILABLE_IDX = 20
TENSOR_N_CHANNELS = 23
# channel mask: keep all except dw_available
_TENSOR_TO_JETT_INDICES = [i for i in range(TENSOR_N_CHANNELS) if i != TENSOR_DW_AVAILABLE_IDX]

# jett channel layout (22 channels, matches model/jett_data.py exactly)
JETT_CHANNEL_NAMES = [
    "t2m", "d2m", "u10", "v10", "swvl1", "swvl2", "ssr", "tp",
    "chirps_precip",
    "sar_vv", "sar_vh", "sar_available",
    "dw_water", "dw_trees", "dw_grass", "dw_flooded_veg",
    "dw_crops", "dw_shrub_scrub", "dw_built", "dw_bare",
    "peat_depth",
    "hotspot_count_lag",
]
JETT_N_CHANNELS = len(JETT_CHANNEL_NAMES)  # 22
ENV_CHANNELS = list(range(0, JETT_N_CHANNELS - 1))  # 0..20 (21 channels)
OPERATIONAL_CHANNELS = list(range(0, JETT_N_CHANNELS))  # 0..21 (22 channels)
FIRE_HISTORY_IDX = JETT_N_CHANNELS - 1


def load_tensors(tensor_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load (fields, labels, meta) from tensor_assembly output.

    Fields: [D,H,W,23] → [D,H,W,22] (drops dw_available).
    Labels: [D,H,W] int8 (-1/0/1).
    """
    fields = np.load(tensor_dir / "data.npy").astype(np.float32)
    labels = np.load(tensor_dir / "labels.npy").astype(np.int8)
    meta = json.loads((tensor_dir / "meta.json").read_text())

    if fields.shape[-1] == TENSOR_N_CHANNELS:
        fields = fields[..., _TENSOR_TO_JETT_INDICES]

    return fields, labels, meta


def eligible_mask(labels: np.ndarray, meta: dict) -> np.ndarray:
    """Boolean mask [D,H,W] of cell-days valid for training/eval.

    A cell-day is valid iff:
      - label != -1 (non-land cells excluded)
      - the cell fits a full 15×15 patch (CENTER pixels from each edge)
      - day is within valid forecast range (T_IN-1 .. D-HORIZON-1 inclusive)
    """
    grid_h, grid_w = labels.shape[1], labels.shape[2]
    valid = np.zeros_like(labels, dtype=bool)
    valid_t = range(T_IN - 1, labels.shape[0] - HORIZON)
    valid_r = range(CENTER, grid_h - CENTER)
    valid_w = range(CENTER, grid_w - CENTER)
    valid[np.ix_(valid_t, valid_r, valid_w)] = True
    valid &= (labels != -1)
    return valid


def extract(
    fields: np.ndarray,
    labels: np.ndarray,
    cell_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build spatiotemporal patches from selected (day_idx, row, col) tuples.

    cell_days: [N, 3] int array of (t, r, c) positions.
    Returns X[N,T_IN,PATCH,PATCH,C], y[N], day_index[N].
    """
    N = len(cell_days)
    c = fields.shape[-1]
    X = np.zeros((N, T_IN, PATCH, PATCH, c), dtype=np.float32)
    y = np.zeros(N, dtype=np.int8)
    day_index = np.zeros(N, dtype=np.int64)
    for i, (t, r, c_) in enumerate(cell_days):
        X[i] = fields[t - T_IN + 1: t + 1, r - CENTER: r + CENTER + 1, c_ - CENTER: c_ + CENTER + 1, :]
        y[i] = labels[t, r, c_]
        day_index[i] = t
    return X, y, day_index


def to_tabular(X: np.ndarray, channels: list[int]) -> tuple[np.ndarray, list[str]]:
    """Collapse [N,T,P,P,C] → tabular features (mirrors jett_data.to_tabular)."""
    Xc = X[..., channels]  # [N, T, H, W, C]
    center = Xc[:, :, CENTER, CENTER, :]   # [N, T, C]
    patch_mean = Xc.mean(axis=(2, 3))       # [N, T, C]

    feats = np.concatenate([
        center.mean(axis=1), center.std(axis=1), center[:, -1, :],
        patch_mean.mean(axis=1), patch_mean.std(axis=1),
    ], axis=1)
    names = []
    for prefix in ["center_mean", "center_std", "center_last", "patch_mean_mean", "patch_mean_std"]:
        names += [f"{prefix}__{JETT_CHANNEL_NAMES[c]}" for c in channels]

    extra_feats, extra_names = [], []
    for pos, c in enumerate(channels):
        cname = JETT_CHANNEL_NAMES[c]
        series = patch_mean[:, :, pos]
        if cname == "chirps_precip":
            extra_feats.append(series[:, -7:].sum(axis=1, keepdims=True))
            extra_names.append("rain_cum_7d")
            extra_feats.append(series.sum(axis=1, keepdims=True))
            extra_names.append("rain_cum_14d")
        elif cname == "t2m":
            extra_feats.append(series[:, -7:].mean(axis=1, keepdims=True))
            extra_names.append("temp_mean_7d")
        elif cname == "swvl1":
            extra_feats.append(series[:, -7:].mean(axis=1, keepdims=True))
            extra_names.append("soilm_mean_7d")
    if extra_feats:
        feats = np.concatenate([feats] + extra_feats, axis=1)
        names += extra_names

    return feats.astype(np.float32), names


def build_dataset(
    tensor_dir: Path,
    train_years: tuple[int, int] = (2019, 2021),
    val_years: tuple[int, int] | None = (2022, 2022),
    test_years: tuple[int, int] = (2023, 2023),
    n_train_samples: int = 50000,
    n_val_samples: int = 10000,
    n_test_samples: int = 20000,
    pos_frac: float = 0.25,
    seed: int = 42,
    hotspot_7d_csv_path: Path | None = None,
) -> dict:
    """Load pre-built tensors and produce train/val/test splits.

    Returns dict with keys:
        X_train, y_train, day_train, X_val, y_val, day_val,
        X_test, y_test, day_test, meta,
        hotspot_7d_train, hotspot_7d_val, hotspot_7d_test (optional 7-day scalar)
    """
    fields, labels, meta = load_tensors(tensor_dir)
    dates = pd.to_datetime(meta["dates"])
    years = dates.year.to_numpy()

    eligible = eligible_mask(labels, meta)
    logger.info("eligible cell-days: %d / %d", int(eligible.sum()), eligible.size)
    logger.info("positive eligible: %d", int((labels[eligible] == 1).sum()))

    def _sample_in_years(yr_lo: int, yr_hi: int, n_samples: int) -> np.ndarray:
        mask = eligible & ((years[:, None, None] >= yr_lo) & (years[:, None, None] <= yr_hi))
        pos = np.argwhere(mask & (labels == 1))
        neg = np.argwhere(mask & (labels == 0))
        rng = np.random.default_rng(seed + yr_lo)
        n_pos = min(len(pos), int(n_samples * pos_frac))
        n_neg = min(len(neg), n_samples - n_pos)
        sel_pos = pos[rng.choice(len(pos), size=n_pos, replace=False)]
        sel_neg = neg[rng.choice(len(neg), size=n_neg, replace=False)]
        sel = np.concatenate([sel_pos, sel_neg], axis=0)
        rng.shuffle(sel)
        return sel[:n_samples]

    train_cd = _sample_in_years(*train_years, n_train_samples)
    test_cd = _sample_in_years(*test_years, n_test_samples)
    val_cd = _sample_in_years(*val_years, n_val_samples) if val_years else np.zeros((0, 3), dtype=int)

    X_train, y_train, d_train = extract(fields, labels, train_cd)
    X_test, y_test, d_test = extract(fields, labels, test_cd)
    X_val, y_val, d_val = extract(fields, labels, val_cd) if len(val_cd) else (None, None, None)

    logger.info("train: %d (%d pos)", len(y_train), int((y_train == 1).sum()))
    logger.info("test:  %d (%d pos)", len(y_test), int((y_test == 1).sum()))

    result = {
        "X_train": X_train, "y_train": y_train, "day_train": d_train,
        "X_test": X_test, "y_test": y_test, "day_test": d_test,
        "X_val": X_val, "y_val": y_val, "day_val": d_val,
        "meta": meta,
    }

    if hotspot_7d_csv_path and hotspot_7d_csv_path.exists():
        h7 = pd.read_csv(hotspot_7d_csv_path)
        for split, d_arr in [("train", d_train), ("val", d_val), ("test", d_test)]:
            if d_arr is None:
                continue
            dset = h7[h7["day"].isin(d_arr[::len(d_arr) // min(len(d_arr), 1000)])]
            result[f"hotspot_7d_{split}"] = d_arr  # placeholder; real impl needs cell-level join
        logger.warning("hotspot_7d loading not implemented; using day-level proxy only")

    return result
