"""train_weather.py
==================
Per-cell weather-context forecasters for the Phase-5 agent insight layer.

Predicts the 9 meteorological channels (ERA5-Land ch0-7 + CHIRPS precip ch8)
for the next HORIZON=7 days at every cell, using EXACTLY the same feature
engineering as the fire-risk model (``extract`` → z-score norm → ``to_tabular``
over the 22-channel operational layout → 158 tabular features).

Architecture: one LGBMRegressor per weather channel, wrapped in
``MultiOutputRegressor`` over the 7 forecast days (regression, not
classification). Baseline: persistence (the channel's last observed center
value at t, held constant across the 7 days) — the standard weather baseline.

Split is identical to the fire model: train 2019-2021 / val 2022 / test 2023.

Usage::

    uv run --python 3.12 python model/train_weather.py --out-dir outputs_weather
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from model.data import (
    load_tensors,
    eligible_mask,
    extract,
    to_tabular,
    compute_norm_stats,
    apply_norm,
    OPERATIONAL_CHANNELS,
    JETT_CHANNEL_NAMES,
    T_IN,
    PATCH,
    HORIZON,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_weather")

# Weather channels to forecast (indices in the 22-channel jett layout):
# 0 t2m, 1 d2m, 2 u10, 3 v10, 4 swvl1, 5 swvl2, 6 ssr, 7 tp, 8 chirps_precip
WEATHER_CHANNELS = list(range(9))
WEATHER_NAMES = [JETT_CHANNEL_NAMES[c] for c in WEATHER_CHANNELS]


def _sample_uniform(
    eligible: np.ndarray,
    years: np.ndarray,
    yr_lo: int,
    yr_hi: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed + yr_lo)
    yr_mask = (years[:, None, None] >= yr_lo) & (years[:, None, None] <= yr_hi)
    mask = eligible & yr_mask
    idx = np.argwhere(mask)
    if len(idx) < n_samples:
        logger.warning("only %d eligible cell-days in %d-%d; sampling all", len(idx), yr_lo, yr_hi)
        return idx
    sel = idx[rng.choice(len(idx), size=n_samples, replace=False)]
    return sel


def _weather_targets(fields: np.ndarray, cell_days: np.ndarray) -> np.ndarray:
    """Future weather at the center cell: y[i, d-1, ch] = fields[t+d, r, c, ch].

    d ∈ 1..HORIZON, ch ∈ WEATHER_CHANNELS. Fully vectorized over samples.
    ``fields`` must be the RAW (unnormalised) tensor; targets are never z-scored.
    """
    t = cell_days[:, 0]
    r = cell_days[:, 1]
    c = cell_days[:, 2]
    y = np.empty((len(cell_days), HORIZON, len(WEATHER_CHANNELS)), dtype=np.float32)
    for d in range(1, HORIZON + 1):
        y[:, d - 1, :] = fields[t + d, r, c][:, WEATHER_CHANNELS]
    return y


def _persistence(fields: np.ndarray, cell_days: np.ndarray) -> np.ndarray:
    """Persistence baseline: last observed center value at t, held across 7 days."""
    t = cell_days[:, 0]
    r = cell_days[:, 1]
    c = cell_days[:, 2]
    last = fields[t, r, c][:, WEATHER_CHANNELS]  # [N, 9]
    return np.repeat(last[:, None, :], HORIZON, axis=1)  # [N, 7, 9]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-cell 7-day weather forecasters (LGBM, same features as fire model)")
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--train", type=int, nargs=2, default=[2019, 2021])
    parser.add_argument("--val", type=int, nargs=2, default=[2022, 2022])
    parser.add_argument("--test", type=int, nargs=2, default=[2023, 2023])
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_weather"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    channels = OPERATIONAL_CHANNELS  # same 22-channel feature space as fire model

    # -- 1. load tensors ---------------------------------------------------
    logger.info("loading tensors from %s", args.tensor_dir)
    fields, labels, meta = load_tensors(args.tensor_dir)
    dates = pd.to_datetime(meta["dates"])
    years = dates.year.to_numpy()
    logger.info("fields %s, dates %s → %s", fields.shape, dates[0].date(), dates[-1].date())

    # -- 2. sample cell-days (uniform — regression, no class balancing) ----
    eligible = eligible_mask(labels, meta)
    train_cd = _sample_uniform(eligible, years, *args.train, args.n_train, args.seed)
    val_cd   = _sample_uniform(eligible, years, *args.val,   args.n_val,   args.seed)
    test_cd  = _sample_uniform(eligible, years, *args.test,  args.n_test,  args.seed)
    logger.info("samples: train=%d val=%d test=%d", len(train_cd), len(val_cd), len(test_cd))

    # -- 3. patches (features) ----------------------------------------------
    X_train, _, _ = extract(fields, labels, train_cd)
    X_val, _, _   = extract(fields, labels, val_cd)
    X_test, _, _  = extract(fields, labels, test_cd)
    logger.info("patches: train %s val %s test %s", X_train.shape, X_val.shape, X_test.shape)

    # -- 3b. targets + persistence baseline from RAW fields (before del) ----
    y_train = _weather_targets(fields, train_cd)
    y_val   = _weather_targets(fields, val_cd)
    y_test  = _weather_targets(fields, test_cd)
    pers_train = _persistence(fields, train_cd)
    pers_test  = _persistence(fields, test_cd)
    logger.info("targets y_test %s (days × channels) | persistence last-day mae example ok")
    del fields

    # -- 3c. z-score normalisation (train stats only, in place) ------------
    norm_stats = compute_norm_stats(X_train)
    X_train = apply_norm(X_train, norm_stats)
    X_val   = apply_norm(X_val, norm_stats)
    X_test  = apply_norm(X_test, norm_stats)

    # -- 3d. tabular features (identical to fire model) ---------------------
    X_train_tab, tab_names = to_tabular(X_train, channels)
    X_val_tab, _           = to_tabular(X_val, channels)
    X_test_tab, _          = to_tabular(X_test, channels)
    logger.info("tabular features: %d (channel space %d ch)",
                X_train_tab.shape[1], len(channels))

    # -- 4. evaluate per channel --------------------------------------------
    def mae_rmse(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
        err = pred - true
        return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))

    rows = []
    models: dict[str, object] = {}
    for ci, ch_name in enumerate(WEATHER_NAMES):
        logger.info("  channel %-14s fitting LGBM (7-day multi-output)", ch_name)
        # Train one regressor per channel, multi-output over the 7 days.
        mdl = MultiOutputRegressor(
            LGBMRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                          learning_rate=args.lr, num_leaves=31, subsample=0.8,
                          colsample_bytree=0.8, random_state=args.seed, verbose=-1)
        )
        mdl.fit(X_train_tab, y_train[:, :, ci])
        models[ch_name] = mdl
        pred = mdl.predict(X_test_tab)  # [N, 7]
        pers = pers_test[:, :, ci]      # [N, 7]
        true = y_test[:, :, ci]
        p_mae, p_rmse = mae_rmse(pers, true)
        m_mae, m_rmse = mae_rmse(pred, true)
        for d in range(HORIZON):
            rows.append({
                "channel": ch_name, "day": d + 1,
                "lgbm_mae": mae_rmse(pred[:, d], true[:, d])[0],
                "lgbm_rmse": mae_rmse(pred[:, d], true[:, d])[1],
                "pers_mae": mae_rmse(pers[:, d], true[:, d])[0],
                "pers_rmse": mae_rmse(pers[:, d], true[:, d])[1],
            })
        logger.info("    LGBM  MAE=%.4f RMSE=%.4f | Persist MAE=%.4f RMSE=%.4f",
                    m_mae, m_rmse, p_mae, p_rmse)

    table = pd.DataFrame(rows)
    table_path = args.out_dir / "comparison_weather.csv"
    table.to_csv(table_path, index=False, float_format="%.5f")
    logger.info("comparison table → %s", table_path)

    # -- 5. persist models + metadata ---------------------------------------
    for ch_name, mdl in models.items():
        joblib.dump(mdl, args.out_dir / f"model_weather_{ch_name}.joblib")
    meta = {
        "task": "weather_forecast",
        "weather_channels": WEATHER_NAMES,
        "weather_channel_indices": WEATHER_CHANNELS,
        "channels": channels,
        "norm_stats": norm_stats,
        "tab_names": tab_names,
        "t_in": T_IN,
        "patch": PATCH,
        "horizon": HORIZON,
        "n_train": len(train_cd), "n_val": len(val_cd), "n_test": len(test_cd),
        "models_saved": [f"model_weather_{c}.joblib" for c in WEATHER_NAMES],
    }
    with open(args.out_dir / "checkpoint_weather.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("checkpoint metadata → %s", args.out_dir / "checkpoint_weather.json")
    logger.info("all done — %d weather channels", len(WEATHER_NAMES))


if __name__ == "__main__":
    main()
