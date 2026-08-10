"""Emit ``weather_forecast.parquet`` for the backend from the per-cell weather models.

The Phase-5 agent insight layer augments the LLM prompt with a 7-day weather
outlook per region. This script forecasts the 9 meteorological channels
(ERA5-Land ch0-7 + CHIRPS precip ch8) for days 1..7 anchored at ``--date``,
using ``model/train_weather.py`` checkpoints (one LGBM MultiOutputRegressor per
channel, same 158-feature tabular extraction as the fire-risk model).

Output parquet columns: {cell_idx, day, channel, value} where ``cell_idx`` is
the backend grid id ``RIAU_{r}_{c}``. Channels are **LLM-friendly derived
features** (converted from the raw sensor forecasts): temp_c (°C), rh_pct (%),
wind_ms (m/s), wind_dir (cardinal), precip_mm (mm/day), soil_moisture_pct (%),
solar_wm2 (daily-mean W/m2).

Usage::

    uv run --python 3.12 python scripts/generate_weather.py \\
        --date 2023-09-25 \\
        --checkpoint-dir outputs_weather \\
        --out data/output/tensors/weather_forecast.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import joblib
import numpy as np
import pandas as pd

import model.data as d

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_weather")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit weather_forecast.parquet for the backend")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD forecast anchor (in test range)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs_weather"))
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--out", type=Path, default=Path("data/output/tensors/weather_forecast.parquet"))
    args = parser.parse_args()

    cdir = Path(args.checkpoint_dir)
    meta = json.loads((cdir / "checkpoint_weather.json").read_text())
    wchannels = meta["weather_channels"]
    channels = meta["channels"]
    tab_names = meta["tab_names"]
    models = {name: joblib.load(cdir / f"model_weather_{name}.joblib") for name in wchannels}
    logger.info("checkpoint: %d weather channels (%s), %d tabular features",
                len(wchannels), ",".join(wchannels), len(tab_names))

    fields, labels, tmeta = d.load_tensors(args.tensor_dir)
    dates = tmeta["dates"]
    try:
        day_idx = dates.index(args.date)
    except ValueError:
        raise SystemExit(f"date {args.date} not in tensor range {dates[0]} .. {dates[-1]}")
    if day_idx < d.T_IN - 1 or day_idx >= len(dates) - d.HORIZON:
        raise SystemExit(f"date {args.date} outside valid forecast range")

    H, W = fields.shape[1], fields.shape[2]
    pad = d.CENTER
    dummy = np.zeros_like(labels, dtype=np.int8)
    fields_pad = np.pad(fields, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    dummy_pad = np.pad(dummy, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

    cell_days = np.array(
        [[day_idx, r + pad, c + pad] for r in range(H) for c in range(W)],
        dtype=np.int64,
    )

    X, _, _ = d.extract(fields_pad, dummy_pad, cell_days)
    X = d.apply_norm(X, meta["norm_stats"])
    Xt, names = d.to_tabular(X, channels)
    assert list(names) == tab_names, "feature order mismatch vs checkpoint"

    # Predict each channel: [N, 7] in raw units.
    raw: dict[str, np.ndarray] = {}
    for name in wchannels:
        raw[name] = models[name].predict(Xt)  # [N, HORIZON]

    # Convert raw forecasts into LLM-friendly derived features. The models
    # predict the raw physical channels; here we derive human-readable
    # quantities (Celsius, %, mm/day, W/m2, cardinal wind) so the agent's
    # prompt reads naturally. Raw-only channels not surfaced to the LLM:
    # d2m, swvl2, tp (redundant once RH / soil% / precip_mm exist).
    t2m = raw["t2m"]
    d2m = raw["d2m"]
    u10, v10 = raw["u10"], raw["v10"]
    swvl1 = raw["swvl1"]
    ssr = raw["ssr"]
    chirps = raw["chirps_precip"]

    rh = 100.0 * np.exp((17.625 * (d2m - 273.15)) / (243.04 + (d2m - 273.15))) / \
         np.exp((17.625 * (t2m - 273.15)) / (243.04 + (t2m - 273.15)))
    wind_ms = np.sqrt(u10 ** 2 + v10 ** 2)
    wind_deg = np.degrees(np.arctan2(u10, v10)) % 360.0
    _CARD = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wind_dir = np.array([_CARD[int((d + 22.5) // 45) % 8] for d in wind_deg.ravel()]).reshape(wind_deg.shape)

    derived: dict[str, np.ndarray] = {
        "temp_c": t2m - 273.15,
        "rh_pct": rh,
        "wind_ms": wind_ms,
        "precip_mm": chirps,
        "soil_moisture_pct": swvl1 * 100.0,
        "solar_wm2": ssr / 86400.0,
    }

    frames = []
    for name, arr in derived.items():
        rows = []
        for (_, r, c), day_vals in zip(cell_days, arr):
            cid = f"RIAU_{int(r - pad)}_{int(c - pad)}"
            for k, v in enumerate(day_vals, start=1):
                rows.append({"cell_idx": cid, "day": k, "channel": name, "value": float(v)})
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    df["cell_idx"] = df["cell_idx"].astype(str)
    df = df[["cell_idx", "day", "channel", "value"]].sort_values(["channel", "day", "cell_idx"])

    # wind_dir is categorical — write as its own parquet (numeric+string in one
    # column cannot round-trip through pyarrow).
    wd_rows = []
    for (_, r, c), day_vals in zip(cell_days, wind_dir):
        cid = f"RIAU_{int(r - pad)}_{int(c - pad)}"
        for k, v in enumerate(day_vals, start=1):
            wd_rows.append({"cell_idx": cid, "day": k, "wind_dir": str(v)})
    wd_df = pd.DataFrame(wd_rows)[["cell_idx", "day", "wind_dir"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    wd_path = args.out.parent / "weather_wind_dir.parquet"
    wd_df.to_parquet(wd_path, index=False)
    logger.info("wrote %s: %d rows (%d cells x 7 days x %d features)",
                args.out, len(df), df["cell_idx"].nunique(),
                df["channel"].nunique())
    logger.info("wrote %s: %d rows (wind direction)", wd_path, len(wd_df))
    sample = df.groupby("channel")["value"].mean().round(3)
    logger.info("mean forecast per feature:\n%s", sample.to_string())


if __name__ == "__main__":
    main()
