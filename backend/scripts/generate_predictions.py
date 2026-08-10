"""Emit ``predictions.parquet`` for the backend from a real 2023 checkpoint.

The backend (``app/predictor.py``) loads a parquet with columns
{cell_idx, day, probability} from the HF model repo. ``cell_idx`` must match
the ids produced by ``app/grid.decode_cells()`` = ``RIAU_{r}_{c}``.

The model maps input window [t-13, t] -> risk for [t+1, t+7]; it has no
per-day resolution, so the SAME probability is written for day=1..7 (this is
the correct/honest interpretation — the forecast is a 7-day window).

Only Riau cells with a full 15x15 patch (rows 7..74 x cols 7..77) are scored;
edge cells are omitted from the parquet (the backend skips unknown cell_idx).

Usage::

    uv run --python 3.12 python scripts/generate_predictions.py \
        --date 2023-09-25 \
        --checkpoint-dir outputs_tabonly \
        --out data/output/tensors/predictions.parquet
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

from data import data as d

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit predictions.parquet for the backend")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD forecast anchor (in test range)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs_tabonly"))
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--out", type=Path, default=Path("data/output/tensors/predictions.parquet"))
    args = parser.parse_args()

    cdir = Path(args.checkpoint_dir)
    meta = json.loads((cdir / "checkpoint_operational.json").read_text())
    model = joblib.load(cdir / "model_lgbm_operational.joblib")
    logger.info("checkpoint: %d channels, %d tabular features", meta["n_channels"], len(meta["tab_names"]))

    fields, labels, tmeta = d.load_tensors(args.tensor_dir)
    dates = tmeta["dates"]
    try:
        day_idx = dates.index(args.date)
    except ValueError:
        raise SystemExit(f"date {args.date} not in tensor range {dates[0]} .. {dates[-1]}")
    if day_idx < d.T_IN - 1 or day_idx >= len(dates) - d.HORIZON:
        raise SystemExit(f"date {args.date} outside valid forecast range")

    H, W = fields.shape[1], fields.shape[2]
    channels = meta["channels"]
    tab_names = meta["tab_names"]

    # Spatial edge-padding (mirrors model/risk_map.py): cells within CENTER=7
    # of the grid edge cannot form a full 15x15 patch; pad the fields with
    # replicated edge values so EVERY cell can be scored.
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

    probs = model.predict_proba(Xt)[:, 1]
    rows = []
    for (_, r, c), p in zip(cell_days, probs):
        rows.append({"cell_idx": f"RIAU_{int(r - pad)}_{int(c - pad)}", "day": 1, "probability": p})

    df = pd.DataFrame(rows)
    df["cell_idx"] = df["cell_idx"].astype(str)
    # Same probability for every day 1..7 (single 7-day-window forecast).
    df = pd.concat([df.assign(day=k) for k in range(1, 8)], ignore_index=True)
    df = df[["cell_idx", "day", "probability"]].sort_values(["day", "cell_idx"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("wrote %s: %d rows (%d cells x 7 days), prob range [%.3f, %.3f]",
                args.out, len(df), df["cell_idx"].nunique(),
                df["probability"].min(), df["probability"].max())


if __name__ == "__main__":
    main()
