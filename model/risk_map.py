"""Render a predicted-fire-risk map for a single date using a saved checkpoint.

Loads the production checkpoint (``model_lgbm_operational.joblib`` +
``checkpoint_operational.json`` from a training ``--out-dir``), runs inference
over every eligible Riau cell on the requested date, and draws the resulting
risk probabilities on the 5 km Albers grid in the style of ``riau_grid.png``.

Example (from the datathon project root)::

    uv run --python 3.12 python model/risk_map.py \\
        --date 2023-09-25 \\
        --checkpoint-dir outputs_tabonly \\
        --tensor-dir data/output/tensors \\
        --out data/output/maps

The date must be inside the forecast range of the tensors (2019-01-15 ..
2023-12-24 for the default 14-in/7-out assembly). Riau cells with no valid
patch are left blank (white).
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
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import model.data as d

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("risk_map")

RISK_CMAP = "YlOrRd"
RISK_VMIN, RISK_VMAX = 0.0, 1.0


def load_checkpoint(checkpoint_dir: Path):
    """Return (model, meta) for the production operational LightGBM pair."""
    cdir = Path(checkpoint_dir)
    meta = json.loads((cdir / "checkpoint_operational.json").read_text())
    model = joblib.load(cdir / "model_lgbm_operational.joblib")
    logger.info("checkpoint: regime=%s channels=%d tab=%d model=%s",
                meta["regime"], meta["n_channels"], len(meta["tab_names"]),
                type(model).__name__)
    return model, meta


def score_day(model, meta, fields: np.ndarray, day_idx: int) -> np.ndarray:
    """Return a [H, W] risk grid (NaN where no valid patch) for one day."""
    H, W = fields.shape[1], fields.shape[2]
    risk = np.full((H, W), np.nan, dtype=np.float32)
    channels = meta["channels"]
    tab_names = meta["tab_names"]

    # Cell-days: every (r, c) whose patch is fully in-bounds. The tensor
    # already 0-fills non-land cells, but we only score rows/cols where a full
    # 15x15 patch exists (mirrors eligible_mask margins, no label dependency).
    rs = range(d.CENTER, H - d.CENTER)
    cs = range(d.CENTER, W - d.CENTER)
    cell_days = np.array([[day_idx, r, c] for r in rs for c in cs], dtype=np.int64)
    n = len(cell_days)
    logger.info("day %d: %d candidate cells", day_idx, n)

    # Extract all patches for the day at once (bounded by model memory; ~2.5k
    # cells x 14 x 15 x 15 x 22 float32 ≈ 1.7 GB at full grid — acceptable).
    # extract() needs a labels array for its y output; we discard it.
    dummy_labels = np.zeros((fields.shape[0], fields.shape[1], fields.shape[2]), dtype=np.int8)
    X, _, _ = d.extract(fields, dummy_labels, cell_days)
    X = d.apply_norm(X, meta["norm_stats"])
    Xt, names = d.to_tabular(X, channels)
    assert list(names) == tab_names, "feature order mismatch vs checkpoint"

    probs = model.predict_proba(Xt)[:, 1]
    for (_, r, c), p in zip(cell_days, probs):
        risk[r, c] = p
    return risk


def render_risk_grid(risk: np.ndarray, date_str: str, cells: pd.DataFrame,
                     boundary: gpd.GeoDataFrame, out_path: Path,
                     threshold: float | None = None) -> None:
    cs = 5000
    H, W = risk.shape
    cmap = plt.get_cmap(RISK_CMAP)
    norm = Normalize(vmin=RISK_VMIN, vmax=RISK_VMAX)

    patches, facecolors = [], []
    for _, row in cells.iterrows():
        r, c = int(row["row"]), int(row["col"])
        if not (0 <= r < H and 0 <= c < W):
            continue
        v = risk[r, c]
        if np.isnan(v):
            continue
        x, y = row["x_center_m"], row["y_center_m"]
        patches.append(Rectangle((x - cs / 2, y - cs / 2), cs, cs))
        facecolors.append(cmap(norm(v)))

    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.add_collection(PatchCollection(patches, facecolors=facecolors, edgecolors="none"))
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Predicted fire risk (P(hotspot in next 7 d))")

    ttl = f"LightGBM (operational) predicted risk — {date_str}\nRiau, 5 km equal-area grid"
    if threshold is not None:
        ttl += f"\nCells above alert threshold {threshold:.3f}: {(risk >= threshold).sum()} (Riau cells)"
    ax.set_title(ttl, fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.margins(0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Rendered %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fire-risk map from a saved checkpoint")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (within tensor forecast range)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs_tabonly"))
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--grid-csv", type=Path, default=Path("data/output/grid/grid_cells.csv"))
    parser.add_argument("--boundary-gpkg", type=Path, default=Path("data/output/grid/riau_boundary_aea.gpkg"))
    parser.add_argument("--out", type=Path, default=Path("data/output/maps"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="alert threshold to count (default: use checkpoint F1 threshold)")
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint_dir)
    fields, labels, tmeta = d.load_tensors(args.tensor_dir)

    dates = tmeta["dates"]
    try:
        day_idx = dates.index(args.date)
    except ValueError:
        raise SystemExit(f"date {args.date} not in tensor range {dates[0]} .. {dates[-1]}")
    if day_idx < d.T_IN - 1 or day_idx >= len(dates) - d.HORIZON:
        raise SystemExit(f"date {args.date} outside valid forecast range")

    risk = score_day(model, meta, fields, day_idx)
    n_riau = int((labels[day_idx] != -1).sum())
    logger.info("Riau cells with label on %s: %d; risk>0 cells: %d",
                args.date, n_riau, int((~np.isnan(risk)).sum()))

    threshold = args.threshold
    if threshold is None:
        threshold = meta["best_thresholds"].get("lgbm", 0.5)
        logger.info("using checkpoint F1 threshold: %.4f", threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.grid_csv)
    boundary = gpd.read_file(args.boundary_gpkg)
    out_path = args.out / f"risk_{args.date}.png"
    render_risk_grid(risk, args.date, cells, boundary, out_path, threshold=threshold)

    # Also dump the raw risk grid for programmatic use.
    np.save(args.out / f"risk_{args.date}.npy", risk)
    logger.info("risk grid → %s", args.out / f"risk_{args.date}.npy")


if __name__ == "__main__":
    main()
