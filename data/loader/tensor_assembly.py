"""tensor_assembly.py
====================
Assembles the real ingested per-cell, per-day CSVs into dense daily field
rasters ``[n_days, grid_h, grid_w, N_CHANNELS]`` + labels, the exact input
layout the jett model pipeline expects (``model/data.py``), but with the
real environmental channels filled in (no synthetic placeholders).

Channel layout (``CHANNEL_NAMES``, N_CHANNELS=23):

  0..7   ERA5-Land:        t2m, d2m, u10, v10, swvl1, swvl2, ssr, tp
  8      CHIRPS:           chirps_precip
  9..11  Sentinel-1:       sar_vv, sar_vh, sar_available
  12..19 Dynamic World:    dw_water, dw_trees, dw_grass, dw_flooded_veg,
                           dw_crops, dw_shrub_scrub, dw_built, dw_bare
  20     Dynamic World:    dw_available      (0+mask channel, NEW)
  21     Peat:             peat_depth
  22     Fire history:     hotspot_count_lag (rolling sum, T_IN window)

  ENV_CHANNELS        = 0..20   (21 channels, no fire history)
  OPERATIONAL_CHANNELS= 0..22   (23 channels, incl. fire history)
  FIRE_HISTORY_IDX    = 22

Missing-data policy (0+mask, per user decision):
  * Dynamic World is a sparse product (Sentinel-2 cloud cover). A cell/date
    with no DW row gets ALL nine class probs set to 0 AND dw_available=0.
    Cells with a real row keep their probs and dw_available=1.
  * Sentinel-1: forward-filled CSVs carry NaN where the gap exceeded the
    14-day cap. sar_vv/sar_vh = 0 where NaN, sar_available = 1 where a
    valid value exists, else 0.
  * CHIRPS is a complete daily product (missing = sea cells, out of CHIRPS
    land mask): filled with 0.
  * ERA5-Land covers a static 5,821-cell footprint (cells whose center falls
    outside the 0.1-degree grid are never returned). Missing cells are
    spatially filled from the nearest valid cell by projected distance
    (values are smooth at ~11 km native resolution) — NOT 0-filled.
  * Peat is static, broadcast across all days.
  * Fire history (channel 22) = rolling sum of per-cell daily hotspot
    counts over the previous T_IN=14 days (lag, no future leakage).

Labels:
  * labels.npy int8 [n_days, grid_h, grid_w]: 1 = fire in (t, t+HORIZON],
    0 = no fire, -1 = cell has no label (non-land bbox cell).

Run (from datathon root):

    uv run --python 3.12 python data/tensor_assembly.py \
        --start 2019-01-01 --end 2023-12-31 --out data/output/tensors

Produces ``data.npy``, ``labels.npy``, ``meta.json`` in the out dir.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tensor_assembly")

# --------------------------------------------------------------------------- #
# Canonical channel layout (matches jett model/data.py + dw_available)
# --------------------------------------------------------------------------- #

T_IN = 14
HORIZON = 7
PATCH = 15
CENTER = 7

ERA5_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]
ERA5_CHANNELS = ["t2m", "d2m", "u10", "v10", "swvl1", "swvl2", "ssr", "tp"]

DW_CSV_COLS = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare",
]
DW_CHANNELS = [
    "dw_water", "dw_trees", "dw_grass", "dw_flooded_veg",
    "dw_crops", "dw_shrub_scrub", "dw_built", "dw_bare",
]

CHANNEL_NAMES = (
    ERA5_CHANNELS
    + ["chirps_precip"]
    + ["sar_vv", "sar_vh", "sar_available"]
    + DW_CHANNELS
    + ["dw_available", "peat_depth", "hotspot_count_lag"]
)
N_CHANNELS = len(CHANNEL_NAMES)
FIRE_HISTORY_IDX = CHANNEL_NAMES.index("hotspot_count_lag")
PEAT_DEPTH_IDX = CHANNEL_NAMES.index("peat_depth")
ENV_CHANNELS = list(range(21))           # 0..20 (excludes fire history)
OPERATIONAL_CHANNELS = list(range(N_CHANNELS))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_paths(start: date, end: date, dirname: str, prefix: str) -> list[Path]:
    """Paths for every calendar month overlapping [start, end]."""
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(Path(dirname) / f"{prefix}_{y:04d}{m:02d}.csv")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def _load_frame(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no data files matched {paths[0].parent}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Per-source daily rasters -> [n_days, H, W, C] fields array
# --------------------------------------------------------------------------- #


class FieldAssembler:
    """Builds the dense [n_days, H, W, N_CHANNELS] fields array from CSVs."""

    def __init__(
        self,
        data_dir: Path = Path("data/output"),
        grid_h: int = 82,
        grid_w: int = 85,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.grid_h = grid_h
        self.grid_w = grid_w

    def _empty_day(self, c: int = N_CHANNELS) -> np.ndarray:
        return np.zeros((self.grid_h, self.grid_w, c), dtype=np.float32)

    def _rasterize(
        self,
        df: pd.DataFrame,
        value_col: str,
        days: list[date],
        cells: pd.DataFrame,
    ) -> np.ndarray:
        """df (cell_idx,row,col,date,value) -> [n_days,H,W] float32."""
        out = np.zeros((len(days), self.grid_h, self.grid_w), dtype=np.float32)
        cell_rc = cells.set_index("cell_idx")[["row", "col"]]
        by_date = {d: g for d, g in df.groupby("date")}
        for di, d in enumerate(days):
            sub = by_date.get(d.isoformat())
            if sub is None or sub.empty:
                continue
            merged = sub.set_index("cell_idx")[[value_col]].join(cell_rc, how="inner")
            out[di, merged["row"].to_numpy(), merged["col"].to_numpy()] = (
                merged[value_col].to_numpy()
            )
        return out

    def _rasterize_multi(
        self,
        df: pd.DataFrame,
        value_cols: list[str],
        days: list[date],
        cells: pd.DataFrame,
    ) -> np.ndarray:
        """df (cell_idx,row,col,date,<cols...>) -> [n_days,H,W,len(cols)]."""
        out = np.zeros(
            (len(days), self.grid_h, self.grid_w, len(value_cols)), dtype=np.float32
        )
        cell_rc = cells.set_index("cell_idx")[["row", "col"]]
        by_date = {d: g for d, g in df.groupby("date")}
        for di, d in enumerate(days):
            sub = by_date.get(d.isoformat())
            if sub is None or sub.empty:
                continue
            merged = sub.set_index("cell_idx")[value_cols].join(cell_rc, how="inner")
            r, c = merged["row"].to_numpy(), merged["col"].to_numpy()
            for vi, vcol in enumerate(value_cols):
                out[di, r, c, vi] = merged[vcol].to_numpy()
        return out

    def chirps(self, days: list[date], cells: pd.DataFrame) -> np.ndarray:
        paths = _month_paths(days[0], days[-1], self.data_dir / "chirpsv3", "chirps_v3sat")
        df = _load_frame(paths)
        return self._rasterize(df, "precip_mm", days, cells)

    def _static_nearest_fill_map(
        self,
        cells: pd.DataFrame,
        present_cell_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (fill_row, fill_col) 2D grids of shape (H, W).

        fill_row[r, c] / fill_col[r, c] give the (row, col) of the nearest
        cell that has real data (``present_cell_idx``, the static 5,821-cell
        ERA5-Land footprint). Cells that are present map to themselves;
        missing cells map to their nearest present neighbour by projected
        distance (cKDTree over x/y center metres).
        """
        valid = cells[cells["cell_idx"].isin(present_cell_idx)]
        valid_row = valid["row"].to_numpy()
        valid_col = valid["col"].to_numpy()
        tree = cKDTree(
            np.c_[valid["x_center_m"].to_numpy(), valid["y_center_m"].to_numpy()]
        )
        all_xy = np.c_[cells["x_center_m"].to_numpy(), cells["y_center_m"].to_numpy()]
        _, idx = tree.query(all_xy, k=1)

        fill_row = np.zeros((self.grid_h, self.grid_w), dtype=np.int64)
        fill_col = np.zeros((self.grid_h, self.grid_w), dtype=np.int64)
        fill_row[cells["row"].to_numpy(), cells["col"].to_numpy()] = valid_row[idx]
        fill_col[cells["row"].to_numpy(), cells["col"].to_numpy()] = valid_col[idx]
        return fill_row, fill_col

    def era5(self, days: list[date], cells: pd.DataFrame) -> np.ndarray:
        """Returns [n_days,H,W,8] with ERA5-Land channels.

        ERA5-Land only covers 5,821 of 6,970 cells (cells whose center falls
        outside the 0.1-degree grid footprint at scale 9000 are never
        returned). The missing set is static geography; each missing cell is
        filled from its nearest valid cell (values are spatially smooth at
        ~11 km native resolution). Two missingness modes are both filled:
          * cells absent on a given day -> 0 (from _rasterize_multi init)
          * cells present-but-NaN on a day -> NaN (full-coverage days)
        See module docstring missing-data policy.
        """
        paths = _month_paths(days[0], days[-1], self.data_dir / "era5land", "era5land")
        df = _load_frame(paths)
        out = self._rasterize_multi(df, ERA5_BANDS, days, cells)

        valid = df.dropna(subset=ERA5_BANDS)
        valid_idx = valid["cell_idx"].unique()
        if len(valid_idx) < len(cells):
            fill_row, fill_col = self._static_nearest_fill_map(cells, valid_idx)
            by_date = {d: g for d, g in df.groupby("date")}
            for di, d in enumerate(days):
                present = np.zeros((self.grid_h, self.grid_w), dtype=bool)
                sub = by_date.get(d.isoformat())
                if sub is not None:
                    present[sub["row"].to_numpy(), sub["col"].to_numpy()] = True
                nan_mask = np.isnan(out[di])
                to_fill = nan_mask | (~present)[:, :, None]
                if to_fill.any():
                    src = out[di][fill_row, fill_col, :]
                    out[di] = np.where(to_fill, src, out[di])
            logger.info(
                "era5: spatial-filled %d of %d cells from nearest valid",
                len(cells) - len(valid_idx),
                len(cells),
            )
        return out

    def sentinel1(self, days: list[date], cells: pd.DataFrame) -> np.ndarray:
        """Returns [n_days,H,W,3] = (sar_vv, sar_vh, sar_available).

        Both orbits concatenated; a cell/date is 'available' if either
        orbit has a valid (non-NaN) value. NaN values are replaced by 0.
        """
        out = np.zeros((len(days), self.grid_h, self.grid_w, 3), dtype=np.float32)
        for orbit in ("ASCENDING", "DESCENDING"):
            paths = _month_paths(
                days[0], days[-1], self.data_dir / "sentinel1_filled", f"s1_{orbit}"
            )
            frames = [pd.read_csv(p) for p in paths if p.exists()]
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            # One row per (cell,date): pick the orbit value that is not NaN.
            df = df.sort_values("cell_idx").drop_duplicates(
                ["cell_idx", "date"], keep="last"
            )
            cell_rc = cells.set_index("cell_idx")[["row", "col"]]
            avail_col = f"{orbit.lower()}_avail"
            df[avail_col] = df[["vv_db", "vh_db"]].notna().any(axis=1).astype(np.float32)
            by_date = {d: g for d, g in df.groupby("date")}
            for di, d in enumerate(days):
                sub = by_date.get(d.isoformat())
                if sub is None or sub.empty:
                    continue
                r, c = sub["row"].to_numpy(), sub["col"].to_numpy()
                out[di, r, c, 0] = sub["vv_db"].fillna(0.0).to_numpy()
                out[di, r, c, 1] = sub["vh_db"].fillna(0.0).to_numpy()
                out[di, r, c, 2] = np.maximum(
                    out[di, r, c, 2], sub[avail_col].to_numpy()
                )
        return out

    def dynamic_world(self, days: list[date], cells: pd.DataFrame) -> np.ndarray:
        """Returns [n_days,H,W,9] = 8 class probs + dw_available.

        0+mask: missing (cell,date) -> class probs 0, dw_available 0.
        """
        out = np.zeros((len(days), self.grid_h, self.grid_w, 9), dtype=np.float32)
        paths = _month_paths(days[0], days[-1], self.data_dir / "dynamic_world", "dynamic_world")
        frames = [pd.read_csv(p) for p in paths if p.exists()]
        if not frames:
            logger.warning("no dynamic_world CSVs found; DW channels stay 0")
            return out
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        by_date = {d: g for d, g in df.groupby("date")}
        for di, d in enumerate(days):
            sub = by_date.get(d.isoformat())
            if sub is None or sub.empty:
                continue
            r, c = sub["row"].to_numpy(), sub["col"].to_numpy()
            for vi, col in enumerate(DW_CSV_COLS):
                out[di, r, c, vi] = sub[col].to_numpy()
            out[di, r, c, 8] = 1.0
        return out

    def peat(self, cells: pd.DataFrame) -> np.ndarray:
        """Static [H,W] peat depth, broadcast by caller across days."""
        path = self.data_dir / "peat" / "peat_cell.csv"
        df = pd.read_csv(path)
        out = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        out[df["row"].to_numpy(), df["col"].to_numpy()] = df["peat_depth_m"].to_numpy()
        return out

    def fire_history(self, daily_count: np.ndarray) -> np.ndarray:
        """Rolling sum of daily hotspot counts over T_IN days (lag window).

        daily_count: [n_days, H, W] int. Returns [n_days, H, W] where out[t]
        = sum(daily_count[t-T_IN+1 .. t]) -- strictly past, no future info.
        """
        n_days = daily_count.shape[0]
        out = np.zeros_like(daily_count, dtype=np.float32)
        csum = np.concatenate(
            [np.zeros((1,) + daily_count.shape[1:], dtype=daily_count.dtype),
             daily_count.cumsum(axis=0)], axis=0
        )
        for t in range(n_days):
            lo = max(0, t - T_IN + 1)
            out[t] = csum[t + 1] - csum[lo]
        return out

    def assemble(
        self,
        days: list[date],
        daily_count: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Build [n_days,H,W,N_CHANNELS] fields array.

        daily_count: [n_days,H,W] per-cell daily hotspot counts (from VIIRS),
        used only for the fire-history channel.
        """
        grid_csv = self.data_dir / "grid" / "grid_cells.csv"
        cells = pd.read_csv(grid_csv)

        f_era = self.era5(days, cells)          # [D,H,W,8]
        f_ch = self.chirps(days, cells)         # [D,H,W]
        f_s1 = self.sentinel1(days, cells)      # [D,H,W,3]
        f_dw = self.dynamic_world(days, cells)  # [D,H,W,9]
        p_peat = self.peat(cells)               # [H,W]
        f_fire = self.fire_history(daily_count)  # [D,H,W]

        d, h, w = len(days), self.grid_h, self.grid_w
        fields = np.zeros((d, h, w, N_CHANNELS), dtype=np.float32)
        fields[:, :, :, 0:8] = f_era
        fields[:, :, :, 8] = f_ch
        fields[:, :, :, 9:12] = f_s1
        fields[:, :, :, 12:21] = f_dw
        fields[:, :, :, PEAT_DEPTH_IDX] = p_peat[np.newaxis, :, :]
        fields[:, :, :, FIRE_HISTORY_IDX] = f_fire

        meta = {
            "n_days": d,
            "grid_h": h,
            "grid_w": w,
            "n_channels": N_CHANNELS,
            "channel_names": CHANNEL_NAMES,
            "env_channels": ENV_CHANNELS,
            "operational_channels": OPERATIONAL_CHANNELS,
            "fire_history_idx": FIRE_HISTORY_IDX,
            "t_in": T_IN,
            "horizon": HORIZON,
            "dates": [str(x) for x in days],
        }
        return fields, meta


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def build_labels(
    days: list[date],
    data_dir: Path,
    cells: pd.DataFrame,
    k: int = 2,
    horizon: int = HORIZON,
) -> np.ndarray:
    """labels.npy [n_days,H,W] int8 from VIIRS k-over-horizon labels.

    -1 = no label cell (non-land), 0 = no fire, 1 = fire in (t, t+horizon].
    Re-reads the precomputed per-day label CSVs.
    """
    paths = sorted((data_dir / "viirs").glob("labels_*.csv"))
    df = _load_frame(paths)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    cell_rc = cells.set_index("cell_idx")[["row", "col"]]
    out = np.full((len(days), cells["row"].max() + 1, cells["col"].max() + 1), -1, dtype=np.int8)
    by_date = {d: g for d, g in df.groupby("date")}
    for di, d in enumerate(days):
        sub = by_date.get(d.isoformat())
        if sub is None or sub.empty:
            continue
        merged = sub.set_index("cell_idx")[["fire_label"]].join(cell_rc, how="inner")
        r, c = merged["row"].to_numpy(), merged["col"].to_numpy()
        out[di, r, c] = merged["fire_label"].to_numpy().astype(np.int8)
    return out


def _assign_cell_idx(
    lon: np.ndarray,
    lat: np.ndarray,
    grid_meta: dict,
) -> np.ndarray:
    """Project lon/lat -> Albers -> floor-divide -> cell_idx (-1 outside bbox).

    Mirrors grid_definition.EqualAreaGrid.assign_cell_idx exactly.
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", grid_meta["crs"], always_xy=True)
    x, y = transformer.transform(lon, lat)
    cell = grid_meta["cell_size_m"]
    col = np.floor((np.asarray(x) - grid_meta["x0"]) / cell).astype(np.int64)
    row = np.floor((np.asarray(y) - grid_meta["y0"]) / cell).astype(np.int64)
    valid = (col >= 0) & (col < grid_meta["cols"]) & (row >= 0) & (row < grid_meta["rows"])
    idx = np.full(len(lon), -1, dtype=np.int64)
    idx[valid] = row[valid] * grid_meta["cols"] + col[valid]
    return idx


def build_daily_counts(
    days: list[date],
    data_dir: Path,
    cells: pd.DataFrame,
    raw_dir: Path = Path("real_data/viirs-snpp"),
) -> np.ndarray:
    """Per-cell daily FIRMS hotspot counts [n_days,H,W] from raw extracts.

    Reads the untouched ``viirs-snpp_{year}_Indonesia.csv`` extracts (written
    by ``viirs.py --raw-csv-dir``), applies the nominal/high confidence
    filter, assigns each detection to its grid cell via exact Albers
    floor-division binning (same as the label pipeline), and counts
    detections per (cell, day).

    This is the correct fire-history source: it is a lag input feature
    (counts of fire detections in the preceding T_IN days), NOT the k=2
    horizon label -- using the labels here would leak the future.
    """
    grid_meta = json.loads((data_dir / "grid" / "grid_meta.json").read_text())
    cols = grid_meta["cols"]
    rows = grid_meta["rows"]
    out = np.zeros((len(days), rows, cols), dtype=np.float32)

    years = sorted({d.year for d in days})
    for year in years:
        path = raw_dir / str(year) / f"viirs-snpp_{year}_Indonesia.csv"
        if not path.exists():
            logger.warning("no raw VIIRS extract %s; counts stay 0", path)
            continue
        df = pd.read_csv(path, low_memory=False)
        conf = df["confidence"].astype(str).str.strip().str.lower()
        df = df[conf.isin({"n", "nominal", "h", "high"})].copy()
        if df.empty:
            continue
        idx = _assign_cell_idx(df["longitude"].to_numpy(), df["latitude"].to_numpy(), grid_meta)
        df["cell_idx"] = idx
        df = df[df["cell_idx"] >= 0]
        df["date"] = pd.to_datetime(df["acq_date"]).dt.date.astype(str)
        cell_rc = cells.set_index("cell_idx")[["row", "col"]]
        merged = df[["cell_idx", "date"]].merge(
            cell_rc, left_on="cell_idx", right_index=True, how="left"
        )
        merged = merged.dropna(subset=["row", "col"])
        counts_by_day = {
            d: g.groupby(["row", "col"]).size()
            for d, g in merged.groupby("date")
        }
        for (di), d in enumerate(days):
            counts = counts_by_day.get(d.isoformat())
            if counts is None or counts.empty:
                continue
            out[di, counts.index.get_level_values(0).to_numpy().astype(int),
                counts.index.get_level_values(1).to_numpy().astype(int)] = counts.to_numpy()
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble dense tensor from ingested CSVs")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--data-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--out", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--raw-dir", type=Path, default=Path("real_data/viirs-snpp"))
    args = parser.parse_args()

    days = pd.date_range(args.start, args.end, freq="D").date.tolist()
    grid_csv = args.data_dir / "grid" / "grid_cells.csv"
    cells = pd.read_csv(grid_csv)
    grid_meta = json.loads((args.data_dir / "grid" / "grid_meta.json").read_text())
    h, w = grid_meta["rows"], grid_meta["cols"]

    daily_count = build_daily_counts(days, args.data_dir, cells, args.raw_dir)
    assembler = FieldAssembler(data_dir=args.data_dir, grid_h=h, grid_w=w)
    fields, meta = assembler.assemble(days, daily_count)
    labels = build_labels(days, args.data_dir, cells)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "data.npy", fields)
    np.save(args.out / "labels.npy", labels)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s (data %s, labels %s)", args.out, fields.shape, labels.shape)
    logger.info("positive label days: %d", int((labels == 1).sum()))
    logger.info("nan in fields: %d", int(np.isnan(fields).sum()))


if __name__ == "__main__":
    main()
