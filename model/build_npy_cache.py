"""
build_npy_cache.py
===================
Precompute the dense per-day data & label arrays that build_real_dataset()
(in real_data.py) builds on every run, and cache them as .npy files -- so
main_real.py doesn't need to reparse gigabytes of raw FIRMS CSVs and rerun
rasterization every single time.

This is a THIN WRAPPER: it calls the exact same functions real_data.py uses
internally (load_viirs_riau, rasterize, labels_from_counts,
fire_history_from_counts, generate_riau_fields, real_grid.load_peat_depth_grid),
so the cached arrays are guaranteed identical to what build_real_dataset()
would compute from scratch -- nothing is reimplemented separately here.

Produces, in --out-dir (default: real_data/cache/):
  - data.npy   : float32 [n_days, grid_h, grid_w, N_CHANNELS]
                 Same `fields` tensor build_real_dataset() builds:
                 channels 0-19 synthetic placeholder env fields, the real
                 causal fire-history channel, and the real peat channel
                 (if real_grid_data/peat_cell.csv is present).
  - labels.npy : int8 [n_days, grid_h, grid_w]
                 Real VIIRS-derived labels (k=2 detections in the next
                 7-day window), from labels_from_counts().
  - meta.json  : n_days/grid_h/grid_w/channel_names/n_real_detections/etc,
                 so downstream code can sanity-check shapes without
                 re-importing data.py.

NOTE ON SIZE: data.npy will be roughly n_days * grid_h * grid_w * N_CHANNELS
* 4 bytes. For the full 2019-2023 window (~1826 days) x 82x85 grid x 22
channels, that's on the order of 1+ GB. If that's too large, consider only
caching `daily_count` (int16, ~25 MB) instead and deriving labels/fields on
the fly at load time -- ask and I'll write that lighter variant instead.

Usage (run from model/, with venv active, AFTER copying grid_definition.py
into model/ so real_data.py's cross-folder import resolves -- see chat):

    python build_npy_cache.py --years 2019 2023 \
        --raw-csv-dir real_data/viirs-snpp \
        --out-dir real_data/cache
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from data import CHANNEL_NAMES, N_CHANNELS, T_IN, FIRE_HISTORY_IDX, generate_riau_fields
import real_grid
from real_data import (
    load_viirs_riau, rasterize, labels_from_counts, fire_history_from_counts,
    _PEAT_CSV, PEAT_IDX, DATE_START,
)


def build_csv_paths(raw_csv_dir: str, years: list[int]) -> dict[int, str]:
    """Matches the layout viirs.py's --raw-csv-dir produces:
    <raw_csv_dir>/<year>/viirs-snpp_<year>_Indonesia.csv"""
    paths = {}
    for y in years:
        p = os.path.join(raw_csv_dir, str(y), f"viirs-snpp_{y}_Indonesia.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Tidak ketemu {p} -- pastikan viirs.py sudah dijalankan dengan "
                f"--raw-csv-dir {raw_csv_dir} untuk tahun {y}"
            )
        paths[y] = p
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs=2, default=[2019, 2023],
                     metavar=("START", "END"), help="Rentang tahun inklusif")
    ap.add_argument("--raw-csv-dir", default="real_data/viirs-snpp")
    ap.add_argument("--out-dir", default="real_data/cache")
    ap.add_argument("--seed", type=int, default=42,
                     help="Seed sintetis env channels -- HARUS sama dengan yang "
                          "dipakai build_real_dataset() (default 42) supaya cache "
                          "cocok dengan run non-cached")
    args = ap.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))
    csv_paths = build_csv_paths(args.raw_csv_dir, years)

    print(f"[1/4] Loading + filtering real VIIRS detections ({years[0]}-{years[-1]})...")
    df = load_viirs_riau(csv_paths)

    print("[2/4] Rasterizing to 5km grid (daily counts)...")
    daily_count, years_arr, grid_h, grid_w, n_days = rasterize(df)

    print("[3/4] Building labels (k=2 persistence rule) + fire-history channel...")
    labels = labels_from_counts(daily_count, k=2)
    fire_history = fire_history_from_counts(daily_count, window=T_IN)

    print("[4/4] Building environmental channels (synthetic placeholders + real peat)...")
    fields, _, _ = generate_riau_fields(n_days, grid_h, grid_w, seed=args.seed, include_seasonal=False)
    fields[..., FIRE_HISTORY_IDX] = fire_history

    peat_coverage = None
    if os.path.exists(_PEAT_CSV):
        peat_depth_grid, peat_coverage = real_grid.load_peat_depth_grid(_PEAT_CSV, grid_h, grid_w)
        fields[..., PEAT_IDX] = peat_depth_grid[None, :, :]
        print(f"  [peat] real peat_depth_m loaded ({peat_coverage:.1%} of Riau cells)")
    else:
        print(f"  [peat] {_PEAT_CSV} tidak ditemukan -- channel peat tetap sintetis")

    os.makedirs(args.out_dir, exist_ok=True)
    data_path = os.path.join(args.out_dir, "data.npy")
    labels_path = os.path.join(args.out_dir, "labels.npy")
    meta_path = os.path.join(args.out_dir, "meta.json")

    fields = fields.astype(np.float32)
    labels = labels.astype(np.int8)

    np.save(data_path, fields)
    np.save(labels_path, labels)

    meta = {
        "n_days": int(n_days),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "n_channels": int(N_CHANNELS),
        "channel_names": list(CHANNEL_NAMES),
        "years": years,
        "date_start": DATE_START,
        "n_real_detections": int(len(df)),
        "peat_coverage": peat_coverage,
        "seed": args.seed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  {data_path}   shape={fields.shape} ({fields.nbytes / 1e6:.1f} MB)")
    print(f"  {labels_path} shape={labels.shape} ({labels.nbytes / 1e6:.1f} MB)")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
