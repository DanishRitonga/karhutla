"""
real_data.py
============
Replaces the synthetic LABEL SOURCE with the real FIRMS VIIRS-SNPP data the
user uploaded (viirs-snpp_2019..2023_all_countries.zip -> Indonesia.csv).
This is exactly the label source in Table 1 of the paper ("FIRMS VIIRS 375m
... Hotspot labels ... Daily").

What is real here:
  - hotspot detections (lat/lon/date/confidence), gridded via the PROJECT'S
    OFFICIAL grid definition (`grid_definition.py`, design log section 4):
    a 5 km equal-area grid in the Indonesia Equal Area Conic (Albers)
    projection, built over the real Riau boundary. Points are DROPPED (not
    clipped) if they fall outside the bounding box, and only cells whose
    centre lies inside the real province polygon (`is_riau`) receive labels
    -- see "Grid source" note below.
  - the 5 km grid rasterization of those detections
  - the k=2 persistence label rule (>=2 valid detections in the 7-day
    target window), applied exactly as Section 3.1 states it
  - the fire-history ("hotspot_count_lag") channel used in the operational
    regime -- a real, causal, past-only rolling count of real detections
  - the train/test split -- real calendar years, train=2019-2022,
    test=2023, exactly Section 3.5's primary evaluation protocol

What is still NOT real: ERA5-Land, CHIRPS, Sentinel-1 SAR, and Dynamic
World (channels 0-19) are still the synthetic, spatially/temporally
smoothed placeholder fields from data.py, generated independently of the
real fire occurrence (i.e. NOT engineered to correlate with it). This is
an important honesty point: any skill the environmental regime shows on
those channels is either noise or a coincidental artifact of the
placeholder generator, not real meteorological signal. The operational
regime's fire-history channel, by contrast, carries a genuine signal
because real fires really do cluster in time and space. gee.py (teammate's
ingestion script for CHIRPS v3/Sentinel-1) exists and is functional but
requires an authenticated Earth Engine run per month, per source -- that
hasn't been executed for the full 2019-2023 window yet, only spot-checked
for single dates (see the *_20190131_*.png / *_20190115_*.png previews),
so those four channels stay synthetic here until that CSV output exists.

Peat (channel 20) IS now real -- see "Grid + peat source" below.

Confidence filter matches Table 2 ("Confidence filter (nominal + high)"):
we keep confidence in {'n', 'h'}, dropping 'l' (low).

Grid + peat source
-------------------
`grid_definition.py` is the project's locked design (design log section 4):
an equal-area Albers grid (NOT a degree-based lat/lon grid -- Riau spans two
UTM zones, so a degree grid distorts cell area east-to-west), built from the
Riau boundary fetched from BIG's (Badan Informasi Geospasial) official
ArcGIS REST service.

That service (kspservices.big.go.id) is NOT reachable from this sandbox --
it is outside the network allowlist and returns HTTP 403 here, so this
module can't build that grid itself. But a teammate's environment DID
reach it: they ran `grid_definition.py` + `peat.py` there and handed over
the results in `real_grid_data/` (grid_cells.csv, grid_meta.json,
riau_boundary_aea.gpkg, peat_cell.csv). `_get_grid()` below loads that
prebuilt grid via `real_grid.py` when present -- no boundary geometry or
network call needed at runtime, since cell assignment is a pure affine
transform once the grid is built (see real_grid.py docstring). If
`real_grid_data/` is missing, it falls back to the old
`riau_boundary_fallback.geojson` path (kept for reproducibility / if this
runs somewhere without the handoff files).

Resulting grid (from the real BIG boundary, via the teammate's handoff):
85 cols x 82 rows bbox cells at 5 km -- same dimensions as the old
fallback grid (same bbox, same 5 km cells) -- but 3,598 of those cells
have centre inside the real Riau polygon (51.6% fill) vs the fallback's
3,356 (48.1%), since the authoritative BIG boundary is more precise than
the GitHub-mirrored fallback. Because dimensions match, this swap needed
no changes to PATCH-margin logic or any downstream tensor shapes.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import grid_definition as gd
import real_grid

from data import (
    PATCH, T_IN, HORIZON, CENTER, N_CHANNELS, CHANNEL_NAMES,
    ENV_CHANNELS, OPERATIONAL_CHANNELS, FIRE_HISTORY_IDX,
    generate_riau_fields,
)

DATE_START, DATE_END = "2019-01-01", "2023-12-31"

_HERE = os.path.dirname(__file__)
_FALLBACK_BOUNDARY = os.path.join(_HERE, "riau_boundary_fallback.geojson")
_REAL_GRID_DIR = os.path.join(_HERE, "real_grid_data")
_PEAT_CSV = os.path.join(_REAL_GRID_DIR, "peat_cell.csv")
PEAT_IDX = CHANNEL_NAMES.index("peat_depth")  # channel 20

_GRID = None       # module-level cache, built once via _get_grid()
_GRID_SOURCE = None  # "real_grid_data" | "fallback_geojson", set by _get_grid()


def _get_grid():
    """Build (once) and cache the official equal-area grid.

    Prefers the teammate's prebuilt real-boundary grid
    (real_grid_data/grid_cells.csv + grid_meta.json, loaded via
    real_grid.PrebuiltGrid -- no network needed). Falls back to
    grid_definition.py's live-fetch-or-fallback-geojson path only if
    real_grid_data/ isn't present.
    """
    global _GRID, _GRID_SOURCE
    if _GRID is not None:
        return _GRID

    prebuilt = real_grid.try_load_prebuilt_grid(_REAL_GRID_DIR)
    if prebuilt is not None:
        _GRID = prebuilt
        _GRID_SOURCE = "real_grid_data"
        print(f"  [grid] using real BIG-boundary grid from {_REAL_GRID_DIR} "
              f"({int(prebuilt.cells['is_riau'].sum())} Riau cells)")
        return _GRID

    config = gd.GridConfig(boundary_fallback=_FALLBACK_BOUNDARY)
    boundary = gd.RiauBoundary(config).load()
    _GRID = gd.EqualAreaGrid(config).build(boundary)
    _GRID_SOURCE = "fallback_geojson"
    print("  [grid] real_grid_data/ not found -- using riau_boundary_fallback.geojson "
          "(run teammate's grid_definition.py + peat.py and drop the output in "
          "real_grid_data/ for the authoritative boundary + real peat channel)")
    return _GRID


def load_viirs_riau(csv_paths_by_year: dict[int, str]) -> pd.DataFrame:
    """Load + filter the yearly Indonesia.csv extracts using the official
    equal-area grid's own point classification: `assign_cell_idx` DROPS
    (does not clip) any detection outside the 5 km bounding-box grid, and
    the paper's confidence filter (nominal + high) is applied on top."""
    grid = _get_grid()
    frames = []
    for year, path in csv_paths_by_year.items():
        df = pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "confidence"])
        cell_idx = grid.assign_cell_idx(df.longitude.values, df.latitude.values)
        n_before = len(df)
        df = df[cell_idx >= 0]
        n_in_bbox = len(df)
        df = df[df.confidence.isin(["n", "h"])]
        print(f"  {year}: {n_before} raw -> {n_in_bbox} inside grid bbox "
              f"({n_before - n_in_bbox} dropped as outside bbox, pre-confidence-filter) "
              f"-> {len(df)} after confidence filter")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["acq_date"] = pd.to_datetime(out["acq_date"])
    return out


def rasterize(df: pd.DataFrame):
    """Grid the filtered detections into a daily [n_days, grid_h, grid_w]
    count array using the official equal-area grid's cell assignment
    (Albers-projected, not degree-based), plus the date index and per-day
    calendar year (used for the real train/test split)."""
    grid = _get_grid()
    grid_h, grid_w = grid.rows, grid.cols

    cell_idx = grid.assign_cell_idx(df.longitude.values, df.latitude.values)
    in_bbox = cell_idx >= 0
    row = (cell_idx[in_bbox] // grid_w).astype(np.int64)
    col = (cell_idx[in_bbox] % grid_w).astype(np.int64)
    df = df[in_bbox]

    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    date_to_idx = {d.date(): i for i, d in enumerate(dates)}
    day_idx = df["acq_date"].dt.date.map(date_to_idx).values
    n_days = len(dates)

    daily_count = np.zeros((n_days, grid_h, grid_w), dtype=np.int16)
    np.add.at(daily_count, (day_idx, row, col), 1)
    years = pd.Series(dates).dt.year.values
    return daily_count, years, grid_h, grid_w, n_days


def is_riau_mask() -> np.ndarray:
    """[grid_h, grid_w] boolean mask: True where the cell CENTRE lies
    inside the real Riau boundary. Only these cells should be sampled for
    labels/evaluation (design log rule -- bbox cells outside the province
    still carry features, for patch context, but are never labelled)."""
    grid = _get_grid()
    by_idx = grid.cells.sort_values("cell_idx")["is_riau"].values
    return by_idx.reshape(grid.rows, grid.cols)


def labels_from_counts(daily_count, k=2, horizon=HORIZON):
    """Exact rule from Section 3.1: '>=2 valid VIIRS hotspot detections
    within the [7-day] prediction window' -- total detection count in the
    future window, not number of days-with-a-detection."""
    n_days = daily_count.shape[0]
    labels = np.zeros(daily_count.shape, dtype=np.int8)
    for t in range(n_days - horizon):
        labels[t] = (daily_count[t + 1: t + 1 + horizon].sum(axis=0) >= k)
    return labels


def fire_history_from_counts(daily_count, window=T_IN):
    """Causal rolling count of real past detections -> operational-regime
    hotspot_count_lag channel. Backward-looking only, no leakage."""
    n_days = daily_count.shape[0]
    cum = np.concatenate([np.zeros((1,) + daily_count.shape[1:]), np.cumsum(daily_count, axis=0)], axis=0)
    start = np.clip(np.arange(n_days) - window + 1, 0, None)
    end = np.arange(n_days) + 1
    return (cum[end] - cum[start]).astype(np.float32)


def _sample_split(labels, years, valid_t, valid_rs, valid_cs, year_filter,
                   n_samples, pos_frac, rng):
    """Stratified sample of (t, r, c) cell-days from one calendar split,
    oversampling positives to `pos_frac` of the requested sample size
    (purely for computational tractability of building [15,15,14,C]
    tensors -- see README for the honest caveat about how this shifts the
    apparent PR-AUC relative to the true, far more extreme, population
    prevalence).

    `valid_rs`/`valid_cs` are PAIRED 1-D arrays (same length, one entry per
    eligible cell) rather than independent row/col ranges, because the set
    of eligible cells is `is_riau AND has room for a full patch` -- an
    irregular shape carved out of the bounding-box grid by the real
    province polygon, not a rectangle. Fancy-indexing `labels[t, rs, cs]`
    pairs them element-wise instead of taking their outer product.
    """
    t_sel = valid_t[year_filter(years[valid_t])]
    sub = labels[t_sel][:, valid_rs, valid_cs]  # [n_t_sel, n_eligible_cells]
    pos_idx = np.argwhere(sub == 1)
    neg_idx = np.argwhere(sub == 0)
    n_pos = min(len(pos_idx), int(n_samples * pos_frac))
    n_neg = min(len(neg_idx), n_samples - n_pos)
    sel_pos = pos_idx[rng.choice(len(pos_idx), size=n_pos, replace=False)]
    sel_neg = neg_idx[rng.choice(len(neg_idx), size=n_neg, replace=False)]
    sel = np.concatenate([sel_pos, sel_neg], axis=0)
    rng.shuffle(sel)
    # map back to global day index / real grid row,col
    out = np.stack([t_sel[sel[:, 0]], valid_rs[sel[:, 1]], valid_cs[sel[:, 1]]], axis=1)
    return out, len(pos_idx), len(neg_idx)


def build_real_dataset(csv_paths_by_year, n_train_samples=1000, n_test_samples=1000,
                        train_pos_frac=0.25, test_pos_frac=0.10, seed=42):
    """
    Full real-label pipeline:
      1. load + filter real VIIRS detections (Riau bbox, confidence n/h)
      2. rasterize to the 5 km grid, daily counts
      3. real k=2 persistence labels + real causal fire-history channel
      4. synthetic placeholder environmental channels 0-20 (see module
         docstring for why), generated independently of the real fire signal
      5. stratified sample from the REAL calendar train (2019-2022) and
         test (2023) periods separately -- exact match to paper Sec. 3.5

    Returns a dict with X_train, y_train, X_test, y_test (both
    [N,14,15,15,22]) plus prevalence metadata for honest reporting.
    """
    rng = np.random.default_rng(seed)
    df = load_viirs_riau(csv_paths_by_year)
    daily_count, years, grid_h, grid_w, n_days = rasterize(df)
    labels = labels_from_counts(daily_count, k=2, horizon=HORIZON)
    fire_history = fire_history_from_counts(daily_count, window=T_IN)

    fields, _, _ = generate_riau_fields(n_days, grid_h, grid_w, seed=seed, include_seasonal=False)
    fields[..., FIRE_HISTORY_IDX] = fire_history

    # Fase 8: a genuine 7-day rolling hotspot count, computed straight from
    # the raw daily counts (NOT derived from the fire_history channel
    # above, which is already a 14-day rolling sum -- summing 7 of its
    # values would double-count overlapping days). Kept OUT of `fields` /
    # the tensor (so ConvLSTM/Transformer are unaffected) and only
    # extracted as a per-sample scalar below, for the tabular baselines.
    fire_history_7d = fire_history_from_counts(daily_count, window=7)

    # Fase 7: overlay real CHIRPS/Sentinel-1 wherever gee_ingest.py has
    # actually been run; any (day, cell) without real coverage yet keeps
    # the synthetic placeholder untouched, so this is a no-op (identical
    # to Fase 6 behaviour) until those CSVs exist.
    try:
        from real_environmental_data import fill_real_environmental_channels
        fields, env_coverage = fill_real_environmental_channels(fields, DATE_START)
    except ImportError:
        env_coverage = None
        print("  [info] real_environmental_data.py tidak ditemukan -- channel ERA5/CHIRPS/S1/DW tetap sintetis")

    # Real peat, from the teammate's real_grid_data/peat_cell.csv (BIG's
    # Peta Fungsi Ekosistem Gambut layer, area-weighted per 5 km cell).
    # Static across time, so broadcast the same [grid_h, grid_w] field to
    # every day. Only runs if the file is present (same handoff as the
    # real grid, so row/col always line up with `fields`' shape here).
    peat_coverage = None
    if os.path.exists(_PEAT_CSV):
        peat_depth_grid, peat_coverage = real_grid.load_peat_depth_grid(_PEAT_CSV, grid_h, grid_w)
        fields[..., PEAT_IDX] = peat_depth_grid[None, :, :]  # broadcast over n_days
        print(f"  [peat] real peat_depth_m loaded from {_PEAT_CSV} "
              f"({(peat_depth_grid > 0).sum()} cells with peat, "
              f"{peat_coverage:.1%} of Riau cells)")
    else:
        print("  [peat] real_grid_data/peat_cell.csv tidak ditemukan -- channel peat tetap sintetis")

    valid_t = np.arange(T_IN - 1, n_days - HORIZON)

    # Eligible cells = is_riau (design-log rule: only these get labelled /
    # evaluated) AND far enough from the bbox edge for a full 15x15 patch.
    # This is an irregular shape (the province outline), not a rectangle,
    # so we enumerate it as paired (row, col) coordinates rather than
    # independent row/col ranges.
    riau_mask = is_riau_mask()  # [grid_h, grid_w]
    margin_mask = np.zeros_like(riau_mask)
    margin_mask[CENTER:grid_h - CENTER, CENTER:grid_w - CENTER] = True
    eligible = riau_mask & margin_mask
    valid_rs, valid_cs = np.where(eligible)
    n_riau_total = int(riau_mask.sum())
    n_eligible = len(valid_rs)
    if n_eligible < n_riau_total:
        print(f"  note: {n_riau_total - n_eligible} Riau cells sit within {CENTER} cells of the "
              f"bbox edge and were dropped (can't fit a full {PATCH}x{PATCH} patch), "
              f"leaving {n_eligible}/{n_riau_total} eligible Riau cells")

    train_sel, n_pos_train_pool, n_neg_train_pool = _sample_split(
        labels, years, valid_t, valid_rs, valid_cs, lambda y: y <= 2022,
        n_train_samples, train_pos_frac, rng)
    test_sel, n_pos_test_pool, n_neg_test_pool = _sample_split(
        labels, years, valid_t, valid_rs, valid_cs, lambda y: y == 2023,
        n_test_samples, test_pos_frac, rng)

    def extract(sel):
        N = len(sel)
        X = np.zeros((N, T_IN, PATCH, PATCH, N_CHANNELS), dtype=np.float32)
        y = np.zeros(N, dtype=np.int64)
        day_index = np.zeros(N, dtype=np.int64)
        hotspot_7d = np.zeros(N, dtype=np.float32)
        for i, (t, r, c) in enumerate(sel):
            X[i] = fields[t - T_IN + 1: t + 1, r - CENTER: r + CENTER + 1, c - CENTER: c + CENTER + 1, :]
            y[i] = labels[t, r, c]
            day_index[i] = t
            hotspot_7d[i] = fire_history_7d[t, r, c]
        return X, y, day_index, hotspot_7d

    X_train, y_train, day_train, hotspot_7d_train = extract(train_sel)
    X_test, y_test, day_test, hotspot_7d_test = extract(test_sel)

    meta = {
        "grid_h": grid_h, "grid_w": grid_w, "n_days": n_days,
        "true_prevalence_train": n_pos_train_pool / max(n_pos_train_pool + n_neg_train_pool, 1),
        "true_prevalence_test": n_pos_test_pool / max(n_pos_test_pool + n_neg_test_pool, 1),
        "sample_prevalence_train": float(y_train.mean()),
        "sample_prevalence_test": float(y_test.mean()),
        "n_real_detections": int(len(df)),
        "n_pos_pool_train": int(n_pos_train_pool), "n_pos_pool_test": int(n_pos_test_pool),
        "n_neg_pool_train": int(n_neg_train_pool), "n_neg_pool_test": int(n_neg_test_pool),
        "n_eligible_pool_train": int(n_pos_train_pool + n_neg_train_pool),
        "n_eligible_pool_test": int(n_pos_test_pool + n_neg_test_pool),
        "env_coverage": env_coverage,
        "peat_coverage": peat_coverage,
        "grid_source": _GRID_SOURCE,
    }
    return {
        "X_train": X_train, "y_train": y_train, "day_train": day_train,
        "X_test": X_test, "y_test": y_test, "day_test": day_test,
        "hotspot_count_7d_train": hotspot_7d_train, "hotspot_count_7d_test": hotspot_7d_test,
        "meta": meta,
    }
