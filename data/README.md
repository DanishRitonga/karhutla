# Dataset Description — Karhutla Riau Early-Warning System

Ingested feature and label data for the peatland-fire early-warning model,
all aligned on a **fixed 5 km equal-area grid** over Riau (and the surrounding
bbox). `cell_idx` is the universal join key across every source.

## Repository layout

Local (gitignored, regenerable): `data/output/<source>/` per source.

HuggingFace mirror (`danishritonga/karhutla`, the canonical copy):

```
raw/<source>/       all per-source CSVs (389 files across 8 sources)
tensors/            pre-built tensors: data.npy [1826,82,85,23] float32,
                    labels.npy [1826,82,85] int8 (-1/0/1), meta.json
```

* `raw/chirpsv3/` — 60 monthly CSVs (CHIRPS v3 SAT daily rainfall)
* `raw/dynamic_world/` — 60 monthly CSVs (Dynamic World class probabilities)
* `raw/era5land/` — 60 monthly CSVs (ERA5-Land daily; **corrected sum-precip version**)
* `raw/grid/` — grid definition (cells + meta)
* `raw/peat/` — static peat depth/extent
* `raw/sentinel1/` — raw S1 SAR acquisitions per orbit
* `raw/sentinel1_filled/` — forward-filled daily S1 (with `filled` provenance)
* `raw/viirs/` — VIIRS hotspot labels per year

The `tensors/` folder is the model input: one `data.npy` covering
2019-01-01 → 2023-12-31 with **0 NaN** (all missing data handled via
spatial fill / mask channels — see the per-source notes below).

## Grid

Defined once by `data/grid/grid_definition.py` and frozen before any data pull
(sources are resampled onto it, never the reverse).

| File | Contents |
|---|---|
| `grid/grid_cells.csv` | One row per cell: `row, col, cell_idx, x_center_m, y_center_m, lon, lat, is_riau` |
| `grid/grid_meta.json` | CRS (Albers Indonesia Equal Area Conic, proj4), `x0, y0, cols, rows, cell_size_m, bbox_cells, riau_cells` |
| `grid/riau_boundary_aea.gpkg` | Riau admin boundary, projected |
| `grid/riau_grid.png` | Map of the grid |

Key numbers:

* **6,970 cells** total (85 cols × 82 rows over the bbox), `cell_idx` 0..6969
* **3,598 cells** inside Riau (`is_riau = 1`) — these are the primary prediction targets
* **6,506 land cells** = union of CHIRPS coverage (3,598 Riau + 2,908 non-Riau land);
  the remaining 464 bbox cells are sea (CHIRPS land-only grid skips water)
* 5,000 m cell size; CRS `+proj=aea +lat_1=-5 +lat_2=-1 +lat_0=2 +lon_0=113 ...`

`is_riau` decides which cells are *targets* (get labels and are evaluated).
All 6,970 bbox cells keep features, because neighbouring (even sea) cells carry
context for 15×15 patches in the spatiotemporal model.

## Sources

All sources are downloaded / rasterized by scripts in `data/ingest/`.

### 1. VIIRS hotspots — labels (`data/ingest/viirs.py`)

NASA FIRMS VIIRS S-NPP 375 m active-fire detections (confidence ∈ {nominal, high}),
spatially binned onto the grid with **exact Albers floor-division binning**
(`col = floor((x-x0)/cell_size)`, `row = floor((y-y0)/cell_size)`; out-of-bbox
points are dropped, not clipped). The binary label for a cell on day *t* is 1 iff
the cell has **≥ k = 2 detections within the window (t+1, t+7]** (persistence filter
against smouldering-peat false positives).

* Input: `data/raw/viirs/viirs-snpp_{year}_all_countries.zip` (kept, ~1.7 GB total)
* Raw extracts (untouched FIRMS Indonesia CSVs): `real_data/viirs-snpp/{year}/viirs-snpp_{year}_Indonesia.csv`
* Output: `viirs/labels_{year}.csv` → `cell_idx, row, col, date, fire_label`

| Year | Rows | Days | Cells | Positive labels | Pos. rate |
|---|---|---|---|---|---|
| 2019 | 2,374,690 | 365 | 6,506 | 21,652 | 0.91% |
| 2020 | 2,381,196 | 366 | 6,506 | 4,411 | 0.19% |
| 2021 | 2,374,690 | 365 | 6,506 | 3,345 | 0.14% |
| 2022 | 2,374,690 | 365 | 6,506 | 3,192 | 0.13% |
| 2023 | 2,374,690 | 365 | 6,506 | 4,941 | 0.21% |

* **Train (2019–2022): 32,600 positive** · **Test (2023): 4,941 positive**
* Peak seasons match known Riau fire years (e.g. Sep 2019: 254–277 burning cells/day).

### 2. CHIRPS v3 SAT rainfall (`data/ingest/chirpsv3.py`)

`UCSB-CHC/CHIRPS/V3/DAILY_SAT` (0.05°, daily, mm/day, IMERG-Late-derived daily
disaggregation). Mean over each 5 km cell.

* Output: `chirpsv3/chirps_v3sat_{YYYYMM}.csv` → `cell_idx, row, col, date, precip_mm`
* **60 files, 11.88M rows**, full daily coverage 2019-01-01 → 2023-12-31
* 6,506 cells (all land; sea cells legitimately absent), 0 NaN, 0 negatives
* Values 0–385 mm/day; no gap-fill needed (complete daily product)

### 3. Sentinel-1 SAR backscatter (`data/ingest/sentinel1.py`)

`COPERNICUS/S1_GRD` IW 10 m, VV/VH dB + incidence `angle` (server-side calibrated,
terrain-corrected; median over cell as speckle suppression). Separate files per
orbit (ASCENDING / DESCENDING) and month. Revisit 6–12 days, swath-limited —
**not daily**.

* Raw: `sentinel1/s1_{ORBIT}_{YYYYMM}.csv` → `cell_idx, row, col, date, vv_db, vh_db, angle_deg, vh_vv_db`
* ASC: 41 files, 1.17M rows · DESC: 41 files, 1.01M rows (0 NaN — only real obs)

**Gap-filling** (`--fill`, default on): forward-fill real acquisitions to daily
rows, capped at `max_gap = 14` days. A cell whose last real obs is older than
14 days has its value **invalidated to NaN** (a stale radar value would be
misleading). A `filled` column records provenance:

* `0` = real acquisition
* `1` + valid value = forward-filled within the 14-day cap
* `1` + NaN = gap exceeded the cap, value deliberately erased

* Filled: `sentinel1_filled/s1_{ORBIT}_{YYYYMM}.csv` → `date, row, col, vv_db, vh_db, angle_deg, vh_vv_db, cell_idx, filled`
* 120 files; ASC: 12.69M rows (97.8% valid, ~10% real) · DESC: 12.72M rows (74.7% valid, ~7% real)

**Why NaNs are correct**: they are validity sentinels, not missing-data accidents.
Tensor assembly converts them into a per-channel **validity mask** so the model
learns where radar coverage is weak. The NaN rate itself is informative — DESC is
25.3% invalid vs ASC 2.2%, i.e. a weaker signal channel the model can down-weight.
Forward-fill is causal (no future leakage, unlike backfill/interpolation) and
mirrors live BPBD dashboard operation.

### 4. Peat depth / extent — static (`data/ingest/peat.py`)

BIG / Satupeta **FEG layer 48** "Peta Fungsi Ekosistem Gambut 1:50.000" (PP 57/2016).
Peat-thickness classes (e.g. `'5,0 - 6,0 meter'`) parsed to midpoint metres;
rasterized onto the grid with **area-weighted overlay**.

* Output: `peat/peat_cell.csv` → `cell_idx, row, col, is_riau, peat_frac, peat_depth_m`
* 6,970 rows (one per grid cell); **722 cells with peat** (`peat_frac > 0`)
* Depth 0–13.32 m (area-weighted class midpoint over the peat-covered part of the cell)
* Static channel — broadcast unchanged across time in tensor assembly

### 5. ERA5-Land daily (`data/ingest/era5land.py`)

`ECMWF/ERA5_LAND/HOURLY`, reduced to daily: **state bands (temperature_2m,
dewpoint_temperature_2m, u/v_component_of_wind_10m, volumetric_soil_water_layer_1/2)
are day-meaned; flux bands (total_precipitation, surface_solar_radiation_downwards)
are day-summed** (corrected version — an earlier teammate upload used mean for
precipitation, which understates daily totals ~24×).

* Output: `era5land/era5land_{YYYYMM}.csv` → `cell_idx, row, col, date, temperature_2m, dewpoint_temperature_2m, u_component_of_wind_10m, v_component_of_wind_10m, volumetric_soil_water_layer_1, volumetric_soil_water_layer_2, total_precipitation, surface_solar_radiation_downwards`
* **60 files, 10.83M rows**, full coverage; t2m 289.4–304.1 K; swvl1 0.041–0.607
* 1,149 static edge cells fall outside the ERA5 0.1° footprint at scale 9000 —
  tensor assembly **spatially fills them from the nearest valid cell** (median
  distance 10 km ≈ ERA5 native resolution)

### 6. Dynamic World land cover (`data/ingest/dynamic_world.py`)

`GOOGLE/DYNAMICWORLD/V1` (10 m, Sentinel-2-derived, cloud-masked ≤35%).
Mean class probability per cell; sparse revisit (cloud cover) — some months have
near-zero Riau coverage. Missing (cell, date) rows get class probs = 0 **and** a
`dw_available = 0` mask channel in the tensor (0+mask convention, so the model
does not misread absent clouds as "bare/water").

* Output: `dynamic_world/dynamic_world_{YYYYMM}.csv` → `cell_idx, row, col, date, water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice, top1_class, top1_prob`
* 60 files; 430 unique dates; all 6,970 cells covered at least 11× (median 50)

## Summary

| Source | Role | Cadence | Coverage | Rows (total) | Missing |
|---|---|---|---|---|---|
| Grid | spatial frame | static | 6,970 cells | 6,970 | — |
| VIIRS | **labels** | daily | 6,506 land cells | 11.89M | none |
| CHIRPS | rainfall feature | daily | 6,506 land cells | 11.88M | sea cells only |
| S1 raw | radar feature | 6–12 d | 6,970 cells | 2.18M | swath gaps (filled) |
| S1 filled | radar feature (daily) | daily | 6,970 cells | 25.40M | >14 d gaps → NaN + mask |
| ERA5-Land | weather + soil moisture | daily | 5,821 cells native | 10.83M | edge cells → spatial fill |
| Dynamic World | land cover probs | 2–5 d | 6,970 cells | 0.39M | cloud → 0 + mask |
| Peat | static feature | static | 6,970 cells | 6,970 | none |

## Pre-built tensors (`tensors/`)

Built by `data/loader/tensor_assembly.py` (all sources → daily [D,H,W,23] float32):

* Channel layout: 0-7 ERA5, 8 chirps_precip, 9-11 sar_vv/sar_vh/sar_available,
  12-19 DW class probs, 20 dw_available, 21 peat_depth, 22 hotspot_count_lag
* T_IN = 14 days input, HORIZON = 7-day forecast window, 15×15 patch
* `labels.npy`: 1 = ≥k=2 hotspots in (t+1, t+7], 0 = no fire, -1 = not a land cell
* **0 NaN** (ERA5 spatially filled; S1/DW use mask channels)

## Reproduction commands (from repo root)

```bash
uv run python data/ingest/viirs.py --years 2019 2023 --keep-raw --raw-csv-dir real_data/viirs-snpp
uv run python data/ingest/chirpsv3.py --start 2019-01-01 --end 2023-12-31 --project <ee-project>
uv run python data/ingest/sentinel1.py --start 2019-01-01 --end 2023-12-31 --project <ee-project>
uv run python data/ingest/era5land.py --start 2019-01-01 --end 2023-12-31 --project <ee-project>
uv run python data/ingest/dynamic_world.py --start 2019-01-01 --end 2023-12-31 --project <ee-project>
uv run python data/ingest/peat.py
uv run python data/loader/tensor_assembly.py --start 2019-01-01 --end 2023-12-31
```

Large regenerable outputs (`data/output/*`) are gitignored; canonical copies live
in the HuggingFace dataset repo under `raw/` and `tensors/`.
