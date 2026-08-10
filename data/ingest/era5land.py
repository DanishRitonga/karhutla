"""ERA5-Land hourly weather ingestion over the fixed Riau grid.

Pulls daily ERA5-Land summaries from GEE and writes per-month CSVs
(``data/output/era5land/era5land_YYYYMM.csv``).

<<<<<<< HEAD
=======
Aggregation semantics (per design log §3):
  * **state** bands (temperature, dewpoint, wind, soil moisture) -> daily mean
  * **flux/accumulation** bands (``total_precipitation``, 
    ``surface_solar_radiation_downwards``) -> daily **sum** over the 24 hourly
    images. Taking the mean here would under-represent the daily total by a
    factor of 24.

>>>>>>> origin/master
Run (from project root):

    uv run --python 3.12 python data/ingest/era5land.py \
        --start 2019-01-01 --end 2023-12-31 --project ee-yours

Default bands (override via ``--bands``):
  * temperature_2m
  * dewpoint_temperature_2m
  * u_component_of_wind_10m
  * v_component_of_wind_10m
  * volumetric_soil_water_layer_1
  * volumetric_soil_water_layer_2
  * total_precipitation
  * surface_solar_radiation_downwards
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data.ingest._gee import GeeClient, GeeConfig, RiauGridCells, _iter_months
except ModuleNotFoundError:
    # Allow direct execution via absolute/relative script path.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.ingest._gee import GeeClient, GeeConfig, RiauGridCells, _iter_months

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("era5land")

<<<<<<< HEAD
=======
# Accumulation bands: daily total = sum of the 24 hourly values.
FLUX_BANDS = ["total_precipitation", "surface_solar_radiation_downwards"]

>>>>>>> origin/master

def _parse_stacked_band(name: str) -> tuple[str, str] | None:
    """Extract (YYYYMMDD, field) from a stacked band name.

    ``toBands()`` adds image-id prefixes, so we scan for the first 8-digit token
    and keep the remainder as the field name.
    """
    parts = name.split("_")
    for i, token in enumerate(parts):
        if len(token) == 8 and token.isdigit() and i < len(parts) - 1:
            return token, "_".join(parts[i + 1 :])
    return None


class Era5LandIngester:
    """ERA5-Land hourly data -> per-cell daily table."""

    COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
    DEFAULT_BANDS = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "total_precipitation",
        "surface_solar_radiation_downwards",
    ]

    def __init__(
        self,
        client: GeeClient,
        cells: RiauGridCells,
        config: GeeConfig,
        bands: list[str] | None = None,
    ) -> None:
        self.client = client
        self.cells = cells
        self.config = config
        self.ee = client.ee
        self.bands = bands or self.DEFAULT_BANDS

    def _stack_month(self, month_start: date, month_end: date):
        ee = self.ee
        bbox = self.cells.bounds_geometry()
        hourly = ee.ImageCollection(self.COLLECTION).filterBounds(bbox)

<<<<<<< HEAD
=======
        state_bands = [b for b in self.bands if b not in FLUX_BANDS]
        flux_bands = [b for b in self.bands if b in FLUX_BANDS]

>>>>>>> origin/master
        daily_images = []
        cur = month_start
        while cur <= month_end:
            nxt = cur + timedelta(days=1)
<<<<<<< HEAD
            daily = hourly.filterDate(cur.isoformat(), nxt.isoformat()).select(self.bands).mean()
            rename_to = [f"{cur:%Y%m%d}_{band}" for band in self.bands]
            daily_images.append(daily.rename(rename_to))
=======
            day_ic = hourly.filterDate(cur.isoformat(), nxt.isoformat())
            day_img = None
            if state_bands:
                day_img = day_ic.select(state_bands).mean()
            if flux_bands:
                flux_img = day_ic.select(flux_bands).sum()
                day_img = flux_img if day_img is None else day_img.addBands(flux_img)
            rename_to = [f"{cur:%Y%m%d}_{band}" for band in self.bands]
            daily_images.append(day_img.rename(rename_to))
>>>>>>> origin/master
            cur = nxt

        return ee.ImageCollection(daily_images).toBands()

    def _pull(self, month_start: date, month_end: date) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("ERA5-Land: pulling %s -> %s", month_start, month_end)
        stacked = self._stack_month(month_start, month_end)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked,
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=self.config.era5land_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("ERA5-Land: no data for %s -> %s", month_start, month_end)
            return None

        id_cols = ["cell_idx", "row", "col"]
        for c in id_cols:
            if c not in wide.columns:
                wide[c] = np.nan

        parsed = []
        for col in wide.columns:
            out = _parse_stacked_band(col)
            if out is None:
                continue
            day, field = out
            if field in self.bands:
                parsed.append((col, day, field))

        if not parsed:
            logger.warning("ERA5-Land: no day bands in %s -> %s", month_start, month_end)
            return None

        band_cols = [col for col, _, _ in parsed]
        day_by_band = {col: day for col, day, _ in parsed}
        field_by_band = {col: field for col, _, field in parsed}

        long = wide.melt(id_vars=id_cols, value_vars=band_cols, var_name="band")
        long["date"] = pd.to_datetime(long["band"].map(day_by_band), format="%Y%m%d").dt.date
        long["field"] = long["band"].map(field_by_band)
        long = long.dropna(subset=["value"])
        if long.empty:
            return None

        long = (
            long.pivot_table(index=id_cols + ["date"], columns="field", values="value", aggfunc="mean")
            .reset_index()
        )
        for field in self.bands:
            if field not in long.columns:
                long[field] = np.nan
        return long[id_cols + ["date"] + self.bands]

    def ingest(self, start: date, end: date) -> list[Path]:
        self.config.era5land_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for month_start, month_end in _iter_months(start, end):
<<<<<<< HEAD
            df = self._pull(month_start, month_end)
=======
            try:
                df = self._pull(month_start, month_end)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ERA5-Land: month %s FAILED (%s) — skipping, re-run resumes",
                    month_start,
                    exc,
                )
                continue
>>>>>>> origin/master
            if df is None or df.empty:
                continue

            path = self.config.era5land_dir / f"era5land_{month_start:%Y%m}.csv"
            if path.exists():
                existing = pd.read_csv(path, dtype={"cell_idx": int})
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
                df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
                    subset=["cell_idx", "date"], keep="last"
                )
            df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
            written.append(path)
            logger.info("ERA5-Land: wrote %s (%d rows)", path, len(df))
        return written


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5-Land over the Riau grid")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--project", default=None, help="GEE cloud project ID")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default data/output/era5land)")
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Subset of ERA5-Land bands to export (default: common weather + soil moisture set)",
    )
    args = parser.parse_args()

    config = GeeConfig(project=args.project)
    if args.out:
        config = GeeConfig(project=args.project, era5land_dir=args.out)

    client = GeeClient(config)
    client.initialize()
    cells = RiauGridCells(config)

    written = Era5LandIngester(client, cells, config, bands=args.bands).ingest(args.start, args.end)
    logger.info("Done. %d files written.", len(written))


if __name__ == "__main__":
    main()
