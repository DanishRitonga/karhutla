"""CHIRPS v3.0 SAT daily precipitation ingestion.

Pulls daily rainfall over the fixed Riau grid from GEE and writes per-month
CSVs (``data/output/chirpsv3/chirps_v3sat_YYYYMM.csv``).

Uses ``_gee.py`` for authentication, grid loading, and iteration helpers.

Run (from datathon root):

    uv run --python 3.12 python data/ingest/chirpsv3.py \
        --start 2019-01-01 --end 2023-12-31 --project ee-yours
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.ingest._gee import (
    GeeClient,
    GeeConfig,
    RiauGridCells,
    _iter_months,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chirpsv3")


class ChirpsIngester:
    """CHIRPS v3.0 SAT daily precipitation -> per-cell daily table."""

    COLLECTION = "UCSB-CHC/CHIRPS/V3/DAILY_SAT"
    BAND = "precipitation"

    def __init__(self, client: GeeClient, cells: RiauGridCells, config: GeeConfig) -> None:
        self.client = client
        self.cells = cells
        self.config = config
        self.ee = client.ee

    def _stack_month(self, month_start: date, month_end: date):
        ee = self.ee
        bbox = self.cells.bounds_geometry()
        ic = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(month_start.isoformat(), (month_end + timedelta(days=1)).isoformat())
        )

        def rename_day(img):
            day = img.date().format("yyyyMMdd")
            return img.select([self.BAND]).rename([day])

        return ic.map(rename_day).toBands()

    def _pull(self, month_start: date, month_end: date) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("CHIRPS: pulling %s -> %s", month_start, month_end)
        stacked = self._stack_month(month_start, month_end)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked, collection=fc,
                reducer=ee.Reducer.mean(), scale=self.config.chirps_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("CHIRPS: no data for %s -> %s", month_start, month_end)
            return None

        id_cols = ["cell_idx", "row", "col"]
        for c in id_cols:
            if c not in wide.columns:
                wide[c] = np.nan

        band_cols = [
            c for c in wide.columns
            if "_" in c and c.split("_")[-1].isdigit() and len(c.split("_")[-1]) == 8
        ]
        if not band_cols:
            logger.warning("CHIRPS: no day bands in %s -> %s", month_start, month_end)
            return None

        long = wide.melt(id_vars=id_cols, value_vars=band_cols, var_name="band", value_name="precip_mm")
        long["date"] = long["band"].str.split("_").str[-1]
        long["date"] = pd.to_datetime(long["date"], format="%Y%m%d").dt.date
        long = long.dropna(subset=["precip_mm"])
        return long[id_cols + ["date", "precip_mm"]]

    def ingest(self, start: date, end: date) -> list[Path]:
        self.config.chirps_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for month_start, month_end in _iter_months(start, end):
            df = self._pull(month_start, month_end)
            if df is None or df.empty:
                continue
            path = self.config.chirps_dir / f"chirps_v3sat_{month_start:%Y%m}.csv"
            df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
            written.append(path)
            logger.info("CHIRPS: wrote %s (%d rows)", path, len(df))
        return written


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CHIRPS v3.0 SAT over the Riau grid")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--project", default=None, help="GEE cloud project ID")
    parser.add_argument("--out", type=Path, default=None, help="Base output dir (default data/output)")
    args = parser.parse_args()

    config = GeeConfig(project=args.project)
    if args.out:
        config = GeeConfig(project=args.project, chirps_dir=args.out)

    client = GeeClient(config)
    client.initialize()
    cells = RiauGridCells(config)

    written = ChirpsIngester(client, cells, config).ingest(args.start, args.end)
    logger.info("Done. %d files written.", len(written))


if __name__ == "__main__":
    main()
