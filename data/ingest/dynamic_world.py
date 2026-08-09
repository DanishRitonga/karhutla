"""Dynamic World V1 ingestion over the fixed Riau grid.

Pulls Dynamic World class probabilities from GEE and writes per-month CSVs
(``data/output/dynamic_world/dynamic_world_YYYYMM.csv``).

Run (from project root):

    uv run --python 3.12 python data/ingest/dynamic_world.py \
        --start 2019-01-01 --end 2023-12-31 --project ee-yours
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
logger = logging.getLogger("dynamic_world")


def _parse_stacked_band(name: str) -> tuple[str, str] | None:
    """Extract (YYYYMMDD, field) from a stacked band name."""
    parts = name.split("_")
    for i, token in enumerate(parts):
        if len(token) == 8 and token.isdigit() and i < len(parts) - 1:
            return token, "_".join(parts[i + 1 :])
    return None


class DynamicWorldIngester:
    """Dynamic World class probabilities -> per-cell temporal table."""

    COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
    CLASS_NAMES = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]

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

        def rename_image(img):
            d = img.date().format("yyyyMMdd")
            return img.select(self.CLASS_NAMES).rename([d.cat("_").cat(f) for f in self.CLASS_NAMES])

        return ic.map(rename_image).toBands()

    def _pull(self, month_start: date, month_end: date) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("Dynamic World: pulling %s -> %s", month_start, month_end)
        stacked = self._stack_month(month_start, month_end)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked,
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=self.config.dynamic_world_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("Dynamic World: no data for %s -> %s", month_start, month_end)
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
            if field in self.CLASS_NAMES:
                parsed.append((col, day, field))

        if not parsed:
            logger.warning("Dynamic World: no class bands in %s -> %s", month_start, month_end)
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
        for field in self.CLASS_NAMES:
            if field not in long.columns:
                long[field] = np.nan

        probs = long[self.CLASS_NAMES]
        long["top1_class"] = probs.idxmax(axis=1, skipna=True)
        long["top1_prob"] = probs.max(axis=1, skipna=True)
        no_prob = probs.isna().all(axis=1)
        long.loc[no_prob, "top1_class"] = pd.NA
        long.loc[no_prob, "top1_prob"] = np.nan

        return long[id_cols + ["date"] + self.CLASS_NAMES + ["top1_class", "top1_prob"]]

    def ingest(self, start: date, end: date) -> list[Path]:
        self.config.dynamic_world_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for month_start, month_end in _iter_months(start, end):
            df = self._pull(month_start, month_end)
            if df is None or df.empty:
                continue

            path = self.config.dynamic_world_dir / f"dynamic_world_{month_start:%Y%m}.csv"
            if path.exists():
                existing = pd.read_csv(path, dtype={"cell_idx": int})
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
                df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
                    subset=["cell_idx", "date"], keep="last"
                )
            df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
            written.append(path)
            logger.info("Dynamic World: wrote %s (%d rows)", path, len(df))
        return written


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Dynamic World over the Riau grid")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--project", default=None, help="GEE cloud project ID")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default data/output/dynamic_world)",
    )
    args = parser.parse_args()

    config = GeeConfig(project=args.project)
    if args.out:
        config = GeeConfig(project=args.project, dynamic_world_dir=args.out)

    client = GeeClient(config)
    client.initialize()
    cells = RiauGridCells(config)

    written = DynamicWorldIngester(client, cells, config).ingest(args.start, args.end)
    logger.info("Done. %d files written.", len(written))


if __name__ == "__main__":
    main()
