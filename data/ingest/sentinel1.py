"""Sentinel-1 GRD backscatter: download from GEE + forward-fill sparse gaps.

Two stages, both run by default:

  1. **download**  Pull S1 VV/VH/angle from GEE onto the 5 km Riau grid,
                   one CSV per month (``data/output/sentinel1/``).
  2. **fill**      Forward-fill the sparse (6-12 day revisit) acquisitions
                   to daily rows, capped at ``--max-gap`` days, with a
                   ``filled`` boolean mask (``data/output/sentinel1_filled/``).

Uses ``_gee.py`` for authentication, grid loading, and iteration helpers.

Run (from datathon root):

    # Download + fill (default):
    uv run --python 3.12 python data/ingest/sentinel1.py \\
        --start 2019-01-01 --end 2023-12-31 --project ee-yours

    # Download only:
    uv run --python 3.12 python data/ingest/sentinel1.py --download

    # Fill only (after download):
    uv run --python 3.12 python data/ingest/sentinel1.py --fill --max-gap 14
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
    from data.ingest._gee import (
        GeeClient,
        GeeConfig,
        RiauGridCells,
        _iter_windows,
    )
except ModuleNotFoundError:
    # Allow direct execution via absolute/relative script path.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.ingest._gee import (
        GeeClient,
        GeeConfig,
        RiauGridCells,
        _iter_windows,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sentinel1")

FIELDS = ["vv_db", "vh_db", "angle_deg", "vh_vv_db"]


class Sentinel1Ingester:
    """Sentinel-1 GRD VV/VH/angle backscatter -> per-cell, per-acquisition table."""

    COLLECTION = "COPERNICUS/S1_GRD"
    BANDS = ["VV", "VH", "angle"]

    def __init__(self, client: GeeClient, cells: RiauGridCells, config: GeeConfig) -> None:
        self.client = client
        self.cells = cells
        self.config = config
        self.ee = client.ee

    def _stack_window(self, start: date, end: date, orbit: str):
        ee = self.ee
        bbox = self.cells.bounds_geometry()
        ic = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit.upper()))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )

        def rename_image(img):
            d = img.date().format("yyyyMMdd")
            return img.select(self.BANDS).rename(
                [d.cat("_vv"), d.cat("_vh"), d.cat("_angle")]
            )

        return ic.map(rename_image).toBands()

    def _pull(self, start: date, end: date, orbit: str) -> pd.DataFrame | None:
        ee = self.ee
        logger.info("S1 %s: pulling %s -> %s", orbit, start, end)
        stacked = self._stack_window(start, end, orbit)

        wide_parts = []
        for fc in self.cells.feature_chunks(self.config.feature_chunk_size):
            info = self.client.reduce_regions(
                image=stacked, collection=fc,
                reducer=ee.Reducer.median(), scale=self.config.s1_scale,
            )
            wide_parts.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))

        wide = pd.concat(wide_parts, ignore_index=True)
        if wide.empty:
            logger.warning("S1 %s: no acquisitions in %s -> %s", orbit, start, end)
            return None

        id_cols = ["cell_idx", "row", "col"]
        for c in id_cols:
            if c not in wide.columns:
                wide[c] = np.nan

        band_cols = [
            c for c in wide.columns
            if c.split("_")[-1] in {"vv", "vh", "angle"}
            and c.split("_")[-2].isdigit()
            and len(c.split("_")[-2]) == 8
        ]
        if not band_cols:
            logger.warning("S1 %s: no acquisition bands in %s -> %s", orbit, start, end)
            return None

        long = wide.melt(id_vars=id_cols, value_vars=band_cols, var_name="band")
        long["date"] = long["band"].str.split("_").str[-2]
        long["field"] = long["band"].str.split("_").str[-1]
        long["date"] = pd.to_datetime(long["date"], format="%Y%m%d").dt.date
        long = long.pivot_table(
            index=id_cols + ["date"], columns="field", values="value"
        ).reset_index()
        long = long.drop_duplicates(subset=["cell_idx", "date"])
        long = long.dropna(subset=["vv", "vh"])
        long["vh_vv_db"] = long["vh"] - long["vv"]
        long = long.rename(columns={"vv": "vv_db", "vh": "vh_db", "angle": "angle_deg"})
        return long[id_cols + ["date", "vv_db", "vh_db", "angle_deg", "vh_vv_db"]]

    def ingest(self, start: date, end: date, orbit: str | None = None) -> list[Path]:
        self.config.s1_dir.mkdir(parents=True, exist_ok=True)
        orbits = [orbit] if orbit else ["ASCENDING", "DESCENDING"]
        written = []
        for orb in orbits:
            for win_start, win_end in _iter_windows(start, end, self.config.s1_chunk_days):
                df = self._pull(win_start, win_end, orb)
                if df is None or df.empty:
                    continue
                path = self.config.s1_dir / f"s1_{orb}_{win_start:%Y%m}.csv"
                if path.exists():
                    existing = pd.read_csv(path, dtype={"cell_idx": int})
                    df = pd.concat([existing, df]).drop_duplicates(
                            subset=["cell_idx", "date"], keep="last"
                        )
                df.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
                written.append(path)
                logger.info("S1 %s: wrote %s (%d rows)", orb, path, len(df))
        return written


# --------------------------------------------------------------------------- #
# Forward-fill sparse gaps -> daily
# --------------------------------------------------------------------------- #


class Sentinel1Filler:
    """Forward-fill S1 sparse acquisitions to daily rows, capped at max_gap."""

    def __init__(self, s1_dir: Path, out_dir: Path, max_gap_days: int = 14) -> None:
        self.s1_dir = s1_dir
        self.out_dir = out_dir
        self.max_gap_days = max_gap_days

    def _load_orbit(self, orbit: str) -> pd.DataFrame:
        files = sorted(self.s1_dir.glob(f"s1_{orbit}_*.csv"))
        if not files:
            raise FileNotFoundError(f"No s1_{orbit}_*.csv in {self.s1_dir}")
        frames = [pd.read_csv(f, dtype={"cell_idx": int, "row": int, "col": int})
                  for f in files]
        data = pd.concat(frames, ignore_index=True)
        data["date"] = pd.to_datetime(data["date"]).dt.date
        data = data.drop_duplicates(subset=["cell_idx", "date"])
        logger.info("fill %s: loaded %d rows from %d files", orbit, len(data), len(files))
        return data

    @staticmethod
    def _fill_one(group: pd.DataFrame, date_index: pd.DatetimeIndex, max_gap: int) -> pd.DataFrame:
        cell_idx = group.name
        row_val, col_val = group["row"].iloc[0], group["col"].iloc[0]
        group = group.set_index("date").sort_index().reindex(date_index)
        group["cell_idx"] = cell_idx
        group["row"] = row_val
        group["col"] = col_val

        orig = group[FIELDS].notna().any(axis=1).astype(int)
        group["filled"] = 1
        group.loc[orig == 1, "filled"] = 0
        group[FIELDS] = group[FIELDS].ffill()

        idx = np.arange(len(group))
        oi = idx[orig == 1]
        last = np.searchsorted(oi, idx, side="right") - 1
        gap = idx - oi[last.clip(0)]
        gap[last < 0] = 0
        mask = (gap > max_gap) & (orig == 0)
        group.loc[mask, FIELDS] = pd.NA
        return group.reset_index(names="date")

    def fill_orbit(self, orbit: str) -> pd.DataFrame:
        data = self._load_orbit(orbit)
        data["date"] = pd.to_datetime(data["date"])
        idx = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
        logger.info("fill %s: %s → %s (%d days)", orbit, idx[0].date(), idx[-1].date(), len(idx))
        filled = (
            data.groupby("cell_idx", group_keys=False)
            .apply(lambda g: self._fill_one(g, idx, self.max_gap_days))
            .reset_index(drop=True)
        )
        filled["date"] = filled["date"].dt.date
        n = len(filled)
        r = (filled["filled"] == 0).sum()
        logger.info("fill %s: %d rows (%d real, %d filled, %.1f%% real)", orbit, n, r, n - r, 100 * r / n)
        return filled

    def _write_monthly(self, data: pd.DataFrame, orbit: str) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = data.sort_values(["cell_idx", "date"])
        written = []
        for (y, m), mdf in data.groupby([data["date"].apply(lambda d: d.year),
                                         data["date"].apply(lambda d: d.month)]):
            path = self.out_dir / f"s1_{orbit}_{y:04d}{m:02d}.csv"
            mdf.sort_values(["cell_idx", "date"]).to_csv(path, index=False)
            written.append(path)
            logger.info("fill %s: wrote %s (%d rows)", orbit, path.name, len(mdf))
        return written

    def run(self, orbit: str | None = None) -> list[Path]:
        orbits = [orbit] if orbit else ["ASCENDING", "DESCENDING"]
        written = []
        for orb in orbits:
            daily = self.fill_orbit(orb)
            written += self._write_monthly(daily, orb)
        return written


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel-1: download + forward-fill for Riau grid")
    parser.add_argument("--download", action="store_true", default=None,
                        help="only download (skip fill)")
    parser.add_argument("--fill", action="store_true", default=None,
                        help="only fill (skip download; uses already-downloaded CSVs)")
    parser.add_argument("--start", type=_parse_date, default=date(2019, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 31))
    parser.add_argument("--orbit", choices=["ASCENDING", "DESCENDING"], default=None,
                        help="restrict to one orbit (default: both)")
    parser.add_argument("--project", default=None, help="GEE cloud project ID")
    parser.add_argument("--out", type=Path, default=None, help="Base output dir (default data/output)")
    parser.add_argument("--max-gap", type=int, default=14,
                        help="fill: max consecutive gap days before zeroing (default 14)")
    args = parser.parse_args()

    # Default: run both stages.
    do_download = args.download is not False and args.fill is None or args.download
    do_fill = args.fill is not False and args.download is None or args.fill
    if args.download is None and args.fill is None:
        do_download = do_fill = True

    # -- Stage 1: download --
    if do_download:
        config = GeeConfig(project=args.project)
        if args.out:
            config = GeeConfig(project=args.project, s1_dir=args.out)
        client = GeeClient(config)
        client.initialize()
        cells = RiauGridCells(config)
        Sentinel1Ingester(client, cells, config).ingest(args.start, args.end, args.orbit)

    # -- Stage 2: fill --
    if do_fill:
        s1_dir = (args.out or Path("data/output/sentinel1"))
        out_dir = s1_dir.parent / (s1_dir.name + "_filled")
        filler = Sentinel1Filler(s1_dir=s1_dir, out_dir=out_dir, max_gap_days=args.max_gap)
        written = filler.run(args.orbit)
        logger.info("Fill done. %d files written.", len(written))


if __name__ == "__main__":
    main()
