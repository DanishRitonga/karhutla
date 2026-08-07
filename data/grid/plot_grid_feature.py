"""Render ingested GEE feature data onto the Riau grid map.

Produces a choropleth of one feature field (one date) on the same equal-area
grid used by ``grid_definition.py`` and drawn in the style of ``riau_grid.png``
(Albers projection, 5 km cells, boundary outline).

Data sources (outputs of the ``data/ingest/*.py`` scripts):

  * chirps_v3sat_YYYYMM.csv       -> cell_idx, row, col, date, precip_mm
  * s1_{ORBIT}_YYYYMM.csv         -> cell_idx, row, col, date, vv_db, vh_db,
                                     angle_deg, vh_vv_db
  * era5land_YYYYMM.csv           -> cell_idx, row, col, date, <8 bands>
  * dynamic_world_YYYYMM.csv      -> cell_idx, row, col, date, <9 classes>,
                                     top1_class, top1_prob

Example calls (from the datathon project root):

    uv run --python 3.12 python data/grid/plot_grid_feature.py \
        --source chirps --date 2019-01-15

    uv run --python 3.12 python data/grid/plot_grid_feature.py \
        --source s1 --date 2019-01-10 --orbit ASCENDING --field vh_vv_db

    uv run --python 3.12 python data/grid/plot_grid_feature.py \
        --source era5land --date 2019-01-15 --field total_precipitation

    uv run --python 3.12 python data/grid/plot_grid_feature.py \
        --source dynamic_world --date 2019-01-15 --field trees --riau-only
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("plot_grid_feature")

# Field -> (colormap, unit label) for each source.
FIELD_META: dict[str, dict[str, tuple[str, str]]] = {
    "chirps": {"precip_mm": ("YlGnBu", "mm/day")},
    "s1": {
        "vv_db": ("RdYlBu_r", "dB"),
        "vh_db": ("RdYlBu_r", "dB"),
        "angle_deg": ("viridis", "deg"),
        "vh_vv_db": ("RdYlBu_r", "dB"),
    },
    "era5land": {
        "temperature_2m": ("coolwarm", "K"),
        "dewpoint_temperature_2m": ("coolwarm", "K"),
        "u_component_of_wind_10m": ("RdBu_r", "m/s"),
        "v_component_of_wind_10m": ("RdBu_r", "m/s"),
        "volumetric_soil_water_layer_1": ("YlGnBu", "m3/m3"),
        "volumetric_soil_water_layer_2": ("YlGnBu", "m3/m3"),
        "total_precipitation": ("YlGnBu", "m"),
        "surface_solar_radiation_downwards": ("inferno", "J/m2"),
    },
    "dynamic_world": {
        "water": ("Blues", "probability"),
        "trees": ("Greens", "probability"),
        "grass": ("YlGn", "probability"),
        "flooded_vegetation": ("PuBuGn", "probability"),
        "crops": ("YlOrBr", "probability"),
        "shrub_and_scrub": ("cividis", "probability"),
        "built": ("Reds", "probability"),
        "bare": ("copper", "probability"),
        "snow_and_ice": ("Purples", "probability"),
        "top1_prob": ("viridis", "probability"),
    },
}

# Per-source defaults when --field is omitted.
DEFAULT_FIELD: dict[str, str] = {
    "chirps": "precip_mm",
    "s1": "vv_db",
    "era5land": "temperature_2m",
    "dynamic_world": "top1_prob",
}


@dataclass(frozen=True)
class FeaturePlotConfig:
    """Static configuration for one feature map."""

    source: str
    field: str
    date: date
    orbit: str | None = None
    riau_only: bool = False
    # Number of days ending at ``date`` whose per-cell values are aggregated
    # (median) before plotting. S1 is swath-limited per acquisition, so a
    # window of ~14-30 days gives near-complete coverage over the province.
    window_days: int = 1
    grid_csv: Path = Path("data/output/grid/grid_cells.csv")
    boundary_gpkg: Path = Path("data/output/grid/riau_boundary_aea.gpkg")
    chirps_dir: Path = Path("data/output/chirpsv3")
    s1_dir: Path = Path("data/output/sentinel1")
    era5land_dir: Path = Path("data/output/era5land")
    dynamic_world_dir: Path = Path("data/output/dynamic_world")
    out_dir: Path = Path("data/output/maps")
    dpi: int = 150
    # Percentiles used to clip the colour scale so a few outliers do not
    # wash out the map. Ignored for dynamic_world (fixed 0..1) and angle.
    vmin_pct: float = 1.0
    vmax_pct: float = 99.0

    @property
    def cell_size_m(self) -> int:
        return 5000


class FeatureGridPlotter:
    """Loads one date of feature data and renders it on the grid."""

    def __init__(self, config: FeaturePlotConfig) -> None:
        self.config = config
        self.cells = pd.read_csv(config.grid_csv)
        self.boundary = gpd.read_file(config.boundary_gpkg)

    # -- data ------------------------------------------------------------ #

    def _month_paths(self, start: date, end: date) -> list[Path]:
        """Paths for every calendar month overlapping [start, end]."""
        cfg = self.config
        months = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            if cfg.source == "chirps":
                months.append(cfg.chirps_dir / f"chirps_v3sat_{y:04d}{m:02d}.csv")
            elif cfg.source == "s1":
                if cfg.orbit is None:
                    raise ValueError("--orbit is required for the s1 source")
                months.append(cfg.s1_dir / f"s1_{cfg.orbit}_{y:04d}{m:02d}.csv")
            elif cfg.source == "era5land":
                months.append(cfg.era5land_dir / f"era5land_{y:04d}{m:02d}.csv")
            else:
                months.append(cfg.dynamic_world_dir / f"dynamic_world_{y:04d}{m:02d}.csv")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return months

    def load_field(self) -> pd.Series:
        """Return a Series indexed by cell_idx with the requested field.

        Values are the per-cell median over ``window_days`` ending at the
        configured date.
        """
        cfg = self.config
        start = cfg.date - timedelta(days=cfg.window_days - 1)
        month_paths = [p for p in self._month_paths(start, cfg.date) if p.exists()]
        if not month_paths:
            raise ValueError(f"No data files found for {start} .. {cfg.date}")
        frames = [pd.read_csv(p) for p in month_paths]
        if not frames:
            raise ValueError(f"No data files found for {start} .. {cfg.date}")
        df = pd.concat(frames, ignore_index=True)
        df = df[
            (df["date"] >= start.isoformat()) & (df["date"] <= cfg.date.isoformat())
        ]
        if df.empty:
            raise ValueError(
                f"No data in window {start} .. {cfg.date} "
                f"(source={cfg.source}, orbit={cfg.orbit or '-'})"
            )
        if cfg.field not in FIELD_META[cfg.source]:
            raise ValueError(f"Unknown {cfg.source} field: {cfg.field}")
        if cfg.field not in df.columns:
            raise ValueError(f"Field '{cfg.field}' does not exist in loaded CSV data")
        series = df.groupby("cell_idx")[cfg.field].median()
        return series.dropna()

    # -- rendering -------------------------------------------------------- #

    def _norm(self, values: pd.Series):
        cfg = self.config
        if cfg.field == "angle_deg" or cfg.source == "dynamic_world":
            return Normalize(vmin=0, vmax=1) if cfg.source == "dynamic_world" else Normalize(vmin=0, vmax=90)
        lo = np.percentile(values, cfg.vmin_pct)
        hi = np.percentile(values, cfg.vmax_pct)
        if cfg.source == "chirps":
            lo = max(lo, 0.0)
        return Normalize(vmin=lo, vmax=hi)

    def render(self, output_path: Path) -> None:
        cfg = self.config
        values = self.load_field()
        norm = self._norm(values)
        cmap_name, unit = FIELD_META[cfg.source][cfg.field]
        cmap = plt.get_cmap(cmap_name)

        cells = self.cells.set_index("cell_idx")
        if cfg.riau_only:
            cells = cells[cells["is_riau"]]

        cs = cfg.cell_size_m
        patches, facecolors = [], []
        for idx, row in cells.iterrows():
            if idx not in values.index:
                continue
            x, y = row["x_center_m"], row["y_center_m"]
            patches.append(Rectangle((x - cs / 2, y - cs / 2), cs, cs))
            facecolors.append(cmap(norm(values[idx])))

        if not patches:
            raise ValueError("No cells matched the requested date/field")

        fig, ax = plt.subplots(figsize=(11, 11), dpi=cfg.dpi)
        ax.add_collection(
            PatchCollection(patches, facecolors=facecolors, edgecolors="none")
        )
        self.boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f"{cfg.field} ({unit})")

        orbit = f" ({cfg.orbit})" if cfg.orbit else ""
        win = f", {cfg.window_days}-day median" if cfg.window_days > 1 else ""
        scope = "Riau" if cfg.riau_only else "Riau bbox"
        ax.set_title(
            f"{cfg.source.upper()} {cfg.field} — {cfg.date}{orbit}{win}\n"
            f"{scope}, 5 km equal-area grid (Albers Indonesia Equal Area Conic)",
            fontsize=12,
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.margins(0.02)
        fig.tight_layout()
        fig.savefig(output_path, dpi=cfg.dpi)
        plt.close(fig)
        logger.info("Rendered %s", output_path)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a GEE feature on the Riau grid")
    parser.add_argument(
        "--source", choices=["chirps", "s1", "era5land", "dynamic_world"], required=True
    )
    parser.add_argument("--date", type=_parse_date, required=True)
    parser.add_argument("--orbit", choices=["ASCENDING", "DESCENDING"], default=None)
    parser.add_argument("--field", default=None, help="Feature field to render")
    parser.add_argument("--riau-only", action="store_true")
    parser.add_argument("--window", type=int, default=1,
                        help="days ending at --date to median-composite (default 1; use ~30 for full S1 coverage)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    field = args.field or DEFAULT_FIELD[args.source]
    if field not in FIELD_META[args.source]:
        parser.error(f"--field must be one of {list(FIELD_META[args.source])}")

    config = FeaturePlotConfig(
        source=args.source,
        field=field,
        date=args.date,
        orbit=args.orbit,
        riau_only=args.riau_only,
        window_days=args.window,
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        fname = f"{config.source}_{config.date:%Y%m%d}"
        if config.orbit:
            fname += f"_{config.orbit}"
        fname += f"_{config.field}"
        if config.window_days > 1:
            fname += f"_w{config.window_days}"
        if config.riau_only:
            fname += "_riau"
        out = config.out_dir / f"{fname}.png"
    else:
        out = args.out

    FeatureGridPlotter(config).render(out)


if __name__ == "__main__":
    main()
