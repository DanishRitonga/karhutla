"""Download FIRMS VIIRS hotspot data for karhutla label generation.

Pulls CSV files from firms.modaps.eosdis.nasa.gov for the VIIRS 375 m
active fire product (VNP14IMGTDL_NRT). Each row is a hotspot detection
with lat/lon, confidence, and date/time, which will be joined to the
5 km Riau grid to produce binary hotspot labels per cell per day.

Not yet implemented. Placeholder for Phase 2 ingestion pipeline.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("VIIRS label download not yet implemented")


if __name__ == "__main__":
    main()
