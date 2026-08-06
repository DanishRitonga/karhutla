"""Compatibility entrypoint.

This module previously ingested CHIRPS v3. It now forwards to the ERA5-Land
ingester so existing command invocations keep working.
"""

from data.ingest.era5land import main


if __name__ == "__main__":
    main()
