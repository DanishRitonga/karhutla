"""Compatibility entrypoint.

This module previously ingested Sentinel-1. It now forwards to the Dynamic
World ingester so existing command invocations keep working.
"""

from data.ingest.dynamic_world import main


if __name__ == "__main__":
    main()
