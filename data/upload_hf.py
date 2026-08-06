"""Upload the ingested CSV files to a Hugging Face dataset repo.

Uploads every CSV under ``data/output/`` (grid, chirpsv3, sentinel1,
sentinel1_filled, peat, viirs labels) preserving the sub-directory layout,
e.g. ``data/output/chirpsv3/chirps_v3sat_201901.csv`` becomes
``chirpsv3/chirps_v3sat_201901.csv`` in the dataset repo.

Usage (from the datathon project root):

    # Interactive token login (first run)
    uv run python data/upload_hf.py

    # Login via a read/write token (first run, non-interactive)
    uv run python data/upload_hf.py --token hf_xxxx

    # Upload to a different repo or only a subset of sources
    uv run python data/upload_hf.py --repo-id yourorg/karhutla \\
        --sources chirpsv3,viirs,peat

Requirements: ``huggingface-hub`` (``uv add huggingface-hub``).

Security note: if you pass ``--token``, the token is only used in-process and
is never written to disk by this script.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import shutil
from pathlib import Path

from huggingface_hub import login, upload_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("upload_hf")

# Mapping of top-level name under data/output -> subdir in the dataset repo.
DEFAULT_SOURCES = [
    "grid",
    "chirpsv3",
    "sentinel1",
    "sentinel1_filled",
    "peat",
    "viirs",
]


def _collect_csvs(base: Path, sources: list[str]) -> list[tuple[str, Path]]:
    """Return [(repo-relative path, local path)] for every CSV under `sources`."""
    items: list[tuple[str, Path]] = []
    for src in sources:
        src_dir = base / src
        if not src_dir.is_dir():
            logger.warning("skipping missing source dir: %s", src_dir)
            continue
        for csv in sorted(src_dir.rglob("*.csv")):
            rel = csv.relative_to(base).as_posix()
            items.append((rel, csv))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ingested CSVs to Hugging Face")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output"),
                        help="base directory holding the source sub-dirs (default data/output)")
    parser.add_argument("--repo-id", default="danishritonga/karhutla",
                        help="Hugging Face dataset repo id (default danishritonga/karhutla)")
    parser.add_argument("--token", default=None,
                        help="Hugging Face write token (optional; falls back to interactive login)")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                        help="comma-separated source sub-dirs to include (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list files that would be uploaded")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    items = _collect_csvs(args.data_dir, sources)
    if not items:
        logger.error("No CSVs found under %s for sources %s", args.data_dir, sources)
        raise SystemExit(1)

    total_bytes = sum(p.stat().st_size for _, p in items)
    logger.info("Found %d CSVs (%.1f GB) to upload to %s",
                len(items), total_bytes / 1e9, args.repo_id)
    for rel, _ in items[:20]:
        logger.info("  %s", rel)
    if len(items) > 20:
        logger.info("  ... and %d more", len(items) - 20)

    if args.dry_run:
        logger.info("Dry run — nothing uploaded.")
        return

    if args.token:
        login(token=args.token)
    else:
        logger.info("No token given; starting interactive login (paste a write token).")
        login()

    # Stage into a flat temp dir with the repo-relative layout so the repo has
    # clean sub-directories (upload_folder mirrors the source layout).
    with tempfile.TemporaryDirectory(prefix="karhutla_hf_") as tmp:
        staging = Path(tmp)
        for rel, local in items:
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)
        # Dataset card (repo-root README.md) doubles as the dataset description.
        readme = args.data_dir.parent / "README.md"
        if readme.is_file():
            shutil.copy2(readme, staging / "README.md")
            logger.info("Including dataset card: %s", readme)
        logger.info("Staged %d files in %s; uploading to %s ...",
                    len(items), staging, args.repo_id)
        upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"ingest: upload {len(items)} feature/label CSVs",
        )
        logger.info("Done — https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
