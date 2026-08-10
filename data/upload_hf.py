"""Upload the ingested CSV files and pre-built tensors to Hugging Face.

CSV sources are uploaded under ``raw/<source>/`` in the dataset repo (matching
the canonical layout: all per-source CSVs live in ``raw/``, pre-built tensors
in ``tensors/``). E.g. ``data/output/chirpsv3/chirps_v3sat_201901.csv`` becomes
``raw/chirpsv3/chirps_v3sat_201901.csv``.

Usage (from the datathon project root):

    # Interactive token login (first run)
    uv run python data/upload_hf.py

    # Login via a read/write token (first run, non-interactive)
    uv run python data/upload_hf.py --token hf_xxxx

    # Upload only a subset of sources
    uv run python data/upload_hf.py --sources chirpsv3,viirs,peat

    # Also upload the pre-built tensors (data.output/tensors -> tensors/)
    uv run python data/upload_hf.py --with-tensors

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

# Top-level names under data/output -> subdir in the dataset repo (under raw/).
DEFAULT_SOURCES = [
    "grid",
    "chirpsv3",
    "sentinel1",
    "sentinel1_filled",
    "peat",
    "viirs",
    "era5land",
    "dynamic_world",
]


def _collect_csvs(base: Path, sources: list[str]) -> list[tuple[str, Path]]:
    """Return [(repo-relative path, local path)] for every CSV under `sources`.

    CSVs are staged under ``raw/<source>/`` to match the dataset-repo layout.
    """
    items: list[tuple[str, Path]] = []
    for src in sources:
        src_dir = base / src
        if not src_dir.is_dir():
            logger.warning("skipping missing source dir: %s", src_dir)
            continue
        for csv in sorted(src_dir.rglob("*.csv")):
            rel = Path("raw") / src / csv.name
            items.append((rel.as_posix(), csv))
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
    parser.add_argument("--with-tensors", action="store_true",
                        help="also upload data/output/tensors/ -> tensors/ in the repo")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list files that would be uploaded")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    items = _collect_csvs(args.data_dir, sources)
    if not items:
        logger.error("No CSVs found under %s for sources %s", args.data_dir, sources)
        raise SystemExit(1)

    total_bytes = sum(p.stat().st_size for _, p in items)
    logger.info("Found %d CSVs (%.1f GB) to upload to %s under raw/",
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
        if args.with_tensors:
            tensors_dir = args.data_dir / "tensors"
            if tensors_dir.is_dir():
                for f in sorted(tensors_dir.iterdir()):
                    if f.is_file():
                        dest = staging / "tensors" / f.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
                logger.info("Including tensors: %s", tensors_dir)
            else:
                logger.warning("--with-tensors given but %s not found; skipping", tensors_dir)
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
