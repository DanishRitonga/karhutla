#!/usr/bin/env bash
#
# Deploy the backend/ directory to the HuggingFace Space.
#
# The Space (https://huggingface.co/spaces/danishritonga/karhutla) is a git
# repo hosted on HuggingFace. This script syncs backend/ (source of truth in
# this monorepo) into a fresh clone of that repo and pushes, which triggers a
# rebuild on HF. Model/data are NOT committed — the backend fetches
# predictions.parquet at runtime from an HF model repo (HF_MODEL_REPO env).
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/deploy_space.sh            # push to prod space
#   ./scripts/deploy_space.sh --dry-run                  # show what would change
#
<<<<<<< HEAD
# Requires: git, rsync, git-lfs. Auth: HF_TOKEN env var (or a saved git credential).
=======
# Requires: git, rsync. Auth: HF_TOKEN env var (or a saved git credential).
>>>>>>> origin/master

set -euo pipefail

SPACE_REPO="${SPACE_REPO:-danishritonga/karhutla}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_REPO}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/../backend" && pwd)"
TMP_DIR="$(mktemp -d)"
DRY_RUN=0

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

if [[ -n "${HF_TOKEN:-}" ]]; then
    # Embed the token in the remote so `git push` needs no extra credentials.
    PUSH_URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}"
else
    PUSH_URL="${SPACE_URL}"
fi

<<<<<<< HEAD
if ! git lfs version >/dev/null 2>&1; then
        cat <<'EOF'
ERROR: git-lfs tidak ditemukan, padahal diperlukan untuk mendorong PDF konteks ke Hugging Face Space.

Install dulu git-lfs, lalu ulangi deploy:
    Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y git-lfs
    Arch Linux:     sudo pacman -S git-lfs

Setelah install (sekali saja):
    git lfs install
EOF
        exit 1
fi

echo "==> Cloning ${SPACE_URL} ..."
git clone --quiet "${SPACE_URL}" "${TMP_DIR}/space"

echo "==> Initializing git-lfs in temporary Space clone ..."
git -C "${TMP_DIR}/space" lfs install --local --force >/dev/null

=======
echo "==> Cloning ${SPACE_URL} ..."
git clone --quiet "${SPACE_URL}" "${TMP_DIR}/space"

>>>>>>> origin/master
echo "==> Syncing ${BACKEND_DIR}/ -> space root (rsync --delete) ..."
rsync -a --delete \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.cache/' \
<<<<<<< HEAD
    --exclude 'rag/index/rag_index.json' \
    --exclude 'rag/output/' \
    "${BACKEND_DIR}/" "${TMP_DIR}/space/"

if compgen -G "${TMP_DIR}/space/rag/context/*.pdf" >/dev/null; then
    # Re-apply clean filter so legacy non-LFS tracked PDFs become LFS pointers.
    git -C "${TMP_DIR}/space" add --renormalize rag/context
fi

git -C "${TMP_DIR}/space" add -A

if git -C "${TMP_DIR}/space" diff --cached --quiet; then
=======
    "${BACKEND_DIR}/" "${TMP_DIR}/space/"

if git -C "${TMP_DIR}/space" diff --quiet && \
   git -C "${TMP_DIR}/space" diff --cached --quiet; then
>>>>>>> origin/master
    echo "==> No changes. Space is already up to date."
    exit 0
fi

<<<<<<< HEAD
=======
git -C "${TMP_DIR}/space" add -A
>>>>>>> origin/master
git -C "${TMP_DIR}/space" status --short

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "==> DRY RUN: would commit and push the changes above."
    exit 0
fi

git -C "${TMP_DIR}/space" commit --quiet \
    -m "deploy: sync backend from monorepo ($(date -u +%Y-%m-%dT%H:%MZ))"
echo "==> Pushing to ${PUSH_URL} ..."
git -C "${TMP_DIR}/space" push --quiet "${PUSH_URL}" main
echo "==> Done. HF will rebuild the Space shortly."
