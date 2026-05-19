#!/bin/bash
set -e

cd "$(dirname "$0")"

# Auto-source repo-root .env (e.g. CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD)
# without overriding anything the caller already exported.
ENV_FILE="../../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

DOWNLOAD=0
NOTEBOOK_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --download) DOWNLOAD=1 ;;
        *) NOTEBOOK_FLAG="--notebook" ;;
    esac
done

if [ "$DOWNLOAD" = "1" ]; then
    python download_dataset.py --skip-existing
fi

python generate_dataset.py $NOTEBOOK_FLAG
python filter_dataset.py $NOTEBOOK_FLAG
