#!/bin/bash
set -euo pipefail

TARGET_DIR="/scratch/hpc/07/zhang303/R-TPT-main"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory does not exist: $TARGET_DIR"
  exit 1
fi

echo "Refreshing timestamps under: $TARGET_DIR"
date

COUNT=$(find "$TARGET_DIR" | wc -l)
find "$TARGET_DIR" -exec touch {} +

echo "Done. Refreshed $COUNT paths."
date