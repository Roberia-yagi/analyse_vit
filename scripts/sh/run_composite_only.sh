#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/raw/image_generation"

RUN_ROOT_ARG="${1:-}"
if [[ -z "$RUN_ROOT_ARG" ]]; then
  echo "Error: pass a date directory (e.g., 20260123_170000) or full run root path." >&2
  exit 1
fi

if [[ -d "$RUN_ROOT_ARG" ]]; then
  RUN_ROOT="$RUN_ROOT_ARG"
elif [[ -d "$RESULTS_DIR/$RUN_ROOT_ARG" ]]; then
  RUN_ROOT="$RESULTS_DIR/$RUN_ROOT_ARG"
else
  echo "Error: run root not found: $RUN_ROOT_ARG" >&2
  exit 1
fi

python -m analyse_vit.rare_colour_bias.image_composite \
  --run-root "$RUN_ROOT" \
  --object-name-real "kangaroo" \
  --object-name-toy "toy" \
  --dominant-color "brown" \
  --rare-color "pink" \
  --seed-real 456 \
  --seed-toy 789 \
  --sam-checkpoint "$PROJECT_ROOT/models/sam/sam_vit_h_4b8939.pth" \
  --lang-sam-box-threshold 0.25 \
  --lang-sam-text-threshold 0.25 \
  --num-runs 10
