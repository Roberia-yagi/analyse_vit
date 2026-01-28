#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
RUN_SUBDIR="${1:-}"
if [[ "$RUN_SUBDIR" == /* ]]; then
  echo "Error: RUN_SUBDIR must be relative to RESULTS_DIR: $RUN_SUBDIR" >&2
  exit 1
fi
if [[ -n "$RUN_SUBDIR" ]]; then
  RUN_ROOT="$RESULTS_DIR/$RUN_SUBDIR"
else
  RUN_ROOT="$RESULTS_DIR"
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Error: selected root not found: $RUN_ROOT" >&2
  exit 1
fi

COLORS=(
  red
  orange
  yellow
  green
  cyan
  blue
  purple
  magenta
  pink
  gray
  black
  white
)

echo "=== Recolor all anchors under: $RUN_ROOT/anchors ==="
python -m analyse_vit.rare_colour_bias.recolor.image_recolor \
  --anchors-root "$RUN_ROOT/anchors" \
  --output-root "$RUN_ROOT" \
  --object-name "auto" \
  --colors "$(IFS=,; echo "${COLORS[*]}")" \
  --lang-sam-box-threshold 0.25 \
  --lang-sam-text-threshold 0.25 \
  --mask-dilate-px 9
