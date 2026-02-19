#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# User-configurable values
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
ANGLES_COMPOSITE_SUBDIR="with_composite"
OUTPUT_ROOT=""
GEN_MODEL="qwen"
ANIMALS=""
ANGLES=""
COLOURS=(
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
SAVE_RGBA=0
OVERWRITE=0
# ------------------------------

if [[ -z "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR is required and must be an absolute path." >&2
  exit 1
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
if [[ "$(basename "$RESULTS_DIR")" != "selected" ]]; then
  echo "Error: RESULTS_DIR must point to the selected directory with an absolute path: $RESULTS_DIR" >&2
  exit 1
fi

SELECTED_ROOT="$RESULTS_DIR"
ANGLES_ROOT="$SELECTED_ROOT/angles/$ANGLES_COMPOSITE_SUBDIR"
MASKS_ROOT="$SELECTED_ROOT/masks/angles"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$SELECTED_ROOT/colour_angles/$ANGLES_COMPOSITE_SUBDIR"
fi

if [[ ! -d "$ANGLES_ROOT" ]]; then
  echo "Error: angles root not found: $ANGLES_ROOT" >&2
  exit 1
fi
if [[ ! -d "$MASKS_ROOT" ]]; then
  echo "Error: angle masks root not found: $MASKS_ROOT" >&2
  exit 1
fi

RACE_ROOT="$PROJECT_ROOT/.race"
RACE_KEY="recolor_angles_${ANGLES_COMPOSITE_SUBDIR}_${GEN_MODEL}"
RACE_DIR="$RACE_ROOT/$RACE_KEY.lock"
mkdir -p "$RACE_ROOT"
if ! mkdir "$RACE_DIR" 2>/dev/null; then
  echo "Error: another run is active (lock exists): $RACE_DIR" >&2
  exit 1
fi
cleanup_lock() {
  rmdir "$RACE_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

export PYTHONNOUSERSITE=1
unset PYTHONPATH

ARGS=(
  --selected-root "$SELECTED_ROOT"
  --angles-root "$ANGLES_ROOT"
  --masks-root "$MASKS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --gen-model "$GEN_MODEL"
  --colors "$(IFS=,; echo "${COLOURS[*]}")"
)

if [[ -n "$ANIMALS" ]]; then
  ARGS+=(--animals "$ANIMALS")
fi
if [[ -n "$ANGLES" ]]; then
  ARGS+=(--angles "$ANGLES")
fi
if [[ "$SAVE_RGBA" == "1" ]]; then
  ARGS+=(--save-rgba)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "$PROJECT_ROOT"
uv run --extra generate -- python -m analyse_vit.rare_colour_bias.recolor.angle_recolor_from_masks "${ARGS[@]}"
