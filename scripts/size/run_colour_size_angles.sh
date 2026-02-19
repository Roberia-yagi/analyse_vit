#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# User-configurable values
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
COMPOSITE_SUBDIR="with_composite"
INPUT_ROOT=""
MASKS_ROOT=""
OUTPUT_ROOT=""
SIZES="100,90,80,70,60,50,40,30,20,10"
ANIMALS=""
ANGLES=""
MODELS="qwen"
COLOURS=""
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
if [[ -z "$INPUT_ROOT" ]]; then
  INPUT_ROOT="$SELECTED_ROOT/colour_angles/$COMPOSITE_SUBDIR"
fi
if [[ -z "$MASKS_ROOT" ]]; then
  MASKS_ROOT="$SELECTED_ROOT/masks/angles"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$SELECTED_ROOT/colour_size_angles/$COMPOSITE_SUBDIR"
fi

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Error: input root not found: $INPUT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$MASKS_ROOT" ]]; then
  echo "Error: masks root not found: $MASKS_ROOT" >&2
  exit 1
fi

RACE_ROOT="$PROJECT_ROOT/.race"
RACE_KEY="colour_size_angles_${COMPOSITE_SUBDIR}_${MODELS:-all}"
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
  --target angles
  --composite-subdir "$COMPOSITE_SUBDIR"
  --input-root "$INPUT_ROOT"
  --masks-root "$MASKS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --sizes "$SIZES"
)

if [[ -n "$ANIMALS" ]]; then
  ARGS+=(--animals "$ANIMALS")
fi
if [[ -n "$ANGLES" ]]; then
  ARGS+=(--angles "$ANGLES")
fi
if [[ -n "$MODELS" ]]; then
  ARGS+=(--models "$MODELS")
fi
if [[ -n "$COLOURS" ]]; then
  ARGS+=(--colours "$COLOURS")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "$PROJECT_ROOT"
uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.composite.colour_size_composite "${ARGS[@]}"
