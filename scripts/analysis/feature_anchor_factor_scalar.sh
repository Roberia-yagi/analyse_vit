#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# User-configurable values
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="/data/gpfs/projects/punim2431/projects/image_editing/experiments/analyse_vit/results/selected"
OUTPUT_DIR=""
COMPOSITE_SUBDIR="with_composite"
GEN_MODEL="qwen"
VISION_MODEL="" # empty=all(qwen,siglip2,pe), or one of qwen/siglip/siglip2/pe
POOLING="attention_pooling"
INCLUDE_SIZE_100=0
COLOURS=()
ANGLES=()
SIZES=()
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
if [[ ! -d "$SELECTED_ROOT/anchors/$COMPOSITE_SUBDIR" ]]; then
  echo "Error: anchors root not found: $SELECTED_ROOT/anchors/$COMPOSITE_SUBDIR" >&2
  exit 1
fi
if [[ ! -d "$SELECTED_ROOT/angles/$COMPOSITE_SUBDIR" ]]; then
  echo "Error: angles root not found: $SELECTED_ROOT/angles/$COMPOSITE_SUBDIR" >&2
  exit 1
fi
if [[ ! -d "$SELECTED_ROOT/size" ]]; then
  echo "Error: size root not found: $SELECTED_ROOT/size" >&2
  exit 1
fi
if [[ ! -d "$SELECTED_ROOT/colour/$COMPOSITE_SUBDIR" && ! -d "$SELECTED_ROOT/colours/$COMPOSITE_SUBDIR" ]]; then
  echo "Error: colour root not found: $SELECTED_ROOT/colour(s)/$COMPOSITE_SUBDIR" >&2
  exit 1
fi

if [[ -n "$VISION_MODEL" ]]; then
  case "$VISION_MODEL" in
    qwen|siglip|siglip2|pe) ;;
    *)
      echo "Error: VISION_MODEL must be one of qwen/siglip/siglip2/pe: $VISION_MODEL" >&2
      exit 1
      ;;
  esac
fi

# Simple race guard for direct .sh execution.
RACE_ROOT="$PROJECT_ROOT/.race"
RACE_KEY="feature_anchor_factor_scalar_${GEN_MODEL}_${VISION_MODEL:-all}_${POOLING}_${COMPOSITE_SUBDIR}"
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
  --composite-subdir "$COMPOSITE_SUBDIR"
  --gen-model "$GEN_MODEL"
  --pooling "$POOLING"
)
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-root "$OUTPUT_DIR")
fi
if [[ "$INCLUDE_SIZE_100" == "1" ]]; then
  ARGS+=(--include-size-100)
fi
if (( ${#COLOURS[@]} > 0 )); then
  ARGS+=(--colours "${COLOURS[@]}")
fi
if (( ${#ANGLES[@]} > 0 )); then
  ARGS+=(--angles "${ANGLES[@]}")
fi
if (( ${#SIZES[@]} > 0 )); then
  ARGS+=(--sizes "${SIZES[@]}")
fi

cd "$PROJECT_ROOT"
if [[ -n "$VISION_MODEL" ]]; then
  RUN_ARGS=("${ARGS[@]}" --vision-model "$VISION_MODEL")
  uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.plot.feature_anchor_factor_scalar "${RUN_ARGS[@]}"
else
  VISION_MODELS=("qwen" "siglip2" "pe")
  for VM in "${VISION_MODELS[@]}"; do
    RUN_ARGS=("${ARGS[@]}" --vision-model "$VM")
    uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.plot.feature_anchor_factor_scalar "${RUN_ARGS[@]}"
  done
fi
