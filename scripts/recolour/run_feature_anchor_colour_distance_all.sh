#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
ROOT_SUBDIR="${1:-}"
if [[ "$ROOT_SUBDIR" == /* ]]; then
  echo "Error: ROOT_SUBDIR must be relative to RESULTS_DIR: $ROOT_SUBDIR" >&2
  exit 1
fi
if [[ -n "$ROOT_SUBDIR" ]]; then
  ROOT_DIR="$RESULTS_DIR/$ROOT_SUBDIR"
else
  ROOT_DIR="$RESULTS_DIR/selected"
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi
if [[ "$(basename "$ROOT_DIR")" != "selected" ]]; then
  echo "Error: ROOT_DIR must point to the selected directory: $ROOT_DIR" >&2
  exit 1
fi
PLOT_SCRIPT="${SCRIPT_DIR}/feature_anchor_colour_distance_plot.sh"

if [[ ! -x "$PLOT_SCRIPT" ]]; then
  echo "Error: plot script not found or not executable: $PLOT_SCRIPT" >&2
  exit 1
fi

ANCHORS_DIR="$ROOT_DIR/anchors"
if [[ ! -d "$ANCHORS_DIR" ]]; then
  echo "Error: anchors directory not found: $ANCHORS_DIR" >&2
  exit 1
fi

shopt -s nullglob

declare -A GEN_MODELS=()
for animal_dir in "$ANCHORS_DIR"/*; do
  [[ -d "$animal_dir" ]] || continue
  for model_dir in "$animal_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    GEN_MODELS["$(basename "$model_dir")"]=1
  done
done

for gen_model in "${!GEN_MODELS[@]}"; do
  declare -A VISION_MODELS=()
  for animal_dir in "$ANCHORS_DIR"/*; do
    [[ -d "$animal_dir/$gen_model/features" ]] || continue
    for vision_dir in "$animal_dir/$gen_model/features"/*; do
      [[ -d "$vision_dir" ]] || continue
      VISION_MODELS["$(basename "$vision_dir")"]=1
    done
  done

  for vision_model in "${!VISION_MODELS[@]}"; do
    declare -A POOLINGS=()
    for animal_dir in "$ANCHORS_DIR"/*; do
      [[ -d "$animal_dir/$gen_model/features/$vision_model" ]] || continue
      for pooling_dir in "$animal_dir/$gen_model/features/$vision_model"/*; do
        [[ -d "$pooling_dir" ]] || continue
        POOLINGS["$(basename "$pooling_dir")"]=1
      done
    done

    for pooling in "${!POOLINGS[@]}"; do
      SELECTED_ROOT="$ROOT_DIR" GEN_MODEL="$gen_model" VISION_MODEL="$vision_model" POOLING="$pooling" "$PLOT_SCRIPT"
    done
  done
done
