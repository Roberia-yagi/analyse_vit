#!/usr/bin/env bash
#SBATCH --job-name=run_recolor_only_all_anchors
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
CACHE_ROOT="$PROJECT_ROOT/.cache"
HF_CACHE_ROOT="$PROJECT_ROOT/.hf_cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
source "$PROJECT_ROOT/scripts/utils/race_guard.sh"
RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results"
RUN_SUBDIR=""
LANG_SAM_BOX_THRESHOLD=0.25
LANG_SAM_TEXT_THRESHOLD=0.25
MASK_DILATE_PX=9
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
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR (absolute path required)" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

if [[ "$RUN_SUBDIR" == /* ]]; then
  echo "Error: RUN_SUBDIR must be relative to RESULTS_DIR: $RUN_SUBDIR" >&2
  exit 1
fi

copy_sources() {
  local results_dir="$1"
  local run_script="$2"
  local dest_dir="$results_dir/source"
  local src_dir="$PROJECT_ROOT/src"

  mkdir -p "$dest_dir"
  cp -p "$run_script" "$dest_dir/"
  if [[ -d "$src_dir" ]]; then
    cp -a "$src_dir" "$dest_dir/"
  else
    echo "Warning: source directory not found: $src_dir" >&2
  fi
}

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

if [[ -n "$RUN_SUBDIR" ]]; then
  RUN_ROOT="$RESULTS_DIR/$RUN_SUBDIR"
else
  RUN_ROOT="$RESULTS_DIR"
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Error: selected root not found: $RUN_ROOT" >&2
  exit 1
fi

echo "=== Recolor all anchors under: $RUN_ROOT/anchors ==="
uv run --extra generate -- python -m analyse_vit.rare_colour_bias.recolor.image_recolor \
  --anchors-root "$RUN_ROOT/anchors" \
  --output-root "$RUN_ROOT" \
  --object-name "auto" \
  --colors "$(IFS=,; echo "${COLORS[*]}")" \
  --lang-sam-box-threshold "$LANG_SAM_BOX_THRESHOLD" \
  --lang-sam-text-threshold "$LANG_SAM_TEXT_THRESHOLD" \
  --mask-dilate-px "$MASK_DILATE_PX"

copy_sources "$RUN_ROOT" "$0"
