#!/usr/bin/env bash
#SBATCH --job-name=run_recolor_only_all_anchors
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
source "$PROJECT_ROOT/scripts/sbatch/utils/race_guard.sh"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/run_recolor_only_all_anchors.sh"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

RUN_SUBDIR="${RUN_SUBDIR:-}"
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
  uv run --extra generate -- bash "$RUN_SCRIPT" "$RUN_SUBDIR"
  RUN_ROOT="$RESULTS_DIR/$RUN_SUBDIR"
else
  uv run --extra generate -- bash "$RUN_SCRIPT"
  RUN_ROOT="$RESULTS_DIR"
fi

if [[ -d "$RUN_ROOT" ]]; then
  copy_sources "$RUN_ROOT" "$RUN_SCRIPT"
else
  echo "Warning: results run directory not found: $RUN_ROOT" >&2
fi
