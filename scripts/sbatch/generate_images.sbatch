#!/usr/bin/env bash
#SBATCH --job-name=gen10_kangaroos
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

source /home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/scripts/sbatch/utils/race_guard.sh

PROJECT_ROOT="$(pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/generate_images.sh"
PY_MODULE="analyse_vit.rare_colour_bias.image_generation.generate_images"

find_latest_run_dir() {
  local base_dir="$1"
  local latest
  latest="$(ls -1dt "$base_dir"/[0-9]*_[0-9]*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf "%s" "${latest%/}"
}

copy_sources() {
  local run_root="$1"
  local run_script="$2"
  local py_module="$3"
  local dest_dir="$run_root/source"
  local py_file="$PROJECT_ROOT/src/${py_module//./\/}.py"

  mkdir -p "$dest_dir"
  cp -p "$run_script" "$dest_dir/"
  if [[ -f "$py_file" ]]; then
    cp -p "$py_file" "$dest_dir/"
  else
    echo "Warning: python module file not found: $py_file" >&2
  fi
}

source .venv/bin/activate
export PYTHONUNBUFFERED=1

bash "$RUN_SCRIPT"

run_root="$(find_latest_run_dir "$PROJECT_ROOT/results/image_generation" || true)"
if [[ -n "$run_root" ]]; then
  copy_sources "$run_root" "$RUN_SCRIPT" "$PY_MODULE"
else
  echo "Warning: results run directory not found under $PROJECT_ROOT/results" >&2
fi
