#!/usr/bin/env bash
#SBATCH --job-name=run_all
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
source "$PROJECT_ROOT/scripts/sbatch/utils/race_guard.sh"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/run_all.sh"
PY_MODULE="analyse_vit.rare_colour_bias.image_generation"
RESULTS_DIR="$PROJECT_ROOT/results/raw/image_generation"

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

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1

bash "$RUN_SCRIPT"

run_root="$(find_latest_run_dir "$RESULTS_DIR" || true)"
if [[ -n "$run_root" ]]; then
  copy_sources "$run_root" "$RUN_SCRIPT" "$PY_MODULE"
else
  echo "Warning: results run directory not found under $PROJECT_ROOT/results" >&2
fi
