#!/usr/bin/env bash
#SBATCH --job-name=run_generate_only
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
CACHE_ROOT="$PROJECT_ROOT/.cache"
HF_CACHE_ROOT="$PROJECT_ROOT/.hf_cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
source "$PROJECT_ROOT/scripts/utils/race_guard.sh"
RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results"
BG_ROOT="${RESULTS_DIR}/raw"
PROMPT_ELEMENTS_JSON="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/data/angles/back/kangaroo_back.json"
PIPELINES=("flux" "qwen")
SEED=0
NUM_RUNS=10

if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR (absolute path required)" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

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
  local dest_dir="$run_root/source"
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
export RESULTS_DIR
unset UV_EXTRAS UV_ALL_EXTRAS

for pipeline in "${PIPELINES[@]}"; do
  uv run --extra generate -- python -m analyse_vit.rare_colour_bias.generation.image_generation \
    --pipeline "$pipeline" \
    --output-dir "$RESULTS_DIR/raw/$pipeline" \
    --prompt-elements-json "$PROMPT_ELEMENTS_JSON" \
    --seed "$SEED" \
    --num-runs "$NUM_RUNS"
done

for pipeline in "${PIPELINES[@]}"; do
  run_root="$(find_latest_run_dir "$BG_ROOT/$pipeline" || true)"
  if [[ -n "$run_root" ]]; then
    copy_sources "$run_root" "$0"
  else
    echo "Warning: results run directory not found under $BG_ROOT/$pipeline" >&2
  fi
done
