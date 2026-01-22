#!/usr/bin/env bash
#SBATCH --job-name=gen10_kangaroos
#SBATCH --output=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/slurm/%x_%j.out
#SBATCH --error=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/slurm/%x_%j.err
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --partition=gpu-a100,gpu-a100-preempt,gpu-l40s-preempt,gpu-a100-short,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

# --chdir sets the working directory for the job.
PROJECT_ROOT="$(pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/generate_10_kangaroos.sh"
PY_MODULE="analyse_vit.rare_colour_bias.flux_sam_composite"

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

# Activate local virtual environment
source .venv/bin/activate

# Avoid buffered logs in Slurm output
export PYTHONUNBUFFERED=1

mkdir -p results/slurm

bash "$RUN_SCRIPT"

run_root="$(find_latest_run_dir "$PROJECT_ROOT/results" || true)"
if [[ -n "$run_root" ]]; then
  copy_sources "$run_root" "$RUN_SCRIPT" "$PY_MODULE"
else
  echo "Warning: results run directory not found under $PROJECT_ROOT/results" >&2
fi
