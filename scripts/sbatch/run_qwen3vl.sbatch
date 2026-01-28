#!/usr/bin/env bash
#SBATCH --job-name=run_qwen3vl
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
source "$PROJECT_ROOT/scripts/sbatch/utils/race_guard.sh"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/run_qwen3vl.sh"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

export RESULTS_DIR
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
uv sync --extra analysis-qwen
uv run --extra analysis-qwen -- bash "$RUN_SCRIPT"
