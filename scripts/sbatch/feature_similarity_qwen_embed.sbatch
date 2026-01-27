#!/usr/bin/env bash
#SBATCH --job-name=feature_similarity_qwen_embed
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
source "$PROJECT_ROOT/scripts/sbatch/utils/race_guard.sh"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/feature_similarity_qwen_embed.sh"
RESULTS_DIR="${RESULTS_DIR:-}"
if [[ -z "$RESULTS_DIR" ]]; then
  echo "Error: set RESULTS_DIR to an absolute results date path (e.g., /.../results/20260123_170000) via sbatch --export." >&2
  exit 1
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

export RESULTS_DIR
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/analysis/qwen3-vl-8b-embed}"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

uv run --extra analysis-qwen -- bash "$RUN_SCRIPT"
