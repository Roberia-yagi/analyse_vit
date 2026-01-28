#!/usr/bin/env bash
#SBATCH --job-name=feature_extract_anchor_colour
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
source "$PROJECT_ROOT/scripts/sbatch/utils/race_guard.sh"
RUN_SCRIPT="$PROJECT_ROOT/scripts/sh/feature_extract_anchor_colour.sh"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

INPUT_SUBDIR="${INPUT_SUBDIR:-selected}"
if [[ -n "$INPUT_SUBDIR" ]]; then
  if [[ "$INPUT_SUBDIR" == /* ]]; then
    echo "Error: INPUT_SUBDIR must be relative to RESULTS_DIR: $INPUT_SUBDIR" >&2
    exit 1
  fi
  INPUT_DIR="$RESULTS_DIR/$INPUT_SUBDIR"
else
  INPUT_DIR="$RESULTS_DIR/selected"
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: INPUT_DIR not found: $INPUT_DIR" >&2
  exit 1
fi
if [[ "$(basename "$INPUT_DIR")" != "selected" ]]; then
  echo "Error: INPUT_DIR must point to the selected directory: $INPUT_DIR" >&2
  exit 1
fi

if [[ -n "${OUTPUT_SUBDIR:-}" ]]; then
  echo "Error: OUTPUT_SUBDIR is not supported. Features must be saved under selected/anchors or selected/colour(s)." >&2
  exit 1
fi

export RESULTS_DIR INPUT_SUBDIR
export INPUT_DIR
export MODELS="${MODELS:-qwen3-vl-8b-embed}"

export PYTHONUNBUFFERED=1

QWEN_MODELS=()
PE_MODELS=()
read -r -a MODEL_ARGS <<< "$MODELS"
for model in "${MODEL_ARGS[@]}"; do
  model_lower="${model,,}"
  case "$model_lower" in
    pe|pe-core|pe-core-l14|pe-core-l14-336|core|l14|pe:*|perception:*|facebook/pe-*)
      PE_MODELS+=("$model")
      ;;
    qwen|qwen-embed|qwen3-vl-8b|qwen3-vl-8b-embed|qwen/*|qwen3-*)
      QWEN_MODELS+=("$model")
      ;;
    *)
      QWEN_MODELS+=("$model")
      ;;
  esac
done

cd "$PROJECT_ROOT"

if (( ${#QWEN_MODELS[@]} )); then
  uv sync --extra analysis-qwen
  export MODELS="${QWEN_MODELS[*]}"
  uv run --extra analysis-qwen -- bash "$RUN_SCRIPT"
fi

if (( ${#PE_MODELS[@]} )); then
  uv sync --extra analysis-perception
  export MODELS="${PE_MODELS[*]}"
  uv run --extra analysis-perception -- bash "$RUN_SCRIPT"
fi
