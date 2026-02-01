#!/usr/bin/env bash
#SBATCH --job-name=feature_extract_angles
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
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR (absolute path required)" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

INPUT_SUBDIR="${INPUT_SUBDIR:-selected/angles}"
if [[ -n "$INPUT_SUBDIR" ]]; then
  if [[ "$INPUT_SUBDIR" == /* ]]; then
    echo "Error: INPUT_SUBDIR must be relative to RESULTS_DIR: $INPUT_SUBDIR" >&2
    exit 1
  fi
  INPUT_DIR="$RESULTS_DIR/$INPUT_SUBDIR"
else
  INPUT_DIR="$RESULTS_DIR/selected/angles"
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: INPUT_DIR not found: $INPUT_DIR" >&2
  exit 1
fi
if [[ "$(basename "$INPUT_DIR")" != "angles" || "$(basename "$(dirname "$INPUT_DIR")")" != "selected" ]]; then
  echo "Error: INPUT_DIR must point to the selected/angles directory: $INPUT_DIR" >&2
  exit 1
fi

if [[ -n "${OUTPUT_SUBDIR:-}" ]]; then
  echo "Error: OUTPUT_SUBDIR is not supported. Features must be saved under selected/angles/*/features." >&2
  exit 1
fi

export RESULTS_DIR INPUT_SUBDIR
export INPUT_DIR

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

QWEN_EMBED_MODELS=()
PE_MODELS=()
SIGLIP2_MODELS=()
MODELS="${MODELS:-pe qwen-embed siglip2}"
USER_POOLINGS="${POOLINGS:-}"
if [[ -z "$MODELS" ]]; then
  echo "Error: MODELS is empty. Provide MODELS." >&2
  exit 1
fi
read -r -a MODEL_ARGS <<< "$MODELS"
for model in "${MODEL_ARGS[@]}"; do
  model_lower="${model,,}"
  case "$model_lower" in
    pe|pe-core|pe-core-l14|pe-core-l14-336|core|l14|pe:*|perception:*|facebook/pe-*)
      PE_MODELS+=("$model")
      ;;
    qwen-embed|qwen_embed|qwen3-embed|qwen3-vl-8b-embed|qwen-embed:*|qwen_embed:*|qwen3-embed:*|qwen3-vl-embed:*)
      QWEN_EMBED_MODELS+=("$model")
      ;;
    siglip2|siglip2-*|siglip2:*|google/siglip2-*)
      SIGLIP2_MODELS+=("$model")
      ;;
    *)
      SIGLIP2_MODELS+=("$model")
      ;;
  esac
done

cd "$PROJECT_ROOT"

build_args() {
  local models_str="$1"
  local poolings_str="$2"
  read -r -a MODEL_ARGS <<< "$models_str"
  ARGS=(
    --input-root "$INPUT_DIR"
    --models "${MODEL_ARGS[@]}"
  )
  if [[ -n "${DEVICE:-}" ]]; then
    ARGS+=(--device "$DEVICE")
  fi
  if [[ -n "${BATCH_SIZE:-}" ]]; then
    ARGS+=(--batch-size "$BATCH_SIZE")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    ARGS+=(--hf-token "$HF_TOKEN")
  fi
  if [[ -n "$poolings_str" ]]; then
    read -r -a POOLING_ARGS <<< "$poolings_str"
    ARGS+=(--poolings "${POOLING_ARGS[@]}")
  fi
  if [[ -n "${OVERWRITE:-}" ]]; then
    ARGS+=(--overwrite)
  fi
  if [[ -n "${FEATURE_EXTRACT_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$FEATURE_EXTRACT_ARGS"
    ARGS+=("${EXTRA_ARGS[@]}")
  fi
}

if (( ${#QWEN_EMBED_MODELS[@]} )); then
  uv sync --extra analysis-qwen
  MODELS_STR="${QWEN_EMBED_MODELS[*]}"
  POOLINGS_STR="$USER_POOLINGS"
  if [[ -z "$POOLINGS_STR" ]]; then
    POOLINGS_STR="attention_pooling"
  fi
  build_args "$MODELS_STR" "$POOLINGS_STR"
  uv run --extra analysis-qwen -- python -m analyse_vit.rare_colour_bias.feature_extract_angles "${ARGS[@]}"
fi

if (( ${#SIGLIP2_MODELS[@]} )); then
  uv sync --extra analysis-qwen
  MODELS_STR="${SIGLIP2_MODELS[*]}"
  POOLINGS_STR="$USER_POOLINGS"
  if [[ -z "$POOLINGS_STR" ]]; then
    POOLINGS_STR="cls mean_patch"
  fi
  build_args "$MODELS_STR" "$POOLINGS_STR"
  uv run --extra analysis-qwen -- python -m analyse_vit.rare_colour_bias.feature_extract_angles "${ARGS[@]}"
fi

if (( ${#PE_MODELS[@]} )); then
  uv sync --extra analysis-perception
  MODELS_STR="${PE_MODELS[*]}"
  POOLINGS_STR="$USER_POOLINGS"
  if [[ -z "$POOLINGS_STR" ]]; then
    POOLINGS_STR="cls mean_patch"
  fi
  build_args "$MODELS_STR" "$POOLINGS_STR"
  uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.feature_extract_angles "${ARGS[@]}"
fi
