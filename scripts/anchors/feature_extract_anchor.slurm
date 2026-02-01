#!/usr/bin/env bash
#SBATCH --job-name=feature_extract_anchor
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail


# ------------------------------
# Pushover notification setup
SCRIPT_BASENAME="$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PWD}"
source "${PROJECT_ROOT}/.env"

pushover_send() {
  local title="$1"
  local message="$2"
  curl -fsS --retry 3 --retry-delay 2 \
    -F "token=${PUSHOVER_TOKEN}" \
    -F "user=${PUSHOVER_USER}" \
    -F "title=${title}" \
    -F "message=${message}" \
    https://api.pushover.net/1/messages.json >/dev/null || true
}

start_ts="$(date +%s)"

notify_exit() {
  local code=$?
  local end_ts="$(date +%s)"
  local elapsed="$((end_ts - start_ts))"

  local jid="${SLURM_JOB_ID:-unknown}"
  local name="${SLURM_JOB_NAME:-job}"
  local host="$(hostname)"

  local array=""
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    array=" array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  local status="COMPLETED"
  local title="Slurm job finished: ${SCRIPT_BASENAME}"
  local kind="slurm"
  if [ "$code" -ne 0 ]; then
    status="FAILED (exit ${code})"
    title="Slurm job failed: ${SCRIPT_BASENAME}"
  fi

  if [ "${jid}" = "unknown" ]; then
    kind="script"
    title="Script finished: ${SCRIPT_BASENAME}"
    if [ "$code" -ne 0 ]; then
      title="Script failed: ${SCRIPT_BASENAME}"
    fi
  fi

  pushover_send "${title}" \
    "type=${kind} file=${SCRIPT_BASENAME} job=${name} id=${jid}${array} host=${host} status=${status} runtime=${elapsed}s"
}
trap notify_exit EXIT
# ------------------------------
PROJECT_ROOT="${PWD}"

DEFAULT_RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
INPUT_SUBDIR="${INPUT_SUBDIR:-}"
COMPOSITES="${COMPOSITES:-without_composite}"
MODELS="${MODELS:-siglip2}"
FEATURE_EXTRACT_ARGS=""

CACHE_ROOT="$PROJECT_ROOT/.cache"
HF_CACHE_ROOT="$PROJECT_ROOT/.hf_cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
source "$PROJECT_ROOT/scripts/utils/race_guard.sh"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR (absolute path required)" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

if [[ -n "$INPUT_SUBDIR" ]]; then
  if [[ "$INPUT_SUBDIR" == /* ]]; then
    echo "Error: INPUT_SUBDIR must be relative to RESULTS_DIR: $INPUT_SUBDIR" >&2
    exit 1
  fi
  INPUT_DIR="$RESULTS_DIR/$INPUT_SUBDIR"
else
  if [[ "$RESULTS_DIR" == */selected ]]; then
    INPUT_DIR="$RESULTS_DIR"
  else
    INPUT_DIR="$RESULTS_DIR/selected"
  fi
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
  echo "Error: OUTPUT_SUBDIR is not supported. Features must be saved under selected/anchors/{without_composite,with_composite} or selected/colour(s)/{without_composite,with_composite}." >&2
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
  read -r -a MODEL_ARGS <<< "$models_str"
  ARGS=(
    --input-root "$INPUT_DIR"
    --models "${MODEL_ARGS[@]}"
  )
  if [[ -n "${COMPOSITES:-}" ]]; then
    read -r -a COMPOSITE_ARGS <<< "$COMPOSITES"
    ARGS+=(--composites "${COMPOSITE_ARGS[@]}")
  fi
  ARGS+=(--targets "anchors")
  if [[ -n "${DEVICE:-}" ]]; then
    ARGS+=(--device "$DEVICE")
  fi
  if [[ -n "${BATCH_SIZE:-}" ]]; then
    ARGS+=(--batch-size "$BATCH_SIZE")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    ARGS+=(--hf-token "$HF_TOKEN")
  fi
  if [[ -n "${POOLINGS:-}" ]]; then
    read -r -a POOLING_ARGS <<< "$POOLINGS"
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
  build_args "$MODELS_STR"
  uv run --extra analysis-qwen -- python -m analyse_vit.rare_colour_bias.feature_extraction.feature_extract_anchor_colour "${ARGS[@]}"
fi

if (( ${#SIGLIP2_MODELS[@]} )); then
  uv sync --extra analysis-qwen
  MODELS_STR="${SIGLIP2_MODELS[*]}"
  build_args "$MODELS_STR"
  uv run --extra analysis-qwen -- python -m analyse_vit.rare_colour_bias.feature_extraction.feature_extract_anchor_colour "${ARGS[@]}"
fi

if (( ${#PE_MODELS[@]} )); then
  uv sync --extra analysis-perception
  MODELS_STR="${PE_MODELS[*]}"
  build_args "$MODELS_STR"
  uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.feature_extraction.feature_extract_anchor_colour "${ARGS[@]}"
fi
