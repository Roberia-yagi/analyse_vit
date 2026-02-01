#!/usr/bin/env bash
#SBATCH --job-name=verify_selected_features
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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
MODELS="${MODELS:-pe qwen-embed siglip2}"
ANIMAL="${ANIMAL:-}"
RUN_ID="${RUN_ID:-}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
RTOL="${RTOL:-}"
ATOL="${ATOL:-}"

CACHE_ROOT="$PROJECT_ROOT/.cache"
HF_CACHE_ROOT="$PROJECT_ROOT/.hf_cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
source "$PROJECT_ROOT/scripts/utils/race_guard.sh"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
export RESULTS_DIR

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

cd "$PROJECT_ROOT"
if [[ -z "$MODELS" ]]; then
  echo "Error: MODELS is empty. Provide MODELS." >&2
  exit 1
fi
read -r -a MODEL_ARGS <<< "$MODELS"

PE_MODELS=()
QWEN_MODELS=()
SIGLIP2_MODELS=()
for model in "${MODEL_ARGS[@]}"; do
  model_lower="${model,,}"
  case "$model_lower" in
    pe|pe-core|pe-core-l14|pe-core-l14-336|core|l14|pe:*|perception:*|facebook/pe-*)
      PE_MODELS+=("$model")
      ;;
    qwen-embed|qwen_embed|qwen3-embed|qwen3-vl-8b-embed|qwen-embed:*|qwen_embed:*|qwen3-embed:*|qwen3-vl-embed:*)
      QWEN_MODELS+=("$model")
      ;;
    siglip2|siglip2-*|siglip2:*|google/siglip2-*)
      SIGLIP2_MODELS+=("$model")
      ;;
    *)
      SIGLIP2_MODELS+=("$model")
      ;;
  esac
done

build_args() {
  local models_str="$1"
  local -a args
  args=(--results-dir "$RESULTS_DIR")
  if [[ -n "${ANIMAL:-}" ]]; then
    args+=(--animal "$ANIMAL")
  fi
  if [[ -n "${RUN_ID:-}" ]]; then
    args+=(--run-id "$RUN_ID")
  fi
  if [[ -n "${DEVICE:-}" ]]; then
    args+=(--device "$DEVICE")
  fi
  if [[ -n "${BATCH_SIZE:-}" ]]; then
    args+=(--batch-size "$BATCH_SIZE")
  fi
if [[ -n "${RTOL:-}" ]]; then
  args+=(--rtol "$RTOL")
fi
if [[ -n "${ATOL:-}" ]]; then
  args+=(--atol "$ATOL")
fi
args+=(--diagnose)
  if [[ -n "$models_str" ]]; then
    read -r -a MODEL_ARGS <<< "$models_str"
    args+=(--models "${MODEL_ARGS[@]}")
  fi
  ARGS=("${args[@]}")
}

run_qwen_family() {
  local models=()
  if (( ${#QWEN_MODELS[@]} )); then
    models+=("${QWEN_MODELS[@]}")
  fi
  if (( ${#SIGLIP2_MODELS[@]} )); then
    models+=("${SIGLIP2_MODELS[@]}")
  fi
  if (( ${#models[@]} )); then
    uv sync --extra analysis-qwen
    build_args "${models[*]}"
    uv run --extra analysis-qwen -- \
      python -m analyse_vit.rare_colour_bias.feature_extraction.verify_selected_features "${ARGS[@]}"
  fi
}

run_pe_family() {
  if (( ${#PE_MODELS[@]} )); then
    uv sync --extra analysis-perception
    build_args "${PE_MODELS[*]}"
    uv run --extra analysis-perception -- \
      python -m analyse_vit.rare_colour_bias.feature_extraction.verify_selected_features "${ARGS[@]}"
  fi
}

run_qwen_family
run_pe_family
