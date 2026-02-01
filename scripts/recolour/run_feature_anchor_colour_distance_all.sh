#!/usr/bin/env bash
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi
ROOT_SUBDIR="${1:-}"
if [[ "$ROOT_SUBDIR" == /* ]]; then
  echo "Error: ROOT_SUBDIR must be relative to RESULTS_DIR: $ROOT_SUBDIR" >&2
  exit 1
fi
if [[ -n "$ROOT_SUBDIR" ]]; then
  ROOT_DIR="$RESULTS_DIR/$ROOT_SUBDIR"
else
  if [[ "$RESULTS_DIR" == */selected ]]; then
    ROOT_DIR="$RESULTS_DIR"
  else
    ROOT_DIR="$RESULTS_DIR/selected"
  fi
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi
if [[ "$(basename "$ROOT_DIR")" != "selected" ]]; then
  echo "Error: ROOT_DIR must point to the selected directory: $ROOT_DIR" >&2
  exit 1
fi
PLOT_SCRIPT="${SCRIPT_DIR}/feature_anchor_colour_distance_plot.sh"

if [[ ! -x "$PLOT_SCRIPT" ]]; then
  echo "Error: plot script not found or not executable: $PLOT_SCRIPT" >&2
  exit 1
fi

ANCHORS_DIR="$ROOT_DIR/anchors"
if [[ ! -d "$ANCHORS_DIR" ]]; then
  echo "Error: anchors directory not found: $ANCHORS_DIR" >&2
  exit 1
fi

shopt -s nullglob

declare -A GEN_MODELS=()
for animal_dir in "$ANCHORS_DIR"/*; do
  [[ -d "$animal_dir" ]] || continue
  for model_dir in "$animal_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    GEN_MODELS["$(basename "$model_dir")"]=1
  done
done

for gen_model in "${!GEN_MODELS[@]}"; do
  declare -A VISION_MODELS=()
  for animal_dir in "$ANCHORS_DIR"/*; do
    [[ -d "$animal_dir/$gen_model/features" ]] || continue
    for vision_dir in "$animal_dir/$gen_model/features"/*; do
      [[ -d "$vision_dir" ]] || continue
      VISION_MODELS["$(basename "$vision_dir")"]=1
    done
  done

  for vision_model in "${!VISION_MODELS[@]}"; do
    declare -A POOLINGS=()
    for animal_dir in "$ANCHORS_DIR"/*; do
      [[ -d "$animal_dir/$gen_model/features/$vision_model" ]] || continue
      for pooling_dir in "$animal_dir/$gen_model/features/$vision_model"/*; do
        [[ -d "$pooling_dir" ]] || continue
        POOLINGS["$(basename "$pooling_dir")"]=1
      done
    done

    for pooling in "${!POOLINGS[@]}"; do
      SELECTED_ROOT="$ROOT_DIR" GEN_MODEL="$gen_model" VISION_MODEL="$vision_model" POOLING="$pooling" "$PLOT_SCRIPT"
    done
  done
done
