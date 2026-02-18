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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
if [[ "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

if [[ "$RESULTS_DIR" == */selected ]]; then
  SELECTED_ROOT="$RESULTS_DIR"
else
  SELECTED_ROOT="$RESULTS_DIR/selected"
fi
if [[ ! -d "$SELECTED_ROOT" ]]; then
  echo "Error: selected root not found: $SELECTED_ROOT" >&2
  exit 1
fi
COMPOSITE_SUBDIR="without_composite"
if [[ ! -d "$SELECTED_ROOT/anchors/$COMPOSITE_SUBDIR" ]]; then
  echo "Error: anchors root not found: $SELECTED_ROOT/anchors/$COMPOSITE_SUBDIR" >&2
  exit 1
fi
if [[ ! -d "$SELECTED_ROOT/angles/$COMPOSITE_SUBDIR" ]]; then
  echo "Error: angles root not found: $SELECTED_ROOT/angles/$COMPOSITE_SUBDIR" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-}"
GEN_MODEL="qwen"
POOLING="attention_pooling"
DPI=200
FIG_WIDTH=8.0
FIG_HEIGHT=11.7
VISION_MODELS=(
  "pe-core-l14-336"
  "siglip2-giant-opt-patch16-384"
  "qwen3-vl-8b-embed"
)

export PYTHONNOUSERSITE=1
unset PYTHONPATH

BASE_ARGS=(
  --selected-root "$SELECTED_ROOT"
  --gen-model "$GEN_MODEL"
  --pooling "$POOLING"
  --dpi "$DPI"
  --fig-width "$FIG_WIDTH"
  --fig-height "$FIG_HEIGHT"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  BASE_ARGS+=(--output-root "$OUTPUT_DIR")
fi
if [[ -n "${PLOT_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$PLOT_ARGS"
  BASE_ARGS+=("${EXTRA_ARGS[@]}")
fi

cd "$PROJECT_ROOT"
for VISION_MODEL in "${VISION_MODELS[@]}"; do
  ARGS=(
    "${BASE_ARGS[@]}"
    --vision-model "$VISION_MODEL"
  )
  uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.plot.feature_anchor_angle_distance_plot "${ARGS[@]}"
done
