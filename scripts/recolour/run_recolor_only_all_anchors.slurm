#!/usr/bin/env bash
#SBATCH --job-name=run_recolor_only_all_anchors
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

RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
RUN_SUBDIR=""
LANG_SAM_BOX_THRESHOLD=0.25
LANG_SAM_TEXT_THRESHOLD=0.25
MASK_DILATE_PX=9
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
COMPOSITE_SUBDIRS=(
  without_composite
  with_composite
)
COLORS=(
  red
  orange
  yellow
  green
  cyan
  blue
  purple
  magenta
  pink
  gray
  black
  white
)

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

if [[ "$RUN_SUBDIR" == /* ]]; then
  echo "Error: RUN_SUBDIR must be relative to RESULTS_DIR: $RUN_SUBDIR" >&2
  exit 1
fi
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "Error: NUM_SHARDS must be >= 1." >&2
  exit 1
fi
if [[ "$SHARD_INDEX" -lt 0 || "$SHARD_INDEX" -ge "$NUM_SHARDS" ]]; then
  echo "Error: SHARD_INDEX must be within [0, NUM_SHARDS): $SHARD_INDEX" >&2
  exit 1
fi

copy_sources() {
  local results_dir="$1"
  local run_script="$2"
  local dest_dir="$results_dir/source"
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

if [[ -n "$RUN_SUBDIR" ]]; then
  RUN_ROOT="$RESULTS_DIR/$RUN_SUBDIR"
else
  RUN_ROOT="$RESULTS_DIR"
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Error: selected root not found: $RUN_ROOT" >&2
  exit 1
fi

for composite_subdir in "${COMPOSITE_SUBDIRS[@]}"; do
  ANCHORS_ROOT="$RUN_ROOT/anchors/$composite_subdir"
  if [[ ! -d "$ANCHORS_ROOT" ]]; then
    echo "Error: anchors root not found: $ANCHORS_ROOT" >&2
    exit 1
  fi

  echo "=== Recolor all anchors under: $ANCHORS_ROOT ==="
  uv run --extra generate -- python -m analyse_vit.rare_colour_bias.recolor.image_recolor \
    --anchors-root "$ANCHORS_ROOT" \
    --output-root "$RUN_ROOT" \
    --composite-subdir "$composite_subdir" \
    --object-name "auto" \
    --colors "$(IFS=,; echo "${COLORS[*]}")" \
    --lang-sam-box-threshold "$LANG_SAM_BOX_THRESHOLD" \
    --lang-sam-text-threshold "$LANG_SAM_TEXT_THRESHOLD" \
    --mask-dilate-px "$MASK_DILATE_PX" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
done

copy_sources "$RUN_ROOT" "$0"
