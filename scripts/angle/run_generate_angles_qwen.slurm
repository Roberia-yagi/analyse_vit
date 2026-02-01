#!/usr/bin/env bash
#SBATCH --job-name=run_generate_angles_flux_qwen
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

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

RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results"
ANGLES_ROOT="$PROJECT_ROOT/data/angles"
OUTPUT_ROOT="$RESULTS_DIR/raw/angles"
SEED=456
NUM_RUNS=10
MODELS="qwen"
BATCH_SIZE=1

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

if [[ ! -d "$ANGLES_ROOT" ]]; then
  echo "Error: angles directory not found: $ANGLES_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
unset UV_EXTRAS UV_ALL_EXTRAS

uv sync --extra generate
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" --force-reinstall --only-binary=:all: pillow
PIL_LIBS_DIR="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/pillow.libs"
if [[ ! -d "$PIL_LIBS_DIR" ]]; then
  echo "Error: Pillow shared libraries not found in $PIL_LIBS_DIR; reinstall Pillow with binary wheels." >&2
  exit 1
fi
if ! ls "$PIL_LIBS_DIR"/libtiff-*.so* >/dev/null 2>&1; then
  echo "Error: libtiff not found in $PIL_LIBS_DIR; reinstall Pillow with binary wheels." >&2
  exit 1
fi
if ! ls "$PIL_LIBS_DIR"/libjpeg-*.so* >/dev/null 2>&1; then
  echo "Error: libjpeg not found in $PIL_LIBS_DIR; reinstall Pillow with binary wheels." >&2
  exit 1
fi

uv run --extra generate -- python -m analyse_vit.rare_colour_bias.generation.angle_batch_generate \
  --angles-root "$ANGLES_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --models "$MODELS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --num-runs "$NUM_RUNS"
