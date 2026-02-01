#!/usr/bin/env bash
#SBATCH --job-name=run_generate_only_all_animals
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00

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
TARGET_RESULTS_DIR="$RESULTS_DIR/raw/anchors"
PIPELINE="flux"
SEED_BG=123
SEED=456
NUM_RUNS=10

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

# data/*.json determines animals.
mapfile -t PROMPT_JSONS < <(find "$PROJECT_ROOT/data" -maxdepth 1 -type f -name "*.json" | sort)
if [[ ${#PROMPT_JSONS[@]} -eq 0 ]]; then
  echo "No prompt JSON files found in $PROJECT_ROOT/data" >&2
  exit 1
fi

ANIMALS=()
for prompt_json in "${PROMPT_JSONS[@]}"; do
  ANIMALS+=("$(basename "$prompt_json" .json)")
done

find_latest_run_dir() {
  local base_dir="$1"
  local latest
  latest="$(ls -1dt "$base_dir"/[0-9]*_[0-9]*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf "%s" "${latest%/}"
}

copy_sources() {
  local run_root="$1"
  local run_script="$2"
  local dest_dir="$run_root/source"
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
unset UV_EXTRAS UV_ALL_EXTRAS

for prompt_json in "${PROMPT_JSONS[@]}"; do
  animal="$(basename "$prompt_json" .json)"
  echo "=== Generating: ${animal} ==="
  uv run --extra generate -- python -m analyse_vit.rare_colour_bias.generation.image_generation \
    --pipeline "$PIPELINE" \
    --output-dir "$TARGET_RESULTS_DIR/${animal}/${PIPELINE}" \
    --prompt-elements-json "$prompt_json" \
    --seed-bg "$SEED_BG" \
    --seed "$SEED" \
    --num-runs "$NUM_RUNS"
done

for animal in "${ANIMALS[@]}"; do
  run_root="$(find_latest_run_dir "$TARGET_RESULTS_DIR/$animal/$PIPELINE" || true)"
  if [[ -n "$run_root" ]]; then
    copy_sources "$run_root" "$0"
  else
    echo "Warning: results run directory not found under $TARGET_RESULTS_DIR/$animal/$PIPELINE" >&2
  fi
done
