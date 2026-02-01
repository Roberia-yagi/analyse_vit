#!/usr/bin/env bash
#SBATCH --job-name=run_generate_angles_flux_qwen
#SBATCH --chdir=/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${PWD}"
CACHE_ROOT="$PROJECT_ROOT/.cache"
HF_CACHE_ROOT="$PROJECT_ROOT/.hf_cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
source "$PROJECT_ROOT/scripts/utils/race_guard.sh"

RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR (absolute path required)" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

ANGLES_ROOT="$PROJECT_ROOT/data/angles"
if [[ ! -d "$ANGLES_ROOT" ]]; then
  echo "Error: angles directory not found: $ANGLES_ROOT" >&2
  exit 1
fi

OUTPUT_ROOT="$RESULTS_DIR/raw/angles"
SEED=456
NUM_RUNS=10
MODELS="qwen"
BATCH_SIZE=1

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
