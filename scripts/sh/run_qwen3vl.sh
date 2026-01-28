#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
if [[ -z "$RESULTS_DIR" || "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
NUM_DUMMY="${NUM_DUMMY:-1}"
IMAGE_ROOT="${IMAGE_ROOT:-$RESULTS_DIR}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH

ARGS=(
  --model-id "$MODEL_ID"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "$DEVICE")
fi

IMAGE_ARGS=()
if [[ -n "${IMAGES:-}" ]]; then
  read -r -a IMAGE_LIST <<< "$IMAGES"
  for img in "${IMAGE_LIST[@]}"; do
    if [[ "$img" != /* ]]; then
      if [[ -n "$IMAGE_ROOT" ]]; then
        img="$IMAGE_ROOT/$img"
      fi
    fi
    IMAGE_ARGS+=(--image "$img")
  done
else
  ARGS+=(--num-dummy "$NUM_DUMMY")
fi

if [[ -n "${QWEN3VL_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$QWEN3VL_ARGS"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

cd "$PROJECT_ROOT"
python -m analyse_vit.rare_colour_bias.qwen3vl "${ARGS[@]}" "${IMAGE_ARGS[@]}"
