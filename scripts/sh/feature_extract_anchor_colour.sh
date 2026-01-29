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

INPUT_SUBDIR="${INPUT_SUBDIR:-selected}"
if [[ -n "$INPUT_SUBDIR" ]]; then
  if [[ "$INPUT_SUBDIR" == /* ]]; then
    echo "Error: INPUT_SUBDIR must be relative to RESULTS_DIR: $INPUT_SUBDIR" >&2
    exit 1
  fi
  INPUT_DIR="$RESULTS_DIR/$INPUT_SUBDIR"
else
  INPUT_DIR="$RESULTS_DIR/selected"
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
  echo "Error: OUTPUT_SUBDIR is not supported. Features must be saved under selected/anchors or selected/colour(s)." >&2
  exit 1
fi

export MODELS="${MODELS:-siglip2}"
if [[ -z "${MODELS:-}" ]]; then
  echo "Error: MODELS is empty. Provide MODELS." >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH

read -r -a MODEL_ARGS <<< "$MODELS"
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

cd "$PROJECT_ROOT"
python -m analyse_vit.rare_colour_bias.feature_extract_anchor_colour "${ARGS[@]}"
