#!/usr/bin/env bash
set -euo pipefail

: "${RESULTS_DIR:?RESULTS_DIR (absolute path) is required (set by sbatch).}"

if [[ "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DATE="$(basename "$RESULTS_DIR")"
RESULTS_ROOT="$(cd "$(dirname "$RESULTS_DIR")" && pwd)"

export MODELS="${MODELS:-qwen3-vl-8b-embed}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/analysis/qwen3-vl-8b-embed}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

read -r -a MODEL_ARGS <<< "$MODELS"
ARGS=(
  --results-date "$RESULTS_DATE"
  --results-root "$RESULTS_ROOT"
  --output-dir "$OUTPUT_DIR"
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
if [[ -n "${PAIRS:-}" ]]; then
  read -r -a PAIR_ARGS <<< "$PAIRS"
  ARGS+=(--pairs "${PAIR_ARGS[@]}")
fi
if [[ -n "${FEATURE_SIMILARITY_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$FEATURE_SIMILARITY_ARGS"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

cd "$PROJECT_ROOT"
python -m analyse_vit.rare_colour_bias.feature_analysis "${ARGS[@]}"
