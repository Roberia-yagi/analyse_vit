#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-}"
if [[ -z "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR is required and must be an absolute path." >&2
  exit 1
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  echo "Error: RESULTS_DIR must be an absolute path: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: RESULTS_DIR not found (absolute path required): $RESULTS_DIR" >&2
  exit 1
fi

SELECTED_ROOT="$RESULTS_DIR/selected"
if [[ ! -d "$SELECTED_ROOT" ]]; then
  echo "Error: selected root not found: $SELECTED_ROOT" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH

ARGS=(
  --selected-root "$SELECTED_ROOT"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-root "$OUTPUT_DIR")
fi
if [[ -n "${DPI:-}" ]]; then
  ARGS+=(--dpi "$DPI")
fi
if [[ -n "${PLOT_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$PLOT_ARGS"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

cd "$PROJECT_ROOT"
uv sync --extra analysis-perception
uv run --extra analysis-perception -- python -m analyse_vit.rare_colour_bias.feature_anchor_size_distance_plot "${ARGS[@]}"
