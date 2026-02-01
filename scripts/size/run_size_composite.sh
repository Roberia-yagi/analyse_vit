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

SELECTED_ROOT="$RESULTS_DIR/selected"
if [[ ! -d "$SELECTED_ROOT" ]]; then
  echo "Error: selected directory not found (absolute path required): $SELECTED_ROOT" >&2
  exit 1
fi

SIZES="${SIZES:-100,90,80,70,60,50,40,30,20,10}"

python -m analyse_vit.rare_colour_bias.composite.size_composite \
  --selected-root "$SELECTED_ROOT" \
  --sizes "$SIZES" \
  "$@"
