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
RAW_ROOT="$RESULTS_DIR/raw"

# Use all available JSON prompt files under data/*.json
mapfile -t PROMPT_JSONS < <(find "$PROJECT_ROOT/data" -maxdepth 1 -type f -name "*.json" | sort)

if [[ ${#PROMPT_JSONS[@]} -eq 0 ]]; then
  echo "No prompt JSON files found in $PROJECT_ROOT/data" >&2
  exit 1
fi

for prompt_json in "${PROMPT_JSONS[@]}"; do
  animal="$(basename "$prompt_json" .json)"
  echo "=== Generating: ${animal} ==="
  python -m analyse_vit.rare_colour_bias.generation.image_generation \
    --pipeline flux \
    --output-dir "$RAW_ROOT/anchors/${animal}/flux" \
    --prompt-elements-json "$prompt_json" \
    --seed-bg 123 \
    --seed 456 \
    --num-runs 10
done
