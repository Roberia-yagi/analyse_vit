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
PROMPT_JSON="$PROJECT_ROOT/data/eagle.json"
ANIMAL="$(basename "$PROMPT_JSON" .json)"

python -m analyse_vit.rare_colour_bias.generation.image_generation \
  --pipeline flux \
  --output-dir "$RAW_ROOT/anchors/$ANIMAL/flux" \
  --prompt "photorealistic studio photo, single adult eagle, full body, centered, uncropped, solid gray seamless background, wide shot, subject occupies 15% of the frame, detailed feathers" \
  --negative-prompt "close-up, macro, zoomed in, cropped, out of frame, multiple animals, perch, branch, scenery, text, watermark, logo, toy, cartoon, illustration" \
  --seed-bg 123 \
  --seed 456 \
  --num-runs 1