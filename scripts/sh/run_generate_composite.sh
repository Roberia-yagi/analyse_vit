#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/raw/image_generation"

find_latest_run_dir() {
  local base_dir="$1"
  local latest
  latest="$(ls -1dt "$base_dir"/[0-9]*_[0-9]*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf "%s" "${latest%/}"
}

python -m analyse_vit.rare_colour_bias.image_generation \
  --pipeline flux \
  --output-dir "$PROJECT_ROOT/results/raw/image_generation" \
  --background-prompt "A grass field" \
  --base-prompt-elements-json "$PROJECT_ROOT/data/prompt_seeds/kangaroo/real_kangaroo_prompt_seeds.json" \
  --paired-prompt-elements-json "$PROJECT_ROOT/data/prompt_seeds/kangaroo/paired_kangaroo_prompt_seeds.json" \
  --seed-bg 123 \
  --seed-real 456 \
  --seed-toy 789 \
  --num-runs 10

RUN_ROOT="$(find_latest_run_dir "$RESULTS_DIR" || true)"
if [[ -z "$RUN_ROOT" ]]; then
  echo "Error: run root not found after generation." >&2
  exit 1
fi

python -m analyse_vit.rare_colour_bias.image_composite \
  --run-root "$RUN_ROOT" \
  --object-name-real "kangaroo" \
  --object-name-toy "toy" \
  --dominant-color "brown" \
  --rare-color "pink" \
  --seed-real 456 \
  --seed-toy 789 \
  --sam-checkpoint "$PROJECT_ROOT/models/sam/sam_vit_h_4b8939.pth" \
  --lang-sam-box-threshold 0.25 \
  --lang-sam-text-threshold 0.25 \
  --num-runs 10
