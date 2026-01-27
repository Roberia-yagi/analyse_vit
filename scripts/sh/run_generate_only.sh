#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python -m analyse_vit.rare_colour_bias.image_generation \
  --pipeline sd3 \
  --output-dir "$PROJECT_ROOT/results/raw/image_generation" \
  --background-prompt "A gray background" \
  --base-prompt-elements-json "$PROJECT_ROOT/data/prompt_seeds/kangaroo/real_kangaroo_prompt_seeds.json" \
  --paired-prompt-elements-json "$PROJECT_ROOT/data/prompt_seeds/kangaroo/paired_kangaroo_prompt_seeds.json" \
  --seed-bg 123 \
  --seed-real 456 \
  --seed-toy 789 \
  --num-runs 3
