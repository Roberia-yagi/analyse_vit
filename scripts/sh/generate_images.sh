python -m analyse_vit.rare_colour_bias.image_generation \
 --pipeline qwen \
 --output-dir results/raw/image_generation \
 --background-prompt "A grass field" \
 --base-prompt-elements-json data/prompt_seeds/kangaroo/real_kangaroo_prompt_seeds.json \
 --paired-prompt-elements-json data/prompt_seeds/kangaroo/paired_kangaroo_prompt_seeds.json \
 --seed-bg 123 \
 --seed-real 456 \
 --seed-toy 789 \
 --num-runs 10 \
