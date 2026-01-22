#!/bin/bash
#SBATCH --job-name=analyse_vit_generate_all
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

# Ensure we run from the submit directory so relative paths resolve as expected.
cd /home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit
PROJECT_ROOT="$(pwd -P)"

echo "Job started at: $(date)"
echo "Project root: $PROJECT_ROOT"

# TODO: Load modules / activate your Python environment if needed.
# Example:
source "$PROJECT_ROOT/.venv/bin/activate"

run_step() {
  local name="$1"
  local script_path="$2"
  echo "--- Running: $name ($(date)) ---"
  bash "$script_path"
}

run_step "generate_10_power_drills" "$PROJECT_ROOT/scripts/generate_10_power_drills.sh"
run_step "generate_10_sharks" "$PROJECT_ROOT/scripts/generate_10_sharks.sh"
run_step "generate_10_strawberries" "$PROJECT_ROOT/scripts/generate_10_strawberries.sh"

echo "Job finished at: $(date)"
