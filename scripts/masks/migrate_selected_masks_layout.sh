#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# User-configurable values
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="/home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/results/selected"
# ------------------------------

RACE_ROOT="$PROJECT_ROOT/.race"
RACE_KEY="migrate_selected_masks_layout"
RACE_DIR="$RACE_ROOT/$RACE_KEY"
mkdir -p "$RACE_ROOT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RACE_DIR
  source "$PROJECT_ROOT/scripts/utils/race_guard.sh"
else
  LOCKDIR="$RACE_DIR.lock"
  if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "Error: another migrate run is active (lock exists): $LOCKDIR" >&2
    exit 1
  fi
  cleanup_lock() {
    rmdir "$LOCKDIR" 2>/dev/null || true
  }
  trap cleanup_lock EXIT
fi

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
if [[ "$(basename "$RESULTS_DIR")" != "selected" ]]; then
  echo "Error: RESULTS_DIR must point to the selected directory with an absolute path: $RESULTS_DIR" >&2
  exit 1
fi

SELECTED_ROOT="$RESULTS_DIR"
MASKS_ROOT="$SELECTED_ROOT/masks"
ANCHOR_MASKS_ROOT="$MASKS_ROOT/anchors"
ANGLE_MASKS_ROOT="$MASKS_ROOT/angles"
ANGLES_ROOT="$SELECTED_ROOT/angles"

if [[ ! -d "$SELECTED_ROOT" ]]; then
  echo "Error: selected root not found: $SELECTED_ROOT" >&2
  exit 1
fi

mkdir -p "$ANCHOR_MASKS_ROOT"
mkdir -p "$ANGLE_MASKS_ROOT"

# Step 1: legacy selected/masks/<animal> -> selected/masks/anchors/<animal>
mapfile -t LEGACY_MASK_DIRS < <(find "$MASKS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
MOVED_LEGACY_ANCHOR_DIRS=0
for src_dir in "${LEGACY_MASK_DIRS[@]}"; do
  base_name="$(basename "$src_dir")"
  if [[ "$base_name" == "anchors" || "$base_name" == "angles" ]]; then
    continue
  fi
  dst_dir="$ANCHOR_MASKS_ROOT/$base_name"
  if [[ -e "$dst_dir" ]]; then
    echo "Error: destination already exists, aborting to avoid mixed state: $dst_dir" >&2
    exit 1
  fi
  mv "$src_dir" "$dst_dir"
  MOVED_LEGACY_ANCHOR_DIRS=$((MOVED_LEGACY_ANCHOR_DIRS + 1))
done

# Step 2: flatten selected/masks/anchors/{with,without}_composite/<animal>/<model>
for composite_subdir in with_composite without_composite; do
  src_root="$ANCHOR_MASKS_ROOT/$composite_subdir"
  if [[ ! -d "$src_root" ]]; then
    continue
  fi
  while IFS= read -r -d '' model_dir; do
    rel="${model_dir#"$src_root"/}"
    animal="${rel%%/*}"
    model="${rel#*/}"
    if [[ -z "$animal" || -z "$model" || "$animal" == "$model" ]]; then
      echo "Error: unexpected anchor mask directory layout: $model_dir" >&2
      exit 1
    fi
    dst_dir="$ANCHOR_MASKS_ROOT/$animal/$model"
    mkdir -p "$dst_dir"
    while IFS= read -r -d '' src_file; do
      file_name="$(basename "$src_file")"
      dst_file="$dst_dir/$file_name"
      if [[ -f "$dst_file" ]]; then
        if ! cmp -s "$src_file" "$dst_file"; then
          echo "Error: conflicting anchor mask file exists: $dst_file" >&2
          exit 1
        fi
      else
        cp -p "$src_file" "$dst_file"
      fi
    done < <(find "$model_dir" -type f -name 'mask_run_*.png' -print0 | sort -z)
  done < <(find "$src_root" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
done

# Step 3: selected/angles/<composite>/<animal>/<angle>/<model>/masks -> selected/masks/angles/<animal>/<angle>/<model>
if [[ ! -d "$ANGLES_ROOT" ]]; then
  echo "Error: angles root not found: $ANGLES_ROOT" >&2
  exit 1
fi

COPIED_ANGLE_MASK_FILES=0
while IFS= read -r -d '' src_file; do
  rel="${src_file#"$ANGLES_ROOT"/}"
  IFS='/' read -r -a parts <<< "$rel"
  if (( ${#parts[@]} != 6 )); then
    echo "Error: unexpected angle mask file layout: $src_file" >&2
    exit 1
  fi
  animal="${parts[1]}"
  angle="${parts[2]}"
  model="${parts[3]}"
  stage="${parts[4]}"
  file_name="${parts[5]}"
  if [[ "$stage" != "masks" ]]; then
    echo "Error: unexpected angle mask stage (expected masks): $src_file" >&2
    exit 1
  fi

  dst_dir="$ANGLE_MASKS_ROOT/$animal/$angle/$model"
  dst_file="$dst_dir/$file_name"
  mkdir -p "$dst_dir"

  if [[ -f "$dst_file" ]]; then
    if ! cmp -s "$src_file" "$dst_file"; then
      echo "Error: conflicting angle mask file exists: $dst_file" >&2
      exit 1
    fi
    continue
  fi
  cp -p "$src_file" "$dst_file"
  COPIED_ANGLE_MASK_FILES=$((COPIED_ANGLE_MASK_FILES + 1))
done < <(find "$ANGLES_ROOT" -type f -path '*/masks/mask_run_*.png' -print0 | sort -z)

echo "Migration complete."
echo "Moved legacy anchor dirs: $MOVED_LEGACY_ANCHOR_DIRS"
echo "Copied angle mask files: $COPIED_ANGLE_MASK_FILES"
echo "Anchor masks root: $ANCHOR_MASKS_ROOT"
echo "Angle masks root: $ANGLE_MASKS_ROOT"
