#!/usr/bin/env bash
set -euo pipefail

: "${RACE_DIR:?RACE_DIR が未設定です（sbatch 側で --export してください）}"

WINNER_FILE="${RACE_DIR}/winner_jobid"
LOCKDIR="${RACE_DIR}/lockdir"

mkdir -p "${RACE_DIR}"

# mkdir は原子的に成功/失敗するため、簡易ロックとして使えます
if mkdir "${LOCKDIR}" 2>/dev/null; then
  # 自分が「勝者」
  echo "${SLURM_JOB_ID}" > "${WINNER_FILE}"
else
  # 自分は「敗者」：即座に自己キャンセル（権限がない環境では exit のみでも可）
  scancel "${SLURM_JOB_ID}" 2>/dev/null || true
  exit 0
fi
