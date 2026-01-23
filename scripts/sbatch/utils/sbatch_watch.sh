#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sbatch_race_watch [options] [--] [sbatch args...] job.sbatch

Options (must come before sbatch args):
  --gpu-ram {40|80|40g|80g}  Select target partition group by GPU RAM (required)
  --race-root DIR            Base directory for race state (default: $HOME/.slurm_race)

  -i, --interval SECONDS     Refresh interval (default: 10)
  -n, --lines LINES          Tail lines for stdout/stderr (default: 30)
  --no-clear                 Do not clear the screen each refresh
  --no-tail                  Do not show stdout/stderr tail
  --no-seff                  Skip seff summary at the end
  -h, --help                 Show this help

Notes:
  - job.sbatch must source race_guard.sh early (before the main workload).
  - RACE_DIR must be on a shared filesystem visible from all target partitions.
  - If sbatch args might conflict with these options, insert "--" before sbatch args.

Examples:
  sbatch_race_watch --gpu-ram 80 -- --gres=gpu:1 -J train job.sbatch
  sbatch_race_watch --gpu-ram 40 -i 5 -n 50 -- -J exp job.sbatch
USAGE
}

# ============================================================
# HARD-CODED TARGET GROUPS
# Format: "partition[:qos][@account]"
#   - "gpu-a100"                 -> partition=gpu-a100, qos omitted, account omitted
#   - "feit-gpu-a100:feit"       -> partition=feit-gpu-a100, qos=feit
#   - "deeplearn@deeplearn"      -> partition=deeplearn, account=deeplearn
#   - "p:q@a"                    -> partition=p, qos=q, account=a
# ============================================================

# 40GB group (edit to your needs)
TARGETS_40=(
  "gpu-a100"
  "gpu-h100"
  "gpu-a100-short"
  "gpu-a100-preempt"
  "gpu-l40s-preempt"
  "deeplearn@deeplearn"
  "feit-gpu-a100:feit"
)

# 80GB group (edit to your needs)
TARGETS_80=(
  "gpu-a100"
  "gpu-h100"
  "feit-gpu-a100:feit"
)

# defaults
gpu_ram="40"
interval=5
lines=30
clear_screen=1
show_tail=1
show_seff=1
race_root="${HOME}/.slurm_race"

# parse our options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ram)
      gpu_ram="${2:-}"
      shift 2
      ;;
    --race-root)
      race_root="${2:-}"
      shift 2
      ;;
    -i|--interval)
      interval="${2:-}"
      shift 2
      ;;
    -n|--lines)
      lines="${2:-}"
      shift 2
      ;;
    --no-clear)
      clear_screen=0
      shift
      ;;
    --no-tail)
      show_tail=0
      shift
      ;;
    --no-seff)
      show_seff=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      # unknown option -> treat as sbatch arg
      break
      ;;
    *)
      # first non-option -> treat as sbatch arg
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

# normalize gpu ram option
norm_ram="${gpu_ram,,}"   # lowercase
norm_ram="${norm_ram%g}"  # allow "40g" / "80g"

declare -a TARGETS=()
case "$norm_ram" in
  40) TARGETS=("${TARGETS_40[@]}") ;;
  80) TARGETS=("${TARGETS_80[@]}") ;;
  "")
    echo "ERROR: --gpu-ram is required (40 or 80)." >&2
    exit 2
    ;;
  *)
    echo "ERROR: invalid --gpu-ram '${gpu_ram}'. Use 40, 80, 40g, or 80g." >&2
    exit 2
    ;;
esac

if [[ "${#TARGETS[@]}" -lt 2 ]]; then
  echo "ERROR: selected target group must contain at least 2 entries. (gpu-ram=${norm_ram})" >&2
  exit 2
fi

is_done_state() {
  local st="$1"
  [[ "$st" == COMPLETED* || "$st" == FAILED* || "$st" == CANCELLED* || "$st" == TIMEOUT* || "$st" == OUT_OF_MEMORY* || "$st" == NODE_FAIL* ]]
}

# race state dir (must be on shared FS)
ts="$(date +%Y%m%dT%H%M%S)"
RACE_DIR="${race_root}/${ts}_$$"
mkdir -p "${RACE_DIR}"
mkdir -p "${RACE_DIR}/slurm"

WINNER_FILE="${RACE_DIR}/winner_jobid"

# submit to all targets
declare -a jobids=()
declare -a bases=()
declare -a meta=()

for t in "${TARGETS[@]}"; do
  part=""
  qos=""
  acct=""

  # extract account: "...@acct"
  t0="$t"
  if [[ "$t0" == *@* ]]; then
    acct="${t0#*@}"
    t0="${t0%@*}"
    if [[ -z "${acct:-}" ]]; then
      echo "ERROR: invalid target '${t}' (empty account after '@')." >&2
      exit 2
    fi
  fi

  # extract qos: "partition:qos"
  if [[ "$t0" == *":"* ]]; then
    IFS=":" read -r part qos <<< "$t0"
    if [[ -z "${qos:-}" ]]; then
      echo "ERROR: invalid target '${t}' (empty qos after ':'). Use 'partition:qos' or omit ':'." >&2
      exit 2
    fi
  else
    part="$t0"
  fi

  if [[ -z "${part:-}" ]]; then
    echo "ERROR: invalid target '${t}' (empty partition)." >&2
    exit 2
  fi

  sbatch_args=(
    --parsable
    --partition="${part}"
    --export=ALL,RACE_DIR="${RACE_DIR}"
    --output="${RACE_DIR}/slurm/%x_%j.out"
    --error="${RACE_DIR}/slurm/%x_%j.err"
  )
  [[ -n "${qos:-}"  ]] && sbatch_args+=( --qos="${qos}" )
  [[ -n "${acct:-}" ]] && sbatch_args+=( --account="${acct}" )

  jid="$(sbatch "${sbatch_args[@]}" "$@")"
  base="${jid%%_*}"

  jobids+=("${jid}")
  bases+=("${base}")
  meta+=("partition=${part} qos=${qos:-<default>} account=${acct:-<inherit>}")

  echo "SUBMITTED job_id=${jid} (base=${base}) ${meta[-1]}"
done

# helper: get top-level state for a base jobid (avoid .batch/.extern/.0)
get_top_state() {
  local b="$1"
  if ! command -v sacct >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  sacct -j "$b" -n -P -o JobID,State 2>/dev/null \
    | awk -F'|' -v B="$b" '$1==B {print $2; exit}' \
    | tr -d ' ' || true
}

# ---- GPU VRAM (used/total) helper ------------------------------------------
get_first_node_for_job() {
  local b="$1"
  local jl batchhost nodelist

  jl="$(scontrol show job "$b" -o 2>/dev/null || true)"
  [[ -z "$jl" ]] && echo "" && return 0

  batchhost="$(sed -n 's/.*BatchHost=\([^ ]*\).*/\1/p' <<<"$jl" | head -n1 || true)"
  if [[ -n "${batchhost:-}" && "$batchhost" != "(null)" && "$batchhost" != "Unknown" && "$batchhost" != "None" ]]; then
    echo "$batchhost"
    return 0
  fi

  nodelist="$(sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' <<<"$jl" | head -n1 || true)"
  if [[ -z "${nodelist:-}" || "$nodelist" == "(null)" || "$nodelist" == "Unknown" || "$nodelist" == "None" ]]; then
    echo ""
    return 0
  fi

  if command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$nodelist" 2>/dev/null | head -n1 || true
  else
    echo "$nodelist"
  fi
}

gpu_vram_block() {
  local b="$1" node="$2"
  local out=""

  [[ -z "${node:-}" ]] && return 0
  command -v srun >/dev/null 2>&1 || return 0

  run_query() {
    local use_overlap="$1"
    local -a cmd=(
      srun
      --jobid="$b"
      --nodes=1
      --ntasks=1
      --nodelist="$node"
      --quiet
    )
    [[ "$use_overlap" -eq 1 ]] && cmd+=( --overlap )
    cmd+=(
      nvidia-smi
      --query-gpu=memory.used,memory.total
      --format=csv,noheader,nounits
    )

    if command -v timeout >/dev/null 2>&1; then
      timeout 5s "${cmd[@]}" 2>/dev/null || true
    else
      "${cmd[@]}" 2>/dev/null || true
    fi
  }

  out="$(run_query 1)"
  [[ -z "$out" ]] && out="$(run_query 0)"

  if [[ -z "$out" ]]; then
    echo "[gpu vram] (unavailable)"
    echo
    return 0
  fi

  echo "[gpu vram] node=${node} (used/total MiB)"
  awk -F',' '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1);
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
      printf("  GPU%d %s/%s MiB\n", NR-1, $1, $2);
    }
  ' <<<"$out"
  echo
}

# Phase A: wait for winner, while showing overview of all submitted jobs
while [[ ! -s "${WINNER_FILE}" ]]; do
  if [[ "$clear_screen" -eq 1 ]]; then
    clear
  fi
  echo "GPU RAM group: ${norm_ram}GB"
  echo "Race directory: ${RACE_DIR}"
  echo "Waiting for winner to start and acquire lock..."
  date
  echo

  echo "[submitted]"
  for i in "${!jobids[@]}"; do
    echo "  ${jobids[$i]} (base=${bases[$i]}) ${meta[$i]}"
  done
  echo

  if command -v squeue >/dev/null 2>&1; then
    echo "[squeue]"
    joblist="$(IFS=','; echo "${bases[*]}")"
    squeue -j "${joblist}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.4D %R" 2>/dev/null || true
    echo
  fi

  if command -v sprio >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
    for b in "${bases[@]}"; do
      st="$(squeue -h -j "$b" -o "%T" 2>/dev/null | head -n1 | tr -d ' ' || true)"
      if [[ "$st" == "PENDING" ]]; then
        echo "[sprio base=${b}]"
        sprio -j "$b" 2>/dev/null || true
        echo
      fi
    done
  fi

  # sacct: show only top-level job lines (no ".batch", ".extern", ".0", ...)
  all_done=1
  if command -v sacct >/dev/null 2>&1; then
    echo "[sacct]"
    joblist="$(IFS=','; echo "${bases[*]}")"
    sacct -j "${joblist}" --format=JobID,State,Elapsed,AllocTRES,ReqMem,MaxRSS,MaxVMSize,ExitCode -P 2>/dev/null \
      | awk -F'|' 'NR==1{print; next} $1 !~ /\./ {print}' \
      | column -t -s'|' || true
    echo

    # detect "no winner possible" to avoid infinite loop
    for b in "${bases[@]}"; do
      st_now="$(get_top_state "$b")"
      if [[ -z "$st_now" ]]; then
        all_done=0
        continue
      fi
      if ! is_done_state "$st_now"; then
        all_done=0
      fi
    done

    if [[ "$all_done" -eq 1 ]]; then
      echo "ERROR: all submitted jobs finished but no winner acquired the lock (winner_jobid not created)." >&2
      echo "Race directory: ${RACE_DIR}" >&2
      exit 3
    fi
  fi

  sleep "$interval"
done

winner="$(cat "${WINNER_FILE}")"
winner_base="${winner%%_*}"

# cancel non-winners (base jobid)
for b in "${bases[@]}"; do
  if [[ "${b}" != "${winner_base}" ]]; then
    scancel "${b}" 2>/dev/null || true
  fi
done

# watch only the winner
jid="${winner}"
base="${winner_base}"
array_id=""
if [[ "$jid" == *"_"* ]]; then
  array_id="${jid#*_}"
fi

# scontrol may lag right after start
for _ in {1..30}; do
  if scontrol show job "$base" -o >/dev/null 2>&1; then break; fi
  sleep 1
done

jobline="$(scontrol show job "$base" -o 2>/dev/null || true)"

stdout_path="$(sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' <<<"$jobline" | head -n1 || true)"
stderr_path="$(sed -n 's/.*StdErr=\([^ ]*\).*/\1/p' <<<"$jobline" | head -n1 || true)"
job_name="$(sed -n 's/.*JobName=\([^ ]*\).*/\1/p' <<<"$jobline" | head -n1 || true)"

# Expand templates: %j/%A jobid, %a array task id, %x job name
expand_path() {
  local path="$1"
  local expanded="$path"
  if [[ -z "$expanded" || "$expanded" == "Unknown" || "$expanded" == "(null)" || "$expanded" == "None" ]]; then
    echo ""
    return
  fi
  expanded="${expanded//%j/$base}"
  expanded="${expanded//%A/$base}"
  if [[ -n "$array_id" ]]; then
    expanded="${expanded//%a/$array_id}"
  fi
  if [[ -n "$job_name" ]]; then
    expanded="${expanded//%x/$job_name}"
  fi
  echo "$expanded"
}

stdout_path="$(expand_path "$stdout_path")"
stderr_path="$(expand_path "$stderr_path")"

last_state=""
while true; do
  if [[ "$clear_screen" -eq 1 ]]; then
    clear
  fi
  echo "GPU RAM group: ${norm_ram}GB"
  echo "WINNER JobID: ${jid} (tracking base: ${base})"
  echo "Race directory: ${RACE_DIR}"
  date
  echo

  pending_state=""
  if command -v squeue >/dev/null 2>&1; then
    echo "[squeue]"
    squeue_out="$(squeue -j "$base" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.4D %R" 2>/dev/null || true)"
    echo "$squeue_out"
    echo
    pending_state="$(squeue -h -j "$base" -o "%T" 2>/dev/null | head -n1 | tr -d ' ' || true)"
  fi

  if [[ "$pending_state" == "PENDING" ]] && command -v sprio >/dev/null 2>&1; then
    echo "[sprio]"
    sprio -j "$base" 2>/dev/null || true
    echo
  fi

  if command -v sacct >/dev/null 2>&1; then
    echo "[sacct]"
    sacct -j "$base" --format=JobID,State,Elapsed,AllocTRES,ReqMem,MaxRSS,MaxVMSize,ExitCode -P 2>/dev/null \
      | awk -F'|' 'NR==1{print; next} $1 !~ /\./ {print}' \
      | column -t -s'|' || true
    echo

    state_now="$(get_top_state "$base")"
    [[ -n "$state_now" ]] && last_state="$state_now"
  fi

  node_now="$(get_first_node_for_job "$base")"
  gpu_vram_block "$base" "$node_now"

  if [[ "$show_tail" -eq 1 ]]; then
    tail_block() {
      local label="$1" path="$2"
      [[ -z "$path" ]] && return 0
      echo "[tail ${label}] ${path}"
      if [[ "$path" == *"%"* ]]; then
        echo "(unresolved path template)"
      elif [[ -f "$path" ]]; then
        tail -n "$lines" "$path" || true
      else
        echo "(not created yet)"
      fi
      echo
    }
    tail_block "stdout" "$stdout_path"
    tail_block "stderr" "$stderr_path"
  fi

  if [[ -n "$last_state" ]] && is_done_state "$last_state"; then
    break
  fi

  sleep "$interval"
done

echo
echo "Final State: ${last_state:-UNKNOWN}"

if [[ "$show_seff" -eq 1 ]] && command -v seff >/dev/null 2>&1; then
  echo
  echo "[seff]"
  seff "$base" || true
fi
