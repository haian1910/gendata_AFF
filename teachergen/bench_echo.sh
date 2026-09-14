#!/usr/bin/env bash
# A/B throughput: same turn_ids, cold prefix cache per mode, fixed window.
#   teachergen/bench_echo.sh IDS OUT_ROOT [MODES...]   (default: thought all)
# Needs replicas started with VLLM_SERVER_DEV_MODE=1 (POST /reset_prefix_cache).
# Env: WINDOW_S (600), WARMUP_S (120), NUM_SHARDS (8), BASE_PORT (8100).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ids=$1 root=$2; shift 2
modes=("$@"); [ ${#modes[@]} -eq 0 ] && modes=(thought all)
WINDOW_S=${WINDOW_S:-600}
WARMUP_S=${WARMUP_S:-120}
NUM_SHARDS=${NUM_SHARDS:-8}
BASE_PORT=${BASE_PORT:-8100}

count() {  # turns written under $1 (data files only)
  local n=0 f
  for f in "$1"/shard_*.jsonl; do
    case "$f" in *.errors.jsonl) continue ;; esac
    [ -f "$f" ] && n=$(( n + $(wc -l < "$f") ))
  done
  echo $n
}

drained() {
  local p
  for (( p = BASE_PORT; p < BASE_PORT + NUM_SHARDS; p++ )); do
    curl -s -m 5 "http://127.0.0.1:$p/metrics" | awk '
      /^vllm:num_requests_(running|waiting)\{/ {s += $2} END {exit s > 0}' || return 1
  done
}

for mode in "${modes[@]}"; do
  out="$root/bench_$mode"
  rm -rf "$out"
  for (( p = BASE_PORT; p < BASE_PORT + NUM_SHARDS; p++ )); do
    curl -s -o /dev/null -X POST "http://127.0.0.1:$p/reset_prefix_cache"
  done
  "$HERE/run_shards.sh" start "$ids" "$out" --echo "$mode" >/dev/null
  t0=$(date +%s); n_warm=0 t_warm=0
  while :; do
    sleep 30
    t=$(( $(date +%s) - t0 )); n=$(count "$out")
    echo "[$mode] t=${t}s turns=$n"
    if [ "$t_warm" -eq 0 ] && [ "$t" -ge "$WARMUP_S" ]; then n_warm=$n t_warm=$t; fi
    [ "$t" -ge "$WINDOW_S" ] && break
  done
  "$HERE/run_shards.sh" stop "$out"
  rate=$(python3 -c "print(f'{($n - $n_warm) / ($t - $t_warm):.3f}')")
  errs=$(cat "$out"/shard_*.errors.jsonl 2>/dev/null | grep -vc turn_id_not_in_corpus)
  tb=$(grep -l Traceback "$out"/shard_*.log 2>/dev/null | wc -l)
  echo "RESULT mode=$mode steady_turns_per_s=$rate window=${t_warm}-${t}s turns=$n errors=$errs tracebacks=$tb"
  until drained; do sleep 5; done
done
