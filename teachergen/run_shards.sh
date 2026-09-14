#!/usr/bin/env bash
# Data-parallel datagen: one gen.py process per replica, shard i -> port
# BASE_PORT+i, so every request of a turn (n=k sample + its echoes) lands on
# the replica that holds the turn's prefix KV.
#
#   teachergen/run_shards.sh start IDS OUT_DIR [extra gen.py args]
#   teachergen/run_shards.sh status OUT_DIR
#   teachergen/run_shards.sh stop OUT_DIR
#
# Env: NUM_SHARDS (8), BASE_PORT (8100), LIMIT (0 = all), CONCURRENCY,
# PER_REPLICA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

NUM_SHARDS=${NUM_SHARDS:-8}
BASE_PORT=${BASE_PORT:-8100}
LIMIT=${LIMIT:-0}
CONCURRENCY=${CONCURRENCY:-192}
PER_REPLICA=${PER_REPLICA:-128}

cmd=${1:-}; shift || true
case "$cmd" in
  start)
    ids=$1 out=$2; shift 2
    mkdir -p "$out"
    for (( i = 0; i < NUM_SHARDS; i++ )); do
      pidf="$out/shard_$i.pid"
      if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "shard $i already running"; continue
      fi
      # Offline: concurrent first gen_prompt calls race past get_tokenizer's
      # lru_cache and each would hit the hub (~90 unauthenticated calls/shard).
      HF_HUB_OFFLINE=1 nohup setsid python "$HERE/gen.py" --turn-ids "$ids" \
        --out "$out/shard_$i.jsonl" \
        --urls "http://127.0.0.1:$(( BASE_PORT + i ))/v1" \
        --shard "$i" --num-shards "$NUM_SHARDS" --limit "$LIMIT" \
        --batch-n --concurrency "$CONCURRENCY" --per-replica "$PER_REPLICA" \
        "$@" >> "$out/shard_$i.log" 2>&1 < /dev/null &
      echo $! > "$pidf"
      echo "shard $i -> port $(( BASE_PORT + i )) pid $!"
    done
    ;;
  status)
    out=$1
    for log in "$out"/shard_*.log; do
      echo "$(basename "$log"): $(grep -E 'turns \(|done:|Error|Traceback' "$log" | tail -1)"
    done
    written=0
    for f in "$out"/shard_*.jsonl; do
      case "$f" in *.errors.jsonl) continue ;; esac
      written=$(( written + $(wc -l < "$f") ))
    done
    echo "turns written: $written"
    ;;
  stop)
    out=$1
    for pidf in "$out"/shard_*.pid; do
      [ -f "$pidf" ] || continue
      kill -- "-$(cat "$pidf")" 2>/dev/null || true
      rm -f "$pidf"
    done
    ;;
  *) echo "usage: $0 start IDS OUT_DIR [args] | status OUT_DIR | stop OUT_DIR" >&2; exit 2 ;;
esac
