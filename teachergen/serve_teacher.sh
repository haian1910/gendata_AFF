#!/usr/bin/env bash
# Teacher replicas for datagen, same serving knobs as the production teacher
# swarm (ops/teacher-swarm/swarm.toml + bootstrap_pod.sh, affine.toml
# [miner_serving].teacher_*): vLLM 0.28.0, max_model_len 131072, util 0.70,
# chunk 8192, FLASH_ATTN, GDN prefill on triton. One replica per TP group of
# GPUS, ports BASE_PORT, BASE_PORT+1, ...
#
#   teachergen/serve_teacher.sh start|stop|status
#   GPUS=0,1,2,3 TP=1 BASE_PORT=8100 teachergen/serve_teacher.sh start
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TP=${TP:-1}
BASE_PORT=${BASE_PORT:-8100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.70}
BATCHED_TOKENS=${BATCHED_TOKENS:-8192}
# 0.70 keeps production's headroom for the fp32 echo spike, but on a 141 GB
# H200 it leaves only 866 Mamba (GDN) cache blocks, under vLLM's default
# max_num_seqs=1024, and the engine refuses to start. Cap the batch instead
# (gen.py keeps well under this per replica). Serving-only knob.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
RUN="$TG_ROOT/run"
LOGS="$TG_ROOT/logs"
mkdir -p "$RUN" "$LOGS"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_REPLICAS=$(( ${#GPU_LIST[@]} / TP ))

group() {  # i -> comma-joined GPU ids of replica i
  local i=$1
  local g=("${GPU_LIST[@]:$(( i * TP )):$TP}")
  (IFS=','; echo "${g[*]}")
}

start() {
  for (( i = 0; i < N_REPLICAS; i++ )); do
    local port=$(( BASE_PORT + i )) gpus
    gpus=$(group "$i")
    if [ -f "$RUN/replica_$port.pid" ] && kill -0 "$(cat "$RUN/replica_$port.pid")" 2>/dev/null; then
      echo "replica $port already running"; continue
    fi
    echo "launch replica port=$port gpus=$gpus tp=$TP util=$GPU_UTIL"
    local extra=()
    [ -n "$MAX_NUM_SEQS" ] && extra=(--max-num-seqs "$MAX_NUM_SEQS")
    CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 nohup setsid vllm serve "$TEACHER_REPO" \
      --revision "$TEACHER_REVISION" \
      --port "$port" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-num-batched-tokens "$BATCHED_TOKENS" \
      --attention-backend FLASH_ATTN \
      --attention-config.use_trtllm_attention 0 \
      --compilation-config.pass_config.fuse_allreduce_rms false \
      --moe-backend triton \
      --additional-config '{"gdn_prefill_backend": "triton"}' \
      "${extra[@]}" \
      >> "$LOGS/vllm_$port.log" 2>&1 < /dev/null &
    echo $! > "$RUN/replica_$port.pid"
  done
}

stop() {
  for pidf in "$RUN"/replica_*.pid; do
    [ -f "$pidf" ] || continue
    pid=$(cat "$pidf")
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    rm -f "$pidf"
    echo "stopped $(basename "$pidf" .pid)"
  done
}

status() {
  local up=0
  for (( i = 0; i < N_REPLICAS; i++ )); do
    local port=$(( BASE_PORT + i ))
    if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "replica $port: ready"; up=$(( up + 1 ))
    elif [ -f "$RUN/replica_$port.pid" ] && kill -0 "$(cat "$RUN/replica_$port.pid")" 2>/dev/null; then
      echo "replica $port: starting ($(tail -1 "$LOGS/vllm_$port.log" 2>/dev/null | cut -c1-120))"
    else
      echo "replica $port: down"
    fi
  done
  echo "$up/$N_REPLICAS ready"
  [ "$up" -eq "$N_REPLICAS" ]
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
