# Source me: `source teachergen/env.sh`
# Everything bulky lives on /dev/shm (root disk is 20 GB). /dev/shm is RAM:
# a reboot wipes the venv, weights and corpus — rerun teachergen/setup.sh.
TG_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TG_REPO
export TG_ROOT=/dev/shm/affine-teachergen

# Frozen teacher, pinned to the HF commit downloaded by setup.sh.
export TEACHER_REPO="Qwen/Qwen3.8-27B"
export TEACHER_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

export VIRTUAL_ENV="$TG_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$TG_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TG_ROOT/uv-python"
export HF_HOME="$TG_ROOT/hf"
export HF_XET_HIGH_PERFORMANCE=1
export XDG_CACHE_HOME="$TG_ROOT/cache"
export VLLM_CACHE_ROOT="$TG_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$TG_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TG_ROOT/cache/inductor"
# The host has no compiler; Triton reads CC, Inductor reads CXX.
export CC="$TG_ROOT/toolchain/bin/gcc"
export CXX="$TG_ROOT/toolchain/bin/g++"
# vLLM 0.28.0 ships torch cu130; an R570 driver (CUDA 12.8) refuses it.
# Datacenter GPUs run it through the CUDA 13 forward-compat libcuda.
CUDA_COMPAT=/usr/local/cuda-13.0/compat
if [ -d "$CUDA_COMPAT" ] && [ "$(nvidia-smi --query-gpu=driver_version \
    --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)" -lt 580 ] 2>/dev/null; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# HF_TOKEN
set -a
# shellcheck disable=SC1091
source "$TG_REPO/.env"
set +a

# Production teacher env (ops/teacher-swarm/bootstrap_pod.sh): FlashInfer JIT
# paths off everywhere.
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
