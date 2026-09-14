#!/usr/bin/env bash
# Idempotent setup for teacher datagen. Everything lands under $TG_ROOT
# (/dev/shm): rerun after a reboot.
#   1. user-space gcc (conda-forge via micromamba) — the host has no cc and
#      Triton / Inductor JIT-compile at runtime
#   2. uv venv with vLLM 0.28.0 (production teacher-swarm pin) + flashinfer
#      prebuilt kernels, wheels only
#   3. teacher weights at the pinned revision
#   4. corpus D, latest manifest, sha-verified (sync_corpus.py)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
mkdir -p "$TG_ROOT"/{logs,run,cache}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== 1. gcc toolchain"
if [ ! -x "$TG_ROOT/toolchain/bin/gcc" ]; then
  if [ ! -x "$TG_ROOT/bin/micromamba" ]; then
    # Plain binary: the host has no bzip2 for the .tar.bz2 distribution.
    mkdir -p "$TG_ROOT/bin"
    curl -fLs -o "$TG_ROOT/bin/micromamba" \
      https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
    chmod +x "$TG_ROOT/bin/micromamba"
  fi
  MAMBA_ROOT_PREFIX="$TG_ROOT/mamba" "$TG_ROOT/bin/micromamba" create -y \
    -p "$TG_ROOT/toolchain" -c conda-forge gcc gxx "sysroot_linux-64=2.28"
fi
"$TG_ROOT/toolchain/bin/gcc" --version | head -1

echo "== 2. venv"
if ! "$VIRTUAL_ENV/bin/python" -c "import vllm, flashinfer_jit_cache" 2>/dev/null; then
  [ -x "$VIRTUAL_ENV/bin/python" ] || uv venv "$VIRTUAL_ENV" --python 3.12
  uv pip install --no-build --override "$HERE/overrides.txt" \
    "vllm==0.28.0" hf_transfer pyarrow orjson httpx boto3
  uv pip install --no-build "flashinfer-cubin==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl
  uv pip install --no-build "flashinfer-jit-cache==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl/cu130
fi
uv pip install --no-deps -e "$TG_REPO/affine"
# Production teacher echo prefix-cache plugin (vllm.general_plugins entry
# point, pairs with the affine_echo_tail xarg sent by vllm_client._echo_span).
uv pip install --no-deps --reinstall "$TG_REPO/ops/teacher-swarm/echo_cache_plugin"

echo "== 3. teacher weights"
"$VIRTUAL_ENV/bin/hf" download "$TEACHER_REPO" --revision "$TEACHER_REVISION" >/dev/null

echo "== 4. corpus"
"$VIRTUAL_ENV/bin/python" "$HERE/sync_corpus.py"

echo "SETUP_OK"
