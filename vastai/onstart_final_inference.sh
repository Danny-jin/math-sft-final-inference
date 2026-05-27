#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export DATA_DIRECTORY="${DATA_DIRECTORY:-/workspace}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

mkdir -p /workspace "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
cd /workspace

apt-get update
apt-get install -y --no-install-recommends git git-lfs curl ca-certificates tmux htop nvtop rsync
git lfs install || true

if [ ! -d /workspace/math-sft-final-inference/.git ]; then
  git clone https://github.com/Danny-jin/math-sft-final-inference.git /workspace/math-sft-final-inference
else
  git -C /workspace/math-sft-final-inference pull --ff-only || true
fi

cd /workspace/math-sft-final-inference
bash scripts/setup_vastai_env.sh 2>&1 | tee /workspace/setup_final_repo.log
