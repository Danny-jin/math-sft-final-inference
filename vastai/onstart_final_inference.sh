#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export DATA_DIRECTORY="${DATA_DIRECTORY:-/workspace}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p /workspace "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
cd /workspace

apt-get update
apt-get install -y --no-install-recommends git git-lfs curl ca-certificates tmux htop nvtop rsync
git lfs install || true

python3 -m pip install --upgrade pip setuptools wheel

if [ ! -d /workspace/math-sft-final-inference/.git ]; then
  git clone https://github.com/Danny-jin/math-sft-final-inference.git /workspace/math-sft-final-inference
else
  git -C /workspace/math-sft-final-inference pull --ff-only || true
fi

cd /workspace/math-sft-final-inference
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade "huggingface_hub[cli]" hf_transfer

python3 - <<'PY'
import importlib
mods = ["torch", "transformers", "vllm", "peft", "huggingface_hub"]
for name in mods:
    try:
        m = importlib.import_module(name)
        print(f"{name}={getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_FAILED: {exc}")
PY

hf download Danny-jin/math-sft-a17-lora --local-dir models/a17_lora

cat > /workspace/READY_FINAL_INFERENCE.txt <<'EOF'
Final inference environment is prepared.

Run from SSH:

cd /workspace/math-sft-final-inference
python run_inference.py

Optional local base cache:

hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir /workspace/models/Qwen3-4B-Thinking-2507
export QWEN_BASE_MODEL=/workspace/models/Qwen3-4B-Thinking-2507
python run_inference.py
EOF

echo "Ready: /workspace/READY_FINAL_INFERENCE.txt"
