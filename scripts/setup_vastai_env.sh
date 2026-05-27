#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/workspace/venv}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

echo "[setup] repo: $REPO_DIR"
echo "[setup] venv: $VENV_DIR"
echo "[setup] creating/updating virtualenv"
python3 -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO_DIR/requirements.txt"

echo "[setup] python package check"
python - <<'PY'
import importlib
mods = ["torch", "transformers", "vllm", "peft", "huggingface_hub"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
PY

echo "[setup] downloading A17 LoRA adapter"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Danny-jin/math-sft-a17-lora",
    local_dir="models/a17_lora",
    allow_patterns=[
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ],
)
PY

cat > /workspace/READY_FINAL_INFERENCE.txt <<EOF
Final inference environment is prepared.

Activate the environment:

source $VENV_DIR/bin/activate

Smoke test:

cd $REPO_DIR
bash scripts/smoke_test.sh

Final run:

cd $REPO_DIR
python run_inference.py
EOF

echo "[setup] ready: /workspace/READY_FINAL_INFERENCE.txt"
