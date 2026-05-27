#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer host driver libcuda over CUDA compat stubs on RTX 5090/Blackwell.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

SMOKE_ROWS="${SMOKE_ROWS:-2}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-1024}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
SMOKE_WORK_DIR="${SMOKE_WORK_DIR:-outputs/smoke/work}"
SMOKE_PRIVATE_JSONL="${SMOKE_PRIVATE_JSONL:-outputs/smoke/private_smoke_${SMOKE_ROWS}.jsonl}"
SMOKE_OUTPUT_CSV="${SMOKE_OUTPUT_CSV:-outputs/smoke/submission_smoke.csv}"

echo "[smoke] repo: $(pwd)"

if [ -z "${PYTHON:-}" ] && [ -x /workspace/venv/bin/python ]; then
  PYTHON=/workspace/venv/bin/python
fi
PYTHON="${PYTHON:-python}"

echo "[smoke] python: $("$PYTHON" -V 2>&1)"
nvidia-smi || true

test -s data/private.jsonl
test -s artifacts/private_all943_codex_A16style_accepted_reference.jsonl

if [ ! -s models/a17_lora/adapter_config.json ] || [ ! -s models/a17_lora/adapter_model.safetensors ]; then
  echo "[smoke] LoRA adapter missing; downloading from HuggingFace..."
  "$PYTHON" - <<'PY'
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
fi

mkdir -p "$(dirname "$SMOKE_PRIVATE_JSONL")"
"$PYTHON" - <<PY
from pathlib import Path
src = Path("data/private.jsonl")
dst = Path("$SMOKE_PRIVATE_JSONL")
n = int("$SMOKE_ROWS")
rows = src.read_text().splitlines()[:n]
dst.write_text("\\n".join(rows) + "\\n")
print(f"[smoke] wrote {dst} rows={len(rows)}")
PY

"$PYTHON" run_inference.py \
  --private-jsonl "$SMOKE_PRIVATE_JSONL" \
  --output-csv "$SMOKE_OUTPUT_CSV" \
  --work-dir "$SMOKE_WORK_DIR" \
  --base-sc-k 1 \
  --max-tokens "$SMOKE_MAX_TOKENS" \
  --max-model-len "$SMOKE_MAX_MODEL_LEN" \
  --max-num-seqs 2

echo "[smoke] report:"
cat "${SMOKE_OUTPUT_CSV%.csv}.report.json"
echo
echo "[smoke] csv preview:"
head -5 "$SMOKE_OUTPUT_CSV"
echo
echo "[smoke] OK"
