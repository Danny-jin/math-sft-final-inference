#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

mkdir -p outputs "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

if [ -e /workspace/SETUP_FINAL_INFERENCE_RUNNING ] || pgrep -f "scripts/setup_vastai_env.sh" >/dev/null 2>&1; then
  echo "[background] setup is still running; waiting for /workspace/READY_FINAL_INFERENCE.txt ..."
  for _ in $(seq 1 180); do
    if [ -s /workspace/READY_FINAL_INFERENCE.txt ]; then
      break
    fi
    if ! pgrep -f "scripts/setup_vastai_env.sh" >/dev/null 2>&1 && [ ! -e /workspace/SETUP_FINAL_INFERENCE_RUNNING ]; then
      break
    fi
    sleep 10
  done
fi

if [ -x /workspace/venv/bin/python ]; then
  PYTHON="${PYTHON:-/workspace/venv/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import huggingface_hub
import transformers
import vllm
import peft
PY
then
  echo "[background] runtime dependencies are missing or incomplete; running setup_vastai_env.sh ..."
  bash scripts/setup_vastai_env.sh
  if [ -x /workspace/venv/bin/python ]; then
    PYTHON=/workspace/venv/bin/python
  fi
fi

SESSION_NAME="${SESSION_NAME:-final_run}"
LOG_FILE="${LOG_FILE:-outputs/final_run.log}"
PID_FILE="${PID_FILE:-outputs/final_run.pid}"
RUN_SCRIPT="${RUN_SCRIPT:-outputs/final_run_command.sh}"
RUN_INFERENCE_ARGS="${RUN_INFERENCE_ARGS:-}"

if [ -s "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[background] run already active: pid=$OLD_PID"
    echo "[background] log: $LOG_FILE"
    exit 0
  fi
fi

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
echo \$\$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
echo "[background] started at \$(date)"
echo "[background] repo: $REPO_DIR"
echo "[background] python: $PYTHON"
echo "[background] args: $RUN_INFERENCE_ARGS"
"$PYTHON" run_inference.py $RUN_INFERENCE_ARGS 2>&1 | tee "$LOG_FILE"
echo "[background] finished at \$(date)"
EOF
chmod +x "$RUN_SCRIPT"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
    echo "[background] tmux session already exists: $SESSION_NAME"
  else
    tmux new-session -d -s "$SESSION_NAME" "$RUN_SCRIPT"
    echo "[background] launched in tmux session: $SESSION_NAME"
  fi
  echo "[background] attach: tmux attach -t $SESSION_NAME"
else
  nohup "$RUN_SCRIPT" >/dev/null 2>&1 &
  echo "[background] tmux not found; launched with nohup"
fi

echo "[background] log: $LOG_FILE"
echo "[background] follow: tail -f $LOG_FILE"
