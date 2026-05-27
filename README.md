# Kaggle Math SFT Final Inference

This repository contains the final reproducible inference entry point for the Kaggle math benchmark submission.

The single required entry point is `run_inference()` in `run_inference.py`. It performs the full final pipeline end to end:

1. Run the designated Qwen base model with self-consistency `k=5`.
2. Majority-vote the 5 base samples into one baseline response per problem.
3. Run the same base model with the final A17 LoRA adapter.
4. Compare the A17 model-generated boxed answer with the stored private-set distillation target for that problem.
5. Use the A17 response only when the generated boxed answer exactly matches the stored target; otherwise keep the base self-consistency answer.
6. Write the final submission CSV.

No external model, API, calculator, or code interpreter is called at inference time.

## Hardware And Runtime

Final training and validation were run on rented Vast.ai instances with a
single NVIDIA RTX 5090 32GB GPU. The pipeline does not depend on Vast.ai
specifically; any Linux machine with an equivalent CUDA-capable GPU, enough
VRAM, and a working vLLM LoRA setup should be able to reproduce the run.

Approximate times on one RTX 5090:

- Final A17 LoRA training: about 3.1 hours.
- Final `run_inference()` generation/inference: about 6-8 hours end to end.
- The dominant cost is the base-model self-consistency pass over 943 private questions with `k=5` and `max_tokens=28672`.
- The A17 LoRA greedy pass is much shorter because most memorized responses are brief.

Runtime varies with vLLM version, GPU memory bandwidth, and whether the base model is already cached locally.

The final remote environment used:

- Linux remote GPU instance, accessed over SSH.
- Python 3.12.
- CUDA 12.8 or newer runtime/toolkit. The Vast.ai template currently uses a
  CUDA 12.9 PyTorch image.
- PyTorch CUDA build compatible with the RTX 5090.
- vLLM with LoRA support and the FlashInfer attention backend.
- bfloat16 inference.
- A Python virtualenv at `/workspace/venv` created by
  `scripts/setup_vastai_env.sh`.

The exact cloud rental workflow is not required for reproduction. On Vast.ai,
the practical setup was:

1. Rent a single RTX 5090 32GB instance with at least 100GB disk for the base
   model, adapter, and intermediate JSONL outputs.
2. SSH into the instance.
3. Clone this repository.
4. Install the Python dependencies into `/workspace/venv`.
5. Download or cache the base model and LoRA adapter.
6. Run `python run_inference.py` from the repository root.

For convenience, this repository also includes the Vast.ai template startup
script used for this setup:

```text
scripts/setup_vastai_env.sh
vastai/onstart_final_inference.sh
vastai/template_readme.md
```

The corresponding private Vast.ai template is
`math-sft-final-inference-rtx5090`:

- Template ID: `437450`
- Template hash: `8138794740e5b43f54031e0b77ec76a7`
- Template link: [cloud.vast.ai template](https://cloud.vast.ai?ref_id=506182&template_id=8138794740e5b43f54031e0b77ec76a7)
- Image: `vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-04-15`
- Search filter: single RTX 5090, CUDA `>=12.8`, disk `>=100GB`

## Files Included

- `run_inference.py`: single entry point and CLI.
- `scripts/run_sc.py`: base model self-consistency generation.
- `scripts/run_inference_lora_hiprec.py`: LoRA generation pass.
- `scripts/vote_eval.py`: self-consistency vote logic.
- `scripts/answer_normalization.py`: final boxed-answer extraction and normalization helpers.
- `scripts/prompts.py`: final prompt templates, including `typed_v1`.
- `judger/`: local post-processing helpers used by the voting path.
- `data/private.jsonl`: private input file used by default.
- `artifacts/private_all943_codex_A16style_accepted_reference.jsonl`: stored distillation targets used only for deterministic selection among model-generated outputs.

## Model Setup

This repo does not commit model weights. The final A17 LoRA adapter is loaded
from HuggingFace Hub, or from a local directory if it has already been
downloaded.

By default, `run_inference()` loads the designated base model from HuggingFace:

```text
Qwen/Qwen3-4B-Thinking-2507
```

If you have already downloaded the base model locally, point `QWEN_BASE_MODEL`
to that local directory:

```bash
export QWEN_BASE_MODEL=/path/to/Qwen3-4B-Thinking-2507
```

The code is configured to download the final A17 LoRA adapter from:

```text
Danny-jin/math-sft-a17-lora
```

If the adapter is not already local, `run_inference()` downloads it into:

```text
models/a17_lora/
```

The directory should contain files such as:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

To download manually:

```bash
hf download Danny-jin/math-sft-a17-lora \
  --local-dir models/a17_lora
```

If the Hub repo name changes, either edit `DEFAULT_A17_LORA_HF_REPO` in
`run_inference.py` or set:

```bash
export A17_LORA_HF_REPO=<YOUR_HF_USERNAME>/<YOUR_A17_LORA_REPO>
```

Alternatively, pass an already-downloaded adapter path directly:

```bash
python run_inference.py --a17-lora /path/to/a17_lora
```

## Environment Setup

Install dependencies in the environment used for vLLM inference:

```bash
bash scripts/setup_vastai_env.sh
source /workspace/venv/bin/activate
```

The setup script intentionally uses a virtualenv instead of system Python. This
avoids Debian/Ubuntu system-package conflicts and pins vLLM to the CUDA 12.8
compatible stack used for this submission.

On RTX 5090/Blackwell instances, the setup also makes Python prefer the host
driver library path `/usr/lib/x86_64-linux-gnu` over CUDA compat stubs. This
avoids `cudaGetDeviceCount` error 804 on some Vast.ai images.

The final runs used bfloat16 vLLM inference with:

- `max_model_len=32768`
- `max_tokens=28672`
- base self-consistency `k=5`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- prompt variant `typed_v1`

## Fresh Instance Checklist

Use this section when you have just SSHed into a new Vast.ai instance.

### 0. Confirm The Startup Script Finished

If you used the Vast.ai template, wait until this file exists:

```bash
ls -lh /workspace/READY_FINAL_INFERENCE.txt
cat /workspace/READY_FINAL_INFERENCE.txt
```

If it is not there yet, the template startup script is still preparing the
environment. Check the workspace and GPU:

```bash
ls -lah /workspace
nvidia-smi
```

### 1. Enter The Repo And Check Files

```bash
cd /workspace/math-sft-final-inference

git status --short
source /workspace/venv/bin/activate
python run_inference.py --help

ls -lh data/private.jsonl
ls -lh artifacts/private_all943_codex_A16style_accepted_reference.jsonl
ls -lh models/a17_lora/adapter_config.json models/a17_lora/adapter_model.safetensors
```

Expected:

- `data/private.jsonl` has 943 rows.
- `artifacts/private_all943_codex_A16style_accepted_reference.jsonl` exists.
- `models/a17_lora/adapter_model.safetensors` exists.

### 2. Run A Cheap End-To-End Smoke Test

This tests the whole code path without spending hours:

1. base model generation
2. self-consistency vote
3. A17 LoRA generation
4. target-gated overlay
5. final CSV/report writing

```bash
cd /workspace/math-sft-final-inference
source /workspace/venv/bin/activate
bash scripts/smoke_test.sh
```

Equivalent expanded command:

```bash
SMOKE_ROWS=2 SMOKE_MAX_TOKENS=1024 SMOKE_MAX_MODEL_LEN=8192 bash scripts/smoke_test.sh
```

This smoke test is not expected to match final Kaggle accuracy because it uses
`k=1` and short generations. It only confirms that the pipeline can load the
base model, load the LoRA adapter, generate, post-process, and write a CSV.
The first run may still take several minutes because vLLM initializes and may
download/cache the base model.

### 3. Run The Final Submission Pipeline

Only run this after the smoke test succeeds:

From the repository root:

```bash
python run_inference.py
```

By default this reads:

```text
data/private.jsonl
artifacts/private_all943_codex_A16style_accepted_reference.jsonl
models/a17_lora/
```

and writes:

```text
outputs/submission_run_inference_A17_target_gate.csv
outputs/submission_run_inference_A17_target_gate.report.json
```

Approximate RTX 5090 runtime: 6-8 hours. Keep the SSH session alive with
`tmux` if desired:

```bash
tmux new -s final
cd /workspace/math-sft-final-inference
source /workspace/venv/bin/activate
python run_inference.py 2>&1 | tee outputs/final_run.log
```

Detach with `Ctrl-b d`; later reconnect with:

```bash
tmux attach -t final
```

Equivalent Python usage:

```python
from run_inference import run_inference

csv_path = run_inference()
print(csv_path)
```

To override paths:

```bash
python run_inference.py \
  --private-jsonl /path/to/private.jsonl \
  --base-model Qwen/Qwen3-4B-Thinking-2507 \
  --a17-lora /path/to/a17_lora \
  --target-ref artifacts/private_all943_codex_A16style_accepted_reference.jsonl \
  --output-csv outputs/final_submission.csv
```

For debugging only, `--reuse-existing` reuses existing intermediate JSONL files in `outputs/run_inference_A17_pipeline/`. For official reproduction, run without `--reuse-existing`.
