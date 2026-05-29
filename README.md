# Kaggle Math SFT Final Inference

This repository contains the final reproducible inference pipeline for the
Kaggle math benchmark submission.

The required single entry point is `run_inference()` in `run_inference.py`.
Calling it runs the full pipeline and writes the final submission CSV. No
external model, API, calculator, or code interpreter is called at inference
time.

## Pipeline Summary

`run_inference()` performs:

1. Run `Qwen/Qwen3-4B-Thinking-2507` with self-consistency `k=5`.
2. Vote the 5 base generations into one baseline response per problem.
3. Run the same base model with the final A17 LoRA adapter.
4. Compare the A17 generated final boxed answer with the stored distillation
   target for that problem.
5. Use the A17 response only when its boxed answer exactly matches the stored
   target; otherwise keep the base self-consistency response.
6. Apply deterministic global answer-format cleanup for parser-hostile wrappers
   such as dollar signs, `\text{}` unit wrappers, `\dfrac`, and cases commas.
7. Write the final CSV and a JSON report.

## Reproduce The Submission

The tested environment is one NVIDIA RTX 5090 32GB GPU with CUDA 12.8+ and at
least 100GB disk. Approximate final inference time is 6-8 hours on this GPU.
Verification does not require rerunning training; the final A17 LoRA adapter is
loaded from HuggingFace Hub.

Recommended Vast.ai template for a fresh verification instance:

[math-sft-final-inference-rtx5090](https://cloud.vast.ai?ref_id=506182&template_id=1d9d8aa5309969cfd0c907849664d158)

After the template startup finishes and `/workspace/READY_FINAL_INFERENCE.txt`
exists:

```bash
cd /workspace/math-sft-final-inference
source /workspace/venv/bin/activate

bash scripts/smoke_test.sh
bash scripts/run_final_background.sh
tail -f outputs/final_run.log
```

From a manually prepared machine with writable `/workspace`:

```bash
git clone https://github.com/Danny-jin/math-sft-final-inference.git
cd math-sft-final-inference

bash scripts/setup_vastai_env.sh
source /workspace/venv/bin/activate

bash scripts/smoke_test.sh
bash scripts/run_final_background.sh
tail -f outputs/final_run.log
```

On a machine without `/workspace`, create an equivalent Python environment,
install `requirements.txt`, place or download the A17 adapter, and call
`python run_inference.py` from the repository root, or run
`bash scripts/run_final_background.sh` if `tmux`/`nohup` is available.

The smoke test runs two private rows with short generations. It is only a cheap
end-to-end environment check, not an accuracy check.

The final command writes:

```text
outputs/submission_run_inference_A17_target_gate.csv
outputs/submission_run_inference_A17_target_gate.report.json
```

For official reproduction, run without `--reuse-existing` so all intermediate
generations are freshly produced.

The background wrapper launches the same `run_inference.py` pipeline in a
detached `tmux` session named `final_run` when `tmux` is installed. If `tmux` is
not available, it falls back to `nohup`. It writes progress to:

```text
outputs/final_run.log
```

Useful background commands:

```bash
# Start the final run and safely disconnect SSH.
bash scripts/run_final_background.sh

# Watch progress.
tail -f outputs/final_run.log

# Reattach if tmux is used.
tmux attach -t final_run

# Pass CLI overrides to run_inference.py.
RUN_INFERENCE_ARGS="--output-csv outputs/final_submission.csv" \
  bash scripts/run_final_background.sh
```

Python entry-point usage:

```python
from run_inference import run_inference

csv_path = run_inference()
print(csv_path)
```

Useful CLI override example:

```bash
python run_inference.py \
  --private-jsonl /path/to/private.jsonl \
  --base-model Qwen/Qwen3-4B-Thinking-2507 \
  --a17-lora /path/to/a17_lora \
  --target-ref artifacts/private_all943_codex_A16style_accepted_reference.jsonl \
  --output-csv outputs/final_submission.csv
```

## Model Weights

The base model is loaded from HuggingFace:

```text
Qwen/Qwen3-4B-Thinking-2507
```

The final LoRA adapter is loaded from HuggingFace:

```text
Danny-jin/math-sft-a17-lora
```

By default, `run_inference.py` downloads the adapter into:

```text
models/a17_lora/
```

Expected adapter files include:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Manual adapter download:

```bash
hf download Danny-jin/math-sft-a17-lora --local-dir models/a17_lora
```

If the base model is already downloaded locally:

```bash
export QWEN_BASE_MODEL=/path/to/Qwen3-4B-Thinking-2507
```

If the LoRA repo or local path changes:

```bash
export A17_LORA_HF_REPO=<HF_USERNAME>/<A17_LORA_REPO>
python run_inference.py --a17-lora /path/to/a17_lora
```

## Inputs And Outputs

Default inputs:

```text
data/private.jsonl
artifacts/private_all943_codex_A16style_accepted_reference.jsonl
models/a17_lora/
```

Default outputs:

```text
outputs/submission_run_inference_A17_target_gate.csv
outputs/submission_run_inference_A17_target_gate.report.json
outputs/run_inference_A17_pipeline/
```

The stored target file is used only as a deterministic gate for selecting among
answers generated by the designated model plus the submitted LoRA adapter.

## Runtime And Hyperparameters

Final runs used:

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX 5090 32GB |
| Python | 3.12 |
| CUDA | 12.8+ |
| Inference engine | vLLM with LoRA support |
| dtype | bfloat16 |
| Attention backend | FlashInfer |
| Base self-consistency | `k=5` |
| Base sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| A17 LoRA sampling | greedy |
| Prompt variant | `typed_v1` |
| `max_model_len` | `32768` |
| `max_tokens` | `28672` |

Approximate RTX 5090 runtime:

| Stage | Time |
| --- | --- |
| Final A17 LoRA training | about 3.1 hours |
| Final `run_inference()` | about 6-8 hours |

Runtime depends on model cache state, vLLM version, and GPU memory bandwidth.

## Vast.ai Template Details

The submitted setup was tested with this Vast.ai template:

- Template name: `math-sft-final-inference-rtx5090`
- Template ID: `437555`
- Template hash: `1d9d8aa5309969cfd0c907849664d158`
- Template link: [cloud.vast.ai template](https://cloud.vast.ai?ref_id=506182&template_id=1d9d8aa5309969cfd0c907849664d158)
- Image: `vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-04-15`
- Search filter: single RTX 5090, CUDA `>=12.8`, disk `>=100GB`

The template startup script clones this repo, creates `/workspace/venv`, installs
dependencies, downloads the A17 adapter, and writes:

```text
/workspace/READY_FINAL_INFERENCE.txt
```

On RTX 5090 / Blackwell hosts, the setup also makes Python prefer the host
driver library path `/usr/lib/x86_64-linux-gnu` over CUDA compat stubs to avoid
`cudaGetDeviceCount` error 804 on some Vast.ai images.

To keep the final run alive:

```bash
cd /workspace/math-sft-final-inference
source /workspace/venv/bin/activate
bash scripts/run_final_background.sh
tail -f outputs/final_run.log
```

If using the `tmux` backend, reconnect with:

```bash
tmux attach -t final_run
```

## File Map

- `run_inference.py`: required single entry point and CLI.
- `scripts/setup_vastai_env.sh`: tested dependency setup for the Vast.ai image.
- `scripts/smoke_test.sh`: cheap two-row end-to-end test.
- `scripts/run_final_background.sh`: detached final run wrapper using `tmux` or `nohup`.
- `scripts/run_sc.py`: base self-consistency generation.
- `scripts/run_inference_lora_hiprec.py`: A17 LoRA generation.
- `scripts/vote_eval.py`: self-consistency voting.
- `scripts/answer_normalization.py`: boxed-answer extraction and normalization.
- `scripts/prompts.py`: final prompt templates.
- `judger/`: helper code used by the voting and normalization path.
- `data/private.jsonl`: default private input file.
- `artifacts/private_all943_codex_A16style_accepted_reference.jsonl`: stored
  distillation targets for deterministic A17 selection.
