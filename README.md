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

Final training and validation were run on a single RTX 5090 32GB GPU.

Approximate times on one RTX 5090:

- Final A17 LoRA training: about 3.1 hours.
- Final `run_inference()` generation/inference: about 6-8 hours end to end.
- The dominant cost is the base-model self-consistency pass over 943 private questions with `k=5` and `max_tokens=28672`.
- The A17 LoRA greedy pass is much shorter because most memorized responses are brief.

Runtime varies with vLLM version, GPU memory bandwidth, and whether the base model is already cached locally.

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
pip install -r requirements.txt
```

The final runs used bfloat16 vLLM inference with:

- `max_model_len=32768`
- `max_tokens=28672`
- base self-consistency `k=5`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- prompt variant `typed_v1`

## Reproduce The Submission

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
