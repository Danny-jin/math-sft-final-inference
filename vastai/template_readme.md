Final inference template for `Danny-jin/math-sft-final-inference`.

This template is intended for one NVIDIA RTX 5090 32GB instance with CUDA
12.8+ support and at least 100GB disk. It prepares:

- the final GitHub repo under `/workspace/math-sft-final-inference`
- Python dependencies from `requirements.txt`
- HuggingFace CLI / `hf_transfer`
- the public A17 LoRA adapter under `models/a17_lora`
- cache directories under `/workspace/.cache`

It does not automatically run the full final inference job, because the full
pipeline takes several hours and should be started intentionally after SSH.

After the instance starts:

```bash
cd /workspace/math-sft-final-inference
python run_inference.py
```

The default base model is:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Optional: pre-download the base model to local disk and point `QWEN_BASE_MODEL`
to that path before running inference.
