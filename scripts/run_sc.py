"""Self-consistency inference: generate k samples per question into JSONL.

Output schema (one row per question):
  {"id": <id>, "type": "mcq"|"ff", "samples": [<text>, ...]}

Usage:
    # final private SC k=5
    python scripts/run_sc.py \
        --input data/private.jsonl \
        --output outputs/private_sc5.jsonl \
        --n 5 --temperature 0.6 --top_p 0.95 --top_k 20 \
        --max_tokens 28672 --prompt_variant typed_v1 --hiprec
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (
    build_prompt_split,
    build_prompt_split_contract_v2,
    build_prompt_split_typed_v1,
    is_mc,
)


def build_prompt(
    ex: dict,
    tokenizer,
    hiprec: bool = False,
    prompt_variant: str = "default",
) -> str:
    """Build a chat-templated prompt."""
    if prompt_variant == "default":
        return build_prompt_split(ex, tokenizer, hiprec=hiprec)
    if prompt_variant == "contract_v2":
        return build_prompt_split_contract_v2(ex, tokenizer, hiprec=hiprec)
    if prompt_variant == "typed_v1":
        return build_prompt_split_typed_v1(ex, tokenizer, hiprec=hiprec)
    raise ValueError(f"unknown prompt_variant: {prompt_variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,
                    help="JSONL with rows {id, question, options?, answer?}")
    ap.add_argument("--output", required=True,
                    help="Output JSONL with k samples per row.")
    ap.add_argument("--model",  default="Qwen/Qwen3-4B-Thinking-2507",
                    help="Local base model directory or HuggingFace model id.")
    ap.add_argument("--lora-path", default=None,
                    help="Optional LoRA adapter directory.")
    ap.add_argument("--n",        type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p",    type=float, default=0.95)
    ap.add_argument("--top_k",    type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=20480)
    ap.add_argument("--hiprec", action="store_true",
                    help="Inject the production high-precision formatting hint for FF prompts.")
    ap.add_argument("--prompt_variant",
                    choices=["default", "contract_v2", "typed_v1"],
                    default="typed_v1",
                    help="Prompt template to use. typed_v1 adds bucket-specific answer-format rules.")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",
        trust_remote_code=True,
    )
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64   # safe upper bound

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    sp = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed,
    )

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"loaded {len(rows)} rows from {args.input}")

    prompts = [
        build_prompt(
            r,
            tok,
            hiprec=args.hiprec,
            prompt_variant=args.prompt_variant,
        )
        for r in rows
    ]

    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("v1", 1, args.lora_path)
        outputs = llm.generate(prompts, sp, lora_request=lora)
    else:
        outputs = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_trunc = 0
    n_total = 0
    with open(args.output, "w") as f:
        for r, out in zip(rows, outputs):
            samples = [c.text for c in out.outputs]
            for c in out.outputs:
                n_total += 1
                if c.finish_reason == "length":
                    n_trunc += 1
            qtype = "mcq" if is_mc(r) else "ff"
            row_out = {"id": r["id"], "type": qtype, "samples": samples}
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    print(f"truncated samples: {n_trunc}/{n_total} = {n_trunc/max(n_total,1):.1%}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
