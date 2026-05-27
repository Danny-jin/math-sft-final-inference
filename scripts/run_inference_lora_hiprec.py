"""Run greedy inference with an optional LoRA adapter and hiprec prompt.

Usage:
    python3 scripts/run_inference_lora_hiprec.py \
        --input data/private.jsonl \
        --output outputs/private_A17_lora_greedy_28k_typed_v1.jsonl \
        --base Qwen/Qwen3-4B-Thinking-2507 \
        --lora models/a17_lora \
        --max_tokens 28672 \
        --prompt_variant typed_v1
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (
    build_prompt_split,
    build_prompt_split_contract_v2,
    build_prompt_split_typed_v1,
)


def build_prompt(ex: dict, tokenizer, prompt_variant: str) -> str:
    """Hiprec inference = build_prompt_split with hiprec=True.

    The shared builder in prompts.py is now the single source of truth for
    PRECISION_HINT wording, so dev eval (`eval_lora_greedy.py --hiprec`)
    and this script can never drift.
    """
    if prompt_variant == "default":
        return build_prompt_split(ex, tokenizer, hiprec=True)
    if prompt_variant == "contract_v2":
        return build_prompt_split_contract_v2(ex, tokenizer, hiprec=True)
    if prompt_variant == "typed_v1":
        return build_prompt_split_typed_v1(ex, tokenizer, hiprec=True)
    raise ValueError(f"unknown prompt_variant: {prompt_variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lora",   default=None)
    ap.add_argument("--base",   default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--max_tokens",   type=int, default=24576)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--prompt_variant",
                    choices=["default", "contract_v2", "typed_v1"],
                    default="typed_v1")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    use_lora = args.lora is not None
    llm_kwargs = dict(
        model=args.base, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.92, max_model_len=args.max_model_len,
        max_num_seqs=16, enforce_eager=False, enable_chunked_prefill=True,
        enable_prefix_caching=True, kv_cache_dtype="fp8", trust_remote_code=True,
    )
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("lora", 1, args.lora); print(f"LoRA: {args.lora}")
    else:
        lora = None

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    print(
        f"loaded {len(rows)} rows; using high-precision hint in user msg; "
        f"prompt_variant={args.prompt_variant}"
    )
    prompts = [build_prompt(r, tok, args.prompt_variant) for r in rows]
    outs = llm.generate(prompts, sp, lora_request=lora) if use_lora else llm.generate(prompts, sp)
    n_trunc = sum(1 for o in outs for c in o.outputs if c.finish_reason == "length")
    print(f"truncated: {n_trunc}/{len(outs)} = {n_trunc/len(outs):.1%}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "response": o.outputs[0].text},
                               ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
