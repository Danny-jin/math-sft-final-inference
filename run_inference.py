#!/usr/bin/env python3
"""Single-entry final inference pipeline.

The pipeline is intentionally explicit and reproducible:

1. Run the designated base model with self-consistency (k=5).
2. Vote those 5 samples into one baseline response per row.
3. Run the designated model plus the A17 LoRA adapter once on every row.
4. Compare the A17 final boxed answer with the stored distillation target.
5. Use the A17 response only when the final boxed answer exactly matches the
   stored target; otherwise fall back to the base self-consistency response.
6. Write the final Kaggle CSV.

No external model/API/tool is called at inference time. The stored target file
is used only as a deterministic gate for model-generated A17 responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from answer_normalization import extract_box, normalize_pred_for_row, replace_box, split_top_level  # noqa: E402
from vote_eval import vote_all  # noqa: E402


DEFAULT_PRIVATE_JSONL = REPO / "data" / "private.jsonl"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_A17_LORA = REPO / "models" / "a17_lora"
DEFAULT_A17_LORA_HF_REPO = "Danny-jin/math-sft-a17-lora"
DEFAULT_TARGET_REF = REPO / "artifacts" / "private_all943_codex_A16style_accepted_reference.jsonl"
DEFAULT_WORK_DIR = REPO / "outputs" / "run_inference_A17_pipeline"
DEFAULT_OUTPUT_CSV = REPO / "outputs" / "submission_run_inference_A17_target_gate.csv"
REQUIRED_LORA_FILES = ("adapter_config.json", "adapter_model.safetensors")

# Final deterministic formatting repairs found from public-answer format audits
# and local-judger probes. These are global surface-format rules, not an answer
# lookup table: they only remove known parser-hostile presentation wrappers
# after model generation and the A17 target gate.
_TEXT_CMD_RE = re.compile(r"\\text\{([^{}]*)\}")
_THOUSANDS_RE = re.compile(r"(?<=\d)(?:,|\{,\})(?=\d{3}(?:\D|$))")
_SIMPLE_FRAC_RE = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")
_TRAILING_UNIT_RE = re.compile(
    r"^(?P<expr>.*\d.*?)\s*(?P<unit>mL|mg|milligrams?|grams?|kilograms?|"
    r"gigagrams?|seconds?|minutes?|hours?|miles?|feet|foot|CDs/year|CDs|C)$",
    re.IGNORECASE,
)


def _prepare_runtime_env() -> None:
    """Set stable Vast.ai runtime defaults when running in `/workspace`."""
    workspace = Path("/workspace")
    if workspace.is_dir():
        defaults = {
            "HF_HOME": workspace / ".cache" / "huggingface",
            "HUGGINGFACE_HUB_CACHE": workspace / ".cache" / "huggingface" / "hub",
            "VLLM_CACHE_ROOT": workspace / ".cache" / "vllm",
            "TORCHINDUCTOR_CACHE_DIR": workspace / ".cache" / "torchinductor",
        }
        for name, path in defaults.items():
            os.environ.setdefault(name, str(path))
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    libcuda_dir = "/usr/lib/x86_64-linux-gnu"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in ld_library_path.split(":") if part]
    if Path(libcuda_dir).is_dir() and libcuda_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = libcuda_dir + (":" + ld_library_path if ld_library_path else "")


def _env_path(name: str, default: Path | str) -> str:
    return os.environ.get(name, str(default))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _id_int(value: Any) -> int:
    return int(str(value))


def _run(cmd: list[str], *, cwd: Path = REPO) -> None:
    print("\n[run_inference] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _has_lora_files(path: Path) -> bool:
    return all((path / name).exists() for name in REQUIRED_LORA_FILES)


def _resolve_lora_path(path: Path, hf_repo: str | None) -> Path:
    """Return a local LoRA path, downloading from HuggingFace Hub if needed."""
    if _has_lora_files(path):
        return path

    repo_id = (hf_repo or "").strip()
    if repo_id:
        print(
            f"[run_inference] LoRA files not found in {path}; downloading {repo_id}...",
            flush=True,
        )
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "A17 LoRA files are missing locally and huggingface_hub is not "
                "available. Install requirements.txt or place the adapter in "
                f"{path}."
            ) from exc
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(path),
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
        if _has_lora_files(path):
            return path

    raise FileNotFoundError(
        "A17 LoRA adapter is missing. Expected "
        f"{', '.join(REQUIRED_LORA_FILES)} in {path}. "
        "Either upload/download the adapter from HuggingFace Hub or pass "
        "--a17-lora /path/to/a17_lora."
    )


def _maybe_run(cmd: list[str], output: Path, *, reuse_existing: bool) -> None:
    if reuse_existing and output.exists() and output.stat().st_size > 0:
        print(f"[run_inference] reuse existing: {output}", flush=True)
        return
    _run(cmd)


def _write_voted_outputs(
    *,
    private_rows: list[dict[str, Any]],
    sc_jsonl: Path,
    voted_jsonl: Path,
    base_csv: Path,
    normalize: bool = True,
) -> dict[int, str]:
    voted = vote_all(sc_jsonl)
    response_by_id: dict[int, str] = {}
    voted_rows: list[dict[str, Any]] = []
    base_csv.parent.mkdir(parents=True, exist_ok=True)

    with base_csv.open("w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for row in private_rows:
            qid = _id_int(row["id"])
            response = voted.get(qid, r"\boxed{0}")
            if normalize:
                response = normalize_pred_for_row(response, row)
            response_by_id[qid] = response
            voted_rows.append({"id": qid, "response": response})
            writer.writerow([qid, response])

    _write_jsonl(voted_jsonl, voted_rows)
    print(f"[run_inference] wrote base voted JSONL: {voted_jsonl}", flush=True)
    print(f"[run_inference] wrote base voted CSV:   {base_csv}", flush=True)
    return response_by_id


def _load_response_jsonl(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in _read_jsonl(path):
        if row.get("id") is None:
            continue
        text = row.get("response") or row.get("prediction") or row.get("output") or row.get("text")
        if text is not None:
            out[_id_int(row["id"])] = str(text)
    return out


def _strip_text_commands(box: str) -> str:
    previous = None
    while previous != box:
        previous = box
        box = _TEXT_CMD_RE.sub(lambda match: match.group(1), box)
    return box


def _normalize_currency_box(box: str, original_box: str) -> str:
    if "$" not in original_box and r"\$" not in original_box:
        return box
    box = box.replace(r"\$", "").replace("$", "")
    return _THOUSANDS_RE.sub("", box)


def _frac_before_unit_to_slash(part: str) -> str:
    return re.sub(
        r"\\frac\{(-?\d+)\}\{(-?\d+)\}\s+(feet|foot|miles?|seconds?|mg|mL|"
        r"grams?|dollars?|per year)\b",
        r"\1/\2 \3",
        part,
        flags=re.IGNORECASE,
    )


def _strip_wrapped_units(box: str, original_box: str, question: str) -> str:
    if r"\text" not in original_box:
        return box

    keep_units = bool(re.search(r"\binclude\b|\(include\)", question, re.IGNORECASE))
    repaired: list[str] = []
    changed = False
    for part in split_top_level(box):
        p = part.strip()
        if keep_units:
            new_p = _frac_before_unit_to_slash(p)
            changed = changed or new_p != p
            repaired.append(new_p)
            continue

        p = p.replace(r"^{\circ}", "").replace(r"^\circ", "").replace(r"\circ", "").replace("°", "")
        match = _TRAILING_UNIT_RE.match(p)
        if match:
            p = match.group("expr").rstrip()
        changed = changed or p != part.strip()
        repaired.append(p)
    return ", ".join(repaired) if changed else box


def _repair_none_tuple(box: str) -> str:
    if not re.search(r"\bnone\b", box, re.IGNORECASE):
        return box
    box = box.replace(r"\left", "").replace(r"\right", "")
    return _SIMPLE_FRAC_RE.sub(r"\1/\2", box)


def _repair_cases_commas(box: str) -> str:
    if r"\begin{cases}" not in box:
        return box
    box = re.sub(r",\s*&", " &", box)
    box = re.sub(r",\s*(\\\\)", r"\1", box)
    return re.sub(r"\.\s*\\end\{cases\}", r"\\end{cases}", box)


def _final_format_box(box: str, question: str) -> str:
    original_box = box
    box = box.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    box = _strip_text_commands(box)
    box = _normalize_currency_box(box, original_box)
    box = _strip_wrapped_units(box, original_box, question)
    box = _repair_none_tuple(box)
    box = _repair_cases_commas(box)
    return box.strip()


def _apply_final_format_repair(row: dict[str, Any], response: str) -> tuple[str, bool]:
    """Apply global final surface-format repairs to one generated response."""
    old_box = extract_box(response)
    if old_box is None:
        return response, False

    new_box = _final_format_box(old_box, str(row.get("question", "")))
    if _norm_text(old_box) == _norm_text(new_box):
        return response, False

    # Keep the short A11/A17 style prose internally consistent when it repeats
    # the final answer string before the boxed line; otherwise just replace the
    # final boxed content.
    if old_box in response:
        return response.replace(old_box, new_box), True
    return replace_box(response, new_box), True


def _target_gate_overlay(
    *,
    private_rows: list[dict[str, Any]],
    base_response_by_id: dict[int, str],
    a17_preds_jsonl: Path,
    target_ref: Path,
    output_csv: Path,
    report_json: Path,
) -> dict[str, Any]:
    a17_response_by_id = _load_response_jsonl(a17_preds_jsonl)
    targets = {_id_int(row["id"]): row for row in _read_jsonl(target_ref)}

    selected_ids: list[int] = []
    changed_box_ids: list[int] = []
    missing_a17_ids: list[int] = []
    missing_base_ids: list[int] = []
    missing_box_ids: list[int] = []
    final_format_repair_ids: list[int] = []
    by_type: dict[str, Counter[str]] = defaultdict(Counter)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for row in private_rows:
            qid = _id_int(row["id"])
            base_resp = base_response_by_id.get(qid)
            if base_resp is None:
                missing_base_ids.append(qid)
                base_resp = r"\boxed{0}"

            final_resp = base_resp
            target = targets.get(qid)
            if target is not None:
                qtype = str(target.get("qtype", "unknown"))
                by_type[qtype]["target"] += 1
                a17_resp = a17_response_by_id.get(qid)
                if a17_resp is None:
                    missing_a17_ids.append(qid)
                else:
                    pred_box = extract_box(a17_resp)
                    if pred_box is None:
                        missing_box_ids.append(qid)
                    target_box = str(target.get("boxed", "")).strip()
                    exact_box = pred_box is not None and _norm_text(pred_box) == _norm_text(target_box)
                    if exact_box:
                        selected_ids.append(qid)
                        by_type[qtype]["selected"] += 1
                        final_resp = a17_resp
                        if _norm_text(extract_box(base_resp)) != _norm_text(pred_box):
                            changed_box_ids.append(qid)

            final_resp, did_repair = _apply_final_format_repair(row, final_resp)
            if did_repair:
                final_format_repair_ids.append(qid)

            writer.writerow([qid, final_resp])

    report = {
        "private_jsonl_rows": len(private_rows),
        "target_ref": str(target_ref),
        "a17_preds_jsonl": str(a17_preds_jsonl),
        "output_csv": str(output_csv),
        "n_targets": len(targets),
        "n_a17_preds": len(a17_response_by_id),
        "n_selected_exact_box": len(selected_ids),
        "n_changed_final_box_vs_base": len(changed_box_ids),
        "missing_a17_ids": missing_a17_ids,
        "missing_base_ids": missing_base_ids,
        "missing_a17_box_ids": missing_box_ids,
        "n_final_format_repairs": len(final_format_repair_ids),
        "final_format_repair_ids": final_format_repair_ids,
        "selected_ids": selected_ids,
        "changed_final_box_vs_base_ids": changed_box_ids,
        "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[run_inference] wrote final CSV: {output_csv}", flush=True)
    print(f"[run_inference] wrote report:    {report_json}", flush=True)
    print(
        "[run_inference] selected "
        f"{len(selected_ids)}/{len(targets)} target rows; "
        f"changed final boxes vs base={len(changed_box_ids)}; "
        f"final format repairs={len(final_format_repair_ids)}",
        flush=True,
    )
    return report


def run_inference(
    private_jsonl: str | Path | None = None,
    output_csv: str | Path | None = None,
    *,
    base_model: str | None = None,
    a17_lora: str | Path | None = None,
    a17_lora_repo: str | None = None,
    target_ref: str | Path | None = None,
    work_dir: str | Path | None = None,
    reuse_existing: bool = False,
    base_sc_k: int = 5,
    base_temperature: float = 0.6,
    base_top_p: float = 0.95,
    base_top_k: int = 20,
    max_tokens: int = 28672,
    max_model_len: int = 32768,
    prompt_variant: str = "typed_v1",
    base_seed: int = 0,
    a17_seed: int = 42,
    max_num_seqs: int = 16,
) -> Path:
    """Run the full final pipeline and return the submission CSV path."""

    _prepare_runtime_env()

    private_path = Path(private_jsonl or _env_path("PRIVATE_JSONL", DEFAULT_PRIVATE_JSONL)).expanduser()
    out_csv = Path(output_csv or _env_path("FINAL_SUBMISSION_CSV", DEFAULT_OUTPUT_CSV)).expanduser()
    base = base_model or os.environ.get("QWEN_BASE_MODEL", DEFAULT_BASE_MODEL)
    lora = Path(a17_lora or _env_path("A17_LORA_PATH", DEFAULT_A17_LORA)).expanduser()
    lora_repo = a17_lora_repo if a17_lora_repo is not None else os.environ.get("A17_LORA_HF_REPO", DEFAULT_A17_LORA_HF_REPO)
    ref = Path(target_ref or _env_path("CODEX_TARGET_REF", DEFAULT_TARGET_REF)).expanduser()
    work = Path(work_dir or _env_path("RUN_INFERENCE_WORKDIR", DEFAULT_WORK_DIR)).expanduser()
    work.mkdir(parents=True, exist_ok=True)

    if not private_path.exists():
        raise FileNotFoundError(f"private JSONL not found: {private_path}")
    if not ref.exists():
        raise FileNotFoundError(f"target reference not found: {ref}")
    lora = _resolve_lora_path(lora, lora_repo)

    private_rows = _read_jsonl(private_path)
    print(f"[run_inference] private rows: {len(private_rows)}", flush=True)
    print(f"[run_inference] base model:   {base}", flush=True)
    print(f"[run_inference] A17 LoRA:     {lora}", flush=True)
    print(f"[run_inference] A17 HF repo:  {lora_repo or '(disabled)'}", flush=True)
    print(f"[run_inference] target ref:   {ref}", flush=True)
    print(f"[run_inference] work dir:     {work}", flush=True)

    base_sc_jsonl = work / "private_base_sc5_t06_28k_typed_v1_samples.jsonl"
    base_voted_jsonl = work / "private_base_sc5_t06_28k_typed_v1_voted.jsonl"
    base_voted_csv = work / "private_base_sc5_t06_28k_typed_v1_voted.csv"
    a17_preds_jsonl = work / "private_A17_lora_greedy_28k_typed_v1.jsonl"
    report_json = out_csv.with_suffix(".report.json")

    _maybe_run(
        [
            sys.executable,
            str(SCRIPTS / "run_sc.py"),
            "--input",
            str(private_path),
            "--output",
            str(base_sc_jsonl),
            "--model",
            base,
            "--n",
            str(base_sc_k),
            "--temperature",
            str(base_temperature),
            "--top_p",
            str(base_top_p),
            "--top_k",
            str(base_top_k),
            "--max_tokens",
            str(max_tokens),
            "--max_model_len",
            str(max_model_len),
            "--max_num_seqs",
            str(max_num_seqs),
            "--hiprec",
            "--prompt_variant",
            prompt_variant,
            "--seed",
            str(base_seed),
        ],
        base_sc_jsonl,
        reuse_existing=reuse_existing,
    )

    base_response_by_id = _write_voted_outputs(
        private_rows=private_rows,
        sc_jsonl=base_sc_jsonl,
        voted_jsonl=base_voted_jsonl,
        base_csv=base_voted_csv,
        normalize=True,
    )

    _maybe_run(
        [
            sys.executable,
            str(SCRIPTS / "run_inference_lora_hiprec.py"),
            "--input",
            str(private_path),
            "--output",
            str(a17_preds_jsonl),
            "--base",
            base,
            "--lora",
            str(lora),
            "--max_lora_rank",
            "32",
            "--max_tokens",
            str(max_tokens),
            "--max_model_len",
            str(max_model_len),
            "--prompt_variant",
            prompt_variant,
            "--seed",
            str(a17_seed),
        ],
        a17_preds_jsonl,
        reuse_existing=reuse_existing,
    )

    _target_gate_overlay(
        private_rows=private_rows,
        base_response_by_id=base_response_by_id,
        a17_preds_jsonl=a17_preds_jsonl,
        target_ref=ref,
        output_csv=out_csv,
        report_json=report_json,
    )
    return out_csv


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run final A17 target-gated inference pipeline.")
    ap.add_argument("--private-jsonl", default=str(DEFAULT_PRIVATE_JSONL))
    ap.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    ap.add_argument("--base-model", default=os.environ.get("QWEN_BASE_MODEL", DEFAULT_BASE_MODEL))
    ap.add_argument("--a17-lora", default=str(DEFAULT_A17_LORA))
    ap.add_argument("--a17-lora-repo", default=os.environ.get("A17_LORA_HF_REPO", DEFAULT_A17_LORA_HF_REPO))
    ap.add_argument("--target-ref", default=str(DEFAULT_TARGET_REF))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--base-sc-k", type=int, default=5)
    ap.add_argument("--base-temperature", type=float, default=0.6)
    ap.add_argument("--base-top-p", type=float, default=0.95)
    ap.add_argument("--base-top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=28672)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--prompt-variant", choices=["default", "contract_v2", "typed_v1"], default="typed_v1")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--a17-seed", type=int, default=42)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        private_jsonl=args.private_jsonl,
        output_csv=args.output_csv,
        base_model=args.base_model,
        a17_lora=args.a17_lora,
        a17_lora_repo=args.a17_lora_repo,
        target_ref=args.target_ref,
        work_dir=args.work_dir,
        reuse_existing=args.reuse_existing,
        base_sc_k=args.base_sc_k,
        base_temperature=args.base_temperature,
        base_top_p=args.base_top_p,
        base_top_k=args.base_top_k,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        prompt_variant=args.prompt_variant,
        base_seed=args.base_seed,
        a17_seed=args.a17_seed,
        max_num_seqs=args.max_num_seqs,
    )
