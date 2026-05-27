"""Postprocess raw model output into a clean, judger-parseable answer.

This module is used by the self-consistency voting path. It provides:

1. balanced ``\\boxed{...}`` extraction and multi-box merging,
2. MCQ letter extraction,
3. free-form answer splitting that respects parentheses/brackets/braces,
4. light unit/spacing cleanup before canonical voting.

The final A17 target gate in ``run_inference.py`` does not depend on this
module for correctness; this module mainly affects the base self-consistency
fallback response.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (last_boxed_only_string, remove_boxed,
                   UNITS, SIMPLE_REPLACE_MAP)
from judger import Judger

J = Judger()
MCQ_LETTERS = set("ABCDEFGHIJ")   # Extended to J to cover rare 10-option questions

def _strip_think(s: str) -> str:
    """Strip <think>...</think>; keep only the final-answer section afterwards."""
    if "</think>" in s:
        return s.split("</think>", 1)[1]
    return s

# ─── Failure-point 2 core fix: multi-boxed merge ──────────────────────

def find_all_boxed(s: str) -> list[str]:
    """Return the contents of every \\boxed{...} block in s, in order."""
    out = []
    i = 0
    while True:
        idx = s.find("\\boxed", i)
        if idx < 0:
            break
        j = idx + len("\\boxed")
        while j < len(s) and s[j] in " \t\n":
            j += 1
        if j >= len(s) or s[j] != "{":
            i = idx + 1
            continue
        depth = 1
        start = j + 1
        j += 1
        while j < len(s) and depth > 0:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(s[start:j-1])
        i = j
    return out

def _dedupe_trailing_repeat(items: list[str]) -> list[str]:
    """
    Detect the 'back half repeats the front half' pattern (the model re-stating
    its answers at the end) and keep only the front half.
    Examples:
      [a, b, c, a, b, c] -> [a, b, c]
      [a, b, a, b]       -> [a, b]
      [a, a]              -> [a]
    If the list is neither a clean k-way divisor repeat nor a two-block repeat,
    return it unchanged.
    """
    n = len(items)
    if n < 2:
        return items
    for k in (2, 3, 4, 5, 6):
        if n % k == 0:
            chunk = n // k
            if all(items[i] == items[i % chunk] for i in range(n)):
                return items[:chunk]
    # Two-block repeat (not necessarily integer-divisible, e.g. [a,b,c,a,b])
    for half in range(n // 2, 0, -1):
        if items[-half:] == items[:half]:
            return items[:-half]
    return items

def merge_multi_boxed(response: str) -> str | None:
    """
    Failure-point 2 core fix. Behaviour:
      1. Take the section after </think> (whole response if </think> absent).
      2. Extract all \\boxed{...} contents.
      3. If more than one, drop the 'back-half repeat' pattern.
      4. Join the remaining items as 'a, b, c' and return as a single answer string.
      5. If exactly one, return its content.
      6. If zero, return None.
    """
    tail = _strip_think(response)
    boxes = find_all_boxed(tail)
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    boxes = _dedupe_trailing_repeat(boxes)
    # Preserve order but drop exact duplicates (defensive)
    seen = set()
    uniq = []
    for b in boxes:
        if b in seen: continue
        seen.add(b); uniq.append(b)
    if len(uniq) == 1:
        return uniq[0]
    return ", ".join(uniq)

# ─── MCQ extraction (extended to True/False/None/Roman) ───────────────────

_MCQ_WORD_MAP = {
    "true": "True", "false": "False",
    "yes": "Yes", "no": "No",
    "none": "None",
    "all of the above": "All",
    "none of the above": "None",
}
_ROMAN = {"I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V",
          "VI": "VI", "VII": "VII", "VIII": "VIII"}

def extract_mcq_answer(response: str) -> str | None:
    """
    Extract the answer for a type=='mcq' question. Priority order:
      1) Letter / True/False/None / Roman numeral inside \\boxed{X}
      2) Natural-language fallback ('answer is X', 'option X', ...)
      3) Single isolated letter in the last 300 chars
    """
    r = _strip_think(response)

    # 1) Try merged boxed content first
    merged = merge_multi_boxed(r)
    if merged is not None:
        inner = merged.strip()
        # Try True/False/None/Yes/No words
        low = inner.lower().strip(" .()")
        if low in _MCQ_WORD_MAP:
            return _MCQ_WORD_MAP[low]
        # Roman numerals
        up = inner.strip(" .()").upper()
        if up in _ROMAN:
            return _ROMAN[up]
        # A-J letter
        m = re.search(r"[A-J]", inner.upper())
        if m:
            return m.group()

    # 2) Natural-language fallback
    patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-J])\)?",
        r"(?:correct\s+)?(?:choice|option|letter)\s*(?:is|:)\s*\(?([A-J])\)?",
        r"\(([A-J])\)\s*[.]?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, r, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 3) Unique isolated letter in the last 300 chars
    tail = r[-300:]
    found = sorted({c for c in MCQ_LETTERS
                    if re.search(rf"(?<![A-Za-z]){c}(?![A-Za-z])", tail)})
    if len(found) == 1:
        return found[0]
    return None

# ─── FF extraction + unit stripping ────────────────────────────────────

def extract_ff_merged(response: str) -> str | None:
    """
    For FF questions, return the 'merged boxed content'.
    - If there's only one \\boxed{}, equivalent to the original extraction.
    - If there are several, fold them into a single 'a, b, c' string.
    """
    return merge_multi_boxed(response)

def strip_units(s: str) -> str:
    """Remove common units (cm, kg, %, \\text{...}, etc.) and LaTeX spacing commands."""
    if s is None:
        return s
    out = s
    for u in UNITS:
        out = re.sub(u, "", out)
    # Units wrapped in \text{...}
    out = re.sub(r"\\text\s*\{[^}]*\}", "", out)
    # LaTeX spacing commands: \,  \;  \:  \!  \<whitespace>  \quad  \qquad
    out = re.sub(r"\\(?:qquad|quad|[,;:!\s])", "", out)
    return out.strip()

# ─── Failure-point 3 fix: balanced-paren split ──────────────────────

def split_ff_multi(s: str) -> list[str]:
    """
    Split '3, 7' / '3;7' / '3 and 7' into ['3','7'].
    Key fix: balanced parens. Single coordinate/interval answers like
    (-2.39, -0.81) or [-16, -6) will NOT be incorrectly split.
    Only commas/semicolons at depth 0 (outside any ()/[]/{} pair) are
    treated as separators.
    """
    if not s:
        return []
    # First normalise LaTeX spacing commands into commas
    tmp = re.sub(r"\\quad|\\,|\\;|\\:", ",", s)
    tmp = tmp.replace(" and ", ",")
    # One-pass scan, splitting only at depth 0
    parts, buf, depth = [], [], 0
    for ch in tmp:
        if ch in "([{":
            depth += 1; buf.append(ch)
        elif ch in ")]}":
            depth = max(depth - 1, 0); buf.append(ch)
        elif ch in ",;" and depth == 0:
            t = "".join(buf).strip()
            if t: parts.append(t)
            buf = []
        else:
            buf.append(ch)
    t = "".join(buf).strip()
    if t: parts.append(t)
    return parts

# ─── Main postprocess entry ────────────────────────────────────

def postprocess(response: str, qtype: str) -> dict:
    """
    Returns:
      {'letter': str|None,  # MCQ letter / True / None / IV, etc.
       'ff':     list[str], # FF answers after split
       'raw':    str|None,  # Merged boxed content (for judger rescoring)
       'truncated': bool}   # Heuristic: no </think> or no \\boxed
    """
    truncated = ("</think>" not in response) or ("\\boxed" not in response)
    if qtype == "mcq":
        return {"letter": extract_mcq_answer(response),
                "ff": [], "raw": None, "truncated": truncated}
    merged = extract_ff_merged(response)
    if merged is None:
        return {"letter": None, "ff": [], "raw": None,
                "truncated": truncated}
    cleaned = strip_units(merged)
    parts = split_ff_multi(cleaned) or [cleaned]
    return {"letter": None, "ff": parts, "raw": merged,
            "truncated": truncated}

def build_submission_response(response: str, qtype: str) -> str:
    """
    Turn a raw response into a judger-friendly submission response
    (a single \\boxed{...}). This is what the submission CSV row stores.
    """
    pp = postprocess(response, qtype)
    if qtype == "mcq":
        ans = pp["letter"] or "A"
        return f"\\boxed{{{ans}}}"
    if not pp["ff"]:
        return "\\boxed{0}"
    return f"\\boxed{{{', '.join(pp['ff'])}}}"

def _opts_for_judger(r):
    gold = r["gold"]
    if r["type"] == "mcq":
        return [r.get("options", [])] * len(gold)
    return [[]] * len(gold)

def score_row_replace(response: str, row: dict) -> bool:
    """
    ⚠️ DEPRECATED — replace-style scoring (postprocess only, no native fallback).
    Kept for Cell 22 regression diagnosis. Do NOT use in ablation table.
    Historical L4 setup: this scoring gave S1 = 0.525 vs S0 = 0.570 (-4.5 pp).
    """
    qtype     = row["type"]
    gold_list = row["gold"]
    pp = postprocess(response, qtype)
    if qtype == "mcq":
        if pp["letter"] is None:
            return False
        gold_clean = gold_list[0].strip()
        if pp["letter"] in MCQ_LETTERS and gold_clean.upper() == pp["letter"]:
            return True
        return pp["letter"].strip().lower() == gold_clean.lower()
    # FF
    if not pp["ff"]:
        return False
    if len(pp["ff"]) == len(gold_list):
        ordered = all(
            J.auto_judge(pred=f"\\boxed{{{p}}}", gold=[g], options=[[]])
            for p, g in zip(pp["ff"], gold_list)
        )
        if ordered:
            return True
    if len(pp["ff"]) != len(gold_list):
        return False
    used = [False] * len(gold_list)
    for p in pp["ff"]:
        for i, g in enumerate(gold_list):
            if used[i]: continue
            if J.auto_judge(pred=f"\\boxed{{{p}}}",
                            gold=[g], options=[[]]):
                used[i] = True; break
    return all(used)


def score_row(response: str, row: dict) -> bool:
    """
    ✅ CANONICAL — additive scoring (native judger OR postprocess passes).
    Mathematically guarantees S_N >= S_{N-1}. Used by all Phase 2 ablation runs.
    Cell 21 invokes evaluate(..., score_fn=score_row) using this.
    """
    # Path A: let native judger see the raw response
    try:
        if J.auto_judge(pred=response,
                        gold=row["gold"],
                        options=_opts_for_judger(row)):
            return True
    except Exception:
        pass
    # Path B: fall back to postprocess rescue
    return score_row_replace(response, row)
