"""Lightweight answer extraction helpers for self-consistency voting.

This module runs only in the base-model self-consistency fallback path:

``run_inference.py`` -> ``vote_all(...)`` -> functions in this file.

It does not score answers, call the judger, or participate in the A17 target
gate. The target gate compares A17 model-generated boxed answers with stored
distillation targets using ``scripts/answer_normalization.py``.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import UNITS  # noqa: E402


MCQ_LETTERS = set("ABCDEFGHIJ")


def _strip_think(text: str) -> str:
    """Keep only the final-answer section after ``</think>`` when present."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def find_all_boxed(text: str) -> list[str]:
    """Return every balanced ``\\boxed{...}`` payload in order."""
    boxes: list[str] = []
    i = 0
    while True:
        idx = text.find(r"\boxed", i)
        if idx < 0:
            break
        j = idx + len(r"\boxed")
        while j < len(text) and text[j] in " \t\n":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = idx + 1
            continue

        depth = 1
        start = j + 1
        j += 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            boxes.append(text[start : j - 1])
        i = j
    return boxes


def _dedupe_trailing_repeat(items: list[str]) -> list[str]:
    """Drop repeated answer blocks such as ``[a, b, a, b]`` -> ``[a, b]``."""
    n = len(items)
    if n < 2:
        return items
    for k in (2, 3, 4, 5, 6):
        if n % k == 0:
            chunk = n // k
            if all(items[i] == items[i % chunk] for i in range(n)):
                return items[:chunk]
    for half in range(n // 2, 0, -1):
        if items[-half:] == items[:half]:
            return items[:-half]
    return items


def merge_multi_boxed(response: str) -> str | None:
    """Return one answer string from the final section of a model response."""
    boxes = find_all_boxed(_strip_think(response))
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]

    boxes = _dedupe_trailing_repeat(boxes)
    unique_boxes: list[str] = []
    seen = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique_boxes.append(box)
    if len(unique_boxes) == 1:
        return unique_boxes[0]
    return ", ".join(unique_boxes)


_MCQ_WORD_MAP = {
    "true": "True",
    "false": "False",
    "yes": "Yes",
    "no": "No",
    "none": "None",
    "all of the above": "All",
    "none of the above": "None",
}
_ROMAN = {
    "I": "I",
    "II": "II",
    "III": "III",
    "IV": "IV",
    "V": "V",
    "VI": "VI",
    "VII": "VII",
    "VIII": "VIII",
}


def extract_mcq_answer(response: str) -> str | None:
    """Extract an MCQ answer letter/value from boxed or final-answer text."""
    tail = _strip_think(response)

    merged = merge_multi_boxed(tail)
    if merged is not None:
        inner = merged.strip()
        low = inner.lower().strip(" .()")
        if low in _MCQ_WORD_MAP:
            return _MCQ_WORD_MAP[low]
        up = inner.strip(" .()").upper()
        if up in _ROMAN:
            return _ROMAN[up]
        match = re.search(r"[A-J]", inner.upper())
        if match:
            return match.group()

    patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-J])\)?",
        r"(?:correct\s+)?(?:choice|option|letter)\s*(?:is|:)\s*\(?([A-J])\)?",
        r"\(([A-J])\)\s*[.]?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, tail, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    final_window = tail[-300:]
    found = sorted(
        letter
        for letter in MCQ_LETTERS
        if re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", final_window)
    )
    return found[0] if len(found) == 1 else None


def extract_ff_merged(response: str) -> str | None:
    """Return merged boxed free-form answer content from a raw response."""
    return merge_multi_boxed(response)


def strip_units(text: str) -> str:
    """Remove common units and LaTeX spacing before free-form voting."""
    out = text
    for unit_pattern in UNITS:
        out = re.sub(unit_pattern, "", out)
    out = re.sub(r"\\text\s*\{[^}]*\}", "", out)
    out = re.sub(r"\\(?:qquad|quad|[,;:!\s])", "", out)
    return out.strip()


def split_ff_multi(text: str) -> list[str]:
    """Split free-form answers on top-level commas/semicolons only."""
    if not text:
        return []

    tmp = re.sub(r"\\quad|\\,|\\;|\\:", ",", text)
    tmp = tmp.replace(" and ", ",")
    parts: list[str] = []
    buf: list[str] = []
    depth = 0

    for ch in tmp:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch in ",;" and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts
