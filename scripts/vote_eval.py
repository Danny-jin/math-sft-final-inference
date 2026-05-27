"""
SC majority-vote + canonical FF matcher (Phase 2 D3 hot path).

Recovered (Phase 2 instance was lost). Re-implemented from:
  - V4 Cell 29 (vote_all import + S3 call)
  - V4 Cell 28 (canon_ff_v2 sanity examples)
  - handoff.md §5 description
  - phase2_plan.md §0.2 (S3 = 0.735 dev = 5 rescued / 2 hurt vs greedy)

`canon_ff_v2(s)` returns a SET of canonical keys for an FF answer string,
multi-key OR'd: literal stripped string, judger.norm_math_str, and a numeric
key from SymPy parse_latex when convertible. Two answers match if their key
sets intersect (any one key in common).

`vote_all(sc_jsonl_path)` reads {id, type, samples=[k strings]} rows and
returns dict {id: voted_response_string}. The voted string is wrapped with
`\\boxed{...}` so eval_dev can score it with score_row additive.

Voting strategy:
  - MCQ: extract letter from each sample, take majority.
  - FF (single answer): canonical key bucket, take largest bucket's
    representative; ties broken by first occurrence.
  - FF (multi-slot): split each sample's boxed payload, vote per-slot
    on canonical buckets, recombine in slot order. Falls back to whole-
    string vote if slot counts disagree.
"""
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "judger"))
from postprocess import (extract_mcq_answer, extract_ff_merged,
                          split_ff_multi, strip_units)


# ─── canon_ff_v2 ───────────────────────────────────────────────────────

def _norm_math(s: str) -> str | None:
    """Try judger.norm_math_str(s); return None on any error/empty."""
    try:
        from judger import Judger
        J = Judger()
        # Both module-level helper and method names have been seen; try both
        for fn_name in ("norm_math_str", "_norm_math_str", "normalize"):
            fn = getattr(J, fn_name, None)
            if fn:
                v = fn(s)
                if v: return str(v).strip()
        # As a fallback, look for a top-level helper in the judger module
        import judger as _jm
        for fn_name in ("norm_math_str", "normalize_math"):
            fn = getattr(_jm, fn_name, None)
            if fn:
                v = fn(s)
                if v: return str(v).strip()
    except Exception:
        return None
    return None


def _sympy_key(s: str) -> str | None:
    """Try SymPy parse_latex → simplify → str. None on failure."""
    if s is None or not s.strip():
        return None
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, nsimplify, Float, Rational
        expr = parse_latex(s)
        if expr is None:
            return None
        try:
            expr2 = simplify(expr)
        except Exception:
            expr2 = expr
        # Convert pure-numeric to Rational/Float for stable string
        try:
            return str(nsimplify(expr2, rational=True))
        except Exception:
            return str(expr2)
    except Exception:
        return None


def _literal_key(s: str) -> str:
    """Cheapest key: stripped, lower-cased, units removed, internal whitespace squeezed."""
    if s is None:
        return ""
    out = strip_units(s) if s else s
    out = (out or "").strip()
    out = re.sub(r"\s+", " ", out)
    out = out.strip(".,;:")
    return out.lower()


def canon_ff_v2(s: str) -> frozenset:
    """Return a frozenset of canonical keys for FF answer s.
    Two answers match iff canon_ff_v2(a) & canon_ff_v2(b) is non-empty.
    """
    if s is None:
        return frozenset()
    keys = set()
    lit = _literal_key(s)
    if lit: keys.add(("lit", lit))
    nm = _norm_math(s)
    if nm: keys.add(("norm", nm))
    sp = _sympy_key(s)
    if sp: keys.add(("sym", sp))
    return frozenset(keys)


# ─── voting helpers ────────────────────────────────────────────────────

def _vote_mcq(samples: list[str]) -> str:
    """Extract letter from each sample, pick majority. Empty → 'A' fallback."""
    letters = []
    for s in samples:
        lt = extract_mcq_answer(s)
        if lt:
            letters.append(lt)
    if not letters:
        return "A"
    return Counter(letters).most_common(1)[0][0]


def _vote_ff_single(samples: list[str]) -> str:
    """Single-answer FF: canonical-bucket vote, return rep of largest bucket.
    Ties broken by first occurrence."""
    answers = []
    for s in samples:
        merged = extract_ff_merged(s)
        if merged is None:
            continue
        merged = strip_units(merged).strip()
        answers.append(merged)
    if not answers:
        return "0"
    # bucket by canonical key intersection (transitive via union-find)
    n = len(answers)
    keys = [canon_ff_v2(a) for a in answers]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if keys[i] & keys[j]:
                union(i, j)
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    # largest bucket; tie → bucket whose first idx is smallest (earliest occurrence)
    best = max(buckets.values(), key=lambda idxs: (len(idxs), -min(idxs)))
    return answers[best[0]]


def _vote_ff_multi(samples: list[str]) -> str:
    """Multi-slot FF: split each sample, vote per-slot. Fall back to single-vote
    if slot counts disagree across most samples."""
    parts_per_sample = []
    for s in samples:
        merged = extract_ff_merged(s)
        if merged is None:
            continue
        merged = strip_units(merged)
        parts = split_ff_multi(merged) or [merged.strip()]
        parts_per_sample.append(parts)
    if not parts_per_sample:
        return "0"
    # majority arity
    arity_counts = Counter(len(p) for p in parts_per_sample)
    target_arity, _ = arity_counts.most_common(1)[0]
    aligned = [p for p in parts_per_sample if len(p) == target_arity]
    if not aligned:
        return ", ".join(parts_per_sample[0])
    # per-slot bucket vote
    voted_slots = []
    for slot in range(target_arity):
        slot_vals = [p[slot] for p in aligned]
        voted_slots.append(_vote_ff_single_strs(slot_vals))
    return ", ".join(voted_slots)


def _vote_ff_single_strs(answers: list[str]) -> str:
    """Same canon-bucket vote as _vote_ff_single but takes raw strings."""
    if not answers: return "0"
    n = len(answers)
    keys = [canon_ff_v2(a) for a in answers]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if keys[i] & keys[j]:
                union(i, j)
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    best = max(buckets.values(), key=lambda idxs: (len(idxs), -min(idxs)))
    return answers[best[0]]


# ─── public API ────────────────────────────────────────────────────────

def vote_one(samples: list[str], qtype: str) -> str:
    """Vote across k samples for one question; return a `\\boxed{...}` response
    suitable for postprocess.score_row."""
    if qtype == "mcq":
        letter = _vote_mcq(samples)
        return f"\\boxed{{{letter}}}"
    # FF
    n_slots_per = []
    for s in samples:
        merged = extract_ff_merged(s)
        if merged is None: continue
        parts = split_ff_multi(strip_units(merged)) or [merged.strip()]
        n_slots_per.append(len(parts))
    if not n_slots_per:
        return "\\boxed{0}"
    target = Counter(n_slots_per).most_common(1)[0][0]
    if target == 1:
        ans = _vote_ff_single(samples)
    else:
        ans = _vote_ff_multi(samples)
    return f"\\boxed{{{ans}}}"


def vote_all(sc_input) -> dict:
    """Read SC inference output and produce {id: voted_response_string}.

    Accepts either a path to a JSONL file (one {id,type,samples} per line)
    or an in-memory list of such row dicts. The list form is what
    `eval_lora_sc.py` actually passes after generation.
    """
    if isinstance(sc_input, (list, tuple)):
        rows = sc_input
    else:
        rows = (json.loads(line) for line in open(sc_input) if line.strip())

    out = {}
    for row in rows:
        qid = row["id"]
        qtype = row.get("type") or "ff"
        samples = row.get("samples") or []
        out[qid] = vote_one(samples, qtype)
    return out


# ─── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="SC JSONL with samples.")
    ap.add_argument("--output", default=None,
                    help="Optional output JSONL of voted responses.")
    args = ap.parse_args()

    voted = vote_all(args.input)
    print(f"voted {len(voted)} questions from {args.input}")
    if args.output:
        with open(args.output, "w") as f:
            for qid, resp in voted.items():
                f.write(json.dumps({"id": qid, "response": resp}) + "\n")
        print(f"wrote {args.output}")
