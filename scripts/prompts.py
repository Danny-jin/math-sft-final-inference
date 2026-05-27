"""Prompt builders used by the final inference pipeline.

The final submitted pipeline uses ``typed_v1``. The older ``default`` and
``contract_v2`` builders are retained only as explicit CLI compatibility
options; they are not selected by ``run_inference()`` unless the caller passes a
different ``--prompt-variant``.
"""

import re

SYSTEM_MC = r"""You are an expert mathematician solving a multiple-choice problem.

Procedure:
1. Read the question and every option carefully.
2. Think step by step inside <think>...</think>. Show the math.
3. After </think>, state the correct option letter.
4. End your response with a single line of the exact form: \boxed{LETTER}
   where LETTER is one uppercase letter from the listed options.

Do NOT repeat the option text inside \boxed{}. Only the letter.
If you are unsure, still commit to the single most likely letter.
"""

SYSTEM_FF = r"""You are an expert mathematician solving a free-form math problem.

Procedure:
1. Read the question carefully. Identify every [ANS] placeholder, in order.
2. Think step by step inside <think>...</think>. Show the math.
3. After </think>, emit EXACTLY ONE \boxed{...} containing ALL answers,
   comma-separated in the same order as the [ANS] placeholders.
   Example (two placeholders): \boxed{8, 4}
   Example (one placeholder):  \boxed{42}
   Do NOT emit multiple \boxed{} blocks. Put everything inside one.

Formatting rules inside \boxed{}:
- No units.
- Use LaTeX.
- Simplify fractions to lowest terms unless asked otherwise.
- Exact over decimal when possible.
"""


# PRECISION_HINT is injected into free-form prompts when hiprec=True. The final
# pipeline enables hiprec for both base self-consistency and A17 LoRA inference.
PRECISION_HINT = (
    "Important formatting note: for decimal answers, give at least 12 "
    "significant figures (e.g. `19.7745967126229`, not `19.77`). For "
    "answers with closed-form expressions like \\sqrt{2}, \\pi, "
    "\\frac{p}{q}, give the exact symbolic form."
)


def is_mc(ex: dict) -> bool:
    return "options" in ex and bool(ex["options"])


def format_mcq_user(q: str, opts: list[str]) -> str:
    """Build the MCQ user message (raw question does NOT include options)."""
    letters = "ABCDEFGHIJ"
    body = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(opts))
    return f"{q}\n\nOptions:\n{body}\n\nAnswer with exactly one letter in \\boxed{{}}."


def build_prompt_split(ex: dict, tokenizer, hiprec: bool = False) -> str:
    """Build the legacy split MCQ/FF prompt.

    Args:
      ex: row dict with "question" (+ "options" for MCQ).
      tokenizer: HF tokenizer whose chat template will be applied.
      hiprec: if True, inject `PRECISION_HINT` into the FF user message
        (MCQ is unaffected because its answer is a letter).
    """
    if is_mc(ex):
        user = format_mcq_user(ex["question"], ex["options"])
        sys_prompt = SYSTEM_MC
    else:
        n_ans = ex["question"].count("[ANS]") or 1
        hint = f"{PRECISION_HINT}\n\n" if hiprec else ""
        if n_ans == 1:
            user = (f"Question:\n{ex['question']}\n\n"
                    f"{hint}"
                    f"Put your final answer inside a single \\boxed{{}}.")
        else:
            user = (f"Question:\n{ex['question']}\n\n"
                    f"{hint}"
                    f"This question has {n_ans} [ANS] placeholders. "
                    f"Put ALL {n_ans} answers inside a SINGLE \\boxed{{...}}, "
                    f"comma-separated and in the same order as the placeholders. "
                    f"Example: \\boxed{{answer1, answer2}}.")
        sys_prompt = SYSTEM_FF
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


SYSTEM_MC_CONTRACT_V2 = r"""You are an expert mathematician solving a multiple-choice problem.

Procedure:
1. Read the question and every option carefully.
2. Think step by step inside <think>...</think>. Compute the answer independently before choosing.
3. Compare your derived result against every listed option, including approximate numeric matches.
4. After </think>, state the correct option letter.
5. End your response with a single line of the exact form: \boxed{LETTER}
   where LETTER is one uppercase letter from the listed options.

Do NOT repeat the option text inside \boxed{}. Only the letter.
If you are unsure, still commit to the single most likely letter.
"""


SYSTEM_FF_CONTRACT_V2 = r"""You are an expert mathematician solving a free-form math problem.

Procedure:
1. Read the question carefully. Identify every [ANS] placeholder, in order.
2. Think step by step inside <think>...</think>. Show the math.
3. After </think>, emit EXACTLY ONE \boxed{...} containing ALL answers,
   comma-separated in the same order as the [ANS] placeholders.
   Example (two placeholders): \boxed{8, 4}
   Example (one placeholder):  \boxed{42}
   Do NOT emit multiple \boxed{} blocks. Put everything inside one.

Formatting rules inside \boxed{}:
- Use LaTeX for formulas.
- Simplify fractions to lowest terms unless asked otherwise.
- Preserve the answer form requested by each blank.
- Do not repeat units if the unit is printed outside the [ANS] blank.
- If the [ANS] blank asks for a unit, direction, word, label, or option letter, output that text exactly.
"""


CONTRACT_PRECISION_HINT = (
    "For decimal answers, give at least 15 significant digits unless the "
    "question explicitly requests rounding to fewer digits. This is especially "
    "important for confidence intervals, p-values, test statistics, critical "
    "values, regression coefficients, roots, and other numeric endpoints."
)


_EMBEDDED_CHOICE_RE = re.compile(r"(?:^|[\s\[(])(?:[A-J][.)]|\([A-J]\))\s")
_MULTI_SELECT_RE = re.compile(
    r"check all|select all|all that apply|more than one|one or more|"
    r"choose all|which of the following.*(?:true|correct|apply)",
    re.IGNORECASE | re.DOTALL,
)
_INTERVAL_LIST_RE = re.compile(
    r"confidence interval|prediction interval|interval|ordered pair|"
    r"coordinate|vector|tuple|list|comma[- ]separated|endpoints?",
    re.IGNORECASE,
)
_HIGH_PRECISION_RE = re.compile(
    r"confidence interval|p-?value|test statistic|critical value|"
    r"regression|correlation|standard deviation|sample sd|variance|anova|"
    r"chi[- ]?square|z[- ]?(?:score|test|statistic)|"
    r"t[- ]?(?:test|interval|statistic)|decimal|significant|"
    r"normal distribution|hypothesis|probability",
    re.IGNORECASE,
)
_EXACT_SYMBOLIC_RE = re.compile(
    r"\bexact\b|\bdo not approximate\b|\bno decimals?\b|\bfractions?\b|\brational\b|"
    r"multiple of\s+(?:pi|\\pi)|in terms of|square root|\bsqrt\b|\\sqrt|"
    r"\\pi|\bpi\b|\\ln|\\log|\bln\(|\blog\(|e\^|\\mathrm\{e\}",
    re.IGNORECASE,
)
_UNIT_WORD_RE = re.compile(
    r"\b(unit|units|feet|foot|meters?|dollars?|percent|percentage|"
    r"direction|label|word|phrase|square feet|cubic|btu)\b",
    re.IGNORECASE,
)
_ROUNDING_RE = re.compile(
    r"round(?:ed)? to|nearest|decimal places?|significant figures?|"
    r"correct to|approximate|approximation|calculator|"
    r"to two decimals?|to three decimals?|to four decimals?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"in percentages?|as a percentage|what percent|percentage|percent|\\%",
    re.IGNORECASE,
)
_CONCLUSION_CHOICE_RE = re.compile(
    r"conclusion|decision|reject|sufficient evidence|support the claim|"
    r"critical region|type [i1]{1,2} error",
    re.IGNORECASE,
)


def _contract_requirement_lines(ex: dict, hiprec: bool = True) -> list[str]:
    q = ex.get("question", "")
    n_ans = q.count("[ANS]") or 1
    lines = [
        "End with exactly one final \\boxed{...}; do not emit any other boxed expression.",
    ]
    if n_ans > 1:
        lines.append(
            f"This question has {n_ans} [ANS] placeholders; put all {n_ans} answers "
            "in the same order as the placeholders."
        )
    else:
        lines.append("This question has one answer; put only that answer in the box.")

    lines.append(
        "Do not convert a symbolic answer to a decimal, or a decimal/statistical answer "
        "to a symbolic expression, unless the question asks for that form."
    )

    if _EMBEDDED_CHOICE_RE.search(q):
        lines.append(
            "If a blank is followed by embedded choices such as A., B., C. or (A), "
            "output the chosen option letter only, not the option text."
        )
    if _MULTI_SELECT_RE.search(q):
        lines.append(
            "For check-all/select-all blanks, concatenate the chosen letters in "
            "alphabetical order with no spaces or commas, e.g. BCEG."
        )
    if _INTERVAL_LIST_RE.search(q):
        lines.append(
            "For intervals, ordered pairs, vectors, or grouped lists, keep internal "
            "commas inside parentheses/brackets, e.g. \\boxed{3, (1.2, 4.5), C}."
        )
    if hiprec or _HIGH_PRECISION_RE.search(q):
        lines.append(CONTRACT_PRECISION_HINT)
        lines.append(
            "If the question explicitly says to round to a smaller number of decimals "
            "or significant figures, obey that requested precision."
        )
    if _EXACT_SYMBOLIC_RE.search(q):
        lines.append(
            "If the question asks for exact form, no decimals, a fraction, a multiple "
            "of pi, radicals, logs, or exponentials, do not give a decimal approximation; "
            "this overrides the decimal precision rule."
        )
    if _UNIT_WORD_RE.search(q):
        lines.append(
            "If the blank itself asks for a unit, direction, word, phrase, or label, "
            "include that text; otherwise do not repeat units printed outside the blank."
        )
    return lines


def _format_contract_requirements(ex: dict, hiprec: bool = True) -> str:
    return "\n".join(f"- {line}" for line in _contract_requirement_lines(ex, hiprec=hiprec))


def build_prompt_split_contract_v2(ex: dict, tokenizer, hiprec: bool = True) -> str:
    """Build the legacy type-aware contract prompt."""
    if is_mc(ex):
        user = format_mcq_user(ex["question"], ex["options"])
        user += "\n\nCompute independently, compare against all options, and finish with exactly one boxed letter."
        sys_prompt = SYSTEM_MC_CONTRACT_V2
    else:
        n_ans = ex["question"].count("[ANS]") or 1
        reqs = _format_contract_requirements(ex, hiprec=hiprec)
        if n_ans == 1:
            user = (
                f"Question:\n{ex['question']}\n\n"
                f"Answer contract:\n{reqs}\n\n"
                f"Put your final answer inside a single \\boxed{{}}."
            )
        else:
            user = (
                f"Question:\n{ex['question']}\n\n"
                f"Answer contract:\n{reqs}\n\n"
                f"Put ALL {n_ans} answers inside a SINGLE \\boxed{{...}}, "
                f"comma-separated and in the same order as the placeholders. "
                f"Example: \\boxed{{answer1, answer2}}."
            )
        sys_prompt = SYSTEM_FF_CONTRACT_V2
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


SYSTEM_MC_TYPED_V1 = r"""You are an expert mathematician solving a multiple-choice problem.

Procedure:
1. Read the problem and every option carefully.
2. Think step by step inside <think>...</think>, but keep the reasoning focused.
3. Compute the answer independently before choosing.
4. Compare your derived result against every option, including approximate numeric or algebraic matches.
5. End with exactly one final line: \boxed{LETTER}

Do NOT repeat the option text inside \boxed{}. Only the letter.
If multiple options look close, choose the option that matches the derived value best.
"""


SYSTEM_FF_TYPED_V1 = r"""You are an expert mathematician solving a free-form math problem.

Procedure:
1. Read the question carefully and identify every [ANS] placeholder in order.
2. Think step by step inside <think>...</think>, but keep the reasoning focused.
3. Before the final answer, verify the slot count, slot order, and requested answer form.
4. After </think>, emit EXACTLY ONE \boxed{...} containing all final answers.
5. Do NOT emit multiple boxed expressions.

Use LaTeX for formulas. Preserve the form requested by the blank.
"""


def _detect_typed_traits(ex: dict, hiprec: bool = True) -> list[str]:
    q = ex.get("question", "")
    traits = []
    if is_mc(ex):
        traits.append("mcq_option_match")
        if _HIGH_PRECISION_RE.search(q) or _ROUNDING_RE.search(q) or _PERCENT_RE.search(q):
            traits.append("mcq_numeric_match")
        return traits
    n_ans = q.count("[ANS]") or 1
    if n_ans > 1:
        traits.append("ff_multi_slot")
    else:
        traits.append("ff_single_slot")
    if _HIGH_PRECISION_RE.search(q) or _ROUNDING_RE.search(q) or _PERCENT_RE.search(q):
        traits.append("high_precision_numeric")
    if _CONCLUSION_CHOICE_RE.search(q) and _EMBEDDED_CHOICE_RE.search(q):
        traits.append("stats_conclusion_letters")
    if _EMBEDDED_CHOICE_RE.search(q):
        traits.append("embedded_choice_letters")
    if _MULTI_SELECT_RE.search(q):
        traits.append("multi_select_letters")
    if _INTERVAL_LIST_RE.search(q):
        traits.append("interval_or_grouped_list")
    if _EXACT_SYMBOLIC_RE.search(q):
        traits.append("exact_symbolic")
    if _UNIT_WORD_RE.search(q):
        traits.append("unit_or_word_blank")
    if _PERCENT_RE.search(q):
        traits.append("percent_context")
    if _ROUNDING_RE.search(q):
        traits.append("explicit_rounding")
    return traits


def _typed_requirement_lines(ex: dict, hiprec: bool = True) -> list[str]:
    q = ex.get("question", "")
    n_ans = q.count("[ANS]") or 1
    traits = _detect_typed_traits(ex, hiprec=hiprec)
    lines = [
        "Emit one and only one final \\boxed{...}.",
    ]
    if "ff_multi_slot" in traits:
        lines.append(
            f"There are {n_ans} [ANS] placeholders. The final box must contain "
            f"{n_ans} top-level comma-separated answers in exactly that order."
        )
        lines.append(
            "If a single slot is itself an interval, ordered pair, vector, or list, "
            "wrap that slot in parentheses/brackets so its internal comma is not a "
            "top-level separator."
        )
    else:
        lines.append(
            "There is one [ANS] placeholder. If the answer is a set/list/interval, "
            "keep the whole grouped answer as one slot, e.g. (a, b)."
        )

    if "high_precision_numeric" in traits:
        lines.append(
            "For statistical/numeric decimals such as test statistics, p-values, "
            "critical values, regression coefficients, standard deviations, and "
            "confidence-interval endpoints, keep at least 15 significant digits."
        )
        lines.append(
            "Do not round intermediate calculations; round only if the problem explicitly requests it."
        )
    if "explicit_rounding" in traits:
        lines.append(
            "If the problem explicitly requests a rounding precision, the final answer must obey that precision."
        )
    if "stats_conclusion_letters" in traits:
        lines.append(
            "For hypothesis-test decisions or conclusions with A/B/C/D options, output the option letter only."
        )
    if "embedded_choice_letters" in traits:
        lines.append(
            "For any blank followed by embedded options like A., B., C. or (A), output only the chosen letter."
        )
    if "multi_select_letters" in traits:
        lines.append(
            "For select-all/check-all blanks, output all chosen letters concatenated in alphabetical order with no spaces or commas, e.g. BCEG."
        )
    if "unit_or_word_blank" in traits:
        lines.append(
            "If the blank itself asks for a unit, word, phrase, direction, or label, output that text exactly."
        )
        lines.append(
            "If the unit is printed outside the [ANS] blank, do not repeat the unit in the boxed answer."
        )
    if "percent_context" in traits:
        lines.append(
            "For percentage answers, use the scale requested by the prompt: if it asks for percentages or prints % after the blank, output the numeric percent value, not the proportion."
        )
    if "interval_or_grouped_list" in traits:
        lines.append(
            "For interval notation, use parentheses/brackets and infinity/-infinity as requested by the prompt."
        )
    if "exact_symbolic" in traits:
        lines.append(
            "If the question asks for exact form, fractions, radicals, pi, logs, or no decimals, give exact symbolic form; this overrides decimal precision."
        )
    lines.append(
        "Do not change answer form just because another equivalent form exists; match the blank's expected form."
    )
    return lines


def _format_typed_requirements(ex: dict, hiprec: bool = True) -> str:
    traits = ", ".join(_detect_typed_traits(ex, hiprec=hiprec))
    lines = _typed_requirement_lines(ex, hiprec=hiprec)
    body = "\n".join(f"- {line}" for line in lines)
    return f"Detected answer profile: {traits}\n{body}"


def build_prompt_split_typed_v1(ex: dict, tokenizer, hiprec: bool = True) -> str:
    """Build the final typed prompt used by ``run_inference()``."""
    if is_mc(ex):
        user = format_mcq_user(ex["question"], ex["options"])
        user += (
            "\n\nAnswer contract:\n"
            "- Compute independently before choosing.\n"
            "- Compare the derived value with every option.\n"
            "- If options are approximate or algebraically equivalent, choose the closest matching option.\n"
            "- Finish with exactly one boxed uppercase letter."
        )
        sys_prompt = SYSTEM_MC_TYPED_V1
    else:
        n_ans = ex["question"].count("[ANS]") or 1
        reqs = _format_typed_requirements(ex, hiprec=hiprec)
        if n_ans == 1:
            user = (
                f"Question:\n{ex['question']}\n\n"
                f"Answer contract:\n{reqs}\n\n"
                f"Put your final answer inside a single \\boxed{{}}."
            )
        else:
            user = (
                f"Question:\n{ex['question']}\n\n"
                f"Answer contract:\n{reqs}\n\n"
                f"Put all {n_ans} answers inside one \\boxed{{...}}, "
                f"comma-separated and in the same order as the [ANS] placeholders."
            )
        sys_prompt = SYSTEM_FF_TYPED_V1
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
