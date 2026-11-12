"""
Answer extraction and scoring for GSM8K and MATH (Person 3, E3/E4).

Fixed parsing policy (version PARSER_VERSION) -- do NOT change between runs:
  1. If CUT_AT_NEXT_QUESTION, drop everything from the point where the model starts a new
     task: "\nQuestion:", "\n[Question]" or "You are an AI assistant" (v3; the base model
     invents new problems / pastes system prompts after answering).
  2. Precedence:
       a) first "Final answer:" marker that has content -> text on that line, or (v4)
          the block right below an empty marker line (e.g. a \\[ \\boxed{..} \\] display).
          (v5) Markers that START a line are tried first; a marker buried mid-sentence
          ("...to find the final answer: $$2.9 x 100 = 293.8$$") is used only if no
          line-start marker exists. Found by grading Person 1's training targets.
       b) otherwise \\boxed{...}         -> must be exactly ONE distinct boxed value
       c) otherwise                     -> invalid ("no_answer")
     There is deliberately NO "last number anywhere" fallback.
  3. Unparseable / ambiguous outputs count as INCORRECT (and invalid).
  4. (v6) GSM numbers: a Unicode minus sign (U+2212, fullwidth U+FF0D) reads as "-", and a
     decimal without a leading zero (".5") reads as 0.5. Before v6 both lost their sign or
     their decimal point ("-5" -> 5, ".5" -> 5) and were graded wrong. MATH was unaffected
     (math_verify already handled both).
"""
from __future__ import annotations

import re
from fractions import Fraction

PARSER_VERSION = "p3-parser-v6"
CUT_AT_NEXT_QUESTION = True  # confirm with team at meeting; log in eval config

_MARKER = re.compile(r"final\s+answer\s*[:：]", re.IGNORECASE)
_NUM = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?")   # v6: also ".5"
_MINUS = str.maketrans({"−": "-", "－": "-"})      # v6: Unicode minus signs


# ---------------------------------------------------------------- extraction
def _boxed_contents(text: str) -> list[str]:
    """Return contents of every \\boxed{...} / \\fbox{...}, with brace matching."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append(text[m.end(): i - 1].strip())
    return out


def _clean(ans: str) -> str:
    ans = ans.strip()
    boxed = _boxed_contents(ans)
    if len(boxed) == 1:  # "Final answer: \boxed{5}"
        ans = boxed[0]
    ans = ans.strip().strip("$").strip()
    ans = re.sub(r"^\\\(|\\\)$", "", ans).strip()      # \( ... \)
    ans = re.sub(r"[.。]+$", "", ans).strip()            # trailing period
    return ans


# v3: the model starts a new task with "Question:", "[Question]" or a pasted system prompt
_RUNAWAY = re.compile(r"\n\s*(?:\[Question\]|Question\s*:)|You are an AI assistant", re.IGNORECASE)


def answer_region(output: str) -> str:
    """The part of the output that is the model's actual answer:
    up to the end of the first 'Final answer:' line; without a marker, up to the point
    where the model starts a new question / system-prompt text. Used for repetition only."""
    if not output:
        return ""
    text = _RUNAWAY.split(output, maxsplit=1)[0]
    m = _MARKER.search(text)
    if m:
        end = text.find("\n", m.end())
        text = text if end == -1 else text[:end]
    return text


def _unwrap_display(block: str) -> str:
    """'\\[ x \\]' or '$$ x $$' spread over several lines -> 'x'."""
    b = " ".join(block.split())
    b = re.sub(r"^\\\[|\\\]$", "", b).strip()
    b = re.sub(r"^\$\$|\$\$$", "", b).strip()
    return b


def extract_answer(output: str) -> tuple[str | None, str]:
    """Return (answer_string or None, parse_status)."""
    if output is None or not output.strip():
        return None, "empty"
    text = output
    if CUT_AT_NEXT_QUESTION:
        text = _RUNAWAY.split(text, maxsplit=1)[0]

    markers = list(_MARKER.finditer(text))
    # v5: line-start markers (optionally after markdown "**", "#", ">", "-") take precedence
    line_start = [m for m in markers
                  if not text[text.rfind("\n", 0, m.start()) + 1:m.start()].strip(" \t*#>-")]
    for m in (line_start or markers):       # v4: first marker WITH content wins
        after = text[m.end():].lstrip(" \t")
        line = after.split("\n", 1)[0]
        ans = _clean(line)
        if not ans:                         # answer on the lines below (e.g. a \[ \boxed{} \] block)
            block = after.lstrip("\n").split("\n\n", 1)[0]
            if not _MARKER.search(block):
                ans = _clean(_unwrap_display(block))
        if ans:
            return ans, "ok"

    boxed = [_clean(b) for b in _boxed_contents(text)]
    boxed = [b for b in boxed if b]
    if len(set(boxed)) == 1:
        return boxed[0], "ok"
    if len(set(boxed)) > 1:
        return None, "ambiguous"
    return (None, "empty_marker") if markers else (None, "no_answer")


# ---------------------------------------------------------------- GSM8K
def gsm_gold(answer_field: str) -> str:
    """GSM8K gold = text after '####'."""
    return answer_field.split("####")[-1].strip()


def _to_number(s: str) -> Fraction | None:
    """Fixed numeric policy: strip $, %, commas, units text; need exactly one number."""
    s = s.translate(_MINUS)
    s = s.replace(",", "").replace("$", "").replace("\\$", "").replace("%", "")
    s = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", s)   # \frac{a}{b}
    s = s.replace("\\!", "").replace("{,}", "")
    nums = _NUM.findall(s)
    if len(nums) != 1:
        return None
    try:
        return Fraction(nums[0])
    except (ValueError, ZeroDivisionError):
        return None


def score_gsm(output: str, gold_answer_field: str) -> dict:
    pred, status = extract_answer(output)
    gold = gsm_gold(gold_answer_field)
    correct = False
    if status == "ok":
        p, g = _to_number(pred), _to_number(gold)
        if p is None:
            status = "unparseable_number"
        else:
            correct = (g is not None and p == g)
    return {"parsed_answer": pred, "gold_answer": gold,
            "parse_status": status, "correct": bool(correct)}


# ---------------------------------------------------------------- MATH
def math_gold(example: dict) -> str | None:
    """Prefer an explicit 'answer' field; else the last \\boxed{} in 'solution'."""
    if example.get("answer"):
        return str(example["answer"]).strip()
    boxed = _boxed_contents(example.get("solution", ""))
    return boxed[-1] if boxed else None


def _strip_thousands(s: str) -> str:
    """'29,\\!322' or '29{,}322' -> '29322' (thousands separators only, never list commas)."""
    return re.sub(r"(?<=\d)(?:,\\!|\{,\})(?=\d{3})", "", s)


def _plain(s: str) -> str:
    """Formatting-free form for a strict string fallback: drop \\text{}, \\left/\\right, $, spaces, case."""
    s = _strip_thousands(s)
    s = re.sub(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|displaystyle|!|,|;)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("$", "")
    return re.sub(r"\s+", "", s).lower().rstrip(".")


def score_math(output: str, gold: str) -> dict:
    from math_verify import parse, verify  # pinned version; logged in eval config

    pred, status = extract_answer(output)
    correct = False
    if status == "ok" and _plain(pred) == _plain(gold):   # same answer, different formatting
        return {"parsed_answer": pred, "gold_answer": gold, "parse_status": status, "correct": True}
    if status == "ok":
        gold = _strip_thousands(gold)
        pred = _strip_thousands(pred)
        try:
            g = parse(f"${gold}$")
            p = parse(f"${pred}$")
            if not p:
                status = "unparseable_math"      # extraction failure != wrong
            else:
                correct = bool(verify(g, p))
        except Exception:                        # timeout / parser crash
            status = "verify_error"
    return {"parsed_answer": pred, "gold_answer": gold,
            "parse_status": status, "correct": bool(correct)}
