"""Parser fixtures (E4). Run:  pytest -q test_answer_parser.py
Every case was checked by hand. If a test fails, fix the parser -- not the expectation --
unless the team agrees to change the policy (then bump PARSER_VERSION)."""
import pytest
from answer_parser import extract_answer, score_gsm, score_math, math_gold

G = lambda ans: f"Some reasoning\n#### {ans}"   # GSM8K-style gold field

GSM_CASES = [
    # (id, model output, gold, expected_correct, expected_status)
    ("plain_int",        "2+3=5\nFinal answer: 5",                      "5",    True,  "ok"),
    ("negative",         "Final answer: -12",                           "-12",  True,  "ok"),
    ("comma_thousands",  "Final answer: 1,250",                         "1250", True,  "ok"),
    ("gold_comma",       "Final answer: 1250",                          "1,250",True,  "ok"),
    ("dollar_sign",      "Final answer: $18",                           "18",   True,  "ok"),
    ("units_text",       "Final answer: 18 dollars.",                   "18",   True,  "ok"),
    ("decimal_eq_int",   "Final answer: 5.0",                           "5",    True,  "ok"),
    ("decimal",          "Final answer: 2.5",                           "2.5",  True,  "ok"),
    ("fraction",         "Final answer: 3/4",                           "0.75", True,  "ok"),
    ("latex_frac",       "Final answer: \\frac{3}{4}",                  "0.75", True,  "ok"),
    ("percent",          "Final answer: 40%",                           "40",   True,  "ok"),
    ("boxed_only",       "so the total is \\boxed{42}",                 "42",   True,  "ok"),
    ("marker_and_boxed", "Final answer: \\boxed{7}",                    "7",    True,  "ok"),
    ("case_insensitive", "FINAL ANSWER: 9",                             "9",    True,  "ok"),
    ("wrong_answer",     "Final answer: 6",                             "5",    False, "ok"),
    ("marker_beats_intermediate", "3*4=12, 12+8=20\nFinal answer: 20",  "20",   True,  "ok"),
    ("no_last_number_fallback", "3*4=12, then 12+8=20",                 "20",   False, "no_answer"),
    ("truncated",        "Step 1: 3*4=12. Step 2: 12+",                 "20",   False, "no_answer"),
    ("empty",            "",                                            "5",    False, "empty"),
    ("whitespace",       "   \n  ",                                     "5",    False, "empty"),
    ("empty_marker",     "Final answer:\n",                             "5",    False, "empty_marker"),
    ("two_numbers",      "Final answer: 5 or 6",                        "5",    False, "unparseable_number"),
    ("ambiguous_boxed",  "\\boxed{5} ... actually \\boxed{6}",          "5",    False, "ambiguous"),
    ("same_boxed_twice", "\\boxed{5} so again \\boxed{5}",              "5",    True,  "ok"),
    ("first_marker_wins","Final answer: 5\nQuestion: new\nFinal answer: 8", "5", True,  "ok"),
    ("runaway_question", "Final answer: 5\n\nQuestion: Tom has 3...\nSolution: ...", "5", True, "ok"),
    ("answer_only_in_new_q", "work...\nQuestion: x\nFinal answer: 5",   "5",    False, "no_answer"),
    ("bracket_question", "The answer is 5.\n[Question]Tom has...\nFinal answer: 8", "5", False, "no_answer"),
    ("system_prompt_junk", "work\nYou are an AI assistant that helps.\nFinal answer: 8", "5", False, "no_answer"),
    # v5: prose "final answer:" inside reasoning must not beat the terminal marker
    ("prose_marker_mid_sentence", "Multiply to find the final answer:\n$$2.938 \\times 100 = 293.8.$$\n\nFinal answer: 293.8", "293.8", True, "ok"),
    ("prose_marker_same_line", "So the final answer: maybe 3 or 4.\nFinal answer: 4", "4", True, "ok"),
    ("bold_line_marker", "work\n**Final answer:** 12", "12", True, "ok"),
    ("only_prose_marker", "We now give the final answer: 7", "7", True, "ok"),
    ("marker_then_block", "Final Answer:\n\\[\n\\boxed{28}\n\\]\n\nFinal answer: 28", "28", True, "ok"),
    ("marker_block_only", "Final Answer:\n\\[\n\\boxed{11}\n\\]", "11", True, "ok"),
    ("marker_block_wrong", "Final Answer:\n\\[\n\\boxed{5}\n\\]", "8", False, "ok"),
    ("marker_dollar_block", "Final answer:\n$$12$$", "12", True, "ok"),
    ("empty_then_marker", "Final answer:\n\nFinal answer: 7", "7", True, "ok"),
    ("answer_then_junk", "Final answer: 5\nYou are an AI assistant that helps people.", "5", True, "ok"),
]


@pytest.mark.parametrize("cid,out,gold,exp_correct,exp_status", GSM_CASES, ids=[c[0] for c in GSM_CASES])
def test_gsm(cid, out, gold, exp_correct, exp_status):
    r = score_gsm(out, G(gold))
    assert r["parse_status"] == exp_status, r
    assert r["correct"] == exp_correct, r


MATH_CASES = [
    ("frac_vs_decimal", "Final answer: 0.5",                 "\\frac{1}{2}",   True),
    ("latex_frac",      "Final answer: $\\frac{1}{2}$",      "\\frac{1}{2}",   True),
    ("sqrt",            "Final answer: 2\\sqrt{3}",          "2\\sqrt{3}",     True),
    ("boxed_nested",    "thus \\boxed{\\frac{\\sqrt{2}}{2}}", "\\frac{\\sqrt{2}}{2}", True),
    ("pi",              "Final answer: 3\\pi",               "3\\pi",          True),
    ("interval",        "Final answer: (2, 5]",              "(2,5]",          True),
    ("wrong",           "Final answer: 7",                   "8",              False),
    ("negative",        "Final answer: -\\frac{3}{4}",       "-\\frac{3}{4}",  True),
    # cases found in the real dev/test files
    ("text_units",      "Final answer: 108",                 "108\\text{ degrees}", True),
    ("text_choice",     "Final answer: (A)",                 "\\text{(A)}",    True),
    ("text_words",      "Final answer: October 30",          "\\text{October 30}", True),
    ("text_wrong",      "Final answer: October 31",          "\\text{October 30}", False),
    ("thousands_sep",   "Final answer: 21000",               "21,\\!000",      True),
    ("thousands_frac",  "Final answer: 1/29322216",          "\\dfrac{1}{29,\\!322,\\!216}", True),
    ("list_any_order",  "Final answer: -2+9i, 2-9i",         "2 - 9i, -2 + 9i", True),
    ("list_not_thousands", "Final answer: 1, 2",             "1,2",            True),
    ("degrees",         "Final answer: 150",                 "150^\\circ",     True),
]


@pytest.mark.parametrize("cid,out,gold,exp", MATH_CASES, ids=[c[0] for c in MATH_CASES])
def test_math(cid, out, gold, exp):
    r = score_math(out, gold)
    assert r["correct"] == exp, r


def test_math_gold_from_solution():
    ex = {"solution": "We get $x=\\boxed{\\frac{1}{2}}$."}
    assert math_gold(ex) == "\\frac{1}{2}"


def test_math_gold_answer_field_preferred():
    assert math_gold({"answer": "4", "solution": "\\boxed{5}"}) == "4"


def test_extract_handles_none():
    assert extract_answer(None) == (None, "empty")


# ---- v6: Unicode minus and leading-dot decimals (GSM)
@pytest.mark.parametrize("out,gold,exp", [
    ("Final answer: .5", "#### 0.5", True),
    ("Final answer: $.75", "#### 0.75", True),
    ("Final answer: \u22125", "#### -5", True),      # U+2212 minus sign
    ("Final answer: \uff0d12", "#### -12", True),    # fullwidth minus
    ("Final answer: \u22125", "#### 5", False),      # sign now counts: -5 is not 5
    ("Final answer: 3.5", "#### 3.5", True),          # ordinary decimals unchanged
    ("Final answer: 1,250", "#### 1250", True),
    ("Final answer: 5 - 3 = 2", "#### 2", False),     # still strict: 3 numbers -> invalid
])
def test_gsm_v6_numbers(out, gold, exp):
    assert score_gsm(out, gold)["correct"] == exp
