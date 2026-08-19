"""The prompts embed literal JSON examples, so str.format() cannot be used on
them -- it treats every example brace as a field. These tests lock in the
substitution scheme and catch any placeholder we forget to fill."""
import pytest

from metaculus_bot import prompts
from metaculus_bot.forecaster import Forecaster
from metaculus_bot.models import (BINARY, DISCRETE, MULTIPLE_CHOICE, NUMERIC,
                                  Question)


class _Bare(Forecaster):
    def __init__(self):
        pass


def q_of(qtype):
    return Question(
        id_of_post=1, id_of_question=2, type=qtype, title="Will X happen?", url="u",
        resolution_criteria="Resolves YES if X.", fine_print="", background="B",
        options=["A", "B"] if qtype == MULTIPLE_CHOICE else [],
        range_min=0, range_max=100, inbound_outcome_count=200,
        open_lower_bound=False, open_upper_bound=True,
        open_time="2026-08-01T00:00:00Z", resolve_time="2026-12-31T00:00:00Z",
    )


@pytest.mark.parametrize("qtype", [BINARY, MULTIPLE_CHOICE, NUMERIC, DISCRETE])
def test_no_unresolved_placeholders(qtype):
    text = _Bare()._build_prompt(q_of(qtype), "research text")
    assert prompts.unresolved_tokens(text) == []


@pytest.mark.parametrize("qtype", [BINARY, MULTIPLE_CHOICE, NUMERIC, DISCRETE])
def test_shared_blocks_are_inlined(qtype):
    text = _Bare()._build_prompt(q_of(qtype), "research text")
    assert "SHARED BLOCK" not in text
    assert "Forecasting window:" in text          # block A
    assert "Separate facts from opinions" in text  # block B
    assert "Prediction markets are strong evidence" in text  # block C


@pytest.mark.parametrize("qtype", [BINARY, MULTIPLE_CHOICE, NUMERIC, DISCRETE])
def test_json_examples_survive_intact(qtype):
    """If format() were ever reintroduced this would raise or mangle the example."""
    text = _Bare()._build_prompt(q_of(qtype), "r")
    assert '"question_type"' in text


def test_market_block_is_type_specific():
    assert "your percentiles should center on it" in prompts.market_block("numeric")
    assert "should anchor your forecast" in prompts.market_block("binary")
    assert "should anchor your distribution" in prompts.market_block("multiple_choice")


def test_window_anchor_reports_real_day_counts():
    text = _Bare()._build_prompt(q_of(BINARY), "r")
    assert "days ago" in text and "days from now" in text
    assert "{elapsed_days}" not in text


def test_open_bound_message_explains_the_mechanism():
    """Piling percentiles on an open boundary is the top numeric failure mode."""
    text = _Bare()._build_prompt(q_of(NUMERIC), "r")
    assert "not a hard limit" in text
    assert "Do not pile percentiles at the boundary" in text


def test_options_are_listed_for_multiple_choice():
    text = _Bare()._build_prompt(q_of(MULTIPLE_CHOICE), "r")
    assert "- A" in text and "- B" in text


def test_render_leaves_unknown_braces_alone():
    out = prompts.render('keep {"json": 1} and fill {name}', name="X")
    assert out == 'keep {"json": 1} and fill X'


# --- the prompt and the parser must agree on the key name --------------------
# Caught live on 2026-08-18: MULTIPLE_CHOICE_PROMPT demanded "option_probs" but
# the parser only looked for "option_probabilities". Every multiple-choice
# question silently paid for a salvage LLM call it did not need. Nothing failed
# loudly, so only a real run exposed it. These tests close that gap.
import json
import re

from metaculus_bot import parsing


def _demanded_keys(prompt_text: str) -> set[str]:
    i = prompt_text.find("STRUCTURED FORECAST")
    block = re.search(r"```json\s*(\{.*?\})\s*```", prompt_text[i:], re.DOTALL)
    assert block, "prompt has no fenced json example after STRUCTURED FORECAST"
    return set(re.findall(r'"([a-z_]+)"\s*:', block.group(1)))


def test_binary_prompt_key_is_one_the_parser_reads():
    keys = _demanded_keys(_Bare()._build_prompt(q_of(BINARY), "r"))
    assert "posterior_prob" in keys
    assert parsing.parse_binary('```json\n{"posterior_prob": 0.4}\n```') == 0.4


def test_multiple_choice_prompt_key_is_one_the_parser_reads():
    text = _Bare()._build_prompt(q_of(MULTIPLE_CHOICE), "r")
    keys = _demanded_keys(text)
    demanded = keys - {"question_type"}
    payload = json.dumps({"question_type": "multiple_choice",
                          next(iter(demanded)): {"A": 0.6, "B": 0.4}})
    out = parsing.parse_multiple_choice(f"```json\n{payload}\n```", ["A", "B"])
    assert out == {"A": 0.6, "B": 0.4}


def test_numeric_prompt_key_is_one_the_parser_reads():
    keys = _demanded_keys(_Bare()._build_prompt(q_of(NUMERIC), "r"))
    assert "declared_percentiles" in keys
    got = parsing.parse_numeric_percentiles(
        '```json\n{"declared_percentiles":{"0.1":1,"0.5":2,"0.9":3}}\n```', [])
    assert got == {10.0: 1.0, 50.0: 2.0, 90.0: 3.0}


def test_mc_example_carries_the_real_option_names():
    """With a placeholder the model invents its own keys and the forecast cannot
    be bound back to the allowed options."""
    text = _Bare()._build_prompt(q_of(MULTIPLE_CHOICE), "r")
    i = text.find("STRUCTURED FORECAST")
    block = re.search(r"```json\s*(\{.*?\})\s*```", text[i:], re.DOTALL).group(1)
    assert '"A"' in block and '"B"' in block
    assert "REAL option names" not in block


def test_no_build_notes_leak_to_the_model():
    for qtype in (BINARY, MULTIPLE_CHOICE, NUMERIC, DISCRETE):
        text = _Bare()._build_prompt(q_of(qtype), "r")
        assert "BUILD NOTE" not in text
        assert not re.search(r"<[A-Z][^>]{10,90}>", text), "placeholder leaked into the prompt"


def test_token_ceiling_leaves_room_for_the_report_and_reasoning():
    """Measured 2026-08-19: a 4000-token ceiling truncated all three competition
    models before the JSON block, making 18 of 24 outputs unparseable. Reasoning
    models spend the same budget on hidden reasoning, so the ceiling must be
    generous. Raising it costs nothing unless used; truncation costs twice."""
    from metaculus_bot import config
    assert config.LLM_MAX_TOKENS >= 12000, (
        "output ceiling too low: the forecasting prompts request a long "
        "structured report and the JSON block comes last")


def test_numeric_prompt_ships_exactly_ONE_bound_instruction_per_axis():
    """The numeric template used to end with an authoring-notes appendix listing
    OPEN UPPER / OPEN LOWER / CLOSED UPPER / CLOSED LOWER, all four with the real
    bounds substituted in. The model therefore read four authoritative,
    mutually contradictory instructions as the last thing in the prompt --
    on a fully closed question, both 'your P50 must be BELOW x' and 'the outcome
    can not be lower than x'. It corrupted exactly the open/closed-bound axis
    that the CDF tail logic exists to get right."""
    text = _Bare()._build_prompt(q_of(NUMERIC), "r")
    assert "BOUND MESSAGES" not in text
    # q_of(NUMERIC) is closed-lower, open-upper
    assert text.count("The lower bound is closed") == 1
    assert text.count("The lower bound is open") == 0
    assert text.count("The upper bound is open") == 1
    assert text.count("The upper bound is closed") == 0


def test_numeric_prompt_ends_with_the_json_contract():
    """Nothing may follow the 'write nothing after it' instruction."""
    text = _Bare()._build_prompt(q_of(NUMERIC), "r")
    tail = text.rstrip()[-400:]
    assert "json" in tail.lower()


# --- number formatting (S4/S5) ----------------------------------------------
def test_bounds_are_never_scientific_or_rounded():
    """`.4g` produced '4.8e+04' from 48000 and CHANGED 1359.5 into 1,360 -- into a
    prompt whose own Units section forbids scientific notation, and into the
    published comment. A rounded bound is a wrong bound."""
    from metaculus_bot.forecaster import _fmt
    assert _fmt(48000.0) == "48,000"
    assert _fmt(7000000.0) == "7,000,000"
    assert _fmt(240500.0) == "240,500"
    assert _fmt(1359.5) == "1,359.5"
    assert _fmt(2462.5) == "2,462.5"
    assert _fmt(-0.105) == "-0.105"
    for v in (48000.0, 7000000.0, 240500.0, 1359.5, 1e10):
        assert "e" not in _fmt(v).lower()


def test_nominal_bounds_are_used_for_elicitation():
    """The CDF grid runs half a step outside the real answer space. Telling the
    model the number of jurisdictions could be 5.5 is nonsense."""
    from metaculus_bot.models import parse_post_questions
    post = {"id": 1, "question": {
        "id": 2, "type": "discrete", "status": "open", "title": "How many?",
        "open_lower_bound": False, "open_upper_bound": False,
        "scaling": {"range_min": 5.5, "range_max": 13.5, "zero_point": None,
                    "nominal_min": 6, "nominal_max": 13, "inbound_outcome_count": 8}}}
    q = parse_post_questions(post)[0]
    assert q.nominal_min == 6 and q.nominal_max == 13
    assert q.range_min == 5.5 and q.range_max == 13.5   # grid unchanged
    text = _Bare()._build_prompt(q, "r")
    assert "5.5" not in text and "13.5" not in text


def test_nominal_bounds_fall_back_to_range_when_absent():
    from metaculus_bot.models import parse_post_questions
    post = {"id": 1, "question": {
        "id": 2, "type": "numeric", "status": "open", "title": "How much?",
        "scaling": {"range_min": 0, "range_max": 100, "zero_point": None,
                    "inbound_outcome_count": 200}}}
    q = parse_post_questions(post)[0]
    assert q.nominal_min == 0 and q.nominal_max == 100


# --- title-only questions (S8) ----------------------------------------------
def test_title_only_question_gets_an_explicit_instruction():
    """VERIFIED against the last settled MiniBench round: 57/57 questions had
    EMPTY resolution_criteria, description and fine_print. Passing the literal
    word "none" invites the model to invent criteria, and an invented stricter
    reading biases every forecast one way."""
    q = q_of(BINARY)
    q.resolution_criteria = ""
    q.background = ""
    text = _Bare()._build_prompt(q, "research")
    assert "COMPLETE and ONLY specification" in text
    assert "Do NOT invent extra conditions" in text
    assert "(none given)" not in text


def test_question_with_real_criteria_is_unchanged():
    q = q_of(BINARY)
    q.resolution_criteria = "Resolves YES if the CPI print exceeds 3.0%."
    text = _Bare()._build_prompt(q, "r")
    assert "Resolves YES if the CPI print exceeds 3.0%." in text
    assert "COMPLETE and ONLY specification" not in text
