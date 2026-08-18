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
