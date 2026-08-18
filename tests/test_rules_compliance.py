"""These tests guard the two rules whose violation forfeits ALL prize money.

  1. "Bot makers should only submit one forecast per question in these bot-only
     tournaments."  -> we must never forecast a question twice.
  2. "the participating bot needs to have written a comment response (including
     a display of its forecast) under each question that it is forecasting"
     -> every forecast must carry a comment.

Plus the autonomy rule: no human in the loop, anywhere.
"""
import inspect
import pathlib

import pytest

from metaculus_bot import comment as comment_mod
from metaculus_bot import config, run
from metaculus_bot.models import (BINARY, MULTIPLE_CHOICE, NUMERIC, Forecast,
                                  Question, parse_post)

REPO = pathlib.Path(__file__).resolve().parent.parent


def make_post(pid, qid, qtype=BINARY, **q):
    base = {"id": qid, "type": qtype, "title": "T", "status": "open",
            "scaling": {"range_min": 0, "range_max": 100, "zero_point": None,
                        "inbound_outcome_count": 200}}
    base.update(q)
    return {"id": pid, "question": base}


# --- rule 1: exactly one forecast per question -------------------------------
def test_already_forecasted_detected_from_latest_forecast_values():
    p = make_post(1, 2, my_forecasts={"latest": {"forecast_values": [0.3, 0.7]}, "history": []})
    assert parse_post(p).already_forecasted is True


def test_already_forecasted_detected_from_history():
    p = make_post(1, 2, my_forecasts={"latest": {}, "history": [{"x": 1}]})
    assert parse_post(p).already_forecasted is True


def test_not_forecasted_when_both_signals_are_empty():
    p = make_post(1, 2, my_forecasts={"latest": {"forecast_values": None}, "history": []})
    assert parse_post(p).already_forecasted is False


def test_missing_my_forecasts_is_treated_as_not_forecasted():
    assert parse_post(make_post(1, 2)).already_forecasted is False


def test_idempotency_is_server_derived_not_disk_derived():
    """A committed or cached file would be empty on every ephemeral runner and
    would race between concurrent runs. The source of truth must be Metaculus."""
    src = inspect.getsource(run)
    for forbidden in ("open(", "json.load", "pathlib", "shelve", "sqlite3", "pickle"):
        assert forbidden not in src, f"run.py touches local state via {forbidden!r}"


# --- rule 2: a comment on every forecast -------------------------------------
def test_publish_order_is_forecast_then_comment():
    """The forecast POST overwrites and is safe to interrupt before; the comment
    POST accumulates and is not. Reversing this risks duplicate comments."""
    src = inspect.getsource(run)
    f_idx = src.index("await client.post_forecast(")
    c_idx = src.index("await client.post_comment(w.post_id, text)")
    assert f_idx < c_idx, "forecast must be published before the comment"


def test_forecast_without_comment_is_repaired_not_reforecast():
    """If a run dies between the two publishes, the next run must post ONLY the
    missing comment -- re-forecasting would break rule 1."""
    src = inspect.getsource(run)
    # The repair path posts a comment and nothing else.
    assert "repair_comment()" in src
    start = src.index("elif w.needs_comment and not w.to_forecast:")
    branch = src[start: src.index("await client.post_comment(w.post_id, text)", start)]
    assert "post_forecast" not in branch


def test_comment_contains_the_forecast_itself():
    """The rule requires 'including a display of its forecast', so prose alone
    is not compliant."""
    q = Question(id_of_post=1, id_of_question=2, type=BINARY, title="T", url="u")
    fc = Forecast(question=q, payload={"probability_yes": 0.37},
                  summary="37.0% yes", reasoning="",
                  model_outputs=["long rationale " * 40], models_used=["m1"])
    text = comment_mod.render(fc)
    assert "37.0%" in text
    assert "| Yes |" in text
    assert "No human" in text


def test_comment_renders_for_multiple_choice_and_numeric():
    q_mc = Question(id_of_post=1, id_of_question=2, type=MULTIPLE_CHOICE, title="T",
                    url="u", options=["A", "B"])
    fc = Forecast(question=q_mc, payload={"probability_yes_per_category": {"A": 0.7, "B": 0.3}},
                  summary="A 70.0%", reasoning="", model_outputs=["r"], models_used=["m"])
    assert "| A | 70.0% |" in comment_mod.render(fc)

    q_num = Question(id_of_post=1, id_of_question=2, type=NUMERIC, title="T", url="u",
                     range_min=0, range_max=100, open_lower_bound=False,
                     open_upper_bound=False, inbound_outcome_count=200)
    cdf = [i / 200 for i in range(201)]
    fc2 = Forecast(question=q_num, payload={"continuous_cdf": cdf}, summary="median 50",
                   reasoning="", model_outputs=["r"], models_used=["m"])
    assert "P50" in comment_mod.render(fc2)


def test_comment_is_private_by_default():
    assert config.COMMENT_IS_PRIVATE is True


def test_comment_is_truncated_to_the_length_limit():
    q = Question(id_of_post=1, id_of_question=2, type=BINARY, title="T", url="u")
    fc = Forecast(question=q, payload={"probability_yes": 0.5}, summary="50.0% yes",
                  reasoning="", model_outputs=["x" * 100000], models_used=["m"])
    assert len(comment_mod.render(fc)) <= config.COMMENT_MAX_CHARS


# --- autonomy ----------------------------------------------------------------
def test_no_human_in_the_loop_anywhere_in_the_package():
    """'Bots may not have a human in the loop when forecasting.' There must be no
    code path that waits for a person."""
    banned = ("input(", "getpass", "click.confirm", "click.prompt", "sys.stdin.read")
    for path in sorted((REPO / "metaculus_bot").glob("*.py")) + [REPO / "main.py"]:
        text = path.read_text()
        for token in banned:
            assert token not in text, f"{path.name} contains an interactive call: {token}"


def test_unsupported_question_types_are_skipped_not_guessed():
    assert parse_post(make_post(1, 2, qtype="date")) is None
    assert parse_post(make_post(1, 2, qtype="conditional")) is None
    assert parse_post({"id": 1, "notebook": {}}) is None
    assert parse_post({"id": 1}) is None


def test_bounds_default_to_open_when_absent():
    q = parse_post(make_post(1, 2, qtype=NUMERIC))
    assert q.open_lower_bound is True and q.open_upper_bound is True


def test_post_id_and_question_id_are_kept_distinct():
    q = parse_post(make_post(29783, 29635))
    assert q.id_of_post == 29783 and q.id_of_question == 29635
