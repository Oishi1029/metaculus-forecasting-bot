"""Group ("multi-part") posts hold several sub-questions under ONE post id.

Market Pulse 26Q3 -- the only tournament paying money right now -- is composed
ENTIRELY of these, so a bot that skips them forfeits that whole tournament.
The subtlety: forecasts attach to a QUESTION but comments attach to a POST, so
a nine-part group must produce nine forecasts and exactly ONE comment.
"""
import pytest

from metaculus_bot import comment as comment_mod
from metaculus_bot.models import Forecast, parse_post, parse_post_questions

# Shaped exactly like live post 45176 (NVIDIA Q2 FY2027 guidance, 3 discrete parts).
GROUP_POST = {
    "id": 45176,
    "title": "What will be NVIDIA's forward guidance in its Q2 FY2027 earnings release?",
    "published_at": "2026-08-12T15:00:00Z",
    "group_of_questions": {
        "id": 9001,
        "description": "Group background.",
        "resolution_criteria": "Per the earnings release.",
        "fine_print": "Group fine print.",
        "group_variable": "Metric",
        "questions": [
            {"id": 45369, "type": "discrete", "status": "open",
             "label": "Operating expenses (GAAP)", "open_lower_bound": True,
             "open_upper_bound": True,
             "scaling": {"range_min": 8.45, "range_max": 10.05, "zero_point": None,
                         "inbound_outcome_count": 16},
             "my_forecasts": {"latest": {"forecast_values": None}, "history": []}},
            {"id": 45367, "type": "discrete", "status": "open", "label": "Revenue",
             "open_lower_bound": True, "open_upper_bound": True,
             "scaling": {"range_min": 99.5, "range_max": 115.5, "zero_point": None,
                         "inbound_outcome_count": 16},
             "my_forecasts": {"latest": {"forecast_values": None}, "history": []}},
            {"id": 45368, "type": "discrete", "status": "open",
             "label": "Gross margin (GAAP)", "open_lower_bound": True,
             "open_upper_bound": True,
             "scaling": {"range_min": 72.95, "range_max": 76.05, "zero_point": None,
                         "inbound_outcome_count": 31},
             "my_forecasts": {"latest": {"forecast_values": None}, "history": []}},
        ],
    },
}


def test_every_part_becomes_a_question():
    qs = parse_post_questions(GROUP_POST)
    assert len(qs) == 3
    assert {q.id_of_question for q in qs} == {45369, 45367, 45368}
    assert all(q.id_of_post == 45176 for q in qs), "all parts share ONE post id"


def test_parts_inherit_group_context_but_keep_own_scaling():
    qs = {q.group_label: q for q in parse_post_questions(GROUP_POST)}
    assert qs["Revenue"].resolution_criteria == "Per the earnings release."
    assert qs["Revenue"].fine_print == "Group fine print."
    # each part has its OWN range and cdf size -- inheriting the group's would
    # produce a wrong-length array and a server rejection
    assert qs["Revenue"].range_min == 99.5
    assert qs["Gross margin (GAAP)"].inbound_outcome_count == 31
    assert qs["Operating expenses (GAAP)"].inbound_outcome_count == 16


def test_part_title_names_its_own_subject():
    """Without the label the three parts are indistinguishable to the model."""
    q = parse_post_questions(GROUP_POST)[0]
    assert q.is_group_part
    assert "NVIDIA" in q.display_title and q.group_label in q.display_title


def test_already_forecasted_is_tracked_per_part():
    post = {**GROUP_POST, "group_of_questions": {
        **GROUP_POST["group_of_questions"],
        "questions": [
            {**GROUP_POST["group_of_questions"]["questions"][0],
             "my_forecasts": {"latest": {"forecast_values": [0.1]}, "history": [{}]}},
            GROUP_POST["group_of_questions"]["questions"][1],
        ]}}
    qs = parse_post_questions(post)
    assert [q.already_forecasted for q in qs] == [True, False]


def test_single_question_post_still_yields_one():
    plain = {"id": 1, "question": {"id": 2, "type": "binary", "status": "open",
                                   "title": "T", "my_forecasts": {}}}
    assert len(parse_post_questions(plain)) == 1
    assert parse_post(plain).id_of_question == 2


def test_notebook_and_empty_posts_yield_nothing():
    assert parse_post_questions({"id": 1, "notebook": {"x": 1}}) == []
    assert parse_post_questions({"id": 1}) == []


def test_one_comment_covers_every_part():
    """Nine comments on one page would look like spam; the rule needs one record
    displaying the forecast for each part."""
    qs = parse_post_questions(GROUP_POST)
    fcs = [Forecast(question=q, payload={"continuous_cdf": [0.0, 1.0]},
                    summary=f"median {i}", reasoning="", model_outputs=["rationale"],
                    models_used=["m1"]) for i, q in enumerate(qs)]
    text = comment_mod.render_post_comment(fcs)
    for q in qs:
        assert q.group_label in text, f"part {q.group_label!r} missing from the comment"
    assert text.count("Generated autonomously") == 1, "one provenance block, not three"
    assert "No human" in text


def test_group_comment_keeps_all_numbers_when_truncated():
    from metaculus_bot import config
    qs = parse_post_questions(GROUP_POST)
    fcs = [Forecast(question=q, payload={}, summary=f"median {i}", reasoning="",
                    model_outputs=["x" * 60000], models_used=["m1"])
           for i, q in enumerate(qs)]
    text = comment_mod.render_post_comment(fcs)
    assert len(text) <= config.COMMENT_MAX_CHARS
    for q in qs:
        assert q.group_label in text, "truncation dropped a part's forecast entirely"
