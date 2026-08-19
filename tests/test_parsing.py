"""A parse failure is a scoring event, not a logging event: the score sums peer
scores and squares the sum, so an unforecast question is a hard zero."""
import pytest

from metaculus_bot import aggregate
from metaculus_bot.parsing import (ParseError, parse_binary,
                                   parse_multiple_choice,
                                   parse_numeric_percentiles)

OPTS = ["NVDA", "AAPL", "MSFT"]


def test_clean_fenced_block():
    assert parse_binary('blah\n```json\n{"question_type":"binary","posterior_prob":0.28}\n```') == 0.28


def test_takes_the_last_block_when_the_model_shows_its_working():
    text = ('```json\n{"posterior_prob": 0.9}\n```\nOn reflection:\n'
            '```json\n{"question_type":"binary","posterior_prob":0.31}\n```')
    assert parse_binary(text) == 0.31


@pytest.mark.parametrize("bad,expected", [
    ('```json\n{"posterior_prob": 0.28,}\n```', 0.28),            # trailing comma
    ("```json\n{'posterior_prob': 0.28}\n```", 0.28),             # single quotes
    ('```json\n{"posterior_prob": 28%}\n```', 0.28),              # percent sign
    ('```\n{"posterior_prob": 0.28} // final\n```', 0.28),        # comment
    ('{"question_type":"binary","posterior_prob":0.28}', 0.28),   # unfenced
])
def test_repairs_the_malformations_llms_actually_emit(bad, expected):
    assert parse_binary(bad) == pytest.approx(expected)


def test_percentage_form_is_rescaled():
    assert parse_binary('```json\n{"posterior_prob": 35}\n```') == pytest.approx(0.35)


def test_falls_back_to_prose_when_json_is_absent():
    assert parse_binary("After weighing it up, my probability is 42%.") == pytest.approx(0.42)


def test_raises_when_nothing_is_recoverable():
    with pytest.raises(ParseError):
        parse_binary("I decline to give a number.")


def test_multiple_choice_canonicalises_option_names():
    """The server matches option keys EXACTLY and the SDK validates nothing, so a
    mis-cased key is a silent server-side rejection."""
    out = parse_multiple_choice(
        '```json\n{"option_probabilities": {"nvda": 0.5, "aapl": 0.3, "msft": 0.2}}\n```', OPTS)
    assert set(out) == set(OPTS)
    assert out["NVDA"] == 0.5


def test_multiple_choice_accepts_a_bare_list_in_option_order():
    out = parse_multiple_choice('```json\n{"probabilities": [0.5, 0.3, 0.2]}\n```', OPTS)
    assert out["MSFT"] == pytest.approx(0.2)


def test_multiple_choice_fills_an_omitted_option_rather_than_failing():
    out = parse_multiple_choice('```json\n{"option_probabilities": {"NVDA":0.6,"AAPL":0.4}}\n```', OPTS)
    assert set(out) == set(OPTS)
    assert out["MSFT"] == 0.0


def test_percentiles_accept_both_key_styles():
    a = parse_numeric_percentiles('```json\n{"declared_percentiles":{"0.1":5,"0.5":10,"0.9":20}}\n```', [])
    b = parse_numeric_percentiles('```json\n{"declared_percentiles":{"10":5,"50":10,"90":20}}\n```', [])
    assert a == b == {10.0: 5.0, 50.0: 10.0, 90.0: 20.0}


# --- aggregation ------------------------------------------------------------
def test_binary_uses_median_so_one_outlier_cannot_dominate():
    assert aggregate.aggregate_binary([0.30, 0.33, 0.95]) == pytest.approx(0.33)


def test_binary_is_clipped_not_shrunk():
    """Clipping protects against the catastrophic-99% failure mode without
    dragging well-justified confident forecasts toward 0.5."""
    assert aggregate.aggregate_binary([0.999, 0.999, 0.999]) == pytest.approx(0.98)
    assert aggregate.aggregate_binary([0.0, 0.0, 0.0]) == pytest.approx(0.02)
    assert aggregate.aggregate_binary([0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_multiple_choice_sums_to_one_and_respects_clamps():
    out = aggregate.aggregate_multiple_choice(
        [{"NVDA": 0.999, "AAPL": 0.001, "MSFT": 0.0}] * 3, OPTS)
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(0.001 <= v <= 0.99 for v in out.values())


def test_multiple_choice_handles_an_all_zero_input():
    out = aggregate.aggregate_multiple_choice([{o: 0.0 for o in OPTS}], OPTS)
    assert sum(out.values()) == pytest.approx(1.0)


def test_percentile_aggregation_repairs_non_monotonicity():
    merged = aggregate.aggregate_percentiles([{10: 5, 50: 4, 90: 20}])
    keys = sorted(merged)
    vals = [merged[k] for k in keys]
    assert all(b > a for a, b in zip(vals, vals[1:])), "percentiles must be strictly increasing"


def test_P1_survives_a_0_to_100_percentile_dict():
    """Scale must be decided for the whole dict. Deciding per key destroyed P1:
    the key "1" looks like 0-1 style, was scaled to 100.0, then dropped for not
    being < 100 -- losing the open-tail anchor on most continuous questions."""
    out = parse_numeric_percentiles(
        '```json\n{"declared_percentiles":{"1":5,"10":8,"50":20,"90":40,"99":95}}\n```', [])
    assert 1.0 in out, "P1 was silently dropped"
    assert out[1.0] == 5.0
    assert out[99.0] == 95.0


def test_P1_survives_a_0_to_1_percentile_dict():
    out = parse_numeric_percentiles(
        '```json\n{"declared_percentiles":{"0.01":5,"0.1":8,"0.5":20,"0.99":95}}\n```', [])
    assert out[1.0] == 5.0 and out[99.0] == 95.0


# --- ratio prose must never become a probability (S7) ------------------------
@pytest.mark.parametrize("prose", ['"1 in 5"', '"roughly 1 chance in 20"',
                                   '"about 7 in 10"', '"1/5"', '"1 out of 20"'])
def test_ratio_prose_is_rejected_not_misread(prose):
    """Grabbing the first number turned "1 in 5" into 1.0, which the [0.02,0.98]
    clamp then passed through as 0.98 -- a 20% belief published as 98%. The clamp
    cannot catch it because 0.98 is inside the band."""
    text = '```json\n{"question_type":"binary","posterior_prob":%s}\n```' % prose
    try:
        got = parse_binary(text)
    except ParseError:
        return                      # rejected outright: correct
    assert got < 0.9, f"{prose} misread as {got}"


def test_plain_numeric_strings_still_parse():
    assert parse_binary('```json\n{"posterior_prob":"0.35"}\n```') == pytest.approx(0.35)
    assert parse_binary('```json\n{"posterior_prob":"35%"}\n```') == pytest.approx(0.35)


# --- an even ensemble must not readmit the outlier (S6) ----------------------
def test_even_ensemble_uses_median_low():
    """statistics.median averages the two middle values, handing an outlier half
    the weight -- precisely what the median was chosen to prevent."""
    assert aggregate.aggregate_binary([0.90, 0.03]) == pytest.approx(0.03)
    assert aggregate.aggregate_binary([0.90, 0.85, 0.03]) == pytest.approx(0.85)


def test_even_multiple_choice_uses_median_low():
    out = aggregate.aggregate_multiple_choice(
        [{"A": 0.9, "B": 0.1}, {"A": 0.1, "B": 0.9}], ["A", "B"])
    assert sum(out.values()) == pytest.approx(1.0)


def test_median_of_zero_is_not_treated_as_missing():
    """`x or fallback` swallows a legitimate median of 0."""
    merged = aggregate.aggregate_percentiles([{10: -5.0, 50: 0.0, 90: 5.0}])
    assert merged[50.0] == 0.0
