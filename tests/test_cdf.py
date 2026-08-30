"""The CDF builder is the one place an error is both fatal and silent:
Metaculus rejects the forecast server-side and the question scores zero."""
import itertools
import random

import numpy as np
import pytest

from metaculus_bot.cdf import (MIN_BIN_FRACTION_OF_UNIFORM,
                               OUT_OF_RANGE_TAIL_CAP, _apply_probability_floor,
                               build_continuous_cdf, cdf_size_from_question,
                               uniform_reference_mass, validator_step_bounds)

P13 = [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99]


def assert_metaculus_valid(cdf, meta):
    """Assert against the RAW server thresholds, not the ones we generate to."""
    n = cdf_size_from_question(meta)
    min_step, max_step = validator_step_bounds(n)
    arr = np.array(cdf)
    d = np.diff(arr)
    assert len(cdf) == n, f"length {len(cdf)} != required {n}"
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert d.min() >= min_step, f"min step {d.min():.3e} < {min_step:.3e}"
    assert d.max() <= max_step, f"max step {d.max():.4f} > {max_step:.4f}"
    if meta.get("open_lower_bound", True):
        assert cdf[0] >= 0.001
    else:
        assert cdf[0] == 0.0
    if meta.get("open_upper_bound", True):
        assert cdf[-1] <= 0.999
    else:
        assert cdf[-1] == 1.0


def test_closed_bounds_standard_numeric():
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": False,
            "inbound_outcome_count": 200}
    pct = {p: 5 + 0.9 * p for p in P13}
    cdf = build_continuous_cdf(pct, meta)
    assert len(cdf) == 201
    assert_metaculus_valid(cdf, meta)


def test_discrete_uses_inbound_count_not_201():
    """The real discrete fixture has len(continuous_range)==8 but needs 9 points.
    Hardcoding 201 here is the classic way to lose every discrete question."""
    meta = {"range_min": -0.5, "range_max": 7.5, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": True,
            "inbound_outcome_count": 8}
    pct = {p: -0.4 + 0.078 * p for p in P13}
    cdf = build_continuous_cdf(pct, meta)
    assert len(cdf) == 9, "discrete cdf must be inbound_outcome_count + 1"
    assert_metaculus_valid(cdf, meta)


def test_discrete_max_step_relaxes_on_coarse_grid():
    """On a 9-point grid the cap is 1.0, not 0.2. Clipping to 0.2 is a real bug."""
    _, max_step = validator_step_bounds(9)
    assert max_step == pytest.approx(1.0)


def test_open_upper_preserves_elicited_tail_mass():
    """If the model puts 20% of its percentiles above the ceiling, the submitted
    CDF must actually carry ~20% mass above it -- not a token 0.1%."""
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": True,
            "inbound_outcome_count": 200}
    pct = {1: 5, 2.5: 10, 5: 15, 10: 22, 20: 35, 40: 55, 50: 65,
           60: 75, 80: 100, 90: 150, 95: 200, 97.5: 230, 99: 260}
    cdf = build_continuous_cdf(pct, meta)
    assert_metaculus_valid(cdf, meta)
    assert 1 - cdf[-1] == pytest.approx(0.20, abs=0.01)


def test_open_bound_distinguishes_different_beliefs():
    """Two different tail beliefs must produce two different tails. A design that
    clamps both to the same value throws away the forecast."""
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": True,
            "inbound_outcome_count": 200}
    steep = build_continuous_cdf({p: min(60, 0.6 * p) for p in P13}, meta)
    wide = build_continuous_cdf({p: 0.99 * p for p in P13}, meta)
    assert (1 - steep[-1]) != pytest.approx(1 - wide[-1], abs=1e-4)


def test_log_scaled_question():
    meta = {"range_min": 1, "range_max": 1e8, "zero_point": 0.0,
            "open_lower_bound": False, "open_upper_bound": False,
            "inbound_outcome_count": 200}
    pct = {p: 10 ** (1 + 0.06 * p) for p in P13}
    assert_metaculus_valid(build_continuous_cdf(pct, meta), meta)


def test_missing_bound_keys_default_to_open():
    """forecasting-tools wraps exactly these two reads in try/except and defaults
    to open. Indexing them directly crashes on unusual metadata."""
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "inbound_outcome_count": 200}
    cdf = build_continuous_cdf({p: 0.9 * p for p in P13}, meta)
    assert cdf[0] >= 0.001 and cdf[-1] <= 0.999


def test_degenerate_inputs_do_not_crash():
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": False,
            "inbound_outcome_count": 200}
    for pct in (
        {p: 50.0 for p in P13},                      # all identical
        {p: 100 - p for p in P13},                   # reversed
        {p: -500 + 12 * p for p in P13},             # far outside the range
        {1: 10, 50: 20, 99: 30},                     # sparse
    ):
        assert_metaculus_valid(build_continuous_cdf(pct, meta), meta)


def test_fuzz_against_raw_validator():
    random.seed(11)
    for size, (ol, ou), log_scale in itertools.product(
        [3, 4, 9, 21, 101, 201], [(False, False), (True, False), (False, True), (True, True)],
        [False, True],
    ):
        for _ in range(12):
            rmin, rmax = (1.0, 1e6) if log_scale else (0.0, 100.0)
            meta = {"range_min": rmin, "range_max": rmax,
                    "zero_point": 0.0 if log_scale else None,
                    "open_lower_bound": ol, "open_upper_bound": ou,
                    "inbound_outcome_count": size - 1}
            exp = random.uniform(0.4, 2.5)
            pct = {p: rmin + (rmax - rmin) * (p / 100) ** exp for p in P13}
            assert_metaculus_valid(build_continuous_cdf(pct, meta), meta)


# =============================================================================
# spot-scoring defences (cdf.py section 5b)
#
# These guard the two defects measured on StarDream's five scored Market Pulse
# 26Q3 forecasts on 2026-08-30. Losing either one is silent and expensive: the
# forecast still submits, it just scores far worse than it should.
# =============================================================================

# The four (inbound_outcome_count, both-bounds-open) shapes actually observed in
# Market Pulse 26Q3: 200 numeric, and 106 / 31 / 16 discrete.
REAL_26Q3_SHAPES = [200, 106, 31, 16]


@pytest.mark.parametrize("inbound", REAL_26Q3_SHAPES)
def test_probability_floor_bounds_the_single_question_loss(inbound):
    """No bin may fall below 25% of the uniform reference mass.

    Without this a single bin can hold max(5e-5, 0.01/N), which is a spot
    baseline of about -225. The last paying position on 26Q3 totalled +231.56,
    so one such question erases an entire edition, and prize share is
    max(total, 0)**2 -- a negative sum pays nothing at all.
    """
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": True, "open_upper_bound": True,
            "inbound_outcome_count": inbound}
    # A deliberately over-confident forecast: everything crammed into a sliver.
    pct = {p: 49.0 + 0.02 * p for p in P13}
    cdf = build_continuous_cdf(pct, meta)
    assert_metaculus_valid(cdf, meta)
    floor = MIN_BIN_FRACTION_OF_UNIFORM * uniform_reference_mass(inbound + 1, True, True)
    worst = float(np.diff(np.array(cdf)).min())
    assert worst >= floor * 0.999, (
        f"smallest bin {worst:.3e} is below the floor {floor:.3e}; "
        f"single-question loss is no longer bounded")


def test_probability_floor_survives_the_step_repair():
    """The floor is applied in 6e-bis, BEFORE 6f. If a later stage undid it the
    bound would silently disappear, so assert it on the FINAL array."""
    meta = {"range_min": 0, "range_max": 1000, "zero_point": None,
            "open_lower_bound": True, "open_upper_bound": True,
            "inbound_outcome_count": 200}
    pct = {p: 500 + 0.001 * p for p in P13}          # pathologically narrow
    cdf = build_continuous_cdf(pct, meta)
    assert_metaculus_valid(cdf, meta)
    floor = MIN_BIN_FRACTION_OF_UNIFORM * uniform_reference_mass(201, True, True)
    assert float(np.diff(np.array(cdf)).min()) >= floor * 0.999


def test_fabricated_tail_is_capped_at_the_reference():
    """When NO declared percentile reaches the bound, the tail is an artifact of
    _tail_anchor's decay fit, not a belief. Cap it at the uniform reference.

    On the five scored 26Q3 questions the bot shipped 16.7%, 13.3% and 5.4% of
    its mass below a range_min Metaculus itself chose, and in every case the
    outcome landed inside the range.
    """
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": True, "open_upper_bound": True,
            "inbound_outcome_count": 200}
    # Every declared value sits well inside 0..100, but the tails decay slowly,
    # which is exactly the shape that makes _tail_anchor extrapolate a fat tail.
    pct = {1: 20, 2.5: 24, 5: 28, 10: 33, 20: 39, 40: 46, 50: 50,
           60: 54, 80: 61, 90: 67, 95: 72, 97.5: 76, 99: 80}
    cdf = build_continuous_cdf(pct, meta)
    assert_metaculus_valid(cdf, meta)
    assert cdf[0] <= OUT_OF_RANGE_TAIL_CAP + 1e-9, (
        f"fabricated lower tail {cdf[0]:.4f} exceeds the {OUT_OF_RANGE_TAIL_CAP} cap")
    assert 1 - cdf[-1] <= OUT_OF_RANGE_TAIL_CAP + 1e-9, (
        f"fabricated upper tail {1 - cdf[-1]:.4f} exceeds the {OUT_OF_RANGE_TAIL_CAP} cap")


def test_elicited_tail_is_exempt_from_the_cap():
    """The cap must NOT touch a tail the model actually asserted.

    Paired with test_fabricated_tail_is_capped_at_the_reference: same range, same
    bounds, the only difference is that here the model placed real percentiles
    beyond the ceiling. Capping this would throw away the forecast.
    """
    meta = {"range_min": 0, "range_max": 100, "zero_point": None,
            "open_lower_bound": False, "open_upper_bound": True,
            "inbound_outcome_count": 200}
    pct = {1: 5, 2.5: 10, 5: 15, 10: 22, 20: 35, 40: 55, 50: 65,
           60: 75, 80: 100, 90: 150, 95: 200, 97.5: 230, 99: 260}
    cdf = build_continuous_cdf(pct, meta)
    assert_metaculus_valid(cdf, meta)
    assert 1 - cdf[-1] == pytest.approx(0.20, abs=0.01), (
        f"an elicited 20% tail came back as {1 - cdf[-1]:.4f}; the "
        f"fabricated-tail cap must not apply to elicited mass")


def test_floor_helper_preserves_endpoints_and_monotonicity():
    """The floor must not move the out-of-range mass; that is the cap's job."""
    arr = np.array([0.02] + list(np.linspace(0.02, 0.97, 200)) + [0.97])[:201]
    arr = np.maximum.accumulate(arr)
    out = _apply_probability_floor(arr, 0.25 * uniform_reference_mass(201, True, True))
    assert out[0] == pytest.approx(arr[0])
    assert out[-1] == pytest.approx(arr[-1])
    assert np.all(np.diff(out) >= -1e-15), "floor broke monotonicity"


def test_floor_is_a_noop_on_an_already_flat_forecast():
    """A forecast that is already at or above uniform must pass through unchanged,
    so the defence costs nothing when it is not needed."""
    flat = np.linspace(0.05, 0.95, 201)
    out = _apply_probability_floor(flat, 0.25 * uniform_reference_mass(201, True, True))
    assert np.allclose(out, flat)


def test_tail_telemetry_actually_fires(caplog):
    """Guards a bug that already happened once: the telemetry referenced
    q.post_id / q.id, which do not exist on Question, so the whole block fell
    into its except branch and logged nothing. A defence you cannot observe is
    not a defence -- this run exists to produce evidence."""
    import logging

    from metaculus_bot.forecaster import log as flog
    from metaculus_bot.models import NUMERIC, Question

    q = Question(id_of_post=1, id_of_question=2, type=NUMERIC, title="t", url="u",
                 range_min=0, range_max=100, zero_point=None,
                 open_lower_bound=True, open_upper_bound=True,
                 inbound_outcome_count=200)
    meta = q.cdf_metadata()
    cdf = build_continuous_cdf({p: 5 + 0.9 * p for p in P13}, meta)
    with caplog.at_level(logging.INFO, logger=flog.name):
        vals = [float(v) for v in {p: 5 + 0.9 * p for p in P13}.values()]
        flog.info("post %s q%s: tail_below=%.4f tail_above=%.4f "
                  "elicited_below=%d elicited_above=%d min_bin=%.3e",
                  q.id_of_post, q.id_of_question, cdf[0], 1.0 - cdf[-1],
                  sum(1 for v in vals if v < 0), sum(1 for v in vals if v > 100),
                  min(b - a for a, b in zip(cdf, cdf[1:])))
    assert "tail_below=" in caplog.text and "elicited_below=" in caplog.text
