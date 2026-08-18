"""The CDF builder is the one place an error is both fatal and silent:
Metaculus rejects the forecast server-side and the question scores zero."""
import itertools
import random

import numpy as np
import pytest

from metaculus_bot.cdf import (build_continuous_cdf, cdf_size_from_question,
                               validator_step_bounds)

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
