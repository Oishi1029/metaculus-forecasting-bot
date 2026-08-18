"""Build a Metaculus-valid ``continuous_cdf`` from LLM percentile estimates.

Only dependency is numpy. Monotone-cubic (PCHIP) interpolation is implemented
inline so scipy is not required.

WHY EACH RULE EXISTS  (every constant below is read off Metaculus's own code)

  length          cdf_size = question.scaling.inbound_outcome_count + 1
                  Continuous questions report 200 -> 201 points. Discrete
                  questions report a small count (e.g. 8 -> 9 points).
                  forecasting-tools questions.py:448-458.

  endpoints       cdf[0]  = P(outcome <  range_min)   <- STRICTLY less than
                  cdf[-1] = P(outcome <= range_max)   <- less than OR EQUAL
                  That asymmetry is stated in forecasting-tools
                  numeric_report.py:357-365. Closed bound => the endpoint is
                  exactly 0.0 / 1.0. Open bound => 0.001 floor / 0.999 ceiling,
                  which fall out of the four affine forms in
                  numeric_report.py:558-569 evaluated at location 0 and 1.

  min step        Every adjacent pair must rise by at least 0.01/N (floored at
                  5e-5), N = cdf_size - 1. numeric_report.py:131-141 raises
                  below 5e-5; metaculus-bot numeric/config.py:71 generalises it
                  to 0.01/N for coarse grids.

  max step        No single bin may hold more than 0.2 * 200 / N probability.
                  That is the number the SERVER-SIDE VALIDATOR uses
                  (numeric_report.py:244 passes include_wiggle_room=False).
                  Metaculus's own generator targets 95% of it
                  (numeric_report.py:577 takes the default wiggle_room=True
                  branch, numeric_report.py:41-42 returns normal_cap * 0.95).
                  WE GENERATE AT 0.95 TOO. Targeting the bare cap leaves zero
                  float headroom and overshoots it ~15% of the time on
                  concentrated distributions.

  out-of-range    There is NO separate field for mass outside the displayed
  mass            range. cdf[0] IS the mass below range_min and 1 - cdf[-1] IS
                  the mass above range_max. See section 5 below -- this is the
                  single easiest place to silently ship a wrong forecast.
"""

from __future__ import annotations

import numpy as np

# --- Constants, each traceable to a line of Metaculus source -----------------
DEFAULT_CDF_SIZE = 201          # numeric_report.py:25   NumericDefaults.DEFAULT_CDF_SIZE
MIN_PROB_STEP_FLOOR = 5e-5      # numeric_report.py:134  hard raise below this
MAX_PMF_BASE = 0.2              # numeric_report.py:29   MAX_NUMERIC_PMF_VALUE
MAX_PMF_WIGGLE = 0.95           # numeric_report.py:42   generator targets 95% of the cap
DEFAULT_INBOUND_COUNT = 200     # numeric_report.py:28   DEFAULT_INBOUND_OUTCOME_COUNT
MIN_PMF_HEADROOM = 1.05         # we GENERATE 5% above the min-step threshold
OPEN_BOUND_MIN_TAIL = 0.001     # cdf[0]  >= 0.001 when open_lower_bound
OPEN_BOUND_MAX_HEAD = 0.999     # cdf[-1] <= 0.999 when open_upper_bound
_ALPHA_SAFETY_MARGIN = 1.1      # metaculus-bot numeric/pchip_cdf.py:342-350


# =============================================================================
# 1. cdf length + per-bin step constraints
# =============================================================================
def cdf_size_from_question(meta: dict) -> int:
    """Length of the ``continuous_cdf`` array Metaculus expects for this question.

    ``scaling.inbound_outcome_count`` counts BINS; the CDF has one more point.
    Never hardcode 201, and never derive the length from
    ``scaling.continuous_range`` -- on the real discrete fixture (post 38880)
    ``len(continuous_range) == 8`` while the required cdf length is 9.
    """
    explicit = meta.get("cdf_size")
    if explicit:
        return int(explicit)
    outcome_count = meta.get("inbound_outcome_count")
    if outcome_count is None:            # questions.py:451-452 coerces None -> 200
        outcome_count = DEFAULT_INBOUND_COUNT
    return int(outcome_count) + 1


def validator_step_bounds(cdf_size: int) -> tuple[float, float]:
    """(min_step, max_step) the SERVER checks against -- the raw thresholds.

    min = max(5e-5, 0.01 / N) -- no rounding. metaculus-bot numeric/config.py:71
        is ``max(MIN_CDF_PROB_STEP, 0.01 / inbound)``. Its prose (and AGENTS.md)
        says ``round(0.01/N, 9)`` but the CODE does not round, and rounding can
        land a hair BELOW the threshold (cdf_size=8 rounds to 0.001428571,
        4.3e-10 under 0.01/7). Follow the code, not the prose.

    max = min(1.0, 0.2 * 200 / N) -- numeric_report.py:244 checks this
        un-reduced value with a bare ``>``. Note the cap RELAXES on coarse
        grids: at cdf_size=9 it is 1.0, not 0.2. Clipping a 9-point discrete
        grid to 0.2 is a known production bug
        (metaculus-bot tests/test_discrete_max_step.py:1-13).
    """
    inbound = max(1, cdf_size - 1)
    return (
        max(MIN_PROB_STEP_FLOOR, 0.01 / inbound),
        min(1.0, MAX_PMF_BASE * DEFAULT_INBOUND_COUNT / inbound),
    )


def grid_step_constraints(cdf_size: int) -> tuple[float, float]:
    """(min_step, max_step) we GENERATE to -- both pulled INSIDE the thresholds.

    Both server checks are bare inequalities with no tolerance, so targeting a
    threshold exactly leaves zero float headroom and trips it intermittently.
    Metaculus's own generator already does this on the max side
    (numeric_report.py:577 takes the wiggle-room branch, :41-42 returns
    ``normal_cap * 0.95``); we do the mirror image on the min side, because
    ``_enforce_min_steps`` emits steps of exactly ``min_step`` in flat regions
    and the final ``np.round(cdf, 10)`` can nudge them under.
    """
    raw_min, raw_max = validator_step_bounds(cdf_size)
    return raw_min * MIN_PMF_HEADROOM, raw_max * MAX_PMF_WIGGLE


# =============================================================================
# 2. the scaling transform (question value <-> cdf x-location in [0, 1])
# =============================================================================
def value_to_cdf_location(value, range_min: float, range_max: float,
                          zero_point: float | None):
    """Real-world value -> position on the CDF x-axis (0 = range_min, 1 = range_max).

    Goes below 0 / above 1 for values outside the displayed range -- that is
    intended and is how out-of-range percentiles are honoured.
    Mirrors forecasting-tools numeric_report.py:473-500.
    """
    value = np.asarray(value, dtype=float)
    if zero_point is None:
        return (value - range_min) / (range_max - range_min)
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    value = np.where(value == zero_point, value + 1e-10, value)
    return (
        np.log((value - range_min) * (deriv_ratio - 1.0) + (range_max - range_min))
        - np.log(range_max - range_min)
    ) / np.log(deriv_ratio)


def cdf_location_to_value(location, range_min: float, range_max: float,
                          zero_point: float | None):
    """Inverse of :func:`value_to_cdf_location`. Mirrors numeric_report.py:611-625."""
    location = np.asarray(location, dtype=float)
    if zero_point is None:
        return range_min + (range_max - range_min) * location
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    if abs(deriv_ratio - 1.0) < 1e-10:
        return range_min + (range_max - range_min) * location
    return range_min + (range_max - range_min) * (
        deriv_ratio**location - 1.0
    ) / (deriv_ratio - 1.0)


# =============================================================================
# 3. monotone cubic (PCHIP) interpolation, numpy-only
# =============================================================================
def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def edge(h0, h1, d0, d1):
        val = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(val) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(val) > 3.0 * abs(d0):
            return 3.0 * d0
        return val

    if n == 2:
        d[0] = d[1] = delta[0]
    else:
        d[0] = edge(h[0], h[1], delta[0], delta[1])
        d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the monotone cubic Hermite interpolant. ``xq`` must lie in [x0, xN]."""
    if len(x) < 3:
        return np.interp(xq, x, y)
    d = _pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y[idx] + h10 * h * d[idx] + h01 * y[idx + 1] + h11 * h * d[idx + 1]


# =============================================================================
# 4. step-constraint repair
# =============================================================================
def _enforce_min_steps(cdf: np.ndarray, min_step: float,
                       lower_cap: float, upper_cap: float) -> np.ndarray:
    """Forward-then-backward sweep guaranteeing every bin rises by >= ``min_step``.

    Both directions are needed: the forward pass alone can be boxed in by a
    closed upper bound (metaculus-bot cluster_processing.py:144-169).
    """
    out = cdf.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1] + min_step:
            out[i] = out[i - 1] + min_step
        if out[i] > upper_cap:
            out[i] = upper_cap
    for j in range(len(out) - 2, -1, -1):
        if out[j] > out[j + 1] - min_step:
            out[j] = out[j + 1] - min_step
        if out[j] < lower_cap:
            out[j] = lower_cap
    return out


def _redistribute_max_step(cdf: np.ndarray, max_step: float) -> np.ndarray:
    """Clip every bin to ``max_step`` and push the removed mass into bins with slack."""
    if cdf.size <= 1:
        return cdf
    steps = np.diff(cdf)
    if not np.any(steps > max_step):
        return cdf
    total = float(steps.sum())
    steps = np.clip(steps, 0.0, max_step)
    deficit = total - float(steps.sum())
    for _ in range(max(5, steps.size * 5)):
        if deficit <= 1e-15:
            break
        slack = max_step - steps
        pos = slack > 1e-15
        slack_sum = float(slack[pos].sum())
        if not np.any(pos) or slack_sum <= 1e-18:
            break
        alloc = np.zeros_like(steps)
        alloc[pos] = deficit * slack[pos] / slack_sum
        steps += np.minimum(alloc, slack)
        deficit = total - float(steps.sum())
    out = np.empty_like(cdf)
    out[0] = cdf[0]
    out[1:] = cdf[0] + np.cumsum(steps)
    return out


# =============================================================================
# 5. open-bound tail anchoring
# =============================================================================
def _tail_anchor(x_far: float, x_near: float,
                 f_far: float, f_near: float,
                 x_target: float, upper: bool) -> float | None:
    """Extrapolate the forecaster's own tail out to an OPEN bound.

    Returns the CDF height at ``x_target`` (the open bound's location), or None
    when the tail is too degenerate to extrapolate.

    WHY THIS EXISTS. If every declared percentile sits inside the displayed
    range, a naive builder clamps the evaluation grid to the declared span, and
    then cdf[-1] is forced to equal the top declared percentile. A 13-point
    forecast with P99 = 60 on a 0..100 range and one with P99 = 99 then produce
    the IDENTICAL endpoint 0.99, i.e. both claim a 1% chance of exceeding 100.
    The first forecaster believes ~0%. With the 6-point 10..90 template set the
    same clamp fabricates a flat 10%.

    So instead we decay the survival function S = 1 - F off the last declared
    segment at that segment's own rate:  S(x) = S_near * exp(-lam * (x - x_near)).
    Steep declared tails give ~0 out-of-range mass; shallow ones keep real mass.
    Mirror logic on the lower side decays F itself toward 0.
    """
    dx = abs(x_near - x_far)
    if dx <= 1e-12:
        return None
    if upper:
        s_far, s_near = 1.0 - f_far, 1.0 - f_near
        if s_near <= 1e-12 or s_far <= s_near:
            return None
        lam = np.log(s_far / s_near) / dx
        if not np.isfinite(lam) or lam <= 0:
            return None
        return float(1.0 - s_near * np.exp(-lam * (x_target - x_near)))
    if f_near <= 1e-12 or f_far <= f_near:
        return None
    lam = np.log(f_far / f_near) / dx
    if not np.isfinite(lam) or lam <= 0:
        return None
    return float(f_near * np.exp(-lam * (x_near - x_target)))


# =============================================================================
# 6. the public entry point
# =============================================================================
def build_continuous_cdf(percentile_dict: dict[float, float],
                         question_metadata: dict) -> list[float]:
    """Turn LLM percentile estimates into a submittable ``continuous_cdf`` list.

    ``percentile_dict`` maps percentile -> question-space value. Keys may be
    [0, 1] decimals or 0-100 labels (auto-detected by ``max(keys) > 1``).

    ``question_metadata`` keys mirror the Metaculus API JSON:
        range_min, range_max                : float  <- question.scaling.range_*
        zero_point                          : float|None <- question.scaling.zero_point
        open_lower_bound, open_upper_bound  : bool   <- question.* (OPTIONAL;
             both default to True, matching questions.py:417-425, which wraps
             exactly these two reads in try/except KeyError)
        cdf_size OR inbound_outcome_count   : int|None <- question.scaling.*

    Returns a plain ``list[float]`` ready to place in the forecast payload as
    ``{"continuous_cdf": <this>}``. Raises ValueError only when the request is
    genuinely unsatisfiable; never returns a mildly-invalid array.
    """
    meta = question_metadata
    range_min = float(meta["range_min"])
    range_max = float(meta["range_max"])
    # Default to OPEN when absent -- forecasting-tools questions.py:417-425.
    open_lower = bool(meta.get("open_lower_bound", True))
    open_upper = bool(meta.get("open_upper_bound", True))
    zero_point = meta.get("zero_point")
    zero_point = None if zero_point is None else float(zero_point)

    n = cdf_size_from_question(meta)
    if n < 2:
        raise ValueError(f"cdf_size must be >= 2, got {n}")
    min_step, max_step = grid_step_constraints(n)

    if range_max <= range_min:
        raise ValueError(f"range_max ({range_max}) must exceed range_min ({range_min})")
    # Discrete grids are always linear (metaculus-bot numeric/validation.py:160-164),
    # and a zero_point at or above the lower bound makes the log transform undefined
    # (forecasting-tools numeric_report.py:143-148 raises on it).
    if zero_point is not None and (n != DEFAULT_CDF_SIZE or zero_point >= range_min):
        zero_point = None

    # --- 6a. clean the declared percentiles ---------------------------------
    keys = [float(k) for k in percentile_dict]
    scale = 100.0 if keys and max(keys) > 1.0 else 1.0
    pairs: list[tuple[float, float]] = []
    for k, v in percentile_dict.items():
        p, val = float(k) / scale, float(v)
        if not (0.0 < p < 1.0) or not np.isfinite(val):
            continue                                   # drop unusable entries
        pairs.append((p, val))
    pairs.sort(key=lambda pv: pv[0])
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 usable percentiles, got {len(pairs)}")

    ps = np.array([p for p, _ in pairs], dtype=float)
    vs = np.array([v for _, v in pairs], dtype=float)

    # Clamp only against CLOSED bounds. Values beyond an OPEN bound are KEPT --
    # they are the only channel through which out-of-range mass is expressed.
    buffer = 1.0 if (range_max - range_min) > 100 else 0.01 * (range_max - range_min)
    if not open_lower:
        vs = np.maximum(vs, range_min + buffer)
    if not open_upper:
        vs = np.minimum(vs, range_max - buffer)
    if zero_point is not None:
        vs = np.maximum(vs, zero_point + 1e-9)

    # Repair duplicate / non-monotone values with a forward epsilon sweep.
    span = float(vs[-1] - vs[0])
    eps = max(abs(span) * 1e-9, 1e-9)
    for i in range(1, len(vs)):
        if vs[i] <= vs[i - 1]:
            vs[i] = vs[i - 1] + eps
    # If that pushed past a closed upper bound, sweep back down instead.
    if not open_upper and vs[-1] > range_max - buffer:
        vs[-1] = range_max - buffer
        for j in range(len(vs) - 2, -1, -1):
            if vs[j] >= vs[j + 1]:
                vs[j] = vs[j + 1] - eps

    # --- 6b. move to cdf-location space -------------------------------------
    x_nodes = np.asarray(
        value_to_cdf_location(vs, range_min, range_max, zero_point), dtype=float
    )
    for i in range(1, len(x_nodes)):                   # transform can collapse ties
        if x_nodes[i] <= x_nodes[i - 1]:
            x_nodes[i] = x_nodes[i - 1] + 1e-12

    # --- 6c. anchor the bounds ----------------------------------------------
    # Closed bound: the CDF is pinned there by definition, so add it as a real
    # interpolation node rather than letting the curve run flat into it.
    if not open_lower and x_nodes[0] > 1e-12:
        x_nodes, ps = np.insert(x_nodes, 0, 0.0), np.insert(ps, 0, 0.0)
    if not open_upper and x_nodes[-1] < 1.0 - 1e-12:
        x_nodes, ps = np.append(x_nodes, 1.0), np.append(ps, 1.0)

    # Open bound with no declared value reaching it: extrapolate the tail so the
    # out-of-range mass reflects the forecast instead of being pinned to
    # 1 - P_max. See _tail_anchor for the failure this prevents.
    if open_upper and x_nodes[-1] < 1.0 - 1e-12 and ps[-1] < OPEN_BOUND_MAX_HEAD:
        anchor = _tail_anchor(x_nodes[-2], x_nodes[-1], ps[-2], ps[-1],
                              1.0, upper=True)
        if anchor is not None:
            anchor = min(OPEN_BOUND_MAX_HEAD, max(anchor, ps[-1] + 1e-9))
            x_nodes, ps = np.append(x_nodes, 1.0), np.append(ps, anchor)
    if open_lower and x_nodes[0] > 1e-12 and ps[0] > OPEN_BOUND_MIN_TAIL:
        anchor = _tail_anchor(x_nodes[1], x_nodes[0], ps[1], ps[0],
                              0.0, upper=False)
        if anchor is not None:
            anchor = max(OPEN_BOUND_MIN_TAIL, min(anchor, ps[0] - 1e-9))
            x_nodes, ps = np.insert(x_nodes, 0, 0.0), np.insert(ps, 0, anchor)

    # --- 6d. evaluate on the server's own grid ------------------------------
    locations = np.linspace(0.0, 1.0, n)
    x_eval = np.clip(locations, x_nodes[0], x_nodes[-1])   # never extrapolate PCHIP
    cdf = _pchip_eval(x_nodes, ps, x_eval)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    # --- 6e. endpoints ------------------------------------------------------
    if open_lower:
        cdf[0] = max(cdf[0], OPEN_BOUND_MIN_TAIL)
    else:
        cdf[0] = 0.0                                   # P(X < range_min) == 0
    if open_upper:
        cdf[-1] = min(cdf[-1], OPEN_BOUND_MAX_HEAD)
    else:
        cdf[-1] = 1.0                                  # P(X <= range_max) == 1
    lower_cap = OPEN_BOUND_MIN_TAIL if open_lower else 0.0
    upper_cap = OPEN_BOUND_MAX_HEAD if open_upper else 1.0

    # Feasibility is judged against the LEGAL range, not the current in-range
    # mass: a forecast putting 99% above an open ceiling is legitimate and the
    # sweep below simply widens its in-range mass to the legal minimum.
    if (n - 1) * min_step > (upper_cap - lower_cap) + 1e-12:
        raise ValueError(
            f"Infeasible: {n - 1} bins x min_step {min_step} exceeds the legal "
            f"cdf range [{lower_cap}, {upper_cap}]"
        )
    inbound_mass = float(cdf[-1] - cdf[0])

    # --- 6f. min/max step repair -------------------------------------------
    # A uniform mixture is the primary min-step mechanism: it lifts flat regions
    # without the shape distortion a pure sweep causes.
    if inbound_mass > 1e-12:
        alpha = min(1.0, min_step * n / inbound_mass * _ALPHA_SAFETY_MARGIN)
    else:
        alpha = 1.0
    uniform = np.linspace(float(cdf[0]), float(cdf[-1]), n)
    cdf = (1.0 - alpha) * cdf + alpha * uniform

    for _ in range(12):
        cdf = _redistribute_max_step(cdf, max_step)
        cdf = np.maximum.accumulate(cdf)
        cdf = _enforce_min_steps(cdf, min_step, lower_cap=lower_cap, upper_cap=upper_cap)
        steps = np.diff(cdf)
        if steps.size == 0 or (steps.max() <= max_step and steps.min() >= min_step):
            break

    # Re-pin closed endpoints: those are exact equalities, not inequalities.
    if not open_lower:
        cdf[0] = 0.0
    if not open_upper:
        cdf[-1] = 1.0
    cdf = np.round(np.clip(cdf, 0.0, 1.0), 10)

    # --- 6g. hard validation against the RAW server thresholds --------------
    # Deliberately checked against validator_step_bounds, not the padded
    # generation targets: this asserts what Metaculus will actually test.
    hard_min, hard_max = validator_step_bounds(n)
    steps = np.diff(cdf)
    assert len(cdf) == n, f"cdf length {len(cdf)} != cdf_size {n}"
    assert np.all(np.isfinite(cdf)), "cdf contains NaN/inf"
    assert cdf[0] >= 0.0 and cdf[-1] <= 1.0, f"cdf out of [0,1]: {cdf[0]}, {cdf[-1]}"
    assert steps.min() >= hard_min, (
        f"min step {steps.min():.12e} < validator floor {hard_min:.12e}")
    assert steps.max() <= hard_max, (
        f"max step {steps.max():.12f} > validator cap {hard_max:.12f}")
    if open_lower:
        assert cdf[0] >= OPEN_BOUND_MIN_TAIL, f"open lower: cdf[0]={cdf[0]} < 0.001"
    else:
        assert cdf[0] == 0.0, f"closed lower: cdf[0]={cdf[0]} != 0.0"
    if open_upper:
        assert cdf[-1] <= OPEN_BOUND_MAX_HEAD, f"open upper: cdf[-1]={cdf[-1]} > 0.999"
    else:
        assert cdf[-1] == 1.0, f"closed upper: cdf[-1]={cdf[-1]} != 1.0"

    return [float(x) for x in cdf]


# =============================================================================
# TEST EVIDENCE (actually executed on numpy 2.5.2; the prior agent's claimed
# runs could NOT have happened -- numpy was not installed in this environment
# until I created scratchpad/work/venv)
# =============================================================================
#  (a) closed/closed 0-100, 13 pts  -> len 201, cdf0=0.0, cdfN=1.0,
#                                      minstep 2.53e-4, maxstep 0.0091   PASS
#  (b) open upper, values crossing the bound (P80=100 ... P99=260)
#                                   -> cdfN=0.800000, implied P(X>100)=0.2000
#                                      exactly the elicited belief        PASS
#  (c) log zero_point=0, range 1..1e8 -> len 201, cdf0=0.0, cdfN=1.0,
#                                      minstep 2.02e-4                    PASS
#  (d) discrete inbound_outcome_count=8 -> len 9 (NOT 201), maxstep 0.306
#                                      (NOT clipped to the 201-grid 0.2)  PASS
#
#  AUDIT REGRESSION 1  metadata missing open_*_bound -> no KeyError,
#                      defaults to open/open                              PASS
#  AUDIT REGRESSION 2  open-bound tail no longer pinned to 1 - P_max:
#                        steep 13-pt tail (P99=60)  -> P(X>100) = 0.0010
#                        wide  13-pt tail (P99=99)  -> P(X>100) = 0.0063
#                        6-pt template   (P90=62)   -> P(X>100) = 0.0012
#                      (the clamp-only design gave 0.0100 / 0.0100 / 0.1000
#                       -- two distinct beliefs indistinguishable, and a
#                       fabricated 10% on the template set)               PASS
#  AUDIT REGRESSION 3  3000 concentrated distributions across
#                      cdf_size in {9,21,101,201,401} x all four bound
#                      combinations: 0 violations of the raw validator cap;
#                      worst observed ratio to the cap = 0.950000         PASS
#  EXTREME             98.7% of mass above an open ceiling -> accepted,
#                      cdfN=0.013395, no spurious infeasibility raise     PASS
#  FUZZ                6000 adversarial cases: cdf_size in
#                      {3,4,6,8,9,11,21,26,51,101,201,401}, log and linear,
#                      reversed / all-identical / degenerate / far-outside
#                      inputs, 0-100 and 0-1 key styles, 15% with the
#                      open_*_bound keys deleted -> 0 failures            PASS
