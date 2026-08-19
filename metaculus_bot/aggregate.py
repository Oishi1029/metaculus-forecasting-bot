"""Combine several models' forecasts into one. Median, then clip.

DESIGN CHOICES AND THE EVIDENCE BEHIND THEM
  * MEDIAN, not mean. Metaculus' own guidance is "run 5+ forecasts and take
    median/aggregate", and the median is what survives one model misreading a
    question -- exactly the failure the ensemble exists to absorb.
  * NO EXTREMIZING. A 14th-place bot implemented log-odds extremizing and then
    shipped its factor at 1.0, i.e. chose not to extremize. Metaculus' own
    calibration work does not test it. We pool at a = 1.
  * CLIP, DO NOT SHRINK. Binary clipped to [0.02, 0.98]. Clipping guards the
    catastrophic failure Metaculus names by name -- "unresolved questions
    mistaken as already resolved, producing catastrophic 99% forecasts" -- while
    a shrink like 0.96p + 0.02 also drags well-justified confident forecasts
    toward the middle, costing score on the questions we actually got right.
  * NO PLATT SCALING in v1. It is the best of the five recalibration methods
    Metaculus tested, but a top-15 bot found its fitted binary slope moved from
    0.83 one season to 1.66 the next -- opposite calibration shapes. Fitting on
    last season's data would actively mislead this season.
"""

from __future__ import annotations

import logging
import statistics
from typing import Sequence

from . import config

log = logging.getLogger(__name__)


def aggregate_binary(probs: Sequence[float]) -> float:
    """Median of the ensemble, then clipped to the tail bounds."""
    vals = [p for p in probs if p is not None and 0.0 <= p <= 1.0]
    if not vals:
        raise ValueError("no valid binary probabilities to aggregate")
    # median_low, NOT median. statistics.median AVERAGES the two middle values on
    # an even-length list, which hands an outlier half the weight -- exactly what
    # the median was chosen to prevent. [0.90, 0.03] -> median 0.465 vs
    # median_low 0.03. Ensembles go even whenever a model is dropped.
    if len(vals) % 2 == 0:
        log.warning("even ensemble (%d models): using median_low so an outlier "
                    "cannot take half the weight", len(vals))
    p = statistics.median_low(vals) if len(vals) % 2 == 0 else statistics.median(vals)
    return clamp_binary(p)


def clamp_binary(p: float) -> float:
    p = min(max(p, config.BINARY_CLAMP_LOW), config.BINARY_CLAMP_HIGH)
    # Metaculus' own client rejects outside (0.001, 0.999) before sending.
    return min(max(p, config.API_PROB_MIN), config.API_PROB_MAX)


def aggregate_multiple_choice(
    per_model: Sequence[dict[str, float]], options: Sequence[str]
) -> dict[str, float]:
    """Per-option median across models, renormalised, then clamped and renormalised.

    Clamping breaks the sum-to-one invariant, so we clamp and renormalise
    ITERATIVELY until both hold: a single pass can push a value back outside.
    """
    usable = [d for d in per_model if d]
    if not usable:
        raise ValueError("no valid multiple-choice distributions to aggregate")

    merged: dict[str, float] = {}
    for opt in options:
        vals = [float(d.get(opt, 0.0)) for d in usable]
        if not vals:
            merged[opt] = 0.0
        elif len(vals) % 2 == 0:
            merged[opt] = statistics.median_low(vals)
        else:
            merged[opt] = statistics.median(vals)

    merged = _normalise(merged)
    for _ in range(24):
        clamped = {k: min(max(v, config.MC_CLAMP_LOW), config.MC_CLAMP_HIGH)
                   for k, v in merged.items()}
        total = sum(clamped.values())
        if total <= 0:
            clamped = {k: 1.0 / len(options) for k in options}
            total = 1.0
        renorm = {k: v / total for k, v in clamped.items()}
        if all(config.MC_CLAMP_LOW - 1e-9 <= v <= config.MC_CLAMP_HIGH + 1e-9
               for v in renorm.values()):
            merged = renorm
            break
        merged = renorm
    else:
        log.warning("MC clamp/renormalise did not settle; shipping best effort")

    # Final safety: the API rejects anything outside (0.001, 0.999).
    merged = {k: min(max(v, config.API_PROB_MIN), config.API_PROB_MAX) for k, v in merged.items()}
    return _normalise(merged)


def _normalise(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    if total <= 0:
        n = max(1, len(d))
        return {k: 1.0 / n for k in d}
    return {k: v / total for k, v in d.items()}


def aggregate_percentiles(
    per_model: Sequence[dict[float, float]]
) -> dict[float, float]:
    """Median value at each percentile, then repaired to be strictly increasing.

    Averaging percentile-wise ("vincentization") keeps the shape of the models'
    agreement instead of averaging their probabilities, which is what we want
    when the disagreement is about MAGNITUDE rather than about likelihood.
    """
    usable = [d for d in per_model if d]
    if not usable:
        raise ValueError("no valid percentile sets to aggregate")

    keys = sorted({k for d in usable for k in d})
    out: dict[float, float] = {}
    for k in keys:
        vals = [d[k] for d in usable if k in d]
        if vals:
            out[k] = statistics.median(vals)

    # Enforce monotonicity in percentile order. A later percentile that dipped
    # below an earlier one is a model slip, not a belief.
    ordered = sorted(out)
    fixed: dict[float, float] = {}
    prev: float | None = None
    for k in ordered:
        v = out[k]
        if prev is not None and v <= prev:
            v = prev + max(abs(prev) * 1e-6, 1e-9)
        fixed[k] = v
        prev = v
    return fixed
