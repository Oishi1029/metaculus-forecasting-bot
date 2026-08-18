"""Render the reasoning comment. This is a PRIZE-ELIGIBILITY REQUIREMENT.

Verbatim from the tournament rules: "In order to be eligible for the prize, the
participating bot needs to have written a comment response (including a display
of its forecast) under each question that it is forecasting."

So the comment must contain the forecast itself, not just prose, and it must
exist on EVERY question we forecast. No published enforcement threshold exists,
so we assume 100% coverage is required and treat a missing comment as a defect.

Comments are posted PRIVATE. Metaculus auto-publishes them at intervals for the
FutureEval tournaments; posting publicly during an open question would leak our
reasoning to competing bots for no benefit.
"""

from __future__ import annotations

import datetime as _dt

from . import config
from .models import BINARY, DISCRETE, MULTIPLE_CHOICE, NUMERIC, Forecast


def repair_comment() -> str:
    """Used when a previous run published a forecast but died before commenting."""
    return (
        "**Forecast previously submitted via the API.**\n\n"
        "This comment is the required reasoning record for that forecast; the preceding "
        "run submitted the forecast but did not complete its comment. "
        "No human reviewed or altered the forecast."
    )


def render_post_comment(forecasts: list[Forecast]) -> str:
    """One comment covering every part of a post.

    Forecasts attach to a question but comments attach to a POST, and a group
    post can hold nine sub-questions. One comment per sub-question would stack
    nine comments on the same page; this renders them as one.
    """
    if not forecasts:
        return repair_comment()
    if len(forecasts) == 1 and not forecasts[0].question.is_group_part:
        return render(forecasts[0])

    lines = [f"**Forecasts on {len(forecasts)} parts of this question**", ""]
    lines.append("| Part | Forecast |")
    lines.append("|---|---|")
    for fc in forecasts:
        label = fc.question.group_label or fc.question.title
        lines.append(f"| {label} | {fc.summary} |")
    lines.append("")
    lines.append(_provenance(forecasts[0], len(forecasts)))
    lines.append("")
    for fc in forecasts:
        label = fc.question.group_label or fc.question.title
        lines.append("---")
        lines.append("")
        lines.append(f"### {label} - {fc.summary}")
        lines.append("")
        lines.append(_forecast_table(fc))
        lines.append("")
        lines.append(_best_rationale(fc))
        lines.append("")

    text = "\n".join(lines)
    if len(text) > config.COMMENT_MAX_CHARS:
        # Keep every part's NUMBER (required for eligibility) and drop rationale
        # depth rather than dropping a part entirely.
        head = lines[: 6 + len(forecasts)]
        text = "\n".join(head) + (
            "\n\n_Per-part reasoning omitted here to fit the comment length limit._"
        )
    return text


def _provenance(fc: Forecast, parts: int = 1) -> str:
    return (
        f"_Generated autonomously at {_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M UTC}. "
        f"Ensemble of {len(fc.models_used)} model(s): {', '.join(fc.models_used)}. "
        f"Research: {fc.research_chars:,} characters from external sources. "
        f"No human reviewed or altered {'these forecasts' if parts > 1 else 'this forecast'}._"
    )


def render(fc: Forecast) -> str:
    q = fc.question
    lines: list[str] = []

    lines.append(f"**Forecast: {fc.summary}**")
    lines.append("")
    lines.append(_forecast_table(fc))
    lines.append("")
    lines.append(_provenance(fc))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Reasoning")
    lines.append("")
    lines.append(_best_rationale(fc))

    text = "\n".join(lines)
    if len(text) > config.COMMENT_MAX_CHARS:
        keep = config.COMMENT_MAX_CHARS - 120
        text = text[:keep].rstrip() + "\n\n_[reasoning truncated to fit the comment length limit]_"
    return text


def _forecast_table(fc: Forecast) -> str:
    q = fc.question
    p = fc.payload

    if q.type == BINARY:
        prob = p.get("probability_yes", 0.0)
        return f"| Outcome | Probability |\n|---|---|\n| Yes | {prob * 100:.1f}% |\n| No | {(1 - prob) * 100:.1f}% |"

    if q.type == MULTIPLE_CHOICE:
        dist = p.get("probability_yes_per_category", {})
        rows = "\n".join(f"| {opt} | {dist.get(opt, 0.0) * 100:.1f}% |" for opt in q.options)
        return f"| Option | Probability |\n|---|---|\n{rows}"

    if q.type in (NUMERIC, DISCRETE):
        cdf = p.get("continuous_cdf") or []
        if not cdf or q.range_min is None or q.range_max is None:
            return "_(continuous distribution submitted)_"
        rows = []
        for target in (0.05, 0.25, 0.50, 0.75, 0.95):
            rows.append(f"| P{int(target * 100)} | {_value_at(cdf, target, q)} |")
        unit = f" ({q.unit})" if q.unit else ""
        return f"| Percentile | Value{unit} |\n|---|---|\n" + "\n".join(rows)

    return ""


def _value_at(cdf: list[float], target: float, q) -> str:
    """Invert the submitted CDF back into question space, for display only."""
    n = len(cdf)
    if n < 2:
        return "n/a"
    idx = 0
    for i, v in enumerate(cdf):
        if v >= target:
            idx = i
            break
    else:
        idx = n - 1
    loc = idx / (n - 1)
    lo, hi = float(q.range_min), float(q.range_max)
    zp = q.zero_point
    if zp is not None and hi > zp and lo > zp:
        ratio = (hi - zp) / (lo - zp)
        val = lo + (hi - lo) * (ratio ** loc - 1) / (ratio - 1) if ratio != 1 else lo + (hi - lo) * loc
    else:
        val = lo + (hi - lo) * loc
    if idx == 0 and q.open_lower_bound:
        return f"< {val:,.4g}"
    if idx == n - 1 and q.open_upper_bound:
        return f"> {val:,.4g}"
    return f"{val:,.4g}"


def _best_rationale(fc: Forecast) -> str:
    """Publish the longest model rationale, with its machine block stripped.

    We show one full chain of reasoning rather than a synthesis, because a
    synthesis would be a fourth LLM call that no scoring rule rewards, and
    because the rules require us to display the reasoning behind the number we
    actually submitted -- which a post-hoc summary would only approximate.
    """
    if not fc.model_outputs:
        return "_(no model rationale captured)_"
    body = max(fc.model_outputs, key=len)
    for marker in ("STRUCTURED FORECAST", "```json", "```"):
        pos = body.rfind(marker)
        if pos > 200:
            body = body[:pos]
            break
    return body.strip()
