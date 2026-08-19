"""Plain dataclasses. Parses Metaculus post JSON -> Question. No network, no I/O.

THE ONE THING TO REMEMBER: a post id and a question id are different numbers.
    POST /questions/forecast/  takes the QUESTION id
    POST /comments/create/     takes the POST id
Confusing them is the single most likely silent failure in this codebase, which
is why they are never bare ints on this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BINARY = "binary"
MULTIPLE_CHOICE = "multiple_choice"
NUMERIC = "numeric"
DISCRETE = "discrete"
SUPPORTED_TYPES = (BINARY, MULTIPLE_CHOICE, NUMERIC, DISCRETE)


@dataclass
class Question:
    id_of_post: int
    id_of_question: int
    type: str
    title: str
    url: str
    background: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    unit: str = ""
    status: str = "open"
    open_time: str = ""
    close_time: str = ""
    resolve_time: str = ""
    already_forecasted: bool = False
    # multiple choice
    options: list[str] = field(default_factory=list)
    group_variable: str = ""
    # numeric / discrete
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    # The DISPLAYED domain, which is not the CDF grid. On a discrete question the
    # grid runs [5.5, 13.5] while the real answer space is the integers 6..13 --
    # telling a model the number of jurisdictions could be 5.5 is nonsense that
    # costs elicitation quality. Metaculus's own SDK reads these; we did not.
    nominal_min: float | None = None
    nominal_max: float | None = None
    open_lower_bound: bool = True
    open_upper_bound: bool = True
    inbound_outcome_count: int | None = None
    # Group ("multi-part") posts hold several sub-questions under ONE post id.
    # Comments attach to the POST, so a group gets one comment covering all of
    # its parts -- posting one per part would stack duplicates on the same page.
    group_label: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_group_part(self) -> bool:
        return bool(self.group_label)

    @property
    def display_title(self) -> str:
        return f"{self.title} - {self.group_label}" if self.group_label else self.title

    @property
    def is_continuous(self) -> bool:
        return self.type in (NUMERIC, DISCRETE)

    def cdf_metadata(self) -> dict[str, Any]:
        """Exactly the dict shape cdf.build_continuous_cdf expects."""
        return {
            "range_min": self.range_min,
            "range_max": self.range_max,
            "zero_point": self.zero_point,
            "open_lower_bound": self.open_lower_bound,
            "open_upper_bound": self.open_upper_bound,
            "inbound_outcome_count": self.inbound_outcome_count,
        }


def parse_post_questions(post: dict[str, Any]) -> list[Question]:
    """Every forecastable question on one post.

    A plain post yields 0 or 1 questions. A GROUP post ("What will revenue be
    for each of these companies?") yields one per sub-question -- Market Pulse
    is composed entirely of these, so skipping them would forfeit that whole
    tournament. Notebooks and conditional posts yield none.
    """
    if "notebook" in post and post.get("notebook"):
        return []
    post_id = int(post["id"])
    post_title = post.get("title") or ""

    q = post.get("question")
    if isinstance(q, dict):
        one = _question_from(post, q, post_id, post_title)
        return [one] if one else []

    group = post.get("group_of_questions")
    if isinstance(group, dict):
        out: list[Question] = []
        for sub in (group.get("questions") or []):
            if not isinstance(sub, dict):
                continue
            built = _question_from(post, sub, post_id, post_title, group=group)
            if built:
                out.append(built)
        return out

    return []


def parse_post(post: dict[str, Any]) -> Question | None:
    """Single-question convenience wrapper. Returns None for groups of != 1."""
    qs = parse_post_questions(post)
    return qs[0] if len(qs) == 1 else None


def _question_from(post: dict[str, Any], q: dict[str, Any], post_id: int,
                   post_title: str, group: dict[str, Any] | None = None) -> Question | None:
    qtype = q.get("type")
    if qtype not in SUPPORTED_TYPES:
        return None
    if q.get("status") not in ("open", None, ""):
        return None

    src = group if group is not None else q
    scaling = q.get("scaling") or {}
    inbound = scaling.get("inbound_outcome_count")
    if inbound is None:
        inbound = q.get("inbound_outcome_count")

    return Question(
        id_of_post=post_id,
        id_of_question=int(q["id"]),
        type=qtype,
        title=(q.get("title") or post_title or ""),
        url=f"https://www.metaculus.com/questions/{post_id}/",
        background=(src.get("description") or q.get("description") or ""),
        resolution_criteria=(src.get("resolution_criteria") or q.get("resolution_criteria") or ""),
        fine_print=(src.get("fine_print") or q.get("fine_print") or ""),
        unit=q.get("unit") or "",
        status=q.get("status") or "",
        open_time=q.get("open_time") or post.get("published_at") or "",
        close_time=q.get("scheduled_close_time") or "",
        resolve_time=q.get("scheduled_resolve_time") or "",
        already_forecasted=_already_forecasted(q),
        options=list(q.get("options") or []),
        group_variable=(src.get("group_variable") or q.get("group_variable") or ""),
        range_min=_maybe_float(scaling.get("range_min")),
        range_max=_maybe_float(scaling.get("range_max")),
        zero_point=_maybe_float(scaling.get("zero_point")),
        nominal_min=(_maybe_float(scaling.get("nominal_min"))
                     if scaling.get("nominal_min") is not None
                     else _maybe_float(scaling.get("range_min"))),
        nominal_max=(_maybe_float(scaling.get("nominal_max"))
                     if scaling.get("nominal_max") is not None
                     else _maybe_float(scaling.get("range_max"))),
        open_lower_bound=bool(q.get("open_lower_bound", True)),
        open_upper_bound=bool(q.get("open_upper_bound", True)),
        inbound_outcome_count=int(inbound) if inbound is not None else None,
        group_label=(q.get("label") or "") if group is not None else "",
        raw=post,
    )


def _already_forecasted(q: dict[str, Any]) -> bool:
    """Has THIS token already forecasted this question?

    Server-derived, never disk-derived: a fresh ephemeral runner reconstructs
    the full history from the API alone. Two independent signals, either is
    sufficient, because the payload shape differs across question types.
    """
    mine = q.get("my_forecasts") or {}
    latest = mine.get("latest") or {}
    if latest.get("forecast_values") is not None:
        return True
    history = mine.get("history") or []
    return len(history) > 0


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Forecast:
    """What the forecaster produced for one question, ready to publish."""
    question: Question
    payload: dict[str, Any]          # the forecast body minus question/source
    summary: str                     # one-line human-readable prediction
    reasoning: str                   # full comment text
    model_outputs: list[str] = field(default_factory=list)
    research_chars: int = 0
    models_used: list[str] = field(default_factory=list)


@dataclass
class QuestionOutcome:
    question: Question
    forecasted: bool = False
    commented: bool = False
    skipped_reason: str = ""
    error: str = ""
