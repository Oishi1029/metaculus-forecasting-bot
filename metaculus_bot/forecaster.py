"""The per-question pipeline: research -> N models -> aggregate -> payload.

Contract: forecast_question() RETURNS, it does not raise. Scoring sums peer
scores and then squares the sum, so an exception that aborts the run costs every
remaining question a zero. One bad question must never take the run down.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from typing import Any

from . import aggregate, config, parsing, prompts
from .cdf import build_continuous_cdf
from .llm import LLMClient
from .models import BINARY, DISCRETE, MULTIPLE_CHOICE, NUMERIC, Forecast, Question
from .research import ResearchRegistry

log = logging.getLogger(__name__)


class ForecastFailure(Exception):
    pass


class Forecaster:
    def __init__(self, llm: LLMClient, research: ResearchRegistry, models: list[str] | None = None):
        self.llm = llm
        self.research = research
        self.models = models or config.models_for_profile()

    async def forecast_question(self, q: Question) -> Forecast:
        research_text, sources = await self._research(q)
        prompt = self._build_prompt(q, research_text)

        # Every model sees the SAME research. Researching per model would
        # multiply cost without adding source diversity -- the diversity that
        # matters is across search indices, and that already happened above.
        outputs = await asyncio.gather(
            *(self._one_model(m, prompt) for m in self.models), return_exceptions=True
        )
        texts: list[str] = []
        used: list[str] = []
        for model, out in zip(self.models, outputs):
            if isinstance(out, Exception):
                log.warning("model %s failed on post %s: %s", model, q.id_of_post, out)
                continue
            texts.append(out)
            used.append(model)

        if not texts:
            raise ForecastFailure("every model in the ensemble failed")

        payload, summary = await self._aggregate(q, texts)
        return Forecast(
            question=q,
            payload=payload,
            summary=summary,
            reasoning="",              # filled by comment.render
            model_outputs=texts,
            research_chars=len(research_text),
            models_used=used,
        )

    async def _one_model(self, model: str, prompt: str) -> str:
        return await self.llm.complete(model, prompt)

    async def _research(self, q: Question) -> tuple[str, list[str]]:
        text, sources = await self.research.gather(q.title, q.resolution_criteria)
        if text:
            extra = await self.research.gap_fill(
                q.title, q.resolution_criteria, q.fine_print, text, _today()
            )
            if extra:
                text = f"{text}\n\n{extra}"
                sources.append("gap-fill")
        return text, sources

    # -- prompt construction --------------------------------------------------
    def _build_prompt(self, q: Question, research: str) -> str:
        """Assemble the prompt WITHOUT str.format().

        These prompts contain literal JSON examples, so format() would treat
        every example brace as a field and raise KeyError. Explicit token
        substitution leaves unknown braces alone, which is what we want.
        """
        elapsed, remaining = _window_days(q)
        common = dict(
            question_text=q.display_title,
            resolution_criteria=q.resolution_criteria or "(none given)",
            fine_print=q.fine_print or "(none)",
            background_info=q.background or "(none)",
            research=research or "(no external research was available; rely on your own "
                                 "knowledge and say so explicitly)",
        )
        shared = dict(
            today=_today(),
            open_date=(q.open_time or "")[:10] or "unknown",
            resolve_date=(q.resolve_time or q.close_time or "")[:10] or "unknown",
            elapsed_days=elapsed,
            remaining_days=remaining,
        )

        if q.type == BINARY:
            return prompts.assemble(prompts.BINARY_PROMPT, BINARY, **shared, **common)
        if q.type == MULTIPLE_CHOICE:
            return prompts.assemble(
                prompts.MULTIPLE_CHOICE_PROMPT, MULTIPLE_CHOICE, **shared,
                options="\n".join(f"- {o}" for o in q.options), **common)
        if q.type in (NUMERIC, DISCRETE):
            return prompts.assemble(
                prompts.NUMERIC_PROMPT, q.type, **shared,
                unit_str=q.unit or "(not stated -- infer from the question)",
                lower=_fmt(q.range_min), upper=_fmt(q.range_max),
                nom_lower=_fmt(q.range_min), nom_upper=_fmt(q.range_max),
                lower_bound_message=_bound_message(q, upper=False),
                upper_bound_message=_bound_message(q, upper=True),
                **common)
        raise ForecastFailure(f"unsupported question type {q.type!r}")

    # -- aggregation ----------------------------------------------------------
    async def _aggregate(self, q: Question, texts: list[str]) -> tuple[dict[str, Any], str]:
        if q.type == BINARY:
            probs = [p for p in [await self._parse_one(q, t, parsing.parse_binary) for t in texts]
                     if p is not None]
            if not probs:
                raise ForecastFailure("no parseable binary probability from any model")
            p = aggregate.aggregate_binary(probs)
            return {"probability_yes": p}, f"{p * 100:.1f}% yes"

        if q.type == MULTIPLE_CHOICE:
            if not q.options:
                raise ForecastFailure("multiple-choice question has no options")
            dists = []
            for t in texts:
                d = await self._parse_one(q, t, lambda x: parsing.parse_multiple_choice(x, q.options))
                if d:
                    dists.append(d)
            if not dists:
                raise ForecastFailure("no parseable option probabilities from any model")
            merged = aggregate.aggregate_multiple_choice(dists, q.options)
            # The server matches option keys EXACTLY and performs no validation,
            # so assert the key set here rather than discovering it as a silent
            # server-side rejection.
            if set(merged) != set(q.options):
                raise ForecastFailure(
                    f"option key mismatch: {sorted(set(merged) ^ set(q.options))}"
                )
            top = max(merged, key=merged.get)
            return ({"probability_yes_per_category": merged},
                    f"{top} {merged[top] * 100:.1f}%")

        if q.type in (NUMERIC, DISCRETE):
            sets = []
            for t in texts:
                d = await self._parse_one(
                    q, t, lambda x: parsing.parse_numeric_percentiles(x, config.NUMERIC_PERCENTILES)
                )
                if d:
                    sets.append(d)
            if not sets:
                raise ForecastFailure("no parseable percentiles from any model")
            merged = aggregate.aggregate_percentiles(sets)
            cdf = build_continuous_cdf(merged, q.cdf_metadata())
            median = merged.get(50.0) or merged[sorted(merged)[len(merged) // 2]]
            unit = f" {q.unit}" if q.unit else ""
            return {"continuous_cdf": cdf}, f"median {_fmt(median)}{unit}"

        raise ForecastFailure(f"unsupported question type {q.type!r}")

    async def _parse_one(self, q: Question, text: str, parser) -> Any:
        """Deterministic parse, then one cheap LLM salvage. Never raises."""
        try:
            return parser(text)
        except Exception as first:                        # noqa: BLE001
            log.info("post %s: direct parse failed (%s); trying salvage",
                     q.id_of_post, first)
        try:
            repaired = await self.llm.complete(
                config.SALVAGE_MODEL,
                parsing.salvage_prompt(q.type, text),
                temperature=0.0, max_tokens=1200, max_retries=1,
            )
            return parser(repaired)
        except Exception as second:                       # noqa: BLE001
            log.warning("post %s: salvage parse also failed: %s", q.id_of_post, second)
            return None


def _window_days(q: Question) -> tuple[object, object]:
    """(days since open, days until resolution) for the forecasting-window anchor.

    Anchoring the model on the window matters: without it models resolve
    forward-looking questions against all of history and fire spurious 99%s.
    """
    now = _dt.datetime.now(_dt.timezone.utc)

    def _parse(v: str):
        if not v:
            return None
        try:
            return _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None

    opened = _parse(q.open_time)
    resolves = _parse(q.resolve_time) or _parse(q.close_time)
    elapsed = (now - opened).days if opened else "unknown"
    remaining = (resolves - now).days if resolves else "unknown"
    return elapsed, remaining


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _fmt(v: float | None) -> str:
    if v is None:
        return "unknown"
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.4g}"
    return f"{v:,.4g}"


def _bound_message(q: Question, *, upper: bool) -> str:
    """The bound text is load-bearing, not decoration.

    Open-bound questions are where numeric forecasts lose the most score: models
    pile percentiles ON the displayed edge instead of past it, which throws away
    all the tail mass they actually believe in. The message spells out the
    mechanism.
    """
    if upper:
        val = _fmt(q.range_max)
        if q.open_upper_bound:
            return (
                f"The upper bound is open: {val} is the top of the displayed range, not a hard "
                f"limit, so the outcome can resolve above {val}. Your percentiles are the ONLY way "
                f"you express probability mass, including mass beyond the displayed range. To put "
                f"N% of your probability above the open ceiling, place that fraction of your "
                f"percentiles above it: if you believe there is a ~75% chance the outcome exceeds "
                f"{val}, then your P50 (median) must be ABOVE {val} and only your lower percentiles "
                f"sit inside the range. Do not pile percentiles at the boundary."
            )
        return f"The upper bound is closed: the outcome can not be higher than {val}."
    val = _fmt(q.range_min)
    if q.open_lower_bound:
        return (
            f"The lower bound is open: {val} is the bottom of the displayed range, not a hard "
            f"limit, so the outcome can resolve below {val}. Your percentiles are the ONLY way you "
            f"express probability mass, including mass beyond the displayed range. To put N% of "
            f"your probability below the open floor, place that fraction of your percentiles below "
            f"it: if you believe there is a ~75% chance the outcome is below {val}, then your P50 "
            f"(median) must be BELOW {val} and only your upper percentiles sit inside the range. "
            f"Do not pile percentiles at the boundary."
        )
    return f"The lower bound is closed: the outcome can not be lower than {val}."
