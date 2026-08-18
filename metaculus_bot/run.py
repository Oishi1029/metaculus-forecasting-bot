"""One full run: fetch -> group by post -> fan out -> publish -> summarise.

THE THREE PROPERTIES THAT MATTER MOST

1. EVERY RUN IS A FULL CATCH-UP SCAN. We re-fetch all open questions and
   forecast every one the SERVER says we have not forecasted. GitHub's own docs
   warn that scheduled runs "can be delayed during periods of high load" and
   that "some queued jobs may be dropped", with no retry. Because each run is a
   full scan, a dropped tick costs latency, never coverage. Tournament scoring
   sums peer scores and squares the sum, so a question never forecasted is a
   zero -- coverage is the whole game.

2. IDEMPOTENCY IS SERVER-DERIVED, NEVER DISK-DERIVED. GitHub runners are
   ephemeral; a local file would be empty on every run and a committed file
   would race. So "have I already done this?" is answered by Metaculus itself:
   question.my_forecasts for forecasts, and our own comment list for comments.
   A brand-new runner reconstructs the entire history in two API calls.

3. WORK IS BATCHED BY POST, NOT BY QUESTION. Forecasts attach to a QUESTION but
   comments attach to a POST, and a group post can hold nine sub-questions.
   Commenting per sub-question would stack nine comments on one page. So we
   forecast each part separately and post exactly ONE comment per post.

RULE COMPLIANCE: the tournament forbids more than one forecast per question and
forbids re-running a bot on a question whose output was seen. Both are enforced
here, before any model is called.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from . import comment as comment_mod
from . import config
from .forecaster import Forecaster
from .llm import LLMClient
from .metaculus_client import MetaculusClient, MetaculusError
from .models import Forecast, Question, parse_post_questions
from .research import ResearchRegistry

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    tournament: str = ""
    seen: int = 0
    questions: int = 0
    eligible: int = 0
    forecasted: int = 0
    commented: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"RUN_SUMMARY tournament={self.tournament} posts={self.seen} "
            f"questions={self.questions} eligible={self.eligible} "
            f"forecasted={self.forecasted} commented={self.commented} "
            f"skipped={self.skipped} failed={self.failed}"
        )


@dataclass
class _PostWork:
    post_id: int
    to_forecast: list[Question] = field(default_factory=list)
    needs_comment: bool = True


async def run_tournament(tournament: str | int, *, dry_run: bool = False,
                         limit: int = 0) -> RunSummary:
    started = time.monotonic()
    summary = RunSummary(tournament=str(tournament))

    llm = LLMClient()
    research = ResearchRegistry(llm)
    log.info("%s", research.describe())
    log.info("ensemble: %s", ", ".join(config.models_for_profile()))

    async with MetaculusClient() as client:
        me = await client.user_id()
        log.info("authenticated as Metaculus user id %s", me)

        posts = await client.open_questions(tournament)
        summary.seen = len(posts)

        by_post: dict[int, list[Question]] = defaultdict(list)
        for p in posts:
            for q in parse_post_questions(p):
                by_post[q.id_of_post].append(q)
        summary.questions = sum(len(v) for v in by_post.values())
        log.info("tournament %s: %d open post(s), %d forecastable question(s)",
                 tournament, len(posts), summary.questions)

        try:
            commented = await client.commented_post_ids()
        except MetaculusError as exc:
            log.warning("could not read comment ledger (%s); "
                        "treating all posts as uncommented", exc)
            commented = set()

        work: list[_PostWork] = []
        for post_id, qs in by_post.items():
            pending = [q for q in qs if not q.already_forecasted]
            done = [q for q in qs if q.already_forecasted]
            has_comment = post_id in commented
            if not pending and (has_comment or not done):
                summary.skipped += len(qs)
                continue
            work.append(_PostWork(post_id=post_id, to_forecast=pending,
                                  needs_comment=not has_comment))
            summary.skipped += len(qs) - len(pending)

        if limit or config.MAX_QUESTIONS_PER_RUN:
            work = work[: (limit or config.MAX_QUESTIONS_PER_RUN)]
        summary.eligible = sum(len(w.to_forecast) for w in work)
        log.info("%d post(s) need work: %d question(s) to forecast, "
                 "%d comment-only repair(s)",
                 len(work), summary.eligible,
                 sum(1 for w in work if w.needs_comment and not w.to_forecast))

        if not work:
            log.info("%s", summary.line())
            await llm.aclose()
            return summary

        forecaster = Forecaster(llm, research)
        sem = asyncio.Semaphore(config.MAX_CONCURRENT_QUESTIONS)

        async def do_question(q: Question) -> tuple[Question, Forecast | None, str]:
            async with sem:
                try:
                    fc = await asyncio.wait_for(
                        forecaster.forecast_question(q),
                        timeout=config.PER_QUESTION_DEADLINE_S,
                    )
                    return q, fc, ""
                except asyncio.TimeoutError:
                    return q, None, "per-question deadline exceeded"
                except Exception as exc:                  # noqa: BLE001
                    return q, None, f"{type(exc).__name__}: {exc}"

        async def do_post(w: _PostWork) -> dict:
            out = {"post_id": w.post_id, "forecasted": 0, "commented": False,
                   "errors": [], "skipped": 0}
            if time.monotonic() - started > config.RUN_DEADLINE_S:
                out["skipped"] = len(w.to_forecast)
                return out

            results = await asyncio.gather(*(do_question(q) for q in w.to_forecast))
            good: list[tuple[Question, Forecast]] = []
            for q, fc, err in results:
                if err or fc is None:
                    out["errors"].append(f"post {q.id_of_post} q {q.id_of_question}: {err}")
                else:
                    good.append((q, fc))

            if dry_run or not config.PUBLISH:
                for q, fc in good:
                    log.info("[dry-run] post %s q %s (%s): %s",
                             q.id_of_post, q.id_of_question, q.type, fc.summary)
                out["forecasted"] = len(good)
                out["commented"] = bool(good) or w.needs_comment
                return out

            # ORDER MATTERS. Forecast POSTs overwrite and are safe to be
            # interrupted before; the comment POST accumulates and is not. If we
            # die in between, the next run sees "forecasted, not commented" and
            # takes the comment-only path rather than re-forecasting.
            published: list[tuple[Question, Forecast]] = []
            for q, fc in good:
                try:
                    await client.post_forecast(q.id_of_question, fc.payload)
                    published.append((q, fc))
                    log.info("forecast post %s q %s (%s): %s",
                             q.id_of_post, q.id_of_question, q.type, fc.summary)
                except Exception as exc:                  # noqa: BLE001
                    out["errors"].append(
                        f"post {q.id_of_post} q {q.id_of_question} publish: "
                        f"{type(exc).__name__}: {exc}")
            out["forecasted"] = len(published)

            # Exactly one comment per post, covering every part forecasted.
            if published:
                text = comment_mod.render_post_comment([fc for _, fc in published])
            elif w.needs_comment and not w.to_forecast:
                text = comment_mod.repair_comment()
            else:
                return out
            try:
                await client.post_comment(w.post_id, text)
                out["commented"] = True
            except Exception as exc:                      # noqa: BLE001
                out["errors"].append(f"post {w.post_id} comment: "
                                     f"{type(exc).__name__}: {exc}")
            return out

        outcomes = await asyncio.gather(*(do_post(w) for w in work))

    orphans = 0
    for o in outcomes:
        summary.forecasted += o["forecasted"]
        summary.skipped += o["skipped"]
        if o["commented"]:
            summary.commented += 1
        elif o["forecasted"]:
            orphans += 1
        for e in o["errors"]:
            summary.failed += 1
            summary.errors.append(e)
            log.error("%s", e)

    # A forecast without its comment is a prize-eligibility defect, so surface it
    # loudly rather than letting it pass as a success.
    if orphans:
        log.error("%d post(s) carry forecasts WITHOUT a comment -- "
                  "the next run will repair them", orphans)

    await llm.aclose()
    log.info("run finished in %.1fs", time.monotonic() - started)
    log.info("%s", summary.line())
    return summary
