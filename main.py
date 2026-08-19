#!/usr/bin/env python3
"""Entry point. This is the file to open first on the inspection screen-share.

    python main.py --tournament minibench
    python main.py --tournament market-pulse-26q3,minibench
    python main.py --tournament bot-testing-area --profile shakeout --dry-run

There is no interactive path, no confirmation prompt and no approval step
anywhere in this program. That is deliberate and required: the tournament rules
state that "Bots may not have a human in the loop when forecasting."

ORDERING IS LOAD-BEARING IN THIS FILE. metaculus_bot.config reads os.environ
ONCE, at import time. Every environment mutation must therefore happen before
the first import of anything under metaculus_bot -- including an innocent-looking
`from metaculus_bot import config` inside a validation branch, which is exactly
how --force silently became a no-op twice. Hence the literal below and the
deferred import at the bottom of main().
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Duplicated from config.TOURNAMENT_SANDBOX on purpose: reading it from config
# here would import config before the environment is finalised. A test asserts
# the two stay equal.
SANDBOX_SLUG = "bot-testing-area"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Metaculus forecasting bot")
    p.add_argument("--tournament", "-t", default=os.environ.get("TOURNAMENT_ID", "minibench"),
                   help="tournament slug(s) or id(s), comma-separated. One cron can then "
                        "cover a live tournament and one that has not opened yet: a slug "
                        "with no open questions simply contributes nothing.")
    p.add_argument("--profile", choices=["competition", "shakeout"], default=None,
                   help="model/cost profile; overrides env PROFILE")
    p.add_argument("--dry-run", action="store_true",
                   help="run the full pipeline but publish nothing (local use only)")
    p.add_argument("--limit", type=int, default=0, help="cap questions this run (0 = no cap)")
    p.add_argument("--force", action="store_true",
                   help="re-forecast questions already answered. SANDBOX + DRY-RUN ONLY: "
                        "used to time and cost a profile against real questions. Refuses to "
                        "run anywhere else, because re-forecasting a tournament question "
                        "breaks the one-forecast-per-question rule and previewing a "
                        "tournament forecast breaks the no-human-in-the-loop rule.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("main")

    tournaments = [t.strip() for t in str(args.tournament).split(",") if t.strip()]
    if not tournaments:
        log.error("no tournament given")
        return 2

    # ---- every environment mutation happens here, before any package import ----
    if args.profile:
        os.environ["PROFILE"] = args.profile

    if args.force:
        # Two hard gates. The sandbox is unscored and is explicitly where the
        # rules tell you to iterate; anywhere else, previewing a forecast and
        # then changing the bot is a disqualifying violation.
        if not args.dry_run:
            log.error("--force requires --dry-run. Refusing to run.")
            return 2
        if any(t != SANDBOX_SLUG for t in tournaments):
            log.error("--force is only allowed against %r, got %s. Refusing to run.",
                      SANDBOX_SLUG, tournaments)
            return 2
        os.environ["FORCE_REFORECAST"] = "1"
        log.warning("--force: re-forecasting already-answered SANDBOX questions, "
                    "publishing nothing")
    # ---------------------------------------------------------------------------

    from metaculus_bot.run import run_tournament   # noqa: PLC0415 - see module docstring

    async def run_all():
        results = []
        for name in tournaments:
            # Sequential, not concurrent: one shared rate limiter and one token
            # budget, and a failure in one tournament must not abort the others.
            try:
                results.append(await run_tournament(name, dry_run=args.dry_run,
                                                    limit=args.limit))
            except Exception as exc:                      # noqa: BLE001
                log.exception("tournament %s aborted: %s", name, exc)
                results.append(None)
        return results

    try:
        summaries = asyncio.run(run_all())
    except Exception as exc:                              # noqa: BLE001
        log.exception("run aborted: %s", exc)
        return 2

    failed = sum(s.failed for s in summaries if s)
    aborted = sum(1 for s in summaries if s is None)
    total_fc = sum(s.forecasted for s in summaries if s)
    log.info("ALL_DONE tournaments=%d forecasted=%d failed=%d aborted=%d",
             len(tournaments), total_fc, failed, aborted)

    # Publish first, then fail the job. CI going red must never suppress a
    # forecast that was already submitted.
    return 1 if (failed or aborted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
