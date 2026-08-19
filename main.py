#!/usr/bin/env python3
"""Entry point. This is the file to open first on the inspection screen-share.

    python main.py --tournament minibench
    python main.py --tournament bot-testing-area --profile shakeout --dry-run

There is no interactive path, no confirmation prompt and no approval step
anywhere in this program. That is deliberate and required: the tournament rules
state that "Bots may not have a human in the loop when forecasting."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


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
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile:
        os.environ["PROFILE"] = args.profile

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Imported after PROFILE is set, because config reads the environment once.
    from metaculus_bot.run import run_tournament

    tournaments = [t.strip() for t in str(args.tournament).split(",") if t.strip()]
    log = logging.getLogger("main")

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
