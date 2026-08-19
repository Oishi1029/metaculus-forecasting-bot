# Autonomous Metaculus forecasting bot

Polls a Metaculus tournament every 20 minutes, forecasts every open question it
has not already answered, and posts a private comment containing its reasoning
and its forecast. No human is in the loop at any point.

Runtime dependencies are `httpx` and `numpy`. That is deliberate — see
"Why we don't use the official SDK" below.

## Quick start

```bash
pip install -r requirements.txt
cp .env.template .env        # then fill in METACULUS_TOKEN and OPENROUTER_API_KEY
python main.py --tournament bot-testing-area --profile shakeout --dry-run
```

`--dry-run` runs the entire pipeline and prints what it *would* submit without
publishing. It exists for local use only; CI never sets it.

## How a forecast is produced

```
GET /posts/            fetch every OPEN question in the tournament
   |
   +-- skip anything the SERVER says we already forecasted   (rule: one forecast per question)
   |
   v
RESEARCH   all live providers run concurrently, results merged
   |         - OpenRouter native web search
   |         - Perplexity Sonar (via OpenRouter, a different search index)
   |         - AskNews      (only if its keys are present)
   |         - Exa          (only if its key is present)
   |       then one fixed gap-fill pass: name the gaps, search for them
   v
ENSEMBLE   the SAME research goes to 3 models, one per vendor, in parallel
   |         gpt-5.6-sol - claude-opus-5 - gemini-3.1-pro-preview
   v
PARSE      each model must end with a ```json block; if it doesn't, repair it,
   |       then fall back to regex, then to one cheap salvage LLM call
   v
AGGREGATE  median across models, then clip tails (binary to [0.02, 0.98])
   |       numeric: median per percentile, then build the CDF
   v
POST /questions/forecast/    then    POST /comments/create/
```

## The five decisions that matter

**1. Median, not mean; clip, don't shrink.** The median survives one model
misreading a question, which is the failure the ensemble exists to absorb.
Clipping to `[0.02, 0.98]` guards the catastrophic failure Metaculus names
explicitly — "unresolved questions mistaken as already resolved, producing
catastrophic 99% forecasts" — without dragging *well-justified* confident
forecasts toward 0.5 the way a shrink like `0.96p + 0.02` does.

**2. Research diversity over research depth.** In Metaculus' own survey of 39
bot makers, the number of *distinct* research sources was the strongest
correlate of winning (r = 0.42) and the only result robust to sample
sensitivity. No individual provider predicted anything — AskNews ranked 1st one
season and 58th the next. So every provider is optional and they all run
concurrently. With only an OpenRouter key we still get two distinct search
indices, because Perplexity is reachable through OpenRouter.

**3. Idempotency comes from the server, never from disk.** GitHub runners are
ephemeral, so a local file would be empty every run and a committed one would
race. "Have I already forecasted this?" is answered by `question.my_forecasts`
in Metaculus' own response; "have I already commented?" by our own comment list.
A brand-new runner reconstructs the full history in two API calls.

**4. Every run is a full catch-up scan.** GitHub's docs warn scheduled runs
"can be delayed during periods of high load" and that "some queued jobs may be
dropped", with no retry. Because each run re-scans everything, a dropped tick
costs latency, never coverage. That matters enormously: tournament scoring sums
your peer scores and then **squares the sum**, and a question you never
forecasted is a zero inside that sum. Coverage dominates cleverness.

**5. One bad question never takes the run down.** `forecast_question` returns
rather than raises; the run publishes everything that succeeded and only then
exits non-zero so CI goes red.

## Publish order, and why it is not the other way round

Forecast first, comment second. The forecast POST **overwrites**, so being
interrupted before it is harmless. The comment POST **accumulates**, so a blind
retry posts twice. If a run dies between the two, the next run sees
"forecasted but not commented" and takes a comment-only repair path — it never
re-forecasts, because that would break the one-forecast-per-question rule.

## Rules this code enforces

| Rule | Where |
|---|---|
| One forecast per question, ever | `run.py` skips `already_forecasted`; repair path never re-forecasts |
| A comment on every question forecasted | `comment.py`, asserted in `tests/test_rules_compliance.py` |
| No human in the loop | no interactive call anywhere; a test greps for them |
| Comments private (auto-published later by Metaculus) | `config.COMMENT_IS_PRIVATE` |
| Don't blindly copy the community prediction | `with_cp=true` IS sent — it is what makes Metaculus return our own `my_forecasts`, which is the entire duplicate-protection mechanism — but the CP is never read: nothing touches `Question.raw`, and the prompt builder passes only named scalar fields |

Iterate only against **`bot-testing-area`**. Tuning a bot after seeing its
output on open tournament questions is explicitly a rules violation.

## Why we don't use the official SDK

`forecasting-tools` is Metaculus' own package and its source was the reference
for every endpoint here. We reimplemented rather than imported because:

1. Its git main is months ahead of its PyPI release, so either pin drifts.
2. Prize eligibility can require demonstrating and explaining the code on a
   live call. ~1,900 readable lines are explainable; a framework is not.
3. The numeric CDF path is the known failure point and we wanted full control.

We also deliberately diverge in three places. The SDK sleeps 3.5–4.5s before
*every* request, which at ~100 questions is 12–15 minutes of pure sleep — most
of a 20-minute tick; we use a token bucket. The SDK retries every non-2xx three
times, which would post a comment three times; we fail fast on 4xx and never
blind-retry a comment. And the SDK logs "Posted comment" *before* checking the
response, so it reports success on failure.

## The numeric CDF

The most error-prone part of any Metaculus bot, and the one most likely to fail
silently. `cdf.py` documents every constant against the line of Metaculus source
it came from. The rules that bite:

- Length is `inbound_outcome_count + 1`, **not always 201**. A real discrete
  question has `len(continuous_range) == 8` but needs a 9-point CDF.
- The per-bin maximum *relaxes* on coarse grids — at 9 points the cap is 1.0,
  not 0.2. Clipping to 0.2 there is a real, shipped bug in another bot.
- On an open bound there is no separate field for outside-range mass: `cdf[0]`
  *is* P(below range_min) and `1 - cdf[-1]` *is* P(above range_max). If the
  model puts 20% of its percentiles above the ceiling, the submitted array must
  actually carry 20% there. Getting this wrong silently discards the tail.

`tests/test_cdf.py` checks all four question shapes plus fuzz cases against the
**raw server thresholds**, not the softened ones we generate to.

## Cost

Roughly **$0.65/question** on the competition profile (3 models + research) and
**under $0.10** on the shakeout profile. A ~57-question MiniBench round is
therefore ~$35 competitive, ~$5 shakeout. Metaculus reports competitive
entrants average $1–1.5/question, so this sits deliberately below that: coverage
beats per-question spend when the score is a squared sum.

## Deployment

`.github/workflows/forecast.yml` runs at :07, :27 and :47 — three explicit cron
entries rather than `*/20`, and offset off the hour because :00 is the most
contended minute. `timeout-minutes: 18` keeps a hung run from starving later
ticks. `heartbeat.yml` exists because GitHub silently disables scheduled
workflows in a public repo after 60 days of no activity.

Required secrets: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`. Optional:
`ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET`, `EXA_API_KEY`. Repo variable
`TOURNAMENT_ID` retargets the bot without a commit.

## Known unknowns

Verified against Metaculus' source, but not yet against the live API — each has
a safe default and a one-line fix:

- Whether `forecast_type` must be repeated params or a comma string (we send
  repeated, the form the SDK exercises).
- Whether the forecast body may omit unused keys (we omit, as the SDK does).
- Whether `cdf[0] == 0.001` is accepted or must exceed it on open bounds.
- Whether the `/comments/` list endpoint takes `author`/`limit`/`offset`.
- Whether private comments satisfy the prize-eligibility comment rule, or
  whether public ones are required.

`tests/test_end_to_end.py` drives all four question types through the full
pipeline against a mocked API, so these are the only things a token changes.
