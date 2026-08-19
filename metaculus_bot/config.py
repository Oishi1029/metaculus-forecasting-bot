"""Every tunable in one place. No logic, no imports from siblings.

Read this file first on the screen-share: it is the whole configuration surface
of the bot, and nothing below it reads os.environ directly.
"""

from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name).lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# --- Credentials -------------------------------------------------------------
METACULUS_TOKEN = _env("METACULUS_TOKEN")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
ASKNEWS_CLIENT_ID = _env("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = _env("ASKNEWS_SECRET")
EXA_API_KEY = _env("EXA_API_KEY")
PERPLEXITY_API_KEY = _env("PERPLEXITY_API_KEY")

# --- Metaculus ---------------------------------------------------------------
METACULUS_API_BASE_URL = _env("METACULUS_API_BASE_URL", "https://www.metaculus.com/api")

# Tournaments. Slugs and integer ids are both accepted by the API.
#   33022        Summer 2026 FutureEval        (forecasting ended 2026-09-06)
#   "minibench"  the CURRENT MiniBench round; the slug is deliberately stable,
#                so never target a dated one like minibench-2026-08-24
#   "market-pulse-26q3"  $7,500, bot-eligible, forecasting to 2026-09-16
#   "bot-testing-area"   unscored sandbox with all four question types
TOURNAMENT_SUMMER_2026 = 33022
TOURNAMENT_MINIBENCH = "minibench"
TOURNAMENT_MARKET_PULSE = "market-pulse-26q3"
TOURNAMENT_SANDBOX = "bot-testing-area"

# --- LLM ---------------------------------------------------------------------
OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# One model per vendor. Metaculus's own bot-maker survey found source and model
# DIVERSITY to be the differentiator, not any single provider.
COMPETITION_MODELS = [
    m.strip() for m in _env(
        "COMPETITION_MODELS",
        "openai/gpt-5.6-sol,anthropic/claude-opus-5,google/gemini-3.1-pro-preview",
    ).split(",") if m.strip()
]
# Cheap profile for shaking the pipeline out against the sandbox.
SHAKEOUT_MODELS = [
    m.strip() for m in _env(
        "SHAKEOUT_MODELS",
        "openai/gpt-5.6-luna,google/gemini-3.7-flash",
    ).split(",") if m.strip()
]
# Small, cheap model used ONLY to salvage an unparseable forecast block.
SALVAGE_MODEL = _env("SALVAGE_MODEL", "google/gemini-3.7-flash")

# Perplexity reached THROUGH OpenRouter -- a second, independent search index
# for the price of the key we already have. NOTE: "perplexity/sonar-reasoning"
# does NOT exist and returns 404; the reasoning variant is "-reasoning-pro".
# Verified live 2026-08-19.
# Web research uses a FAST, CHEAP model, never the flagship. Measured 2026-08-19:
# routing the web-search call through gpt-5.6-sol timed out on 8 of 8 questions
# at 90s, so every question silently lost one of only two research sources --
# and source diversity is the strongest evidenced predictor of score. Searching
# and summarising does not need a frontier reasoning model.
RESEARCH_MODEL = _env("RESEARCH_MODEL", "openai/gpt-5.6-luna")

PERPLEXITY_MODEL = _env(
    "PERPLEXITY_MODEL",
    "perplexity/sonar" if _env("PROFILE", "competition").lower() == "shakeout"
    else "perplexity/sonar-pro",
)

PROFILE = _env("PROFILE", "competition").lower()   # "competition" | "shakeout"


def models_for_profile() -> list[str]:
    return SHAKEOUT_MODELS if PROFILE == "shakeout" else COMPETITION_MODELS


LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.3)
# MEASURED 2026-08-19: at 4000 this truncated ALL THREE competition models
# mid-sentence, before the required JSON block at the end of the report -- 18 of
# 24 outputs unparseable, each then paid for twice (wasted generation + salvage
# call). gemini-3.1-pro produced only 615 characters because reasoning tokens
# consume the same budget. The prompt asks for a long structured report on a
# ~37k-character input, so the ceiling has to accommodate that plus reasoning.
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 16000)
# 180s was arithmetically incompatible with a 16000-token ceiling: httpx applies
# this as a TOTAL request timeout, llm.py retries twice, so a slow model burned
# 3x180s + backoff ~= 546s against a 420s per-question deadline and returned a
# ZERO -- billed three times over. gemini-3.1-pro is the worst case because
# reasoning tokens draw on the same budget.
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 300.0)

# --- Forecast post-processing ------------------------------------------------
# Tail clipping. Evidenced as cheap insurance against the catastrophic-99%
# failure mode Metaculus names explicitly ("unresolved questions mistaken as
# already resolved"). Clip only -- do NOT shrink toward 0.5.
BINARY_CLAMP_LOW = _env_float("BINARY_CLAMP_LOW", 0.02)
BINARY_CLAMP_HIGH = _env_float("BINARY_CLAMP_HIGH", 0.98)
MC_CLAMP_LOW = _env_float("MC_CLAMP_LOW", 0.01)
MC_CLAMP_HIGH = _env_float("MC_CLAMP_HIGH", 0.99)
# Metaculus's own client rejects outside this before the request is even sent.
API_PROB_MIN = 0.001
API_PROB_MAX = 0.999

# Percentiles elicited for numeric/discrete questions. Deliberately wider in the
# tails than the template's 6-point set: the numeric pipeline is the template's
# weakest point and thin tails are where it loses score.
NUMERIC_PERCENTILES = [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99]

# --- Research ----------------------------------------------------------------
RESEARCH_ENABLED = _env_bool("RESEARCH_ENABLED", True)
RESEARCH_GAP_FILL = _env_bool("RESEARCH_GAP_FILL", PROFILE != "shakeout")
RESEARCH_TIMEOUT_S = _env_float("RESEARCH_TIMEOUT_S", 120.0)
RESEARCH_MAX_CHARS = _env_int("RESEARCH_MAX_CHARS", 24000)
# AskNews free tier: 1,000 calls/month, 4,000 total. "latest news" costs 1 unit,
# archive ("news knowledge") costs 5. Archive is OFF by default to protect quota.
ASKNEWS_USE_ARCHIVE = _env_bool("ASKNEWS_USE_ARCHIVE", False)
ASKNEWS_ARTICLES = _env_int("ASKNEWS_ARTICLES", 6)
# OpenRouter's native web plugin, used as a research source that needs no extra key.
WEB_PLUGIN_ENABLED = _env_bool("WEB_PLUGIN_ENABLED", True)
WEB_PLUGIN_MAX_RESULTS = _env_int("WEB_PLUGIN_MAX_RESULTS", 5 if PROFILE == "shakeout" else 10)

# --- Run shape ---------------------------------------------------------------
# INVARIANT: RUN_DEADLINE_S + PER_QUESTION_DEADLINE_S must stay under the
# workflow's timeout-minutes (18), or a question started just before the run
# deadline can still be running when GitHub kills the job -- potentially between
# the forecast POST and the comment POST, leaving an uncommented forecast.
# 15 + 9 = 24 minutes against a 30-minute job timeout. Enforced by a test.
RUN_DEADLINE_S = _env_float("RUN_DEADLINE_S", 15 * 60)
PER_QUESTION_DEADLINE_S = _env_float("PER_QUESTION_DEADLINE_S", 540.0)
MAX_CONCURRENT_QUESTIONS = _env_int("MAX_CONCURRENT_QUESTIONS", 6)
MAX_QUESTIONS_PER_RUN = _env_int("MAX_QUESTIONS_PER_RUN", 0)  # 0 = no cap

# --- HTTP --------------------------------------------------------------------
HTTP_TIMEOUT_S = _env_float("HTTP_TIMEOUT_S", 60.0)
PUBLISH_TIMEOUT_S = _env_float("PUBLISH_TIMEOUT_S", 20.0)
HTTP_MAX_RETRIES = _env_int("HTTP_MAX_RETRIES", 3)
HTTP_RATE_LIMIT_PER_S = _env_float("HTTP_RATE_LIMIT_PER_S", 1.0)

# --- Safety ------------------------------------------------------------------
# Hard stop: the bot must never post twice on one question. This is a tournament
# rule ("only submit one forecast per question"), not merely good hygiene.
# Refuse to start a run below this much remaining OpenRouter credit. Dying
# mid-round zeroes the tail of the question list.
MIN_CREDIT_FLOOR = _env_float("MIN_CREDIT_FLOOR", 1.50)

PUBLISH = _env_bool("PUBLISH", True)
# Set ONLY by main.py's --force, which itself refuses to run outside the sandbox
# and outside dry-run. Never set this in CI.
FORCE_REFORECAST = _env_bool("FORCE_REFORECAST", False)
COMMENT_IS_PRIVATE = _env_bool("COMMENT_IS_PRIVATE", True)
COMMENT_MAX_CHARS = _env_int("COMMENT_MAX_CHARS", 12000)
