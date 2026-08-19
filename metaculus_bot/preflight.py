"""Startup validation of configured model ids against OpenRouter's catalogue.

WHY THIS EXISTS
A model id is a string in a config file, and vendors rename and retire them.
This bot runs unattended for months, so a retired id would fail every call
silently: research would quietly drop to one source, or the ensemble to two
models, and the run would still report success. That is the worst kind of
failure -- degraded scoring with a green checkmark.

Caught for real on 2026-08-19: "perplexity/sonar-reasoning" 404s (the reasoning
variant is "-reasoning-pro"). It had been in the competition profile since the
build, and would have removed one of only two research sources -- source
diversity being the strongest evidenced predictor of tournament score.

The catalogue endpoint needs no auth and is cheap. We fail loudly on a bad
forecasting model, and warn-and-continue on a bad research model.
"""

from __future__ import annotations

import logging

import httpx

from . import config

log = logging.getLogger(__name__)

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"


async def available_model_ids(timeout: float = 20.0) -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(CATALOGUE_URL, headers={"Accept": "application/json"})
            r.raise_for_status()
            return {m["id"] for m in (r.json().get("data") or []) if m.get("id")}
    except Exception as exc:                              # noqa: BLE001
        log.warning("could not fetch the OpenRouter catalogue (%s); "
                    "skipping model validation", exc)
        return set()


async def check_models() -> dict[str, list[str]]:
    """Validate every configured id. Returns {"ok": [...], "missing": [...]}.

    An empty catalogue (network failure) is treated as "cannot check", not as
    "everything is missing" -- we must never refuse to run because a lookup
    endpoint was briefly down.
    """
    ids = await available_model_ids()
    configured = list(dict.fromkeys(
        config.models_for_profile()
        + [config.SALVAGE_MODEL, config.PERPLEXITY_MODEL, config.RESEARCH_MODEL]
    ))
    if not ids:
        return {"ok": configured, "missing": []}

    ok = [m for m in configured if m in ids]
    missing = [m for m in configured if m not in ids]
    for m in missing:
        log.error("configured model %r is NOT in the OpenRouter catalogue -- "
                  "every call to it will fail", m)
    return {"ok": ok, "missing": missing}


def usable_ensemble(models: list[str], missing: list[str]) -> list[str]:
    """Drop unavailable models, but never return an empty ensemble."""
    live = [m for m in models if m not in missing]
    if not live:
        log.error("NO configured forecasting model is available; "
                  "attempting the configured list anyway")
        return models
    if len(live) < len(models):
        log.warning("forecasting with %d of %d configured models: %s",
                    len(live), len(models), ", ".join(live))
    return live


async def openrouter_credit(timeout: float = 15.0) -> float | None:
    """Spendable credit: the MINIMUM of the account balance and the key's own cap.

    Two different numbers, and either can bind. /api/v1/key reports
    `limit_remaining`, which is only the KEY's monthly cap minus its usage -- it
    knows nothing about whether the account has money. Measured 2026-08-19: key
    said $62.93 remaining while the account held $42.93. Trusting the key figure
    would let the bot start a round it cannot pay to finish, and the tail of an
    unfinished round is zeros in a SQUARED sum.
    """
    if not config.OPENROUTER_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
    balance = cap = None
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            try:
                r = await c.get("https://openrouter.ai/api/v1/credits", headers=headers)
                r.raise_for_status()
                d = (r.json() or {}).get("data") or {}
                if d.get("total_credits") is not None:
                    balance = float(d["total_credits"]) - float(d.get("total_usage") or 0)
            except Exception as exc:                      # noqa: BLE001
                log.warning("could not read OpenRouter balance (%s)", exc)
            try:
                r = await c.get("https://openrouter.ai/api/v1/key", headers=headers)
                r.raise_for_status()
                d = (r.json() or {}).get("data") or {}
                if d.get("limit_remaining") is not None:
                    cap = float(d["limit_remaining"])
            except Exception as exc:                      # noqa: BLE001
                log.warning("could not read OpenRouter key cap (%s)", exc)
    except Exception as exc:                              # noqa: BLE001
        log.warning("could not reach OpenRouter (%s); continuing", exc)
        return None

    vals = [v for v in (balance, cap) if v is not None]
    if not vals:
        return None
    if balance is not None and cap is not None and abs(balance - cap) > 0.01:
        log.info("OpenRouter: account balance $%.2f, key cap remaining $%.2f "
                 "-- using the lower", balance, cap)
    return min(vals)
