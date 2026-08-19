"""Research provider registry: 0..N sources, key-gated, failures isolated.

WHY THIS IS THE MOST IMPORTANT MODULE IN THE BOT
In Metaculus' own survey of 39 bot makers, the number of DISTINCT research
sources was the strongest correlate of prize-winning (r = 0.42, p ~ 0.006) and
"the only [result] with an effect size large enough to remain robust to sample
sensitivity". Winners averaged 1.75 sources; non-winners 1.00. No individual
provider predicted anything -- AskNews ranked 1st one season and 58th the next.
Diversity is the lever, not any particular vendor.

So: every provider is optional, all live providers run CONCURRENTLY, and one
provider failing never fails the question. With only OPENROUTER_API_KEY set we
still get two genuinely distinct search indices, because Perplexity's Sonar
models are reachable through OpenRouter without a separate Perplexity key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from . import config, prompts
from .llm import LLMClient

log = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    source: str
    text: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


class ResearchRegistry:
    """Detects which providers are live at startup and fans out to all of them."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._providers: list[tuple[str, Callable[[str, str], Awaitable[str]]]] = []

        if config.WEB_PLUGIN_ENABLED and config.OPENROUTER_API_KEY:
            self._providers.append(("openrouter-web", self._openrouter_web))
            # Perplexity Sonar via OpenRouter: a DIFFERENT search index reached
            # with the SAME key. Cheapest possible source diversity.
            self._providers.append(("perplexity-sonar", self._perplexity_via_openrouter))
        if config.ASKNEWS_CLIENT_ID and config.ASKNEWS_SECRET:
            self._providers.append(("asknews", self._asknews))
        if config.EXA_API_KEY:
            self._providers.append(("exa", self._exa))

    @property
    def provider_names(self) -> list[str]:
        return [n for n, _ in self._providers]

    def describe(self) -> str:
        if not self._providers:
            return "NO research providers live -- the bot will forecast on model priors alone"
        return f"{len(self._providers)} research provider(s) live: {', '.join(self.provider_names)}"

    # -- the fan-out ----------------------------------------------------------
    async def gather(self, question_text: str, resolution_criteria: str) -> tuple[str, list[str]]:
        """Run every live provider concurrently. Returns (merged_text, sources_used)."""
        if not config.RESEARCH_ENABLED or not self._providers:
            return "", []

        async def guarded(name: str, fn: Callable[[str, str], Awaitable[str]]) -> ResearchResult:
            try:
                text = await asyncio.wait_for(
                    fn(question_text, resolution_criteria), timeout=config.RESEARCH_TIMEOUT_S
                )
                return ResearchResult(name, text or "")
            except asyncio.TimeoutError:
                return ResearchResult(name, "", "timed out")
            except Exception as exc:                      # noqa: BLE001 - isolation is the point
                return ResearchResult(name, "", f"{type(exc).__name__}: {exc}")

        results = await asyncio.gather(*(guarded(n, f) for n, f in self._providers))

        chunks: list[str] = []
        used: list[str] = []
        for r in results:
            if r.ok:
                used.append(r.source)
                chunks.append(f"### Source: {r.source}\n{r.text.strip()}")
            else:
                log.warning("research provider %s unusable: %s", r.source, r.error or "empty")

        merged = "\n\n".join(chunks)[: config.RESEARCH_MAX_CHARS]
        return merged, used

    async def gap_fill(self, question_text: str, resolution_criteria: str,
                       fine_print: str, first_pass: str, today: str) -> str:
        """One fixed second pass: name the gaps, then search for them.

        Deliberately a FIXED two-pass shape, not a free-roaming agent. Metaculus'
        survey is blunt about the alternative: "agentic researcher with a bunch
        of tools just doesn't work very well compared to dedicated pipelines."
        """
        if not config.RESEARCH_GAP_FILL or not self._providers or len(first_pass) < 200:
            return ""
        try:
            analysis = await self.llm.complete(
                config.SALVAGE_MODEL,
                prompts.render(
                    prompts.GAP_ANALYZER_PROMPT,
                    question_text=question_text,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print or "None",
                    first_pass_research=first_pass[:12000],
                    current_year=today[:4],
                    max_gaps=3,
                ),
                max_tokens=900,
            )
        except Exception as exc:                          # noqa: BLE001
            log.warning("gap analysis failed: %s", exc)
            return ""

        queries = _extract_queries(analysis, limit=3)
        if not queries:
            return ""

        async def one(q: str) -> str:
            try:
                return await asyncio.wait_for(
                    self._openrouter_web(q, ""), timeout=config.RESEARCH_TIMEOUT_S
                )
            except Exception:                             # noqa: BLE001
                return ""

        extra = await asyncio.gather(*(one(q) for q in queries))
        filled = [f"### Gap-fill: {q}\n{t.strip()}" for q, t in zip(queries, extra) if t.strip()]
        return "\n\n".join(filled)[: config.RESEARCH_MAX_CHARS // 2]

    # -- providers ------------------------------------------------------------
    async def _openrouter_web(self, question_text: str, resolution_criteria: str) -> str:
        prompt = prompts.render(
            prompts.RESEARCH_PROMPT,
            question_text=question_text, resolution_criteria=resolution_criteria or "N/A",
        )
        return await self.llm.complete(config.RESEARCH_MODEL, prompt,
                                       web_search=True, max_tokens=2500)

    async def _perplexity_via_openrouter(self, question_text: str, resolution_criteria: str) -> str:
        prompt = prompts.render(
            prompts.RESEARCH_PROMPT,
            question_text=question_text, resolution_criteria=resolution_criteria or "N/A",
        )
        model = config.PERPLEXITY_MODEL
        return await self.llm.complete(model, prompt, max_tokens=2000, max_retries=1)

    async def _asknews(self, question_text: str, resolution_criteria: str) -> str:
        """AskNews via raw HTTP so we keep the dependency set to httpx + numpy.

        QUOTA DISCIPLINE: the tournament free tier is 1,000 calls/month and 4,000
        total. strategy="latest news" costs 1 unit; strategy="news knowledge"
        (the archive) costs 5, and is therefore OFF unless ASKNEWS_USE_ARCHIVE=1.
        VERIFIED LIVE 2026-08-19 against a real tournament key: OAuth2
        client_credentials -> https://auth.asknews.app/oauth2/token (scope=news,
        HTTP Basic, 2h token), then GET /v1/news/search with
        strategy="latest news" and return_type="string". The usable text is at
        the response key "as_string", and the response reports usage.credits == 1,
        confirming the cheap strategy. Still fails soft: a provider failing must
        cost one source, never the question.
        """
        async with httpx.AsyncClient(timeout=45.0) as client:
            tok = await client.post(
                "https://auth.asknews.app/oauth2/token",
                data={"grant_type": "client_credentials", "scope": "news"},
                auth=(config.ASKNEWS_CLIENT_ID, config.ASKNEWS_SECRET),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            tok.raise_for_status()
            access = tok.json()["access_token"]
            headers = {"Authorization": f"Bearer {access}"}

            out: list[str] = []
            hot = await client.get(
                "https://api.asknews.app/v1/news/search",
                params={
                    "query": question_text,
                    "n_articles": config.ASKNEWS_ARTICLES,
                    "return_type": "string",
                    "strategy": "latest news",
                    # historical is the ACTUAL 1-vs-5 credit switch; "latest news"
                    # is documented to set it False implicitly, but sending it
                    # explicitly means a change to the strategy defaults upstream
                    # can never quintuple our billing without us noticing.
                    "historical": "false",
                    # Cached responses bill at 0.25x. Our catch-up scans re-research
                    # a question whenever a previous tick failed on it, so this is a
                    # real saving; 1h is short enough that the news is still fresh
                    # for a question whose window is only ~3 hours.
                    "try_cache": "1h",
                },
                headers=headers,
            )
            if hot.status_code < 300:
                payload = hot.json()
                _log_credits(payload, "latest")
                out.append("Latest news (last 48h):\n" + _asknews_text(payload))
            else:
                log.warning("asknews hot search -> HTTP %s: %s",
                            hot.status_code, (hot.text or "")[:160])

            if config.ASKNEWS_USE_ARCHIVE:
                await asyncio.sleep(10)   # free tier is rate limited
                arch = await client.get(
                    "https://api.asknews.app/v1/news/search",
                    params={"query": question_text, "n_articles": 10,
                            "return_type": "string", "strategy": "news knowledge"},
                    headers=headers,
                )
                if arch.status_code < 300:
                    payload = arch.json()
                    _log_credits(payload, "archive")
                    # 60 days, not 160 -- the "160" in Metaculus' own SDK comment
                    # is stale against the live API spec.
                    out.append("Archive (past ~60 days):\n" + _asknews_text(payload))
            return "\n\n".join(out)

    async def _exa(self, question_text: str, resolution_criteria: str) -> str:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                headers={"accept": "application/json", "content-type": "application/json",
                         "x-api-key": config.EXA_API_KEY},
                json={"query": question_text, "type": "auto", "useAutoprompt": True,
                      "numResults": 8, "livecrawl": "always",
                      "contents": {"text": {"maxCharacters": 1200},
                                   "highlights": {"query": question_text, "numSentences": 3}}},
            )
            resp.raise_for_status()
            parts: list[str] = []
            for r in (resp.json().get("results") or [])[:8]:
                title = r.get("title") or "(untitled)"
                url = r.get("url") or ""
                date = r.get("publishedDate") or ""
                body = " ".join(r.get("highlights") or []) or (r.get("text") or "")[:800]
                parts.append(f"- {title} ({date}) {url}\n  {body}")
            return "\n".join(parts)


def _log_credits(payload: Any, label: str) -> None:
    """Surface what a call actually cost. The free tournament tier is 1,000/month
    and 4,000 total, and the difference between the hot and archive paths is 5x,
    so silent overspend is a real way to run out mid-round."""
    try:
        credits = (payload or {}).get("usage", {}).get("credits")
        if credits is not None:
            log.info("asknews %s search cost %s credit(s)", label, credits)
    except Exception:                                     # noqa: BLE001
        pass


def _asknews_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload[:6000]
    if isinstance(payload, dict):
        for key in ("as_string", "response", "text", "content"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v[:6000]
        arts = payload.get("articles") or payload.get("as_dicts") or []
        if isinstance(arts, list):
            return "\n".join(
                f"- {a.get('eng_title') or a.get('title')} ({a.get('pub_date','')}) "
                f"{a.get('article_url','')}\n  {(a.get('summary') or '')[:500]}"
                for a in arts[:10] if isinstance(a, dict)
            )[:6000]
    return ""


def _extract_queries(analysis: str, limit: int = 3) -> list[str]:
    """Pull search queries out of the gap analyser's output, however it framed them."""
    try:
        m = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", analysis, re.DOTALL)
        blob = json.loads(m.group(1)) if m else json.loads(analysis)
        items = blob.get("gaps", blob) if isinstance(blob, dict) else blob
        out = []
        for it in items if isinstance(items, list) else []:
            q = it.get("search_query") or it.get("query") if isinstance(it, dict) else str(it)
            if q:
                out.append(str(q).strip())
        if out:
            return out[:limit]
    except Exception:                                     # noqa: BLE001
        pass
    lines = [re.sub(r'^[\s\-\*\d\.\)"]+|"$', "", ln).strip()
             for ln in analysis.splitlines() if 12 < len(ln.strip()) < 220]
    return [ln for ln in lines if ln][:limit]
