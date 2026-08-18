"""One async chat call against any OpenAI-compatible endpoint. httpx only.

We talk to OpenRouter directly rather than through a vendor SDK so that the
runtime dependency set stays {httpx, numpy} and nothing drifts underneath us.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = (api_key or config.OPENROUTER_API_KEY or "").strip()
        if not self.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. The bot needs at least one LLM key."
            )
        self.base_url = (base_url or config.OPENROUTER_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.LLM_TIMEOUT_S),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # OpenRouter uses these for attribution; harmless elsewhere.
                "HTTP-Referer": "https://github.com/",
                "X-Title": "metaculus-forecasting-bot",
            },
        )
        self.total_calls = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: bool = False,
        max_retries: int = 2,
    ) -> str:
        """Return the assistant's text. Raises LLMError if every attempt fails."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        }
        if web_search:
            # OpenRouter's native web plugin. This is a research source that
            # requires no key beyond the OpenRouter one, which matters because
            # source DIVERSITY is the strongest evidenced predictor of score.
            body["plugins"] = [{
                "id": "web",
                "max_results": config.WEB_PLUGIN_MAX_RESULTS,
                "engine": "native",
            }]

        last: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(f"{self.base_url}/chat/completions", json=body)
                if resp.status_code >= 300:
                    text = (resp.text or "")[:300]
                    # 4xx other than 429 will not improve on retry.
                    if resp.status_code != 429 and resp.status_code < 500:
                        raise LLMError(f"{model} -> HTTP {resp.status_code}: {text}")
                    raise httpx.HTTPError(f"HTTP {resp.status_code}: {text}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise httpx.HTTPError(f"{model} returned no choices: {str(data)[:200]}")
                content = (choices[0].get("message") or {}).get("content") or ""
                if not content.strip():
                    raise httpx.HTTPError(f"{model} returned empty content")
                self.total_calls += 1
                return content
            except LLMError:
                raise
            except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as exc:
                last = exc
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(2.0 * (2 ** attempt), 20.0) + random.uniform(0, 1.0))

        raise LLMError(f"{model} failed after {max_retries + 1} attempts: {last}")
