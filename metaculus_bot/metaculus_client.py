"""Our own thin Metaculus HTTP client. httpx only.

Every endpoint, header and payload below was read off Metaculus's own SDK
(forecasting-tools) and bot template, then re-audited against that source.
We do not depend on the SDK at runtime, for three reasons:
  1. its git main is months ahead of its PyPI release, so either pin drifts;
  2. prize eligibility requires the operator to explain this code on a live
     screen-share, and 350 readable lines are explainable where a framework
     is not;
  3. the numeric CDF path is the known failure point and we want full control.

DELIBERATE DIVERGENCES FROM THE SDK, each for a reason:
  * The SDK sleeps uniform(3.5, 4.5)s before EVERY request. At ~100 questions
    that is 12-15 minutes of pure sleep, most of a 20-minute tick. We use a
    token bucket instead and honour Retry-After.
  * The SDK retries every non-2xx three times, including 400s. Blind-retrying a
    comment POST posts it three times, because comments ACCUMULATE while
    forecasts overwrite. We fail fast on 4xx and never blind-retry a comment.
  * The SDK logs "Posted comment" before raise_for_status, so it logs success
    on failure. We check first.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)

# Fail fast: the request was wrong, or we are not allowed. Retrying cannot help
# and, for comments, actively harms.
NO_RETRY_STATUS = {400, 401, 404, 405, 409, 422}
# 403 IS retried: a Cloudflare/WAF-flavoured 403 on a healthy key is a real,
# observed transient on this host.
RETRY_STATUS = {403, 408, 429, 500, 502, 503, 504}


class MetaculusError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class _RateLimiter:
    """Simple async token bucket. Smooths bursts without the SDK's flat sleep."""

    def __init__(self, per_second: float):
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_at = now + self._min_interval


class MetaculusClient:
    def __init__(self, token: str | None = None, base_url: str | None = None):
        self.token = (token or config.METACULUS_TOKEN or "").strip()
        if not self.token:
            raise MetaculusError(
                "METACULUS_TOKEN is not set. The Metaculus API returns 403 to "
                "unauthenticated clients, so nothing can run without it. "
                "Create one at Settings -> My Forecasting Bots -> Create a Bot."
            )
        self.base_url = (base_url or config.METACULUS_API_BASE_URL).rstrip("/")
        self._limiter = _RateLimiter(config.HTTP_RATE_LIMIT_PER_S)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.HTTP_TIMEOUT_S),
            headers={
                # "Token", NOT "Bearer". Getting this wrong yields a 403 that
                # looks exactly like an unauthenticated request.
                "Authorization": f"Token {self.token}",
                "Accept-Language": "en",
                "Content-Type": "application/json",
                "User-Agent": "metaculus-forecasting-bot/1.0",
            },
            follow_redirects=True,
        )
        self._user_id: int | None = None

    async def __aenter__(self) -> "MetaculusClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- core request ---------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        json_body: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        attempts = config.HTTP_MAX_RETRIES if max_retries is None else max_retries
        last: Exception | None = None

        for attempt in range(attempts + 1):
            await self._limiter.acquire()
            try:
                resp = await self._client.request(
                    method, url, params=params, json=json_body,
                    timeout=timeout or config.HTTP_TIMEOUT_S,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt >= attempts:
                    raise MetaculusError(f"{method} {path} transport failure: {exc}") from exc
                await self._backoff(attempt, None)
                continue

            if resp.status_code < 300:
                if not resp.content:
                    return None
                try:
                    return resp.json()
                except ValueError:
                    return resp.text

            body = (resp.text or "")[:400]
            if resp.status_code in NO_RETRY_STATUS or attempt >= attempts:
                raise MetaculusError(
                    f"{method} {path} -> HTTP {resp.status_code}: {body}",
                    status=resp.status_code, body=body,
                )
            if resp.status_code not in RETRY_STATUS:
                raise MetaculusError(
                    f"{method} {path} -> HTTP {resp.status_code}: {body}",
                    status=resp.status_code, body=body,
                )
            log.warning("%s %s -> HTTP %s, retrying (%d/%d)",
                        method, path, resp.status_code, attempt + 1, attempts)
            await self._backoff(attempt, resp.headers.get("Retry-After"))

        raise MetaculusError(f"{method} {path} failed after {attempts} retries: {last}")

    @staticmethod
    async def _backoff(attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(2.0 * (2 ** attempt), 30.0) + random.uniform(0, 1.5))

    # -- identity -------------------------------------------------------------
    async def user_id(self) -> int:
        if self._user_id is None:
            data = await self._request("GET", "/users/me/")
            self._user_id = int(data["id"])
        return self._user_id

    # -- reads ----------------------------------------------------------------
    async def open_questions(self, tournament: str | int) -> list[dict[str, Any]]:
        """Every OPEN post in a tournament, following DRF pagination.

        The tournament may be a numeric id or a slug; the API accepts both.

        WE DELIBERATELY DO NOT SEND forecast_type. Verified live on 2026-08-18:
        adding it drops GROUP posts from the response entirely, because a group
        post carries no top-level question object to match against. Market Pulse
        26Q3 is composed ENTIRELY of group posts, so the filter silently reduced
        that whole tournament to zero questions. We fetch everything and filter
        in parse_post_questions(), which is strictly more inclusive and cannot
        hide a question type from us.
        """
        results: list[dict[str, Any]] = []
        offset = 0
        page_size = 100  # hard server maximum
        while True:
            params = [
                ("limit", str(page_size)),
                ("offset", str(offset)),
                ("order_by", "-published_at"),
                ("statuses", "open"),
                ("tournaments", str(tournament)),
                ("with_cp", "true"),
                ("include_description", "true"),
            ]
            data = await self._request("GET", "/posts/", params=params)
            if not isinstance(data, dict):
                break
            page = data.get("results") or []
            results.extend(page)
            # Terminate on either signal. Correct under both envelope shapes:
            # if "next" is absent, .get() is None and we stop here anyway.
            if not page or data.get("next") is None or len(page) < page_size:
                break
            offset += page_size
            if offset > 5000:  # runaway guard
                log.warning("pagination guard tripped at offset %d", offset)
                break
        return results

    async def commented_post_ids(self) -> set[int]:
        """Post ids that already carry a comment from THIS bot.

        This is the durable half of the ledger. It comes from Metaculus, not
        from disk, so an ephemeral GitHub runner reconstructs it in one pass.

        🔴 TWO QUERIES ARE REQUIRED, NOT ONE. Verified live on 2026-08-18:
        GET /comments/?author=<id> silently EXCLUDES private comments, and we
        post private comments by design. With a single query the bot believes
        it has never commented and re-comments on every run -- and comments
        ACCUMULATE where forecasts overwrite, so that is duplicate spam on
        every question, three times an hour. There is no combined filter:
        is_private=true returns private, the default returns public, and
        repeating the parameter returns neither.
        """
        uid = await self.user_id()
        seen: set[int] = set()
        for params_extra in ({"is_private": "true"}, {}):
            offset = 0
            while True:
                params = {"author": uid, "limit": 100, "offset": offset, **params_extra}
                data = await self._request("GET", "/comments/", params=params)
                if not isinstance(data, dict):
                    break
                page = data.get("results") or []
                for c in page:
                    pid = c.get("on_post")
                    if pid is None and isinstance(c.get("post"), dict):
                        pid = c["post"].get("id")
                    if pid is not None:
                        try:
                            seen.add(int(pid))
                        except (TypeError, ValueError):
                            pass
                if not page or data.get("next") is None or len(page) < 100:
                    break
                offset += 100
                if offset > 20000:
                    break
        return seen

    # -- writes ---------------------------------------------------------------
    async def post_forecast(self, question_id: int, payload: dict[str, Any]) -> None:
        """POST /questions/forecast/ — body is a JSON ARRAY, takes the QUESTION id.

        Idempotent server-side: a later forecast overwrites an earlier one. That
        is why forecast goes first and the comment second.
        """
        body = [{"question": int(question_id), "source": "api", **payload}]
        await self._request(
            "POST", "/questions/forecast/", json_body=body,
            timeout=config.PUBLISH_TIMEOUT_S, max_retries=1,
        )

    async def post_comment(self, post_id: int, text: str, *, private: bool | None = None) -> None:
        """POST /comments/create/ — body is a JSON OBJECT, takes the POST id.

        NOT idempotent: comments accumulate. Never blind-retry this. The caller
        re-reads commented_post_ids() before any repair attempt.
        """
        is_private = config.COMMENT_IS_PRIVATE if private is None else private
        body = {
            "on_post": int(post_id),
            "text": text,
            "is_private": bool(is_private),
            "included_forecast": True,
        }
        await self._request(
            "POST", "/comments/create/", json_body=body,
            timeout=config.PUBLISH_TIMEOUT_S, max_retries=0,
        )
