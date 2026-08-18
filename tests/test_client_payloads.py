"""Assert the exact bytes we put on the wire.

Every shape here was read off Metaculus' own SDK source. A wrong key name or a
list-vs-object mistake fails server-side with no useful client error, so these
tests exist to catch it on the laptop instead of in the tournament.
"""
import httpx
import pytest

from metaculus_bot.metaculus_client import MetaculusClient


def client_with(handler):
    c = MetaculusClient(token="tok")
    c._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers=dict(c._client.headers),
        base_url="",
    )
    return c


@pytest.mark.asyncio
async def test_auth_header_is_Token_not_Bearer():
    """'Bearer' yields a 403 indistinguishable from being unauthenticated."""
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": 7})

    c = client_with(handler)
    await c.user_id()
    assert seen["auth"] == "Token tok"
    await c.aclose()


@pytest.mark.asyncio
async def test_forecast_body_is_a_list_and_uses_question_id():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={})

    c = client_with(handler)
    await c.post_forecast(29635, {"probability_yes": 0.37})
    assert captured["url"].endswith("/questions/forecast/")
    body = captured["body"]
    assert isinstance(body, list), "forecast body MUST be a JSON array"
    assert body[0]["question"] == 29635
    assert body[0]["source"] == "api"
    assert body[0]["probability_yes"] == 0.37
    # Sparse form: unused keys are omitted, matching the SDK.
    assert "continuous_cdf" not in body[0]
    await c.aclose()


@pytest.mark.asyncio
async def test_comment_body_is_an_object_and_uses_post_id():
    captured = {}

    def handler(request):
        import json
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={})

    c = client_with(handler)
    await c.post_comment(29783, "reasoning here")
    assert captured["url"].endswith("/comments/create/")
    body = captured["body"]
    assert isinstance(body, dict), "comment body MUST be a JSON object, not a list"
    assert body["on_post"] == 29783          # POST id, not question id
    assert body["is_private"] is True
    assert body["included_forecast"] is True
    await c.aclose()


@pytest.mark.asyncio
async def test_comment_is_never_blind_retried():
    """Comments ACCUMULATE where forecasts overwrite. Retrying a comment that
    actually landed posts it twice, which looks like spam on a public leaderboard."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    c = client_with(handler)
    with pytest.raises(Exception):
        await c.post_comment(1, "x")
    assert calls["n"] == 1, f"comment POST was attempted {calls['n']} times; must be exactly 1"
    await c.aclose()


@pytest.mark.asyncio
async def test_4xx_fails_fast_without_retrying():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    c = client_with(handler)
    with pytest.raises(Exception):
        await c.post_forecast(1, {"probability_yes": 0.5})
    assert calls["n"] == 1
    await c.aclose()


@pytest.mark.asyncio
async def test_open_questions_paginates_and_stops():
    pages = {"n": 0}

    def handler(request):
        pages["n"] += 1
        if pages["n"] == 1:
            return httpx.Response(200, json={
                "results": [{"id": i} for i in range(100)], "next": "http://x/?offset=100"})
        return httpx.Response(200, json={"results": [{"id": 999}], "next": None})

    c = client_with(handler)
    out = await c.open_questions("minibench")
    assert len(out) == 101
    assert pages["n"] == 2
    await c.aclose()


@pytest.mark.asyncio
async def test_forecast_type_is_never_sent():
    """Verified live: sending forecast_type drops GROUP posts from the response,
    which silently zeroes any tournament built from them (e.g. Market Pulse)."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"results": [], "next": None})

    c = client_with(handler)
    await c.open_questions(33022)
    url = seen["url"]
    assert "forecast_type" not in url
    assert "tournaments=33022" in url
    assert "statuses=open" in url
    await c.aclose()


async def test_comment_ledger_queries_private_AND_public():
    """Verified live 2026-08-18: /comments/?author= EXCLUDES private comments,
    and we post private ones. A single query makes the bot think it never
    commented, so it re-comments every run -- and comments accumulate. This is
    the highest-consequence bug the live smoke test caught."""
    queries = []

    def handler(request):
        queries.append(str(request.url))
        if "users/me" in request.url.path:
            return httpx.Response(200, json={"id": 307005})
        priv = "is_private=true" in str(request.url)
        return httpx.Response(200, json={
            "results": [{"on_post": 111 if priv else 222}], "next": None})

    c = client_with(handler)
    seen = await c.commented_post_ids()
    comment_queries = [q for q in queries if "/comments/" in q]
    assert len(comment_queries) == 2, "ledger must make BOTH a private and a public pass"
    assert any("is_private=true" in q for q in comment_queries), "no private pass"
    assert any("is_private" not in q for q in comment_queries), "no public pass"
    assert seen == {111, 222}, "ledger must union both result sets"
    await c.aclose()
