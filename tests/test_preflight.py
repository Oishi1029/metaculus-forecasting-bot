"""A model id is a string in a config file, and vendors rename and retire them.

Caught live 2026-08-19: "perplexity/sonar-reasoning" 404s (the real id is
"-reasoning-pro"). It had shipped in the competition profile and would have cut
research from two sources to one -- silently, with the run still reporting
success. Source diversity is the strongest evidenced predictor of score, so
that is real points lost behind a green checkmark.
"""
import httpx
import pytest

from metaculus_bot import config, preflight


def _patch_catalogue(monkeypatch, ids):
    async def fake(timeout=20.0):
        return set(ids)
    monkeypatch.setattr(preflight, "available_model_ids", fake)


async def test_missing_model_is_reported(monkeypatch):
    _patch_catalogue(monkeypatch, ["openai/real-model"])
    out = await preflight.check_models()
    assert out["missing"], "a bogus id must be reported as missing"


async def test_all_present_reports_nothing_missing(monkeypatch):
    every = (set(config.models_for_profile())
             | {config.SALVAGE_MODEL, config.PERPLEXITY_MODEL, config.RESEARCH_MODEL})
    _patch_catalogue(monkeypatch, every)
    out = await preflight.check_models()
    assert out["missing"] == []


async def test_unreachable_catalogue_does_not_block_the_run(monkeypatch):
    """A lookup endpoint being briefly down must never stop us forecasting."""
    _patch_catalogue(monkeypatch, [])
    out = await preflight.check_models()
    assert out["missing"] == []
    assert out["ok"], "must fall through to the configured list"


def test_unavailable_models_are_dropped_from_the_ensemble():
    live = preflight.usable_ensemble(["a", "b", "c"], ["b"])
    assert live == ["a", "c"]


def test_ensemble_is_never_emptied():
    """Better to try and fail loudly than to forecast nothing at all."""
    live = preflight.usable_ensemble(["a", "b"], ["a", "b"])
    assert live == ["a", "b"]


def test_perplexity_model_id_is_not_the_dead_one():
    assert config.PERPLEXITY_MODEL != "perplexity/sonar-reasoning", \
        "that id 404s; the reasoning variant is perplexity/sonar-reasoning-pro"
    assert config.PERPLEXITY_MODEL.startswith("perplexity/")


async def test_credit_uses_the_lower_of_balance_and_key_cap(monkeypatch):
    """/key reports the KEY's monthly cap minus usage; it knows nothing about
    whether the account has money. Measured 2026-08-19: key said $62.93 while the
    account held $42.93. Trusting the higher number lets the bot start a round it
    cannot finish, and an unfinished round's tail is zeros in a squared sum."""
    import httpx
    from metaculus_bot import preflight as pf

    def handler(request):
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"total_credits": 60, "total_usage": 17.07}})
        return httpx.Response(200, json={"data": {"limit_remaining": 62.93, "limit": 80}})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(pf.config, "OPENROUTER_API_KEY", "k")
    got = await pf.openrouter_credit()
    assert got == pytest.approx(42.93, abs=0.01), f"took the wrong number: {got}"


async def test_credit_returns_none_when_unreachable(monkeypatch):
    """A lookup outage must never stop the bot forecasting."""
    import httpx
    from metaculus_bot import preflight as pf

    def handler(request):
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(pf.config, "OPENROUTER_API_KEY", "k")
    assert await pf.openrouter_credit() is None


def test_asknews_registers_only_when_both_credentials_present():
    """VERIFIED live 2026-08-19 against a real key: OAuth2 client_credentials at
    https://auth.asknews.app/oauth2/token with scope=news, then
    GET /v1/news/search with strategy="latest news" -> usable text at the
    response key "as_string", and usage.credits == 1 (the cheap strategy).
    AskNews needs BOTH id and secret; registering on one is a guaranteed 401
    that silently costs a research source."""
    import importlib
    from metaculus_bot import research as rmod
    from metaculus_bot import config as cmod

    class _FakeLLM:
        pass

    for cid, sec, expect in (("id", "sec", True), ("id", "", False),
                             ("", "sec", False), ("", "", False)):
        old_id, old_sec = cmod.ASKNEWS_CLIENT_ID, cmod.ASKNEWS_SECRET
        cmod.ASKNEWS_CLIENT_ID, cmod.ASKNEWS_SECRET = cid, sec
        try:
            reg = rmod.ResearchRegistry(_FakeLLM())
            assert ("asknews" in reg.provider_names) is expect, (cid, sec)
        finally:
            cmod.ASKNEWS_CLIENT_ID, cmod.ASKNEWS_SECRET = old_id, old_sec


def test_archive_strategy_stays_off_by_default():
    """The archive strategy costs 5 quota credits per call against a 1,000/month
    and 4,000-total tournament allowance; latest-news costs 1 (measured:
    usage.credits == 1). A 57-question round is ~57 credits, not ~285."""
    from metaculus_bot import config
    assert config.ASKNEWS_USE_ARCHIVE is False
