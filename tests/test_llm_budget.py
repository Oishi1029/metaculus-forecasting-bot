"""The ensemble-member retry budget.

This setting has failed in both directions in production -- unbounded retries
ate the per-question deadline (2026-10-16), then zero retries let a 0.3s HTTP
503 kill whole questions (2026-11-02). These tests pin both ends down so the
next edit cannot quietly reintroduce either.
"""

import asyncio

import httpx
import pytest

from metaculus_bot import config
from metaculus_bot.llm import LLMClient, LLMError


class _Stub:
    """Stands in for httpx.AsyncClient, recording what each attempt was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.timeouts = []

    async def post(self, url, json=None, timeout=None):
        self.timeouts.append(timeout)
        status, body = self.responses.pop(0) if self.responses else (200, "late")
        if status == 200:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": body}}]},
                request=httpx.Request("POST", url))
        return httpx.Response(status, text=body, request=httpx.Request("POST", url))

    async def aclose(self):
        pass

    @property
    def calls(self):
        return len(self.timeouts)


@pytest.fixture
def client(monkeypatch):
    # Capture the real sleep BEFORE patching: llm.py shares the asyncio module
    # object, so a lambda that called asyncio.sleep would call itself.
    real_sleep = asyncio.sleep
    monkeypatch.setattr("metaculus_bot.llm.asyncio.sleep",
                        lambda *_a, **_k: real_sleep(0))
    return LLMClient(api_key="k")


def test_fast_transient_503_is_retried(client):
    """The 2026-11-02 failure. A 503 that arrives in milliseconds costs almost
    none of the budget, so giving up on it throws away a whole question."""
    stub = _Stub([(503, "Service temporarily overloaded"), (200, "RECOVERED")])
    client._client = stub
    out = asyncio.run(client.complete("m", "p", max_retries=2, budget_s=600.0))
    assert out == "RECOVERED"
    assert stub.calls == 2


def test_retry_is_abandoned_when_the_budget_is_spent(client):
    """The 2026-10-16 failure. With no time left a retry can only overrun the
    per-question deadline, so it must not be started."""
    stub = _Stub([(503, "overloaded"), (200, "TOO LATE")])
    client._client = stub
    with pytest.raises(LLMError):
        asyncio.run(client.complete("m", "p", max_retries=2, budget_s=0.5))
    assert stub.calls == 1, "a retry was started with no budget left"


def test_each_attempt_timeout_is_clamped_to_the_remaining_budget(client):
    """A retry must never be allowed to run longer than the question has left."""
    stub = _Stub([(503, "overloaded"), (200, "ok")])
    client._client = stub
    asyncio.run(client.complete("m", "p", max_retries=2, budget_s=90.0))
    assert all(t <= 90.0 for t in stub.timeouts), stub.timeouts
    assert stub.timeouts[0] == pytest.approx(90.0, abs=1.0)


def test_a_4xx_is_not_retried(client):
    """A bad request will not improve on retry; spending budget on it steals
    time from models that could still answer."""
    stub = _Stub([(400, "bad request"), (200, "unreachable")])
    client._client = stub
    with pytest.raises(LLMError):
        asyncio.run(client.complete("m", "p", max_retries=2, budget_s=600.0))
    assert stub.calls == 1


def test_model_budget_fits_inside_the_per_question_deadline():
    """Research runs before the ensemble and salvage/aggregation/publishing
    after it. If the budget consumed the whole deadline we would be back to a
    slow model killing the question."""
    used = config.RESEARCH_TIMEOUT_S + config.MODEL_BUDGET_S
    assert used < config.PER_QUESTION_DEADLINE_S, (
        f"research {config.RESEARCH_TIMEOUT_S}s + model budget "
        f"{config.MODEL_BUDGET_S}s = {used}s leaves nothing of the "
        f"{config.PER_QUESTION_DEADLINE_S}s question deadline")
    assert config.PER_QUESTION_DEADLINE_S - used >= 90.0, (
        "under 90s of headroom for salvage, aggregation and the forecast + "
        "comment POSTs")


def test_the_ensemble_actually_retries():
    """Guards the regression directly: _one_model must pass a non-zero retry
    count AND a budget. Either one alone has already caused an outage."""
    import inspect

    from metaculus_bot.forecaster import Forecaster
    src = inspect.getsource(Forecaster._one_model)
    assert "max_retries=0" not in src, (
        "zero retries lets a 0.3s 503 kill the question (2026-11-02)")
    assert "budget_s=" in src, (
        "retries without a budget ate the per-question deadline (2026-10-16)")
    assert config.ENSEMBLE_MAX_RETRIES >= 1


def test_free_ensemble_survives_losing_one_vendor():
    """A retired model halved the ensemble and left it single-vendor, so one
    NVIDIA 503 took every member down at once. Keep more than one vendor and
    enough members that one retirement is not fatal."""
    models = config.FREE_MODELS
    assert len(models) == len(set(models)), f"duplicate entries: {models}"
    assert len(models) >= 3, (
        f"only {len(models)} free models; one retirement leaves too few")
    vendors = {m.split("/", 1)[0] for m in models}
    assert len(vendors) >= 2, (
        f"every free model is from {vendors} -- one provider outage ends the "
        "run, which is what happened on 2026-11-02")
