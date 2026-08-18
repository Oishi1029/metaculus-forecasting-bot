"""Full-pipeline test with both HTTP layers mocked.

This is the closest we can get to a real run without a Metaculus token, and it
is what proves the bot actually functions rather than merely imports. It drives
one binary, one multiple-choice, one numeric and one discrete question all the
way from GET /posts/ to the exact bytes of POST /questions/forecast/ and
POST /comments/create/.
"""
import json

import httpx
import pytest

from metaculus_bot import run as run_mod
from metaculus_bot.llm import LLMClient
from metaculus_bot.metaculus_client import MetaculusClient

# --- fixture questions, shaped like real Metaculus payloads -------------------
POSTS = [
    {"id": 29783, "question": {
        "id": 29635, "type": "binary", "status": "open",
        "title": "Will X happen before 2027?",
        "resolution_criteria": "Resolves YES if X.", "fine_print": "",
        "description": "Background.", "scheduled_close_time": "2026-09-01T00:00:00Z",
        "my_forecasts": {"latest": {"forecast_values": None}, "history": []}}},
    {"id": 31858, "question": {
        "id": 31365, "type": "multiple_choice", "status": "open",
        "title": "Which company leads?", "options": ["NVDA", "AAPL", "MSFT"],
        "resolution_criteria": "Largest by market cap.", "fine_print": "",
        "description": "", "my_forecasts": {"latest": {}, "history": []}}},
    {"id": 31656, "question": {
        "id": 31207, "type": "numeric", "status": "open",
        "title": "How many widgets in 2026?", "unit": "widgets",
        "resolution_criteria": "Per the official report.", "fine_print": "",
        "description": "", "open_lower_bound": False, "open_upper_bound": True,
        "scaling": {"range_min": 0, "range_max": 100, "zero_point": None,
                    "inbound_outcome_count": 200},
        "my_forecasts": {"latest": {}, "history": []}}},
    {"id": 38880, "question": {
        "id": 38195, "type": "discrete", "status": "open",
        "title": "How many rate cuts?",
        "resolution_criteria": "Count of cuts.", "fine_print": "",
        "description": "", "open_lower_bound": False, "open_upper_bound": True,
        "scaling": {"range_min": -0.5, "range_max": 7.5, "zero_point": None,
                    "inbound_outcome_count": 8},
        "my_forecasts": {"latest": {}, "history": []}}},
    # Already forecasted AND already commented -> must be skipped entirely.
    {"id": 40000, "question": {
        "id": 40001, "type": "binary", "status": "open", "title": "Done already",
        "resolution_criteria": "", "fine_print": "", "description": "",
        "my_forecasts": {"latest": {"forecast_values": [0.4, 0.6]}, "history": [{}]}}},
    # Forecasted but NOT commented -> comment-repair path, never re-forecast.
    {"id": 40002, "question": {
        "id": 40003, "type": "binary", "status": "open", "title": "Missing comment",
        "resolution_criteria": "", "fine_print": "", "description": "",
        "my_forecasts": {"latest": {"forecast_values": [0.4, 0.6]}, "history": [{}]}}},
    # Unsupported type -> silently ignored.
    {"id": 41000, "question": {"id": 41001, "type": "date", "status": "open",
                               "title": "When?", "my_forecasts": {}}},
]

LLM_REPLIES = {
    "binary": 'Reasoning...\nMy base rate was 20%. After considering current evidence, '
              "I'm moving to 30% because of Y.\n```json\n"
              '{"question_type":"binary","posterior_prob":0.30}\n```',
    "multiple_choice": 'Reasoning...\n```json\n{"question_type":"multiple_choice",'
                       '"option_probabilities":{"NVDA":0.5,"AAPL":0.3,"MSFT":0.2}}\n```',
    "numeric": 'Reasoning...\n```json\n{"question_type":"numeric","declared_percentiles":'
               '{"0.01":5,"0.025":8,"0.05":12,"0.1":18,"0.2":25,"0.4":38,"0.5":45,'
               '"0.6":52,"0.8":68,"0.9":82,"0.95":95,"0.975":110,"0.99":130}}\n```',
}


class Recorder:
    def __init__(self):
        self.forecasts = []
        self.comments = []


@pytest.fixture
def wired(monkeypatch):
    rec = Recorder()

    def metac_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/users/me/"):
            return httpx.Response(200, json={"id": 12345})
        if path.endswith("/posts/"):
            return httpx.Response(200, json={"results": POSTS, "next": None})
        if path.endswith("/comments/") and request.method == "GET":
            # 40000 already has our comment; 40002 does not.
            return httpx.Response(200, json={"results": [{"on_post": 40000}], "next": None})
        if path.endswith("/questions/forecast/"):
            rec.forecasts.append(json.loads(request.content))
            return httpx.Response(201, json={})
        if path.endswith("/comments/create/"):
            rec.comments.append(json.loads(request.content))
            return httpx.Response(201, json={})
        return httpx.Response(404, text=f"unexpected {path}")

    def llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        # Route on the question title, which is unambiguous. Routing on words
        # like "percentile" fails: the binary prompt's calibration checklist
        # mentions percentiles too.
        if "research assistant" in prompt.lower():
            reply = "Recent reporting indicates a stable trend. Source: example.com, 2026-08-15."
        elif "widgets" in prompt or "rate cuts" in prompt:
            reply = LLM_REPLIES["numeric"]
        elif "Which company leads" in prompt:
            reply = LLM_REPLIES["multiple_choice"]
        else:
            reply = LLM_REPLIES["binary"]
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    real_mc_init = MetaculusClient.__init__
    real_llm_init = LLMClient.__init__

    def mc_init(self, *a, **k):
        real_mc_init(self, token="test-token")
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(metac_handler),
                                         headers=dict(self._client.headers))

    def llm_init(self, *a, **k):
        real_llm_init(self, api_key="test-key")
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(llm_handler),
                                         headers=dict(self._client.headers))

    monkeypatch.setattr(MetaculusClient, "__init__", mc_init)
    monkeypatch.setattr(LLMClient, "__init__", llm_init)
    return rec


async def test_full_run_publishes_every_question_type(wired):
    summary = await run_mod.run_tournament("minibench")

    assert summary.failed == 0, summary.errors
    # 4 new questions forecasted; the already-done one skipped; the orphan repaired.
    assert summary.forecasted == 4, f"expected 4 forecasts, got {summary.forecasted}"
    assert summary.commented == 5, "4 new comments + 1 repair"
    # Both already-forecasted questions are skipped for FORECASTING; one of them
    # still gets a comment repair because its comment is missing.
    assert summary.skipped == 2

    by_qid = {f[0]["question"]: f[0] for f in wired.forecasts}
    assert set(by_qid) == {29635, 31365, 31207, 38195}

    # binary
    assert by_qid[29635]["probability_yes"] == pytest.approx(0.30)
    assert by_qid[29635]["source"] == "api"
    # multiple choice: exact option keys, sums to 1
    mc = by_qid[31365]["probability_yes_per_category"]
    assert set(mc) == {"NVDA", "AAPL", "MSFT"}
    assert sum(mc.values()) == pytest.approx(1.0)
    # numeric: 201 points, monotone, open upper honoured
    cdf = by_qid[31207]["continuous_cdf"]
    assert len(cdf) == 201
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))
    assert cdf[0] == 0.0 and cdf[-1] <= 0.999
    # discrete: 9 points, NOT 201
    dcdf = by_qid[38195]["continuous_cdf"]
    assert len(dcdf) == 9


async def test_every_forecast_carries_a_comment(wired):
    """Hard prize-eligibility rule: a comment on every question forecasted."""
    await run_mod.run_tournament("minibench")
    forecasted_qids = {f[0]["question"] for f in wired.forecasts}
    commented_posts = {c["on_post"] for c in wired.comments}
    qid_to_post = {29635: 29783, 31365: 31858, 31207: 31656, 38195: 38880}
    for qid in forecasted_qids:
        assert qid_to_post[qid] in commented_posts, f"question {qid} forecasted without a comment"
    assert all(c["is_private"] is True for c in wired.comments)
    assert all(c["included_forecast"] is True for c in wired.comments)


async def test_never_forecasts_an_already_forecasted_question(wired):
    """Violating this forfeits all prize money."""
    await run_mod.run_tournament("minibench")
    qids = [f[0]["question"] for f in wired.forecasts]
    assert 40001 not in qids, "re-forecast a completed question"
    assert 40003 not in qids, "re-forecast during comment repair"
    assert len(qids) == len(set(qids)), "same question forecasted twice in one run"


async def test_orphaned_forecast_gets_comment_only_repair(wired):
    await run_mod.run_tournament("minibench")
    assert 40002 in {c["on_post"] for c in wired.comments}
    assert 40000 not in {c["on_post"] for c in wired.comments}


async def test_dry_run_publishes_nothing(wired):
    summary = await run_mod.run_tournament("minibench", dry_run=True)
    assert wired.forecasts == [] and wired.comments == []
    assert summary.forecasted == 4


async def test_one_broken_question_does_not_abort_the_run(wired, monkeypatch):
    """Coverage dominates: an exception that kills the run zeroes every remaining
    question, which is strictly worse than one mediocre forecast."""
    from metaculus_bot.forecaster import Forecaster
    real = Forecaster.forecast_question

    async def flaky(self, q):
        if q.id_of_post == 31858:
            raise RuntimeError("simulated model meltdown")
        return await real(self, q)

    monkeypatch.setattr(Forecaster, "forecast_question", flaky)
    summary = await run_mod.run_tournament("minibench")
    assert summary.failed == 1
    assert summary.forecasted == 3, "the other questions must still publish"
