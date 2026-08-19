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
