"""tests/test_proposer_call.py — the request shape, and what happens when there is no answer.

Every call below patches `providers.complete_json` — the seam the proposer now talks to —
and runs with memory disabled and `prices={}`, so nothing here touches a network. The Gemini
and Ollama request shapes are tested separately, in `test_providers.py`.
That is not only speed: a unit test that quietly depends on a live price feed fails for
reasons that have nothing to do with the behaviour it claims to pin.

These run against a fake client, so they pin how the call is BUILT, not that the API accepts
it. The failures they cover are the ones that would otherwise surface as a trade:

  * a non-answer read as a book. A `finish_reason` of SAFETY or MAX_TOKENS comes back as a
    normal 200 with a body that may well parse — treating either as a proposal would place
    orders off a response the model never finished, or deliberately declined to make;
  * the universe leaking. The schema's symbol enum is the mechanism that stops a hallucinated
    ticker; if it ever stopped being generated from the universe, the closed list would
    silently become advisory;
  * `additionalProperties` reaching Gemini. It is not in the OpenAPI subset Gemini accepts
    and is a hard 400 — verified against the live API, so the stripping is load-bearing;
  * a malformed request buried under the fallback chain. Only availability failures deserve
    another model; a 400 means every model will say the same thing, and hiding it behind
    "no model was available" would send you looking in the wrong place;
  * prompt drift. The model is told the sleeve's capital and the views already held; without
    those it proposes every view in isolation and stacks three duration trades.
"""
import json

import pytest

from agentic_macro import config, proposer, universe
from agentic_macro.store import Leg, Worldview


@pytest.fixture(autouse=True)
def no_memory(monkeypatch):
    """These pin the request SHAPE. Letting them retrieve real context would make them
    depend on a live embedding call and on whatever happens to be in the news store."""
    monkeypatch.setattr(config, "MEMORY_ENABLED", False)


class Captured:
    """Stands in for providers.complete_json, recording what the proposer asked for."""

    def __init__(self, reply=None, raises=None):
        self.reply = reply if reply is not None else dict(GOOD)
        self.raises = raises
        self.calls = []

    def __call__(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.raises:
            raise self.raises
        return self.reply

    @property
    def last(self):
        return self.calls[-1]


GOOD = {"expressible": True, "structure": "steepener", "weighting": "dv01_neutral",
        "reasoning": "cuts front-load", "confidence": "high", "conflicts": "",
        "legs": [{"symbol": "SHY", "direction": "long", "weight": 1, "rationale": "front"},
                 {"symbol": "TLT", "direction": "short", "weight": 1, "rationale": "back"}]}


def patch(monkeypatch, cap):
    from agentic_macro import providers
    monkeypatch.setattr(providers, "complete_json", cap)
    return cap


# --------------------------------------------------------------------------- request shape
def test_the_schema_restricts_symbols_to_the_universe():
    """The enum IS the closed list. If this ever stops matching, a hallucinated ticker
    becomes a runtime rejection instead of an impossibility."""
    enum = proposer.SCHEMA["properties"]["legs"]["items"]["properties"]["symbol"]["enum"]
    assert set(enum) == set(universe.symbols())
    assert "NVDA" not in enum


def test_the_proposer_hands_the_schema_to_the_provider(monkeypatch):
    cap = patch(monkeypatch, Captured())
    proposer.propose("cuts coming", [], 100_000.0, 0.0, prices={})
    assert cap.last["schema"] is proposer.SCHEMA


def test_the_prompt_states_the_capital_and_the_views_already_held(monkeypatch):
    held = Worldview(id=7, thesis="oil goes up", reasoning="", status="active",
                     created_at="2026-01-01T00:00:00", created_by="me",
                     legs=(Leg("XLE", 100, 90.0, ""),))
    cap = patch(monkeypatch, Captured())
    proposer.propose("dollar falls", [held], 250_000.0, 40_000.0, prices={})
    prompt = cap.last["user"]
    assert "#7" in prompt and "oil goes up" in prompt and "long XLE" in prompt
    assert "250,000" in prompt and "40,000" in prompt


def test_the_system_prompt_carries_the_universe_and_the_playbooks(monkeypatch):
    cap = patch(monkeypatch, Captured())
    proposer.propose("curve steepens", [], 100_000.0, 0.0, prices={})
    system = cap.last["system"]
    assert "SHY" in system and "TLT" in system
    assert "steepener" in system and "curve_butterfly" in system
    assert "duration 16.5y" in system      # the data the hedge ratio comes from


# --------------------------------------------------------------------------- failures
def test_a_provider_failure_becomes_a_proposal_error(monkeypatch):
    """Every way a model call can fail must arrive as one error type the bot can render,
    never as a half-built proposal."""
    from agentic_macro import providers
    patch(monkeypatch, Captured(raises=providers.ProviderError("the model declined")))
    with pytest.raises(proposer.ProposalError, match="declined"):
        proposer.propose("x", [], 100_000.0, 0.0, prices={})


def test_a_non_object_reply_is_refused(monkeypatch):
    patch(monkeypatch, Captured(reply=["not", "an", "object"]))
    with pytest.raises(proposer.ProposalError, match="not an object"):
        proposer.propose("x", [], 100_000.0, 0.0, prices={})


def test_a_good_reply_comes_back_intact(monkeypatch):
    patch(monkeypatch, Captured())
    out = proposer.propose("x", [], 100_000.0, 0.0, prices={})
    assert out["structure"] == "steepener"
    assert out["_context"] == [] and "_context_note" in out


# --------------------------------------------------------------------------- budget
def test_conviction_scales_the_budget_but_the_ceiling_holds():
    alloc = 200_000.0
    assert proposer.default_budget(alloc, "low") < proposer.default_budget(alloc, "high")
    assert proposer.default_budget(alloc, "high") <= alloc * config.MAX_WORLDVIEW_WEIGHT + 1e-9


def test_an_unknown_confidence_falls_back_to_the_middle():
    """A confidence string the schema did not anticipate must not read as maximum size."""
    alloc = 200_000.0
    assert proposer.default_budget(alloc, "certain") == proposer.default_budget(alloc, "medium")
