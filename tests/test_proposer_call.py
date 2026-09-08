"""tests/test_proposer_call.py — the request shape, and what happens when there is no answer.

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


class FakeCandidate:
    def __init__(self, finish_reason="STOP"):
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, payload=None, finish_reason="STOP", text=None):
        self.candidates = [FakeCandidate(finish_reason)]
        self.prompt_feedback = None
        self.text = text if text is not None else json.dumps(payload or {})


class FakeModels:
    def __init__(self, outer):
        self._outer = outer

    def generate_content(self, **kwargs):
        self._outer.calls.append(kwargs)
        behaviour = self._outer.behaviour.get(kwargs["model"], self._outer.default)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


class FakeClient:
    """Captures the request instead of sending it. `behaviour` maps a model name to either a
    response or an exception, which is how the fallback chain is exercised."""

    def __init__(self, default=None, behaviour=None):
        self.default = default
        self.behaviour = behaviour or {}
        self.calls = []
        self.models = FakeModels(self)

    @property
    def captured(self):
        return self.calls[-1]


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


GOOD = {"expressible": True, "structure": "steepener", "weighting": "dv01_neutral",
        "reasoning": "cuts front-load", "confidence": "high", "conflicts": "",
        "legs": [{"symbol": "SHY", "direction": "long", "weight": 1, "rationale": "front"},
                 {"symbol": "TLT", "direction": "short", "weight": 1, "rationale": "back"}]}


def ok(payload=GOOD, **kw):
    return FakeResponse(payload, **kw)


# --------------------------------------------------------------------------- request shape
def test_the_request_carries_the_model_schema_and_thinking_level():
    client = FakeClient(ok())
    proposer.propose("cuts coming", [], 100_000.0, 0.0, client=client)
    sent = client.captured
    assert sent["model"] == config.MODEL
    cfg = sent["config"]
    assert cfg.response_mime_type == "application/json"
    # the SDK coerces the string into its ThinkingLevel enum, so compare on the value
    level = cfg.thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == config.THINKING_LEVEL.lower()
    assert cfg.max_output_tokens == config.MAX_TOKENS


def test_additional_properties_is_stripped_before_it_reaches_gemini():
    """Gemini takes an OpenAPI subset, not JSON Schema. Sending additionalProperties is a
    400 against the live API, so this is not stylistic — it is the request working."""
    def walk(node):
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(proposer.gemini_schema(proposer.SCHEMA))
    assert "additionalProperties" in proposer.SCHEMA          # kept as documented intent

    client = FakeClient(ok())
    proposer.propose("x", [], 100_000.0, 0.0, client=client)
    walk(client.captured["config"].response_schema)


def test_stripping_preserves_everything_else():
    stripped = proposer.gemini_schema(proposer.SCHEMA)
    assert stripped["required"] == proposer.SCHEMA["required"]
    leg = stripped["properties"]["legs"]["items"]["properties"]
    assert set(leg) == {"symbol", "direction", "weight", "rationale"}
    assert leg["direction"]["enum"] == ["long", "short"]


def test_the_schema_restricts_symbols_to_the_universe():
    """The enum IS the closed list. If this ever stops matching, a hallucinated ticker
    becomes a runtime rejection instead of an impossibility."""
    enum = proposer.SCHEMA["properties"]["legs"]["items"]["properties"]["symbol"]["enum"]
    assert set(enum) == set(universe.symbols())
    assert "NVDA" not in enum


def test_the_prompt_states_the_capital_and_the_views_already_held():
    held = Worldview(id=7, thesis="oil goes up", reasoning="", status="active",
                     created_at="2026-01-01T00:00:00", created_by="me",
                     legs=(Leg("XLE", 100, 90.0, ""),))
    client = FakeClient(ok())
    proposer.propose("dollar falls", [held], 250_000.0, 40_000.0, client=client)
    prompt = client.captured["contents"]
    assert "#7" in prompt and "oil goes up" in prompt and "long XLE" in prompt
    assert "250,000" in prompt and "40,000" in prompt


def test_the_system_instruction_carries_the_universe_and_the_playbooks():
    client = FakeClient(ok())
    proposer.propose("curve steepens", [], 100_000.0, 0.0, client=client)
    system = client.captured["config"].system_instruction
    assert "SHY" in system and "TLT" in system
    assert "steepener" in system and "curve_butterfly" in system
    assert "duration 16.5y" in system      # the data the hedge ratio comes from


# --------------------------------------------------------------------------- non-answers
@pytest.mark.parametrize("reason,match", [
    ("MAX_TOKENS", "ran out of tokens"),
    ("SAFETY", "declined"),
    ("PROHIBITED_CONTENT", "declined"),
    ("RECITATION", "declined"),
    ("OTHER", "without finishing"),
])
def test_a_non_answer_never_becomes_a_book(reason, match):
    client = FakeClient(ok(finish_reason=reason))
    with pytest.raises(proposer.ProposalError, match=match):
        proposer.propose("x", [], 100_000.0, 0.0, client=client)


def test_a_stop_finish_reason_is_accepted():
    client = FakeClient(ok(finish_reason="STOP"))
    assert proposer.propose("x", [], 100_000.0, 0.0, client=client)["structure"] == "steepener"


def test_no_candidates_raises_and_names_the_block():
    response = ok()
    response.candidates = []
    response.prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()
    with pytest.raises(proposer.ProposalError, match="blocked"):
        proposer.propose("x", [], 100_000.0, 0.0, client=FakeClient(response))


def test_unparseable_output_raises_rather_than_returning_empty():
    client = FakeClient(ok(text="I think you should buy bonds."))
    with pytest.raises(proposer.ProposalError, match="JSON"):
        proposer.propose("x", [], 100_000.0, 0.0, client=client)


# --------------------------------------------------------------------------- fallbacks
def test_an_unavailable_model_falls_through_to_the_next():
    """Probing this key found a preview model over quota (429), one under load (503) and one
    retired but still listed (404). None of those mean the worldview was bad."""
    client = FakeClient(behaviour={config.MODEL: FakeAPIError(429)}, default=ok())
    assert proposer.propose("x", [], 100_000.0, 0.0, client=client)["structure"] == "steepener"
    assert len(client.calls) == 2
    assert client.calls[1]["model"] == config.FALLBACK_MODELS[0]


def test_every_model_failing_says_which_and_why():
    client = FakeClient(behaviour={m: FakeAPIError(503) for m in
                                   [config.MODEL] + config.FALLBACK_MODELS})
    with pytest.raises(proposer.ProposalError, match="no model was available") as e:
        proposer.propose("x", [], 100_000.0, 0.0, client=client)
    assert config.MODEL in str(e.value) and "503" in str(e.value)


def test_a_malformed_request_is_not_retried_on_other_models():
    """A 400 is the same on every model. Burying it in the fallback loop would report it as
    an availability problem and send you looking in the wrong place."""
    client = FakeClient(behaviour={config.MODEL: FakeAPIError(400)}, default=ok())
    with pytest.raises(proposer.ProposalError, match="rejected the request"):
        proposer.propose("x", [], 100_000.0, 0.0, client=client)
    assert len(client.calls) == 1, "a 400 must not trigger the fallback chain"


# --------------------------------------------------------------------------- budget
def test_conviction_scales_the_budget_but_the_ceiling_holds():
    alloc = 200_000.0
    assert proposer.default_budget(alloc, "low") < proposer.default_budget(alloc, "high")
    assert proposer.default_budget(alloc, "high") <= alloc * config.MAX_WORLDVIEW_WEIGHT + 1e-9


def test_an_unknown_confidence_falls_back_to_the_middle():
    """A confidence string the schema did not anticipate must not read as maximum size."""
    alloc = 200_000.0
    assert proposer.default_budget(alloc, "certain") == proposer.default_budget(alloc, "medium")
