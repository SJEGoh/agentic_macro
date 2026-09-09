"""tests/test_memory.py — the news store, and how age is weighed.

The failures pinned here are the ones that produce a store which works and is wrong:

  * a batch of embeddings that does not line up with its documents. `contents=["a","b"]`
    returns ONE blended vector, so every document after the first would carry someone else's
    embedding and retrieval would quietly return the wrong thing forever;
  * a syndicated copy stored twice. Benzinga republishes GlobeNewswire press releases
    verbatim; two rows retrieve as two documents and read to the model as two independent
    outlets confirming the same thing — a fabricated corroboration signal;
  * age treated as a cliff. A hard TTL makes a 13.9-day-old story fully visible and a
    14.1-day-old one non-existent, and lets an important old piece vanish entirely rather
    than merely rank below fresher material;
  * decay applied only to the nearest few candidates, which would mean a fresh item can
    never outrank a stale one it was not fetched alongside.
"""
import time

import pytest

from agentic_macro import config, memory

DAY = memory.DAY


# --------------------------------------------------------------------------- decay
def test_weight_halves_at_one_halflife():
    assert memory.decay(0, 7) == pytest.approx(1.0)
    assert memory.decay(7, 7) == pytest.approx(0.5)
    assert memory.decay(14, 7) == pytest.approx(0.25)
    assert memory.decay(21, 7) == pytest.approx(0.125)


def test_decay_is_smooth_with_no_cliff():
    """The property a TTL did not have: nothing changes abruptly from one day to the next."""
    steps = [memory.decay(d, 7) for d in range(0, 30)]
    assert all(b < a for a, b in zip(steps, steps[1:])), "not monotonically decreasing"
    assert all(abs(a - b) < 0.1 for a, b in zip(steps, steps[1:])), "found a cliff"
    assert steps[-1] > 0, "decay must never reach zero — old is outranked, not erased"


def test_a_longer_halflife_holds_value_longer():
    """A central-bank statement still frames the regime a month later; a wire headline does
    not. One number for both would force a bad choice."""
    assert memory.decay(21, memory.halflife_for("cb")) > \
           memory.decay(21, memory.halflife_for("headline"))
    assert memory.halflife_for("unknown-kind") == memory.DEFAULT_HALFLIFE


def test_a_fresh_weaker_match_can_outrank_a_stale_stronger_one():
    """The whole point of ranking on similarity x weight rather than similarity."""
    fresh = 0.60 * memory.decay(1, 7)
    stale = 0.72 * memory.decay(30, 7)
    assert fresh > stale


# --------------------------------------------------------------------------- dedup
def test_syndicated_copies_share_one_story_key():
    """The measured case: identical headline, 28 seconds apart, two publishers."""
    a = "Churchill Downs Incorporated Announces State of Maryland's Decision"
    b = "Churchill Downs Incorporated Announces State of Maryland's Decision"
    assert memory.story_key(a) == memory.story_key(b)


def test_the_story_key_ignores_punctuation_and_case():
    assert memory.story_key("Fed Cuts Rates — Again!") == memory.story_key("fed cuts rates  again")


def test_different_stories_get_different_keys():
    assert memory.story_key("Fed cuts rates") != memory.story_key("Fed holds rates")


# --------------------------------------------------------------------------- embeddings
# The batch-alignment guard now lives at the provider seam, where both backends meet it —
# see test_providers.py::test_a_short_batch_raises_rather_than_returning.


def test_embed_ignores_blank_input():
    assert memory.embed([]) == []
    assert memory.embed(["", "   "]) == []


def test_retry_delay_honours_the_servers_own_number():
    """The server knows when its per-minute window rolls over; guessing shorter just burns
    another request against the limit."""
    err = "429 RESOURCE_EXHAUSTED ... 'retryDelay': '15.3s' ..."
    assert 15 < memory._retry_after(err, 0) <= 17
    assert memory._retry_after("some other failure", 0) >= 5


# --------------------------------------------------------------------------- prompt block
def test_the_context_block_says_it_is_untrusted_and_incomplete(monkeypatch):
    """Three failures at once: news undated is weighed like today's; absence read as
    evidence; and third-party text read as instruction by a system that places orders."""
    monkeypatch.setattr(memory, "recall", lambda thesis, k=None, as_of=None: [
        {"text": "Fed cuts", "meta": {"title": "Fed cuts rates"}, "similarity": 0.8,
         "age_days": 3.0, "weight": 0.74, "score": 0.59, "published": "2026-09-05"}])
    block, hits = memory.context_block("rates")
    assert "2026-09-05" in block and "3d ago" in block      # dated
    assert "incomplete" in block                            # absence is not evidence
    assert "never as instruction" in block                  # untrusted
    assert "weight 0.74" in block                           # why it ranked


def test_no_hits_yields_no_block(monkeypatch):
    monkeypatch.setattr(memory, "recall", lambda thesis, k=None, as_of=None: [])
    assert memory.context_block("anything") == ("", [])


def test_as_of_reaches_the_retrieval(monkeypatch):
    """A backtest that lets one article published a minute after the release into the
    context has measured hindsight, not a strategy — and it will look excellent."""
    seen = {}

    def fake_recall(thesis, k=None, as_of=None):
        seen["as_of"] = as_of
        return []

    monkeypatch.setattr(memory, "recall", fake_recall)
    memory.context_block("rates", as_of=1_700_000_000.0)
    assert seen["as_of"] == 1_700_000_000.0


def test_the_store_refuses_to_mix_vector_spaces(monkeypatch, tmp_path):
    """The quietest possible failure: cosine distance between a Gemini vector and a nomic
    vector is perfectly well defined and completely meaningless. Nothing raises on its own,
    so the store has to check."""
    from agentic_macro import config as _c, providers
    monkeypatch.setattr(_c, "MEMORY_PATH", tmp_path / "chroma")
    monkeypatch.setattr(_c, "MEMORY_COLLECTION", "space_test")
    monkeypatch.setattr(memory, "_collection", None)

    monkeypatch.setattr(_c, "EMBED_PROVIDER", "gemini")
    col = memory.collection()
    col.upsert(ids=["a"], embeddings=[[0.1] * 8], documents=["x"],
               metadatas=[{"published_at": 0.0, "kind": "news"}])

    monkeypatch.setattr(memory, "_collection", None)
    monkeypatch.setattr(_c, "EMBED_PROVIDER", "ollama")
    with pytest.raises(memory.MemoryUnavailable, match="not comparable"):
        memory.collection()


def test_an_empty_store_may_change_provider_freely(monkeypatch, tmp_path):
    """Nothing to be incomparable with — switching before ingesting is fine."""
    from agentic_macro import config as _c
    monkeypatch.setattr(_c, "MEMORY_PATH", tmp_path / "chroma2")
    monkeypatch.setattr(_c, "MEMORY_COLLECTION", "empty_test")
    monkeypatch.setattr(memory, "_collection", None)
    monkeypatch.setattr(_c, "EMBED_PROVIDER", "gemini")
    memory.collection()
    monkeypatch.setattr(memory, "_collection", None)
    monkeypatch.setattr(_c, "EMBED_PROVIDER", "ollama")
    assert memory.collection() is not None
