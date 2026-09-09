"""tests/test_providers.py — the seam between this codebase and whichever model serves it.

What is pinned here is everything that differs between a hosted and a local model, because
each difference is a way the swap breaks quietly rather than loudly:

  * a batch that comes back short. Gemini blends a bare list of strings into ONE vector;
    Ollama batches properly. Either way, a length mismatch must raise, because every vector
    after a gap would be attached to the wrong document and cosine distance is perfectly
    well defined between two vectors that mean nothing to each other;
  * mixing vector spaces. A store half-built by Gemini and half by nomic returns confident
    nonsense with no error anywhere. `embedding_id()` is what lets the store notice;
  * the schema reaching the model intact. The symbol enum is what makes a hallucinated
    ticker impossible rather than merely unlikely, and Gemini needs `additionalProperties`
    stripped while Ollama does not;
  * automatic provider fallback, which must NOT exist. A proposal silently answered by a
    different model than the one configured is not the proposal you think you are reading.
"""
import json

import pytest

from agentic_macro import config, providers


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "gemini")


# --------------------------------------------------------------------------- embeddings
def test_a_short_batch_raises_rather_than_returning(ollama, monkeypatch):
    monkeypatch.setattr(providers, "_ollama",
                        lambda path, body, timeout=None: {"embeddings": [[0.1] * 768]})
    with pytest.raises(providers.ProviderError, match="may not match their documents"):
        providers.embed(["one", "two", "three"])


def test_a_matching_batch_passes_through(ollama, monkeypatch):
    monkeypatch.setattr(providers, "_ollama",
                        lambda path, body, timeout=None: {"embeddings": [[0.1]] * 3})
    assert len(providers.embed(["a", "b", "c"])) == 3


def test_documents_and_queries_get_different_prefixes(ollama, monkeypatch):
    """nomic-embed-text is trained with task prefixes. Using one for both costs recall with
    no error — the same asymmetry the hosted models have."""
    sent = {}

    def fake(path, body, timeout=None):
        sent.update(body)
        return {"embeddings": [[0.1]] * len(body["input"])}

    monkeypatch.setattr(providers, "_ollama", fake)
    providers.embed(["the Fed cut rates"], query=False)
    assert sent["input"][0].startswith("search_document: ")
    providers.embed(["the Fed cut rates"], query=True)
    assert sent["input"][0].startswith("search_query: ")


def test_no_texts_needs_no_provider():
    assert providers.embed([]) == []


# --------------------------------------------------------------------------- vector space
def test_the_embedding_id_distinguishes_the_two_vector_spaces(monkeypatch):
    """Vectors from different models are not comparable, and nothing about mixing them
    errors — this id is how a store can refuse."""
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    local = providers.embedding_id()
    monkeypatch.setattr(config, "EMBED_PROVIDER", "gemini")
    hosted = providers.embedding_id()
    assert local != hosted
    assert config.OLLAMA_EMBED_MODEL in local and config.EMBED_MODEL in hosted


def test_the_embedding_id_tracks_the_dimension(monkeypatch):
    """768 truncated from 3072 is a different space from a native 768."""
    monkeypatch.setattr(config, "EMBED_PROVIDER", "gemini")
    monkeypatch.setattr(config, "EMBED_DIM", 768)
    a = providers.embedding_id()
    monkeypatch.setattr(config, "EMBED_DIM", 1536)
    assert providers.embedding_id() != a


# --------------------------------------------------------------------------- completion
def test_ollama_receives_the_schema_unstripped(ollama, monkeypatch):
    """Ollama honours additionalProperties; Gemini rejects it. Passing the schema through
    intact is what keeps the symbol enum doing its job."""
    sent = {}

    def fake(path, body, timeout=None):
        sent.update(body)
        return {"message": {"content": json.dumps({"ok": True})}}

    monkeypatch.setattr(providers, "_ollama", fake)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    assert providers.complete_json("sys", "user", schema) == {"ok": True}
    assert sent["format"] == schema
    assert sent["format"]["additionalProperties"] is False
    assert sent["messages"][0]["role"] == "system"
    assert sent["options"]["num_ctx"] == config.OLLAMA_NUM_CTX


def test_an_empty_ollama_reply_is_an_error_not_an_empty_proposal(ollama, monkeypatch):
    """A model that did not fit in memory returns an empty string, and an empty string is
    not a decision to hold nothing."""
    monkeypatch.setattr(providers, "_ollama",
                        lambda path, body, timeout=None: {"message": {"content": "  "}})
    with pytest.raises(providers.ProviderError, match="empty completion"):
        providers.complete_json("s", "u", {"type": "object"})


def test_unparseable_ollama_output_raises(ollama, monkeypatch):
    monkeypatch.setattr(providers, "_ollama",
                        lambda path, body, timeout=None: {"message": {"content": "not json"}})
    with pytest.raises(providers.ProviderError, match="unparseable"):
        providers.complete_json("s", "u", {"type": "object"})


def test_a_provider_never_silently_falls_back_to_the_other(ollama, monkeypatch):
    """If Ollama is down the answer is an error, never a hosted-model proposal wearing a
    local model's name."""
    def dead(path, body, timeout=None):
        raise providers.ProviderError("ollama unreachable")

    monkeypatch.setattr(providers, "_ollama", dead)
    called = {"gemini": False}
    monkeypatch.setattr(providers, "_complete_gemini",
                        lambda *a, **k: called.update(gemini=True) or {})
    with pytest.raises(providers.ProviderError, match="unreachable"):
        providers.complete_json("s", "u", {"type": "object"})
    assert not called["gemini"], "silently answered with the other provider"


# --------------------------------------------------------------------------- status
def test_status_names_what_is_configured(gemini):
    s = providers.status()
    assert s["llm"].startswith("gemini:")
    assert s["embeddings"].startswith("gemini:")


def test_status_flags_a_model_that_is_not_pulled(ollama, monkeypatch):
    monkeypatch.setattr(providers, "ollama_models", lambda: ["something-else:latest"])
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3:14b")
    s = providers.status()
    assert "ollama pull qwen3:14b" in s["llm_missing"]


# --------------------------------------------------------------------------- host
@pytest.mark.parametrize("raw,expect", [
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),      # ollama's own default, unusable as-is
    ("localhost:11434", "http://localhost:11434"),    # no scheme
    ("http://box:11434/", "http://box:11434"),        # trailing slash
    ("https://remote:443", "https://remote:443"),
])
def test_the_ollama_host_is_normalised_into_something_connectable(monkeypatch, raw, expect):
    """OLLAMA_HOST is a BIND address by convention, so it routinely holds a value no client
    can use: urllib rejects a schemeless URL, and 0.0.0.0 means "every interface", not a
    destination."""
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert config._ollama_host() == expect
