"""
agentic_macro/providers.py — where the model calls actually go.

One seam, two consumers: the proposer needs a chat completion constrained to a JSON schema,
and the memory needs embeddings. Both can be served by Gemini over the network or by Ollama
on this machine, and nothing else in the codebase needs to know which.

Why the split is worth making
-----------------------------
The two have completely different risk profiles, so they are configured separately rather
than by one "use local models" switch.

**Embeddings are safe to move.** Retrieval quality is not what limits this system — the news
corpus is (roughly 90% of the feed is single-stock press releases). What local embeddings buy
is the removal of a hard quota: the hosted free tier allows 100 embed requests per minute,
which is the single thing that has made ingesting history painful.

**The chat model is not safe to move without measuring.** It chooses the structure, the
instruments and the direction — the judgement the whole sleeve is built around. A smaller
model does not fail loudly here; it produces a *plausible* structure, and every guard
downstream then faithfully executes a well-formed bad choice. The mechanical protections all
survive (schema-constrained decoding still makes an off-universe ticker impossible), but a
beta-neutral pair that is the wrong pair looks exactly like a right one.

So: move embeddings, and A/B the proposer before trusting it.

The mismatch that would be silent
---------------------------------
Vectors from different embedding models are not comparable. A store half-filled by Gemini and
half by nomic returns confident nonsense with no error anywhere, because cosine distance is
perfectly well defined between two vectors that mean nothing to each other. `embedding_id()`
exists so the store can record which model built it and refuse to mix.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from . import config

log = logging.getLogger("agentic-macro.providers")


class ProviderError(RuntimeError):
    """A model call failed. Never fall back to another provider automatically — a proposal
    silently answered by a different model than the one configured is not the proposal you
    think you are reading."""


# --------------------------------------------------------------------------- ollama
def _ollama(path: str, body: dict, timeout: float = None) -> dict:
    url = f"{config.OLLAMA_HOST.rstrip('/')}{path}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.OLLAMA_TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ProviderError(f"ollama {path} -> {e.code}: {e.read()[:200].decode(errors='replace')}")
    except Exception as e:
        raise ProviderError(f"ollama {path} unreachable at {config.OLLAMA_HOST}: {e}")


def ollama_models() -> list:
    try:
        return [m["name"] for m in _ollama_get("/api/tags").get("models", [])]
    except ProviderError:
        return []


def _ollama_get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST.rstrip('/')}{path}",
                                    timeout=10) as r:
            return json.load(r)
    except Exception as e:
        raise ProviderError(f"ollama {path} unreachable: {e}")


# --------------------------------------------------------------------------- embeddings
def embedding_id() -> str:
    """Identifies the vector space. Stored with the index so a provider change cannot
    silently mix incomparable vectors."""
    return (f"ollama:{config.OLLAMA_EMBED_MODEL}" if config.EMBED_PROVIDER == "ollama"
            else f"gemini:{config.EMBED_MODEL}:{config.EMBED_DIM}")


def embed(texts: list, query: bool = False) -> list:
    """One vector per text, in order. Raises rather than returning a short list — a batch
    that does not line up attaches every later vector to the wrong document."""
    if not texts:
        return []
    vectors = (_embed_ollama(texts, query) if config.EMBED_PROVIDER == "ollama"
               else _embed_gemini(texts, query))
    if len(vectors) != len(texts):
        raise ProviderError(
            f"asked for {len(texts)} embeddings and got {len(vectors)} — refusing to "
            f"store vectors that may not match their documents")
    return vectors


def _embed_ollama(texts: list, query: bool) -> list:
    # nomic-embed-text is trained with task prefixes; using the wrong one costs recall with
    # no error, the same asymmetry the hosted models have.
    prefix = "search_query: " if query else "search_document: "
    out = []
    for start in range(0, len(texts), config.MEMORY_BATCH):
        chunk = texts[start:start + config.MEMORY_BATCH]
        body = {"model": config.OLLAMA_EMBED_MODEL,
                "input": [prefix + t[:8000] for t in chunk]}
        out.extend(_ollama("/api/embed", body).get("embeddings") or [])
    return out


def _embed_gemini(texts: list, query: bool) -> list:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)
    out = []
    for start in range(0, len(texts), config.MEMORY_BATCH):
        chunk = texts[start:start + config.MEMORY_BATCH]
        # Each document must be its own Content: a bare list of strings comes back as ONE
        # blended vector.
        resp = client.models.embed_content(
            model=config.EMBED_MODEL,
            contents=[types.Content(parts=[types.Part(text=t[:8000])]) for t in chunk],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                output_dimensionality=config.EMBED_DIM))
        out.extend(e.values for e in resp.embeddings)
    return out


# --------------------------------------------------------------------------- completion
def complete_json(system: str, user: str, schema: dict) -> dict:
    """A completion constrained to `schema`. Returns the parsed object.

    Both providers do constrained decoding, so the symbol enum still makes an off-universe
    ticker impossible whichever is in use. What differs is judgement, not validity."""
    if config.LLM_PROVIDER == "ollama":
        return _complete_ollama(system, user, schema)
    return _complete_gemini(system, user, schema)


def _complete_ollama(system: str, user: str, schema: dict) -> dict:
    body = {
        "model": config.OLLAMA_MODEL,
        "format": _ollama_schema(schema),
        "stream": False,
        "options": {"temperature": config.OLLAMA_TEMPERATURE,
                    "num_ctx": config.OLLAMA_NUM_CTX},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if config.OLLAMA_THINK is not None:
        body["think"] = config.OLLAMA_THINK
    resp = _ollama("/api/chat", body)
    text = (resp.get("message") or {}).get("content") or ""
    if not text.strip():
        raise ProviderError("ollama returned an empty completion — the model may not fit "
                            f"in memory at num_ctx={config.OLLAMA_NUM_CTX}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ProviderError(f"ollama returned unparseable JSON: {e}: {text[:200]}")


def _ollama_schema(schema: dict) -> dict:
    """Ollama takes a JSON Schema directly and honours `additionalProperties`, so unlike the
    Gemini path nothing needs stripping. Passed through unchanged and deliberately: the enum
    is what makes a hallucinated symbol impossible rather than merely unlikely."""
    return schema


def _complete_gemini(system: str, user: str, schema: dict) -> dict:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)

    from .proposer import gemini_schema        # strips what Gemini's subset rejects
    request = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=gemini_schema(schema),
        thinking_config=types.ThinkingConfig(thinking_level=config.THINKING_LEVEL),
        max_output_tokens=config.MAX_TOKENS)

    last = []
    for model in [config.MODEL] + config.FALLBACK_MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=user,
                                                  config=request)
        except Exception as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 503, 404):
                raise ProviderError(f"{model} rejected the request: {e}")
            last.append(f"{model} ({code})")
            continue
        _check_finished(resp)
        try:
            return json.loads(resp.text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ProviderError(f"could not read the proposal as JSON: {e}")
    raise ProviderError("no model was available to propose with: " + ", ".join(last))


_REFUSED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _check_finished(response) -> None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise ProviderError("the model returned nothing"
                            + (f" — the prompt was blocked ({blocked})" if blocked else ""))
    name = getattr(getattr(candidates[0], "finish_reason", None), "name", None) \
        or str(getattr(candidates[0], "finish_reason", "") or "")
    if name in ("STOP", "FINISH_REASON_UNSPECIFIED", "None", ""):
        return
    if name == "MAX_TOKENS":
        raise ProviderError("ran out of tokens mid-proposal — raise AGENTIC_MAX_TOKENS")
    if name in _REFUSED:
        raise ProviderError(f"the model declined to answer this one ({name})")
    raise ProviderError(f"the model stopped without finishing ({name})")


# --------------------------------------------------------------------------- diagnostics
def status() -> dict:
    """What is actually configured and reachable. Worth printing before a long run."""
    out = {"llm": f"{config.LLM_PROVIDER}:"
                  f"{config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.MODEL}",
           "embeddings": embedding_id()}
    if "ollama" in (config.LLM_PROVIDER, config.EMBED_PROVIDER):
        have = ollama_models()
        out["ollama_reachable"] = bool(have)
        out["ollama_models"] = have
        for role, want in (("llm", config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else None),
                           ("embeddings", config.OLLAMA_EMBED_MODEL
                            if config.EMBED_PROVIDER == "ollama" else None)):
            if want and have and not any(m == want or m.startswith(want + ":") for m in have):
                out[f"{role}_missing"] = f"`ollama pull {want}`"
    return out
