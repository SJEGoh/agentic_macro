"""
agentic_macro/config.py — every knob, and where it is read from.

Nothing here reaches the broker on its own. The executor enforces its own allocation cap
and drawdown halt server-side, so the limits in this file are the ones that stop a bad
proposal from being SHOWN to you, not the ones that stop it from being filled. Both layers
matter: the server-side cap is what you trust, this one is what keeps the approval prompt
from ever containing a trade the executor would reject anyway.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- identity
#: Must exist in the executor's config.CONFIG, or every intent is rejected as "not active".
#: Register it with `/addstrategy agentic_macro 100k` in the executor's control bot.
STRATEGY_ID = os.environ.get("AGENTIC_STRATEGY_ID", "agentic_macro")

# --------------------------------------------------------------------------- providers
#: Where each kind of model call goes. Two switches rather than one, because the two have
#: very different risk: embeddings are safe to run locally (the news corpus, not the
#: embedding model, is what limits retrieval here), while the chat model chooses the
#: structure and a weaker one fails by producing a PLAUSIBLE wrong answer that every guard
#: downstream then faithfully executes.
LLM_PROVIDER = os.environ.get("AGENTIC_LLM_PROVIDER", "gemini").strip().lower()
EMBED_PROVIDER = os.environ.get("AGENTIC_EMBED_PROVIDER", "gemini").strip().lower()

def _ollama_host() -> str:
    """Normalise OLLAMA_HOST into something a CLIENT can connect to.

    Ollama's own convention is a bind address — `0.0.0.0:11434`, no scheme — because the
    variable configures the server. Handed to a client that means two broken things at once:
    urllib rejects a URL with no scheme, and 0.0.0.0 is "listen on every interface", not an
    address anything can connect to."""
    raw = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").strip()
    if "://" not in raw:
        raw = "http://" + raw
    return raw.replace("://0.0.0.0", "://127.0.0.1").rstrip("/")


OLLAMA_HOST = _ollama_host()
OLLAMA_MODEL = os.environ.get("AGENTIC_OLLAMA_MODEL", "qwen3:14b")
OLLAMA_EMBED_MODEL = os.environ.get("AGENTIC_OLLAMA_EMBED_MODEL", "nomic-embed-text")
#: A local model prefills the whole system prompt (universe + 29 playbooks, ~8k tokens) on
#: every call, so this is minutes-per-call territory on a laptop, not seconds.
OLLAMA_TIMEOUT = float(os.environ.get("AGENTIC_OLLAMA_TIMEOUT", "600"))
#: Must exceed the system prompt plus the context block, or the model silently sees a
#: truncated universe and proposes from whatever survived the window.
OLLAMA_NUM_CTX = int(os.environ.get("AGENTIC_OLLAMA_NUM_CTX", "16384"))
OLLAMA_TEMPERATURE = float(os.environ.get("AGENTIC_OLLAMA_TEMPERATURE", "0.3"))
#: Reasoning models (qwen3, deepseek-r1) accept think=true/false. None leaves it unset.
_think = os.environ.get("AGENTIC_OLLAMA_THINK", "").strip().lower()
OLLAMA_THINK = None if not _think else _think in ("1", "true", "yes", "on")

# --------------------------------------------------------------------------- the model
#: Gemini, via GEMINI_API_KEY. The default is the model actually verified to answer on this
#: key with the response schema below; see FALLBACK_MODELS for why there is a list at all.
MODEL = os.environ.get("AGENTIC_MODEL", "gemini-3.5-flash")

#: Tried in order when the primary is unavailable. This is not belt-and-braces: probing this
#: key found three different ways a Gemini model declines to serve, none of which mean the
#: worldview was bad — a preview model over its free-tier quota (429), a popular model under
#: load (503), and a model still listed by models.list() that has been retired (404). A
#: proposal that dies for any of those reasons is a trade you did not get to consider, so it
#: is worth asking the next model instead. Order is most- to least-capable.
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "AGENTIC_FALLBACK_MODELS", "gemini-3.1-pro-preview,gemini-3.8-flash,gemini-3-flash-preview"
).split(",") if m.strip()]

#: Proposals are short; the reasoning is where the tokens go.
MAX_TOKENS = int(os.environ.get("AGENTIC_MAX_TOKENS", "8000"))

#: Gemini 3.x reasoning depth ("low" | "high"). Macro structure selection is exactly the kind
#: of judgement that repays thinking, so this defaults high.
THINKING_LEVEL = os.environ.get("AGENTIC_THINKING_LEVEL", "high")

# --------------------------------------------------------------------------- Telegram
#: A SEPARATE bot from the executor's control bot. Two processes long-polling getUpdates on
#: one token fight over every update (Telegram hands each update to one consumer and 409s
#: the other), so sharing the executor's token would silently break /status and /kill.
BOT_TOKEN = (os.environ.get("AGENTIC_BOT_TOKEN") or "").strip()
CHAT_ID = (os.environ.get("AGENTIC_CHAT_ID")
           or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
THREAD = (os.environ.get("AGENTIC_THREAD") or "").strip()

#: Who may propose and approve. Empty means nobody — fail closed, exactly as the executor's
#: control bot does, because an unset allowlist must never mean "everyone".
def allowed_users() -> set:
    out = set()
    for part in (os.environ.get("AGENTIC_ALLOWED_USER_IDS", "")
                 .replace(";", ",").split(",")):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------- limits
#: Prefix on every token this bot issues, so `/confirm m7c2e` is visibly ours.
#:
#: This exists because the executor's control bot shares the chat, also implements
#: /confirm, and also issues `secrets.token_hex(2)` — four hex characters, the same shape as
#: ours. Without a prefix the two are indistinguishable, so every confirmation reaches both
#: bots and the one that does not own the token answers "nothing pending with that token".
#: That is not merely noisy: it teaches you to ignore a message that is a REAL error the day
#: a /kill confirmation has genuinely expired. With a prefix, a token that is not ours is
#: recognisably not ours and we stay quiet.
TOKEN_PREFIX = os.environ.get("AGENTIC_TOKEN_PREFIX", "m")

#: A proposal token dies after this long. Deliberately longer than the executor bot's 60s:
#: a macro view is worth re-reading before you approve it, and the price-drift guard below
#: is what actually keeps a stale approval from being filled at the wrong size.
CONFIRM_TTL_SEC = float(os.environ.get("AGENTIC_CONFIRM_TTL", "900"))

#: Refuse to submit an approved proposal if any leg's price has moved more than this since
#: the proposal was priced. Quantities are frozen at approval, so a big move between
#: proposal and confirmation means you would be approving a notional you never saw.
MAX_PRICE_DRIFT = float(os.environ.get("AGENTIC_MAX_PRICE_DRIFT", "0.02"))

#: The largest share of the sleeve's capital ONE worldview may consume, gross. Keeps a
#: single confident view from crowding out the ability to express the next one.
MAX_WORLDVIEW_WEIGHT = float(os.environ.get("AGENTIC_MAX_WORLDVIEW_WEIGHT", "0.35"))

#: Gross notional across ALL active worldviews, as a share of the allocation. Below 1.0 so
#: the local check trips before the executor's hard cap does — a rejection you can read is
#: worth more than one you have to go and find in a log.
MAX_GROSS_WEIGHT = float(os.environ.get("AGENTIC_MAX_GROSS_WEIGHT", "0.95"))

#: Allow fractional share quantities on ETFs. Verified to survive the whole path: the
#: executor's netting compares deltas against an epsilon rather than rounding, and
#: build_order assigns `order.totalQuantity = intent["quantity"]` with no int cast, so a
#: fractional size reaches IB intact.
#:
#: FUTURES ARE NEVER FRACTIONAL regardless of this flag — 0.4 contracts is not a thing you
#: can own, and an order for one would be rejected at the broker rather than rounded.
#:
#: Turn this off if the IB account is not enabled for fractional trading; that entitlement
#: is per-account, and without it IB rejects the order rather than rounding it for you.
FRACTIONAL = (os.environ.get("AGENTIC_FRACTIONAL", "true").strip().lower()
              in ("1", "true", "yes", "on"))

#: Decimal places for a fractional quantity. Four is far finer than any sizing decision
#: needs and keeps the number readable in a confirmation — sending 78.31415926 shares is
#: noise dressed as precision.
FRACTIONAL_DP = int(os.environ.get("AGENTIC_FRACTIONAL_DP", "4"))

#: Most legs a single worldview may have. A view that needs fifteen instruments is not a
#: view, it is an index.
MAX_LEGS = int(os.environ.get("AGENTIC_MAX_LEGS", "6"))

#: A price older than this is not a price. Weekends and holidays are handled by the caller
#: asking for a range, not by trusting a stale quote.
MAX_PRICE_AGE_DAYS = float(os.environ.get("AGENTIC_MAX_PRICE_AGE_DAYS", "5"))

# --------------------------------------------------------------------------- memory
#: Recent-news context the proposer retrieves before choosing a structure. Enrichment, not a
#: dependency: when the store is empty or unreachable the proposal is made without it and
#: SAYS SO, because a proposal that silently lost its context looks exactly like one that
#: never had any.
MEMORY_ENABLED = (os.environ.get("AGENTIC_MEMORY", "true").strip().lower()
                  in ("1", "true", "yes", "on"))
MEMORY_PATH = Path(os.environ.get("AGENTIC_MEMORY_PATH",
                                  Path(__file__).resolve().parents[1] / "db" / "chroma"))
MEMORY_COLLECTION = os.environ.get("AGENTIC_MEMORY_COLLECTION", "macro_memory")

#: gemini-embedding-2 is Matryoshka-trained, so 768 of its 3072 dimensions keeps most of the
#: retrieval quality at a quarter of the storage and distance cost.
EMBED_MODEL = os.environ.get("AGENTIC_EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(os.environ.get("AGENTIC_EMBED_DIM", "768"))

#: Documents per embedding request. Each is sent as its own `Content` — a bare list of
#: strings comes back as ONE blended vector (see memory.embed).
MEMORY_BATCH = int(os.environ.get("AGENTIC_MEMORY_BATCH", "64"))
MEMORY_RECALL_K = int(os.environ.get("AGENTIC_MEMORY_RECALL_K", "6"))
#: Candidates pulled per returned hit before age-decay re-ranking. The index searches on
#: similarity alone, so a fresh item only gets the chance to outrank a stale one if it was
#: fetched in the first place.
MEMORY_OVERFETCH = int(os.environ.get("AGENTIC_MEMORY_OVERFETCH", "8"))
#: Hard age backstop, in days. Decay already makes anything this old unable to compete; this
#: exists so the index does not accumulate forever.
MEMORY_MAX_AGE_DAYS = float(os.environ.get("AGENTIC_MEMORY_MAX_AGE_DAYS", "180"))
#: The free tier allows 100 embed requests per minute and tells you how long to wait, so a
#: 429 during an ingest is worth sitting out rather than abandoning the batch.
MEMORY_RETRIES = int(os.environ.get("AGENTIC_MEMORY_RETRIES", "5"))
MEMORY_INGEST_LIMIT = int(os.environ.get("AGENTIC_MEMORY_INGEST_LIMIT", "400"))
#: Keep only macro-relevant articles. Measured on 25,000 articles from this feed, ~90% are
#: single-stock press releases; storing them means a query about the yield curve retrieves
#: "Carnival's record booking curve" and presents it to the model as retrieved evidence.
MEMORY_MACRO_ONLY = (os.environ.get("AGENTIC_MEMORY_MACRO_ONLY", "true").strip().lower()
                     in ("1", "true", "yes", "on"))
#: Articles scanned per ingest. Much larger than what is kept, because the filter is the
#: whole point.
MEMORY_SCAN_LIMIT = int(os.environ.get("AGENTIC_MEMORY_SCAN_LIMIT", "6000"))

#: Build the news store on startup when it is empty or stale, and refresh it periodically.
#: A container that comes up with an empty store otherwise proposes without context
#: indefinitely, and says so on every proposal — visible, but nobody reads a working bot's
#: logs until something is wrong.
MEMORY_AUTO_INGEST = (os.environ.get("AGENTIC_MEMORY_AUTO_INGEST", "true").strip().lower()
                      in ("1", "true", "yes", "on"))
#: Days of history to pull when bootstrapping an empty store.
MEMORY_BOOTSTRAP_DAYS = float(os.environ.get("AGENTIC_MEMORY_BOOTSTRAP_DAYS", "14"))
#: How often to top it up. 0 disables the refresh but leaves the bootstrap.
MEMORY_REFRESH_HOURS = float(os.environ.get("AGENTIC_MEMORY_REFRESH_HOURS", "6"))
#: Below this the store counts as empty and gets bootstrapped.
MEMORY_MIN_DOCS = int(os.environ.get("AGENTIC_MEMORY_MIN_DOCS", "25"))

# --------------------------------------------------------------------------- storage
DB_PATH = Path(os.environ.get("AGENTIC_DB_PATH",
                              Path(__file__).resolve().parents[1] / "db" / "worldviews.db"))

# --------------------------------------------------------------------------- executor
EXECUTOR_URL = (os.environ.get("EXECUTOR_URL") or "http://127.0.0.1:8000").rstrip("/")

#: PAPER MODE — the executor is never contacted and no order is ever placed.
#:
#: This is a flag rather than commented-out code on purpose. Commenting out the submission
#: fails in the dangerous direction: the code still LOOKS like it trades, every message still
#: says "is on", and the day someone restores it there is no record of what was running
#: without it. A flag is loud instead — it is printed on every confirmation and in the
#: startup banner, so the state you are in is never something you have to remember.
#:
#: In paper mode the "broker" is the sleeve's own stored book: nothing else moves those
#: positions, so the netted book IS what would be held. That makes the order previews
#: truthful about the deltas a real submission would generate.
PAPER = (os.environ.get("AGENTIC_PAPER", "").strip().lower()
         in ("1", "true", "yes", "on"))

#: Capital assumed in paper mode, where there is no executor to ask for an allocation.
PAPER_CAPITAL = float(os.environ.get("AGENTIC_PAPER_CAPITAL", "250000"))
