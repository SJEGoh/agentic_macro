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

#: Most legs a single worldview may have. A view that needs fifteen instruments is not a
#: view, it is an index.
MAX_LEGS = int(os.environ.get("AGENTIC_MAX_LEGS", "6"))

#: A price older than this is not a price. Weekends and holidays are handled by the caller
#: asking for a range, not by trusting a stale quote.
MAX_PRICE_AGE_DAYS = float(os.environ.get("AGENTIC_MAX_PRICE_AGE_DAYS", "5"))

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
