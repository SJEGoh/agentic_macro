"""
agentic_macro/bot.py — submit a worldview from Telegram, approve the trade it implies.

    /worldview the Fed cuts three times into an intact labour market
        -> a structure, its legs, what it costs, and what it would do to the sleeve
    /confirm m3f9
        -> written to the store, then the whole netted book goes to the executor

Two properties this file exists to guarantee
--------------------------------------------
**What you approve is exactly what is sent.** The confirmation token carries the SIZED
PROPOSAL — symbols, signed share counts, the prices they were sized at — not the worldview
text. Re-running the model on approval would be the obvious implementation and a serious
bug: the same thesis can yield a different structure on a second call, and you would be
confirming one trade having read another. Nothing between `/worldview` and `/confirm` calls
the model again.

**An approval that has gone stale is refused, not resized.** Quantities are frozen when you
approve, so if a leg has moved more than MAX_PRICE_DRIFT since it was priced, the notional
you read is no longer the notional you would get. That refuses and asks for a fresh
proposal rather than quietly filling a different-sized trade.

Sharing a chat with the executor's control bot
----------------------------------------------
This runs its OWN bot token. Two processes long-polling getUpdates on one token fight over
every update — Telegram hands each to exactly one consumer and 409s the other — so sharing
that token would silently break /status and /kill on the box that holds the positions.

The CHAT, though, can be shared, and this file is written so it can be. Telegram delivers
every slash command to both bots, so three things are deliberate: tokens carry a prefix
(config.TOKEN_PREFIX) so a /confirm meant for the other bot is recognisably not ours;
unknown commands get silence rather than "unknown command", because /status and /kill belong
to the other bot and it already answers typos; and `/cmd@TheirBot` is honoured rather than
stripped. Without those, every command either bot owns draws a complaint from the other, and
"nothing pending with that token" stops meaning anything on the day it matters.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import NamedTuple

import requests

from . import config, orders, prices, proposer, playbooks
from .executor.executor_client import (ExecutorClient, ExecutorError, ExecutorRejected,
                                       ExecutorUnreachable)
from .store import Store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] agentic-macro: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("agentic-macro.bot")

API = "https://api.telegram.org/bot{token}/{method}"
STALE_COMMAND_SEC = 300.0      # ignore anything Telegram redelivers from before a restart

store = Store()
client = ExecutorClient(strategy_id=config.STRATEGY_ID)

#: Our own @name, so a command addressed to the executor's bot can be left alone. Resolved
#: at startup; None means we could not tell, in which case addressing is not filtered — that
#: costs noise, never correctness.
BOT_USERNAME = None


# --------------------------------------------------------------------------- plumbing
def tg_call(method: str, **payload):
    """One Telegram API call. Returns (ok, result) or (False, description).

    The description matters: `say()` has to tell a bad topic id apart from every other
    reason a send can fail, and it can only do that if the reason survives the call."""
    try:
        r = requests.post(API.format(token=config.BOT_TOKEN, method=method),
                          json=payload, timeout=40)
        body = r.json()
        if not body.get("ok"):
            return False, body.get("description") or "unknown error"
        return True, body.get("result")
    except Exception as e:
        return False, str(e)


def tg(method: str, **payload):
    ok, value = tg_call(method, **payload)
    if not ok:
        log.warning("telegram %s failed: %s", method, value)
        return None
    return value


def say(text: str, thread=None) -> None:
    """Deliver a reply, falling back to the main thread if the configured topic is gone.

    A wrong AGENTIC_THREAD is the worst kind of misconfiguration: every send fails with
    "message thread not found" while each handler logs that it ran, so the bot goes
    completely mute and the logs insist it is working. Falling back means a reply always
    lands SOMEWHERE — a message in the wrong topic is recoverable, a message you never see
    is not — and the error says exactly which variable to fix."""
    payload = {"chat_id": config.CHAT_ID, "text": text, "disable_web_page_preview": True}
    target = thread if thread is not None else (config.THREAD or None)

    # Telegram rejects anything over 4096 characters; a long proposal must not vanish.
    for chunk in (text[i:i + 3900] for i in range(0, len(text), 3900)):
        payload["text"] = chunk
        if target:
            payload["message_thread_id"] = int(target)
        ok, err = tg_call("sendMessage", **payload)

        if not ok and target and "thread not found" in str(err).lower():
            log.error("AGENTIC_THREAD=%s is not a topic in this chat — replying in the main "
                      "thread instead. Set it to a real topic id or leave it unset.", target)
            target = None                      # and for every chunk after this one
            payload.pop("message_thread_id", None)
            ok, err = tg_call("sendMessage", **payload)

        if not ok:
            log.warning("telegram sendMessage failed: %s", err)


def money(v) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{'-' if n < 0 else ''}${abs(n):,.0f}"


def _parse_amount(text: str):
    """'50k' -> 50000.0. Returns None when the token is not an amount, which is how the
    optional size prefix on /worldview is detected without a flag."""
    raw = text.replace(",", "").replace("$", "").strip().lower()
    factor = 1.0
    if raw.endswith("k"):
        factor, raw = 1_000.0, raw[:-1]
    elif raw.endswith("m"):
        factor, raw = 1_000_000.0, raw[:-1]
    try:
        value = float(raw) * factor
    except ValueError:
        return None
    return value if value > 0 else None


# --------------------------------------------------------------------------- sleeve state
def _capital_used(views: list, marks: dict = None) -> float:
    return sum(v.gross(marks) for v in views)


def _banner() -> str:
    """Prepended to anything that would otherwise read as "this traded"."""
    return ("PAPER MODE — nothing is sent to the executor.\n"
            "Unset AGENTIC_PAPER to trade for real.\n\n" if config.PAPER else "")


def _sleeve() -> tuple:
    """(allocation, active views, gross at current prices). One place, because every
    command that quotes capacity must quote the same number."""
    allocation = (config.PAPER_CAPITAL if config.PAPER
                  else float(client.allocation()["capital_allocation"]))
    views = store.active()
    held = store.net_positions()
    marks = prices.fetch(held.keys()) if held else {}
    return allocation, views, _capital_used(views, marks), marks


# --------------------------------------------------------------------------- pending
class Pending(NamedTuple):
    proposal: object
    user_id: int
    expires_at: float


_pending: dict = {}      # awaiting /confirm — a worldview you have not accepted yet
_staged: dict = {}       # awaiting /place  — accepted, orders computed, not yet sent
_lock = threading.Lock()


def new_token() -> str:
    """A token that is recognisably ours — see config.TOKEN_PREFIX."""
    return config.TOKEN_PREFIX + secrets.token_hex(2)


def is_ours(token: str) -> bool:
    """Whether a /confirm token was issued by this bot.

    Used to decide whether to answer at all. In a chat shared with the executor's control
    bot, a token without our prefix belongs to that bot, and the right response is silence —
    not "nothing pending with that token", which is the error message we want to still mean
    something when it appears."""
    return bool(token) and token.startswith(config.TOKEN_PREFIX)


def _stash(proposal, user_id: int) -> str:
    token = new_token()
    with _lock:
        now = time.time()
        for t, p in list(_pending.items()):
            if p.expires_at < now:
                del _pending[t]
        _pending[token] = Pending(proposal, user_id, now + config.CONFIRM_TTL_SEC)
    return token


def _stage(proposal, user_id: int) -> str:
    """Move an accepted proposal to the final gate. Kept separate from _pending so a token
    can only be used for the step it was issued for: a /place token cannot re-approve a
    worldview, and a /confirm token cannot send orders."""
    token = new_token()
    with _lock:
        now = time.time()
        for t, p in list(_staged.items()):
            if p.expires_at < now:
                del _staged[t]
        _staged[token] = Pending(proposal, user_id, now + config.CONFIRM_TTL_SEC)
    return token


def _take_staged(token: str, user_id: int):
    with _lock:
        pending = _staged.pop(token, None)
    if pending is None:
        return None, "nothing staged with that token (or it expired)"
    if pending.user_id != user_id:
        return None, "that order belongs to someone else"
    if pending.expires_at < time.time():
        return None, "that order expired — run /worldview again"
    return pending.proposal, None


def _take(token: str, user_id: int):
    with _lock:
        pending = _pending.pop(token, None)
    if pending is None:
        return None, "nothing pending with that token (or it expired)"
    if pending.user_id != user_id:
        return None, "that proposal belongs to someone else"
    if pending.expires_at < time.time():
        return None, "that proposal expired — run /worldview again"
    return pending.proposal, None


# --------------------------------------------------------------------------- rendering
def _render(proposal, allocation: float, capital_used: float, token: str) -> str:
    p = proposal
    lines = [f"WORLDVIEW\n  {p.thesis}", ""]

    if not p.legs:
        lines.append(f"NOT EXPRESSIBLE in this universe.\n\n{p.reasoning}")
        return "\n".join(lines)

    structure = p.structure.replace("_", " ")
    lines.append(f"STRUCTURE  {structure}  [{p.weighting}]  ·  confidence {p.confidence}")
    lines.append(f"\n{p.reasoning}\n")

    lines.append("LEGS")
    for leg in p.legs:
        notional = leg.quantity * leg.entry_price
        lines.append(f"  {'LONG ' if leg.quantity > 0 else 'SHORT'} {leg.symbol:<5} "
                     f"{abs(leg.quantity):>8,.0f} @ {leg.entry_price:>8,.2f}  "
                     f"= {money(abs(notional)):>12}")
        if leg.rationale:
            lines.append(f"        {leg.rationale}")

    if p.hedge_note:
        lines.append(f"\n  hedge ratio: {p.hedge_note}")
    if p.dropped:
        lines.append("  not included: " + "; ".join(p.dropped))

    after = capital_used + p.gross()
    lines.append(f"\nCAPITAL")
    lines.append(f"  this view    {money(p.gross())} gross (budget {money(p.budget)})")
    lines.append(f"  sleeve after {money(after)} of {money(allocation)} "
                 f"({after / allocation:.0%})")

    if p.conflicts:
        lines.append(f"\nAGAINST WHAT YOU ALREADY HOLD\n  {p.conflicts}")

    # Deliberately does NOT say "this places orders" — it does not. /confirm accepts the
    # view and shows the orders it implies; /place is the only step that sends anything.
    # Overstating what a button does is how a confirmation stops being read.
    lines.append(f"\nNothing is sent yet. To accept this view and see the orders it implies:"
                 f"\n/confirm {token}\n"
                 f"(expires in {config.CONFIRM_TTL_SEC / 60:.0f} min)")
    return _banner() + "\n".join(lines)


# --------------------------------------------------------------------------- commands
def cmd_help(args, ctx) -> str:
    return (
        _banner() + "Discretionary macro sleeve\n"
        "\n  /worldview [size] <your view>\n"
        "        propose a structure for a view. Optional leading size:\n"
        "        /worldview 50k the Fed cuts into an intact labour market\n"
        "  /confirm <token>       accept it, and see the ORDERS it implies\n"
        "  /place <token>         send those orders  <- the last gate\n"
        "\n  /views                 the worldviews currently on, and what they hold\n"
        "  /view <id>             one worldview in full — thesis, legs, reasoning\n"
        "  /close <id>            retire a worldview; its legs unwind on the next book\n"
        "  /book                  the netted book across every active worldview\n"
        "  /sync                  re-send the netted book (self-heals drift)\n"
        "  /playbooks             the structures available\n"
        "\nThe book sent to the executor is the NET of every active worldview. Closing one\n"
        "removes exactly its legs. /close and /sync need /confirm; /worldview always does.")


def cmd_playbooks(args, ctx) -> str:
    if args:
        book = playbooks.BY_NAME.get(args[0].lower())
        if not book:
            return f"no playbook called {args[0]!r} — /playbooks lists them"
        return book.render()
    return ("Structures (/playbooks <name> for one in full):\n\n" + "\n".join(
        f"  {p.name.replace('_', ' '):<24} [{p.weighting}]" for p in playbooks.PLAYBOOKS))


def cmd_views(args, ctx) -> str:
    allocation, views, used, marks = _sleeve()
    if not views:
        return f"no active worldviews. Sleeve is flat, {money(allocation)} available."
    lines = []
    for v in views:
        legs = ", ".join(f"{'+' if l.quantity > 0 else ''}{l.quantity:g} {l.symbol}"
                         for l in v.legs)
        lines.append(f"#{v.id}  {v.thesis}\n"
                     f"     {legs or 'no legs'}\n"
                     f"     {money(v.gross(marks))} gross · opened {v.created_at[:10]}")
    lines.append(f"\n{len(views)} view(s) · {money(used)} of {money(allocation)} "
                 f"({used / allocation:.0%})")
    return "\n".join(lines)


def cmd_view(args, ctx) -> str:
    if not args:
        return "usage: /view <id>"
    view = store.get(int(args[0]))
    if view is None:
        return f"no worldview #{args[0]}"
    lines = [f"#{view.id}  [{view.status}]  opened {view.created_at[:10]} by {view.created_by}",
             f"\n{view.thesis}\n", view.reasoning or "", "\nLEGS"]
    for leg in view.legs:
        lines.append(f"  {'LONG ' if leg.quantity > 0 else 'SHORT'} {leg.symbol:<5} "
                     f"{abs(leg.quantity):>8,.0f} @ {leg.entry_price:,.2f} entry")
        if leg.rationale:
            lines.append(f"        {leg.rationale}")
    return "\n".join(lines)


def cmd_book(args, ctx) -> str:
    targets = store.net_positions()
    if not targets:
        return "the netted book is empty — nothing is held."
    marks = prices.fetch(targets.keys())
    lines = ["NETTED BOOK (what the executor holds for this sleeve)"]
    for symbol, quantity in sorted(targets.items()):
        owners = ", ".join(f"#{wid}" for wid, _ in store.holders(symbol))
        lines.append(f"  {symbol:<5} {quantity:>+9,.0f} @ {marks[symbol]:>8,.2f} "
                     f"= {money(abs(quantity) * marks[symbol]):>12}   from {owners}")
    gross = sum(abs(q) * marks[s] for s, q in targets.items())
    lines.append(f"\n{money(gross)} gross across {len(targets)} names")
    return "\n".join(lines)


def preview_close(args, ctx) -> str:
    if not args:
        raise ValueError("usage: /close <id>")
    view = store.get(int(args[0]))
    if view is None:
        raise ValueError(f"no worldview #{args[0]}")
    if view.status != "active":
        raise ValueError(f"worldview #{view.id} is already closed")

    # The ORDERS, not the view's legs. Another active worldview may hold the same name, in
    # which case closing this one does not flatten it — and the difference between "this
    # sells 400 TLT" and "this sells 150 TLT because #2 still wants 250" is exactly what
    # you need to see before approving.
    rows = orders.deltas(client, store, exclude=view.id)
    marks = prices.fetch([r.symbol for r in rows]) if rows else {}
    return (f"{_banner()}CLOSE #{view.id}\n  {view.thesis}\n\n"
            + orders.render(rows, marks))


def cmd_close(args, ctx) -> str:
    view = store.close_worldview(int(args[0]), closed_by=ctx["user"])
    code = _submit()
    if code != 0:
        return (f"#{view.id} closed in the store, but the book did NOT go through "
                f"(exit {code}). Its positions are still at the broker — run /sync.")
    return f"#{view.id} closed and its legs unwound.\n\n{cmd_views([], ctx)}"


def preview_sync(args, ctx) -> str:
    """What re-sending would actually change. With no drift this is "NO ORDERS", which is
    the useful answer — it says the broker already matches the book."""
    rows = orders.deltas(client, store)
    marks = prices.fetch([r.symbol for r in rows]) if rows else {}
    return (f"{_banner()}RE-SEND THE NETTED BOOK\n\n" + orders.render(rows, marks)
            + "\n\nSafe to repeat — /targets takes absolute positions.")


def cmd_sync(args, ctx) -> str:
    code = _submit()
    return "book re-sent." if code == 0 else f"submission failed (exit {code}) — see the logs."


def cmd_worldview(args, ctx) -> str:
    if not args:
        return "usage: /worldview [size] <your view>"

    size_override = _parse_amount(args[0])
    thesis = " ".join(args[1:] if size_override else args).strip()
    if len(thesis) < 10:
        return "give me a bit more to work with — a sentence or two of actual view."

    allocation, views, used, _ = _sleeve()
    room = allocation * config.MAX_GROSS_WEIGHT - used
    if room <= 0:
        return (f"the sleeve is full — {money(used)} of {money(allocation)} at work. "
                f"Close a worldview first (/views, then /close <id>).")

    say(f"thinking about: {thesis[:120]}…", ctx.get("thread"))
    raw = proposer.propose(thesis, views, allocation, used)

    # Budget: conviction-scaled, then clamped by BOTH the per-view ceiling and the room
    # actually left. An explicit size is honoured but still clamped — a typed 500k must not
    # be able to put the sleeve over its cap.
    budget = size_override or proposer.default_budget(allocation, raw.get("confidence", "medium"))
    budget = min(budget, allocation * config.MAX_WORLDVIEW_WEIGHT, room)

    symbols = [leg["symbol"] for leg in (raw.get("legs") or [])]
    proposal = proposer.size(thesis, raw, prices.fetch(symbols) if symbols else {}, budget)
    if not proposal.legs:
        return _render(proposal, allocation, used, "")

    token = _stash(proposal, ctx["user_id"])
    return _render(proposal, allocation, used, token)


def cmd_confirm(args, ctx):
    """Accept the worldview and show the ORDERS it implies. Sends nothing.

    This is the split that matters: approving a worldview is not approving an order. The
    legs say what the view wants to hold; the orders are the delta against what the broker
    holds right now, netted with every other active view. A view reading "long 1,099 SHY"
    can be a 300-share buy or a 700-share sell depending on the book it joins, so the last
    thing approved before money moves is the order list, computed here from live state."""
    if not args:
        return "usage: /confirm <token>"
    if not is_ours(args[0]):
        return None                 # another bot's token — say nothing
    proposal, err = _take(args[0], ctx["user_id"])
    if err:
        return err

    fresh, drift_error = _check_drift(proposal)
    if drift_error:
        return drift_error

    rows = orders.deltas(client, store, extra=proposal.legs)
    token = _stage(proposal, ctx["user_id"])
    return (f"{_banner()}{orders.render(rows, fresh)}\n\n"
            f"Nothing has been sent yet. To place these:\n/place {token}\n"
            f"(expires in {config.CONFIRM_TTL_SEC / 60:.0f} min)")


def cmd_place(args, ctx):
    """The final gate: write the worldview and send the book."""
    if not args:
        return "usage: /place <token>"
    if not is_ours(args[0]):
        return None
    proposal, err = _take_staged(args[0], ctx["user_id"])
    if err:
        return err

    # Re-check drift at the moment of sending, not only when the orders were shown. Time
    # passes between the two, and the share counts are frozen — so this is the check that
    # actually guards what gets filled.
    _, drift_error = _check_drift(proposal)
    if drift_error:
        return drift_error

    # Store BEFORE submitting: a view at the broker that the store does not know about would
    # be closed by the next netted book, silently. The reverse — stored but not sent — is
    # visible in /views and fixed by /sync.
    view = store.add(proposal.thesis, proposal.legs, reasoning=proposal.reasoning,
                     created_by=ctx["user"])
    code = _submit()
    if code != 0:
        return (f"#{view.id} saved, but the book did NOT reach the executor (exit {code}).\n"
                f"The positions are NOT on. Fix the executor, then /sync.")
    return (f"{_banner()}#{view.id} is on.\n\n" + "\n".join(
        f"  {'LONG ' if l.quantity > 0 else 'SHORT'} {l.symbol:<5} {abs(l.quantity):>8,.0f}"
        for l in proposal.legs))


def _check_drift(proposal):
    """(fresh prices, error). Quantities are frozen at approval, so a leg that has moved is a
    different notional than the one shown — resizing it silently would fill a trade nobody
    read, and proceeding anyway would fill a bigger one than was approved."""
    fresh = prices.fetch([l.symbol for l in proposal.legs])
    moved = {s: d for s, d in prices.drift(
        {l.symbol: l.entry_price for l in proposal.legs}, fresh).items()
        if abs(d) > config.MAX_PRICE_DRIFT}
    if not moved:
        return fresh, None
    return fresh, ("REFUSED — prices moved since this was sized:\n"
                   + "\n".join(f"  {s} {d:+.2%}" for s, d in sorted(moved.items()))
                   + "\n\nThe share counts were fixed at the old prices, so this is no "
                     "longer the trade you read. Run /worldview again.")


def _submit() -> int:
    """Push the netted book, unless we are on paper. Imported lazily so the bot starts
    without yfinance present."""
    if config.PAPER:
        log.info("PAPER MODE — not submitting; the book stays local")
        return 0
    from .strategy import submit
    return submit(store=store)


# --------------------------------------------------------------------------- dispatch
class Command(NamedTuple):
    handler: object
    restricted: bool = False
    confirm: bool = False
    preview: object = None


COMMANDS = {
    "help":      Command(cmd_help),
    "start":     Command(cmd_help),
    "views":     Command(cmd_views),
    "view":      Command(cmd_view),
    "book":      Command(cmd_book),
    "playbooks": Command(cmd_playbooks),
    # /worldview runs its own approval flow — the token carries the sized proposal, which
    # the generic confirm path (which re-runs the handler) could not do without asking the
    # model a second time and getting a different trade.
    "worldview": Command(cmd_worldview, restricted=True),
    "confirm":   Command(cmd_confirm, restricted=True),
    "place":     Command(cmd_place, restricted=True),
    # these move real positions, so they take the generic speed bump
    "close":     Command(cmd_close, restricted=True, confirm=True, preview=preview_close),
    "sync":      Command(cmd_sync, restricted=True, confirm=True, preview=preview_sync),
}

_generic_pending: dict = {}


def _describe_failure(e: Exception, label: str) -> str:
    if isinstance(e, ExecutorUnreachable):
        return (f"{label}: the executor is unreachable at {config.EXECUTOR_URL} — "
                f"nothing was sent. {e}")
    if isinstance(e, ExecutorRejected):
        return f"{label}: the executor refused it — {e}"
    if isinstance(e, proposer.ProposalError):
        return f"{label}: {e}"
    if isinstance(e, prices.PriceUnavailable):
        return f"{label}: {e}"
    if isinstance(e, requests.RequestException):
        return f"{label}: cannot reach {config.EXECUTOR_URL} — {e}"
    return f"{label} failed: {e}"


def handle(message: dict) -> None:
    text = (message.get("text") or "").strip()
    if not text.startswith("/"):
        return
    user = message.get("from") or {}
    user_id = user.get("id")
    user_name = user.get("username") or user.get("first_name") or str(user_id)
    thread = message.get("message_thread_id") if config.THREAD == "here" else None

    parts = text.split()
    name, _, addressed = parts[0][1:].partition("@")
    name = name.lower()
    args = parts[1:]

    # `/cmd@SomeOtherBot` in a shared chat is not ambiguous — it is for them. Telegram
    # delivers it to us anyway, so honouring the suffix is the only way to stay out of it.
    # (The executor's control bot strips this suffix instead, which is why it still answers
    # commands aimed at us; that one is worth fixing there too.)
    if addressed and BOT_USERNAME and addressed.lower() != BOT_USERNAME.lower():
        return

    # A generic confirmation for /close and /sync. /confirm is overloaded: it takes both
    # these tokens and proposal tokens, and tries the generic table first because those
    # entries are consumed by re-running a handler while a proposal token is not.
    if name == "confirm" and args:
        if not is_ours(args[0]):
            return                          # the executor bot's token — not our business
        with _lock:
            staged = _generic_pending.pop(args[0], None)
        if staged is not None:
            inner_name, inner_args, owner, expires = staged
            if owner != user_id:
                say("that confirmation belongs to someone else", thread)
                return
            if expires < time.time():
                say("that confirmation expired — run the command again", thread)
                return
            _run(inner_name, inner_args, user_id, user_name, thread)
            return

    entry = COMMANDS.get(name)
    if entry is None:
        # Silence, not "unknown command". This chat also holds the executor's control bot,
        # so an unrecognised command is far more likely to be /status or /kill meant for it
        # than a typo — and that bot already answers unknown commands, so a genuine typo
        # still gets a reply. Two bots both complaining about every command the other owns
        # is what makes a shared chat unusable.
        log.debug("ignoring /%s — not one of ours", name)
        return

    allowed = config.allowed_users()
    if entry.restricted:
        if not allowed:
            say(f"AGENTIC_ALLOWED_USER_IDS is not set, so nothing that trades is enabled. "
                f"Add your id ({user_id}) to it and restart.", thread)
            return
        if user_id not in allowed:
            log.warning("rejected /%s from %s (%s)", name, user_name, user_id)
            say(f"{user_name}, you are not allow-listed here.", thread)
            return

    if entry.confirm:
        try:
            preview = entry.preview(args, {"user": user_name, "user_id": user_id}) \
                if entry.preview else f"/{name} {' '.join(args)}"
        except Exception as e:
            # A plan the executor would reject must never become a confirmation token.
            say(_describe_failure(e, f"/{name}"), thread)
            return
        token = new_token()
        with _lock:
            _generic_pending[token] = (name, args, user_id,
                                       time.time() + config.CONFIRM_TTL_SEC)
        say(f"{preview}\n\nReply:\n/confirm {token}", thread)
        return

    _run(name, args, user_id, user_name, thread)


def _run(name, args, user_id, user_name, thread) -> None:
    label = f"/{name} {' '.join(args)}".strip()
    try:
        reply = COMMANDS[name].handler(args, {"user": user_name, "user_id": user_id,
                                              "thread": thread})
    except Exception as e:
        if not isinstance(e, (ExecutorError, ValueError, requests.RequestException,
                              proposer.ProposalError, prices.PriceUnavailable)):
            log.exception("handler %s failed", name)
        reply = _describe_failure(e, label)
    if reply:                      # a handler may return None to stay silent
        say(reply, thread)
    log.info("%s ran %s", user_name, label)


# --------------------------------------------------------------------------- loop
def poll_loop() -> None:
    # Acknowledge the backlog without acting on it. Telegram redelivers unacknowledged
    # updates forever, so without this a restart would replay whatever was sent while the
    # bot was down — including a stale /confirm.
    result = tg("getUpdates", offset=-1, timeout=0) or []
    offset = (result[-1]["update_id"] + 1) if result else 0
    log.info("skipped the backlog, starting at %s", offset)

    backoff = 1.0
    while True:
        try:
            ok, updates = tg_call("getUpdates", offset=offset, timeout=30,
                                  allowed_updates=["message"])
            if not ok:
                # A 409 means a SECOND process is polling this same token — usually a local
                # `python run_bot.py` started while the container is up. Telegram hands each
                # update to one consumer and terminates the other's request, so the two take
                # turns losing. Backing off and retrying recovers on its own once the extra
                # instance stops; the point of naming it here is that the log should say
                # what to go and look for.
                if "conflict" in str(updates).lower():
                    log.warning("getUpdates conflict — another process is polling this same "
                                "bot token (a stray `python run_bot.py`, or a second "
                                "container). Retrying; stop the other instance to fix it.")
                else:
                    log.warning("telegram getUpdates failed: %s", updates)
                time.sleep(min(backoff, 30)); backoff *= 2
                continue
            backoff = 1.0
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message") or {}
                if str(message.get("chat", {}).get("id")) != config.CHAT_ID:
                    continue
                if time.time() - float(message.get("date", 0)) > STALE_COMMAND_SEC:
                    log.warning("ignoring stale command: %r", (message.get("text") or "")[:40])
                    continue
                handle(message)
        except Exception as e:
            log.exception("poll loop error: %s", e)
            time.sleep(min(backoff, 30)); backoff *= 2


def main() -> None:
    if not config.BOT_TOKEN:
        raise SystemExit(
            "AGENTIC_BOT_TOKEN is not set. This needs its OWN bot, separate from the "
            "executor's control bot — create one with @BotFather, add it to the group, "
            "and put its token in .env.")
    if not config.CHAT_ID:
        raise SystemExit("AGENTIC_CHAT_ID (or TELEGRAM_CHAT_ID) must be set")
    if not config.allowed_users():
        log.warning("AGENTIC_ALLOWED_USER_IDS is not set — read-only until it is")

    global BOT_USERNAME
    me = tg("getMe") or {}
    BOT_USERNAME = me.get("username")
    if BOT_USERNAME:
        log.info("this bot is @%s; commands addressed to other bots are ignored", BOT_USERNAME)
    else:
        log.warning("could not resolve this bot's username — commands addressed to another "
                    "bot in this chat cannot be filtered out")

    log.info("sleeve %s -> executor %s", config.STRATEGY_ID, config.EXECUTOR_URL)
    if config.PAPER:
        log.warning("PAPER MODE — the executor will not be contacted and no order will be "
                    "placed. Unset AGENTIC_PAPER to trade for real.")
    say(f"{_banner()}macro sleeve online — {len(store.active())} worldview(s) active. /help")
    poll_loop()


if __name__ == "__main__":
    main()
