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

import datetime as dt
import logging
import secrets
import threading
import time
from typing import NamedTuple

import requests

from . import config, orders, prices, proposer, playbooks, universe
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
    #: worldview this would REPLACE (/switch). Carried on the token so the close and the
    #: open commit together — closing first would leave a window where the sleeve holds
    #: neither view and a /sync would trade the gap.
    replaces: int = None


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


def _stash(proposal, user_id: int, replaces: int = None) -> str:
    token = new_token()
    with _lock:
        now = time.time()
        for t, p in list(_pending.items()):
            if p.expires_at < now:
                del _pending[t]
        _pending[token] = Pending(proposal, user_id, now + config.CONFIRM_TTL_SEC, replaces)
    return token


def _stage(proposal, user_id: int, replaces: int = None) -> str:
    """Move an accepted proposal to the final gate. Kept separate from _pending so a token
    can only be used for the step it was issued for: a /place token cannot re-approve a
    worldview, and a /confirm token cannot send orders."""
    token = new_token()
    with _lock:
        now = time.time()
        for t, p in list(_staged.items()):
            if p.expires_at < now:
                del _staged[t]
        _staged[token] = Pending(proposal, user_id, now + config.CONFIRM_TTL_SEC, replaces)
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
    return pending, None


def _take(token: str, user_id: int):
    with _lock:
        pending = _pending.pop(token, None)
    if pending is None:
        return None, "nothing pending with that token (or it expired)"
    if pending.user_id != user_id:
        return None, "that proposal belongs to someone else"
    if pending.expires_at < time.time():
        return None, "that proposal expired — run /worldview again"
    return pending, None


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
        # For a future the size is contracts, and the notional is contracts * price *
        # multiplier. Printing the bare price * quantity would understate a note future by
        # 1,000x — the same mistake the sizing code exists to avoid, made in the one place
        # a human would actually catch it.
        unit = f" x{leg.multiplier:,.0f}" if leg.multiplier != 1 else ""
        lines.append(f"  {'LONG ' if leg.quantity > 0 else 'SHORT'} {leg.symbol:<5} "
                     f"{universe.qty(abs(leg.quantity)):>11} @ {leg.entry_price:>8,.2f}{unit}"
                     f"  = {money(leg.risk):>12}")
        if leg.rationale:
            lines.append(f"        {leg.rationale}")

    if p.hedge_note:
        lines.append(f"\n  hedge ratio: {p.hedge_note}")
    if p.substituted:
        # Loud, because the instrument in the trade is not the one the reasoning above
        # names. The exposure is identical; the fee and roll profile are not.
        lines.append("\n  SUBSTITUTED (contract too large for this budget):")
        lines += [f"    {s}" for s in p.substituted]
    if p.dropped:
        lines.append("  not included: " + "; ".join(p.dropped))

    after = capital_used + p.gross()
    lines.append(f"\nCAPITAL")
    lines.append(f"  this view    {money(p.gross())} gross (budget {money(p.budget)})")
    lines.append(f"  sleeve after {money(after)} of {money(allocation)} "
                 f"({after / allocation:.0%})")

    if p.context:
        lines.append("\nNEWS THE MODEL SAW")
        lines += [f"  [{c['published']}] {c['title'][:88]}" for c in p.context]
    elif p.context_note:
        # Said out loud. A proposal made without context looks identical to one made with it,
        # and the difference decides whether "already priced" meant anything.
        lines.append(f"\nNEWS THE MODEL SAW\n  none — {p.context_note}")

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
        "\nPROPOSE\n"
        "  /worldview [size] <your view>   propose a structure for a view\n"
        "        /worldview 50k the Fed cuts into an intact labour market\n"
        "  /switch <id> <new view>         replace a worldview with a different one\n"
        "\nAPPROVE  (two gates — nothing trades until /place)\n"
        "  /confirm <token>       accept it, and see the ORDERS it implies\n"
        "  /place <token>         send those orders  <- the last gate\n"
        "  /cancel [token]        throw away what is pending\n"
        "\nADJUST WHAT IS ON\n"
        "  /resize <id> <size>    80k | +50% | -30% | x1.5 — every leg by the same\n"
        "        factor, so the hedge ratio is unchanged\n"
        "  /adjust <id> <SYM> <n> set one leg, signed: -250 means short 250.\n"
        "        += / -= to nudge it, 0 to remove it. Breaks the hedge — it shows you\n"
        "  /close <id>            retire a view; its legs unwind on the next book\n"
        "\nLOOK\n"
        "  /views                 what is on, and what each view holds\n"
        "  /view <id>             one worldview in full — thesis, legs, reasoning\n"
        "  /book                  the netted book, and which views each name comes from\n"
        "  /sync                  re-send the netted book (self-heals drift)\n"
        "  /playbooks [name]      the structures available\n"
        "  /universe [what]       the instruments, and what one unit costs\n"
        "  /news [thesis]         recent macro context, and what a thesis retrieves\n"
        "  /ingest                pull the latest news into that store\n"
        "\nThe book sent to the executor is the NET of every active worldview, so closing\n"
        "one removes exactly its legs. Everything that trades needs a confirmation.")


def cmd_playbooks(args, ctx) -> str:
    if args:
        book = playbooks.BY_NAME.get(args[0].lower())
        if not book:
            return f"no playbook called {args[0]!r} — /playbooks lists them"
        return book.render()
    return ("Structures (/playbooks <name> for one in full):\n\n" + "\n".join(
        f"  {p.name.replace('_', ' '):<24} [{p.weighting}]" for p in playbooks.PLAYBOOKS))


def cmd_universe(args, ctx) -> str:
    """What can actually be traded, and what one unit of it costs.

    The size matters more than the list now that the universe holds futures: MCL and ZT are
    both "a futures contract" and one is $9k while the other is $205k. Prices are best
    effort — this is a reference card, and a missing quote costs a number, not a trade."""
    try:
        marks = prices.fetch([i.symbol for i in universe.UNIVERSE
                              if i.sec_type == "FUT"])
    except Exception:
        marks = {}

    query = (args[0] if args else "").strip().lower()

    # one instrument
    if query:
        try:
            i = universe.resolve(query)
        except universe.UnknownSymbol:
            i = None
        if i is not None:
            facts = [f"bucket      {i.bucket}", f"type        {i.sec_type}"]
            if i.duration:
                facts.append(f"duration    {i.duration:g}y")
            if i.beta:
                facts.append(f"beta        {i.beta:g}")
            if i.sec_type == "FUT":
                facts.append(f"exchange    {i.exchange}")
                facts.append(f"multiplier  {i.multiplier:,.0f}")
                if i.symbol in marks:
                    facts.append(f"1 contract  {money(i.notional(marks[i.symbol]))}"
                                 f"  (at {marks[i.symbol]:,.4g})")
            else:
                facts.append(f"fractional  {'yes' if config.FRACTIONAL else 'no'}")
            flags = ([f"DECAYS if held: {i.symbol} loses money while you wait"]
                     if i.symbol in universe.DECAY_PRONE else []) + \
                    (["thin — keep it small"] if i.symbol in universe.THIN else [])
            return (f"{i.symbol}\n  {i.note}\n\n  " + "\n  ".join(facts)
                    + ("\n\n  ! " + "\n  ! ".join(flags) if flags else ""))

    # one bucket
    buckets = {}
    for i in universe.UNIVERSE:
        buckets.setdefault(i.bucket, []).append(i)
    if query in buckets:
        lines = [f"[{query.upper()}]"]
        for i in buckets[query]:
            note = i.note.split(" — ")[0][:46]      # the exposure, not the caveat
            size = (f"  1 contract {money(i.notional(marks[i.symbol])):>9}"
                    if i.symbol in marks else "")
            lines.append(f"  {i.symbol:<5} {note:<48}{size}")
        return "\n".join(lines)
    if query:
        return (f"no instrument or bucket called {args[0]!r}.\n"
                f"buckets: " + ", ".join(sorted(buckets)))

    # the summary
    lines = [f"TRADEABLE UNIVERSE — {len(universe.UNIVERSE)} instruments",
             "/universe <bucket> or <symbol> for detail\n"]
    for bucket, members in buckets.items():
        tag = " (futures)" if members[0].sec_type == "FUT" else ""
        lines.append(f"{bucket}{tag}")
        lines.append("  " + " ".join(i.symbol for i in members))
    if marks:
        cheapest = min(marks, key=lambda s: universe.resolve(s).notional(marks[s]))
        dearest = max(marks, key=lambda s: universe.resolve(s).notional(marks[s]))
        lines.append(f"\ncontract sizes run {money(universe.resolve(cheapest).notional(marks[cheapest]))}"
                     f" ({cheapest}) to {money(universe.resolve(dearest).notional(marks[dearest]))}"
                     f" ({dearest}) — a big one may not fit the sleeve at all")
    lines.append("ETFs size fractionally; futures are whole contracts only.")
    return "\n".join(lines)


def cmd_news(args, ctx) -> str:
    """What the store would retrieve for a thesis — the same call the proposer makes."""
    from . import memory
    if not args:
        try:
            s = memory.stats()
        except memory.MemoryUnavailable as e:
            return f"memory unavailable: {e}"
        return (f"macro memory: {s['total']} document(s) — {s['fresh_7d']} from the last "
                f"week, {s['older']} older\n"
                f"  relevance halves every {memory.DEFAULT_HALFLIFE:.0f} days for news; "
                f"nothing is hidden, only outranked\n"
                f"  /news <a thesis>   what would be retrieved for it\n"
                f"  /ingest            pull the latest news in")
    query = " ".join(args)
    try:
        hits = memory.recall(query)
    except memory.MemoryUnavailable as e:
        return f"memory unavailable: {e}"
    if not hits:
        return f"nothing live in the store for {query!r} — /ingest first?"
    return (f"for {query!r}  (ranked on similarity x age-decay):\n" + "\n".join(
        f"  [{h['published']}, {h['age_days']:.0f}d]  sim {h['similarity']:.2f}"
        f" x w {h['weight']:.2f} = {h['score']:.2f}  "
        f"{(h['meta'].get('title') or h['text'])[:60]}" for h in hits))


def preview_ingest(args, ctx) -> str:
    from . import memory
    try:
        s = memory.stats()
    except memory.MemoryUnavailable as e:
        return f"memory unavailable: {e}"
    return (f"Pull up to {config.MEMORY_INGEST_LIMIT} recent articles into the macro memory.\n"
            f"Currently holding {s['total']} ({s['fresh_7d']} from the last week).\n"
            f"This spends embedding quota; it places no orders.")


def cmd_ingest(args, ctx) -> str:
    from . import memory
    try:
        r = memory.ingest_news()
        sweep = memory.sweep()
    except memory.MemoryUnavailable as e:
        return f"ingest failed: {e}"
    return (f"scanned {r['scanned']}, kept {r['fetched']} macro, stored {r['stored']}\n"
            f"  {r['deduped']} syndicated copies collapsed into their original\n"
            f"  {sweep['deleted']} past the age backstop swept, "
            f"{sweep['remaining']} held")


def cmd_views(args, ctx) -> str:
    allocation, views, used, marks = _sleeve()
    if not views:
        return f"no active worldviews. Sleeve is flat, {money(allocation)} available."
    lines = []
    for v in views:
        legs = ", ".join(f"{'+' if l.quantity > 0 else ''}{universe.qty(l.quantity)} "
                         f"{l.symbol}" for l in v.legs)
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
        inst = universe.resolve(symbol)
        lines.append(f"  {symbol:<5} {'+' if quantity >= 0 else ''}"
                     f"{universe.qty(quantity):>11} @ {marks[symbol]:>8,.2f} "
                     f"= {money(abs(quantity) * inst.notional(marks[symbol])):>12}"
                     f"   from {owners}")
    gross = sum(abs(q) * universe.resolve(s).notional(marks[s])
                for s, q in targets.items())
    lines.append(f"\n{money(gross)} gross across {len(targets)} names")
    return "\n".join(lines)


# --------------------------------------------------------------------------- adjusting
def _live_view(arg) -> object:
    view = store.get(int(arg))
    if view is None:
        raise ValueError(f"no worldview #{arg}")
    if view.status != "active":
        raise ValueError(f"worldview #{view.id} is closed")
    return view


def _scaled(view, factor: float, marks: dict) -> list:
    """The view's legs, all multiplied by `factor`.

    Scaling every leg by the same number is what keeps a resize a RESIZE: the hedge ratio,
    the DV01 balance and the vol weighting are all ratios between legs, and a common factor
    leaves every one of them untouched. Adjusting legs individually is `/adjust`, and it is
    a different operation precisely because it does not preserve them.

    Futures still have to land on whole contracts, so a scaled futures leg is truncated —
    and if that truncation would zero a leg, the caller refuses rather than dropping it."""
    from .store import Leg
    out = []
    for l in view.legs:
        inst = universe.resolve(l.symbol)
        q = l.quantity * factor
        if inst.sec_type == "FUT":
            q = float(int(q))
        else:
            scale = 10 ** config.FRACTIONAL_DP
            q = int(q * scale) / scale
        out.append(Leg(l.symbol, q, l.entry_price, l.rationale, l.multiplier))
    lost = [l.symbol for old, l in zip(view.legs, out) if old.quantity and not l.quantity]
    if lost:
        raise ValueError(
            f"scaling by {factor:.2f}x would zero {', '.join(lost)}, which turns this into a "
            f"different structure rather than a smaller one. Use /close, or a smaller change.")
    return out


def _resize_target(view, arg: str, marks: dict) -> tuple:
    """(new legs, factor, description). Accepts an absolute gross, a percentage, or xN."""
    current = view.gross(marks)
    raw = arg.strip().lower()
    if raw.startswith("x"):
        factor = float(raw[1:])
    elif raw.endswith("%"):
        factor = 1.0 + float(raw[:-1].replace("+", "")) / 100.0
    elif raw.startswith(("+", "-")) and _parse_amount(raw.lstrip("+-")):
        delta = _parse_amount(raw.lstrip("+-")) * (1 if raw[0] == "+" else -1)
        factor = (current + delta) / current
    else:
        target = _parse_amount(raw)
        if not target:
            raise ValueError("usage: /resize <id> <80k | +50% | -30% | x1.5>")
        factor = target / current
    if factor <= 0:
        raise ValueError("that would take the view to zero or flip it — use /close instead")
    return _scaled(view, factor, marks), factor, f"{current:,.0f} -> {current * factor:,.0f}"


def preview_resize(args, ctx) -> str:
    if len(args) < 2:
        raise ValueError("usage: /resize <id> <80k | +50% | -30% | x1.5>")
    view = _live_view(args[0])
    marks = prices.fetch([l.symbol for l in view.legs])
    legs, factor, moved = _resize_target(view, args[1], marks)
    rows = orders.deltas(client, store, extra=legs, exclude=view.id)
    return (f"{_banner()}RESIZE #{view.id} to {factor:.2f}x\n  {view.thesis}\n"
            f"  gross {money(view.gross(marks))} -> {money(view.gross(marks) * factor)}\n\n"
            + orders.render(rows, marks)
            + "\n\nEvery leg scales by the same factor, so the hedge ratio is unchanged.")


def cmd_resize(args, ctx) -> str:
    view = _live_view(args[0])
    marks = prices.fetch([l.symbol for l in view.legs])
    legs, factor, _ = _resize_target(view, args[1], marks)
    store.replace_legs(view.id, legs)
    code = _submit()
    if code != 0:
        return (f"#{view.id} resized in the store, but the book did NOT go through "
                f"(exit {code}). Run /sync.")
    return f"{_banner()}#{view.id} resized to {factor:.2f}x.\n\n{cmd_view([str(view.id)], ctx)}"


def _adjusted(view, symbol: str, spec: str, marks: dict) -> list:
    from .store import Leg
    inst = universe.resolve(symbol)
    legs = {l.symbol: l for l in view.legs}
    current = legs[inst.symbol].quantity if inst.symbol in legs else 0.0
    # ABSOLUTE by default; relative needs an explicit += / -=.
    #
    # "-250" on a leg that is already short 100 is genuinely ambiguous — it reads equally as
    # "set it to short 250" and "reduce it by 250", and those are different positions. Leg
    # quantities are displayed signed, so a signed number setting the leg is the reading that
    # matches what you are looking at; anything relative has to say so.
    raw = spec.strip()
    if raw[:2] in ("+=", "-="):
        target = current + float(raw[0] + raw[2:])
    else:
        target = float(raw)
    if inst.sec_type == "FUT":
        target = float(int(target))
    out = [Leg(l.symbol, l.quantity, l.entry_price, l.rationale, l.multiplier)
           for l in view.legs if l.symbol != inst.symbol]
    if target:
        entry = legs[inst.symbol].entry_price if inst.symbol in legs else marks[inst.symbol]
        out.append(Leg(inst.symbol, target, entry,
                       legs[inst.symbol].rationale if inst.symbol in legs else "manual",
                       inst.multiplier))
    if not out:
        raise ValueError("that would empty the worldview — use /close instead")
    return out


def _balance(legs, marks: dict) -> str:
    """Net duration- and beta-dollars of a set of legs. Shown on /adjust because a manual
    change is the one operation that CAN break a hedge, and the whole point of the structures
    in this sleeve is that they were balanced on something."""
    dv01 = beta = gross = 0.0
    neither = []
    for l in legs:
        i = universe.resolve(l.symbol)
        value = l.quantity * marks.get(l.symbol, l.entry_price) * l.multiplier
        gross += abs(value)
        dv01 += value * (i.duration or 0.0)
        beta += value * (i.beta or 0.0)
        if not i.duration and not i.beta:
            neither.append(i.symbol)
    # Gross first, and name the legs that sit on NEITHER axis. Without that, adding a $16k
    # gold leg to a curve trade prints an identical duration line before and after and reads
    # as "nothing changed", when what actually happened is a whole new exposure the balance
    # numbers are blind to.
    bits = [f"gross {money(gross)}"]
    if dv01:
        bits.append(f"duration-$ {dv01:+,.0f}")
    if beta:
        bits.append(f"beta-$ {beta:+,.0f}")
    if neither:
        bits.append(f"outside both axes: {', '.join(sorted(set(neither)))}")
    return "  ·  ".join(bits)


def preview_adjust(args, ctx) -> str:
    if len(args) < 3:
        raise ValueError("usage: /adjust <id> <SYMBOL> <qty>\n"
                         "  /adjust 3 TLT -250   set the leg to short 250\n"
                         "  /adjust 3 TLT +=500  add 500 to it\n"
                         "  /adjust 3 TLT 0      remove the leg")
    view = _live_view(args[0])
    legs = _adjusted(view, args[1], args[2],
                     prices.fetch(universe.with_fallbacks([args[1]])))
    marks = prices.fetch({l.symbol for l in legs} | {l.symbol for l in view.legs})
    rows = orders.deltas(client, store, extra=legs, exclude=view.id)
    return (f"{_banner()}ADJUST #{view.id}\n  {view.thesis}\n\n"
            + orders.render(rows, marks)
            + f"\n\nBALANCE\n  before  {_balance(view.legs, marks)}"
              f"\n  after   {_balance(legs, marks)}"
              f"\n\nAdjusting one leg does not preserve the hedge ratio — check the line "
              f"above before approving.")


def cmd_adjust(args, ctx) -> str:
    view = _live_view(args[0])
    legs = _adjusted(view, args[1], args[2],
                     prices.fetch(universe.with_fallbacks([args[1]])))
    store.replace_legs(view.id, legs)
    code = _submit()
    if code != 0:
        return (f"#{view.id} adjusted in the store, but the book did NOT go through "
                f"(exit {code}). Run /sync.")
    return f"{_banner()}#{view.id} adjusted.\n\n{cmd_view([str(view.id)], ctx)}"


def cmd_switch(args, ctx) -> str:
    """Replace a worldview with a different one. The old view closes and the new one opens
    in the SAME netted book, so anything the two share is never sold and re-bought."""
    if len(args) < 2:
        return "usage: /switch <id> <the new view>"
    view = _live_view(args[0])
    thesis = " ".join(args[1:]).strip()
    if len(thesis) < 10:
        return "give me a sentence or two of actual view to switch to."

    allocation, views, used, marks = _sleeve()
    # The view being replaced frees its own capital, so the new one is costed against the
    # room it will actually have rather than the room before the swap.
    freed = view.gross(marks)
    room = allocation * config.MAX_GROSS_WEIGHT - used + freed
    others = [v for v in views if v.id != view.id]

    say(f"switching #{view.id} to: {thesis[:100]}…", ctx.get("thread"))
    raw = proposer.propose(thesis, others, allocation, used - freed)
    budget = min(proposer.default_budget(allocation, raw.get("confidence", "medium")),
                 allocation * config.MAX_WORLDVIEW_WEIGHT, room)

    symbols = universe.with_fallbacks([l["symbol"] for l in (raw.get("legs") or [])])
    series = prices.bars(symbols) if symbols else {}
    proposal = proposer.size(thesis, raw, {s: v[-1][1] for s, v in series.items()}, budget,
                             prices.realized_vol(symbols, series=series) if symbols else {})
    if not proposal.legs:
        return _render(proposal, allocation, used, "")
    token = _stash(proposal, ctx["user_id"], replaces=view.id)
    return (f"REPLACING #{view.id}: {view.thesis}\n\n"
            + _render(proposal, allocation, used - freed, token))


def cmd_cancel(args, ctx) -> str:
    """Throw away anything this user has pending. Cheaper than waiting out the TTL, and it
    means an abandoned proposal cannot be confirmed by accident twenty minutes later."""
    uid = ctx["user_id"]
    with _lock:
        if args:
            token = args[0]
            if not is_ours(token):
                return None
            gone = bool(_pending.pop(token, None) or _staged.pop(token, None)
                        or _generic_pending.pop(token, None))
            return f"{token} discarded." if gone else "nothing pending with that token."
        n = 0
        for book in (_pending, _staged, _generic_pending):
            for tok in [k for k, v in book.items()
                        if (v.user_id if isinstance(v, Pending) else v[2]) == uid]:
                del book[tok]
                n += 1
    return f"discarded {n} pending item(s)." if n else "nothing pending."


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

    # One pull serves both: the price that sizes the leg and the volatility that weights it
    # come from the same bars, so they cannot describe two different moments.
    # Price the ETF equivalents alongside the proposed legs. A substitution decided during
    # sizing must not then need its own network call — a price fetched mid-sizing would be
    # from a different moment than the rest of the book.
    symbols = universe.with_fallbacks(
        [leg["symbol"] for leg in (raw.get("legs") or [])])
    series = prices.bars(symbols) if symbols else {}
    marks = {s: v[-1][1] for s, v in series.items()}
    vols = prices.realized_vol(symbols, series=series) if symbols else {}
    proposal = proposer.size(thesis, raw, marks, budget, vols)
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
    pending, err = _take(args[0], ctx["user_id"])
    if err:
        return err
    proposal = pending.proposal

    fresh, drift_error = _check_drift(proposal)
    if drift_error:
        return drift_error

    rows = orders.deltas(client, store, extra=proposal.legs, exclude=pending.replaces)
    token = _stage(proposal, ctx["user_id"], pending.replaces)
    head = (f"REPLACES #{pending.replaces}\n\n" if pending.replaces else "")
    return (f"{_banner()}{head}{orders.render(rows, fresh)}\n\n"
            f"Nothing has been sent yet. To place these:\n/place {token}\n"
            f"(expires in {config.CONFIRM_TTL_SEC / 60:.0f} min)  ·  /cancel to discard")


def cmd_place(args, ctx):
    """The final gate: write the worldview and send the book."""
    if not args:
        return "usage: /place <token>"
    if not is_ours(args[0]):
        return None
    pending, err = _take_staged(args[0], ctx["user_id"])
    if err:
        return err
    proposal = pending.proposal

    # Re-check drift at the moment of sending, not only when the orders were shown. Time
    # passes between the two, and the share counts are frozen — so this is the check that
    # actually guards what gets filled.
    _, drift_error = _check_drift(proposal)
    if drift_error:
        return drift_error

    # Store BEFORE submitting: a view at the broker that the store does not know about would
    # be closed by the next netted book, silently. The reverse — stored but not sent — is
    # visible in /views and fixed by /sync.
    replaced = ""
    if pending.replaces:
        # Close and open in the SAME netted book. Closing as a separate submission would
        # briefly flatten the old view's legs and then re-open the new ones, paying spread
        # twice on anything the two views share.
        store.close_worldview(pending.replaces, closed_by=ctx["user"])
        replaced = f" (replacing #{pending.replaces})"
    view = store.add(proposal.thesis, proposal.legs, reasoning=proposal.reasoning,
                     created_by=ctx["user"])
    code = _submit()
    if code != 0:
        return (f"#{view.id} saved, but the book did NOT reach the executor (exit {code}).\n"
                f"The positions are NOT on. Fix the executor, then /sync.")
    return (f"{_banner()}#{view.id} is on{replaced}.\n\n" + "\n".join(
        f"  {'LONG ' if l.quantity > 0 else 'SHORT'} {l.symbol:<5} "
        f"{universe.qty(abs(l.quantity)):>11}" for l in proposal.legs))


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
    "universe":  Command(cmd_universe),
    "news":      Command(cmd_news),
    # spends embedding quota, so it takes a confirmation even though it cannot trade
    "ingest":    Command(cmd_ingest, restricted=True, confirm=True, preview=preview_ingest),
    # /worldview runs its own approval flow — the token carries the sized proposal, which
    # the generic confirm path (which re-runs the handler) could not do without asking the
    # model a second time and getting a different trade.
    "worldview": Command(cmd_worldview, restricted=True),
    "confirm":   Command(cmd_confirm, restricted=True),
    "place":     Command(cmd_place, restricted=True),
    # these move real positions, so they take the generic speed bump
    "close":     Command(cmd_close, restricted=True, confirm=True, preview=preview_close),
    "resize":    Command(cmd_resize, restricted=True, confirm=True, preview=preview_resize),
    "adjust":    Command(cmd_adjust, restricted=True, confirm=True, preview=preview_adjust),
    "switch":    Command(cmd_switch, restricted=True),
    "cancel":    Command(cmd_cancel, restricted=True),
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


# --------------------------------------------------------------------------- bootstrap
def bootstrap_memory(force: bool = False) -> dict:
    """Fill the news store if it is empty or stale. NEVER fatal.

    A proposal without context is a worse proposal, not a broken one — and it says so on
    every message. A bot that refuses to start because a news feed is down would be the
    strictly worse failure: you lose the ability to trade at all over a degraded input."""
    if not config.MEMORY_ENABLED or not config.MEMORY_AUTO_INGEST:
        return {"skipped": "memory disabled"}
    try:
        from . import memory
        stats = memory.stats()
        if not force and stats["total"] >= config.MEMORY_MIN_DOCS and stats["fresh_7d"]:
            log.info("news store ready: %s documents (%s from the last week)",
                     stats["total"], stats["fresh_7d"])
            return stats

        days = config.MEMORY_BOOTSTRAP_DAYS
        log.info("news store has %s documents — pulling %.0f days", stats["total"], days)
        now = dt.datetime.now(dt.timezone.utc)
        r = memory.ingest_window(before=now, after=now - dt.timedelta(days=days),
                                 limit=config.MEMORY_INGEST_LIMIT)
        memory.sweep()
        after = memory.stats()
        log.info("ingested %s of %s macro articles (%s syndicated copies collapsed) — "
                 "store now %s", r["stored"], r["fetched"], r["deduped"], after["total"])
        return after
    except Exception as e:
        log.error("could not build the news store — proposals will run WITHOUT context "
                  "and will say so: %s", e)
        return {"error": str(e)}


def refresh_loop() -> None:
    """Top the store up on a schedule. Without this a long-running container serves
    fortnight-old news that age-decay has already weighted to nearly nothing, which is the
    same as having none while looking like it has some."""
    interval = config.MEMORY_REFRESH_HOURS * 3600
    if interval <= 0:
        return
    while True:
        time.sleep(interval)
        try:
            bootstrap_memory(force=True)
        except Exception as e:                      # never let this kill the thread
            log.warning("news refresh failed: %s", e)


def preflight() -> None:
    """Say what this process is configured to do, before it does any of it."""
    from . import providers
    status = providers.status()
    log.info("chat: %s   embeddings: %s", status["llm"], status["embeddings"])
    for role in ("llm", "embeddings"):
        if f"{role}_missing" in status:
            log.error("%s model is not available — %s", role, status[f"{role}_missing"])
    if status.get("ollama_reachable") is False:
        log.error("Ollama is configured but unreachable at %s. From a container that is "
                  "usually host.docker.internal, not localhost.", config.OLLAMA_HOST)
    if config.PAPER:
        log.warning("PAPER MODE — no order will reach the executor")


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

    preflight()
    log.info("sleeve %s -> executor %s", config.STRATEGY_ID, config.EXECUTOR_URL)
    bootstrap_memory()
    threading.Thread(target=refresh_loop, daemon=True, name="news-refresh").start()
    if config.PAPER:
        log.warning("PAPER MODE — the executor will not be contacted and no order will be "
                    "placed. Unset AGENTIC_PAPER to trade for real.")
    say(f"{_banner()}macro sleeve online — {len(store.active())} worldview(s) active. /help")
    poll_loop()


if __name__ == "__main__":
    main()
