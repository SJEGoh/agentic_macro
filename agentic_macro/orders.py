"""
agentic_macro/orders.py — what the broker will actually be told to do.

The gap this closes: approving a worldview is not the same as approving an order. A
worldview says "long 1,099 SHY"; the order that reaches IB is the DELTA between what the
strategy holds at the broker right now and the netted book across every active worldview.
Those two numbers differ whenever another view already holds the name, whenever a previous
submission partially filled, and whenever anything has drifted.

So the last thing approved before money moves should be this list, not the legs. A view whose
legs read "long 1,099 SHY" can be a 300-share buy, a 700-share SELL, or nothing at all,
depending on the book it is joining — and the only way to know which is to difference it
against live broker state at the moment of submission.

Everything here is READ-ONLY. It asks the executor what it holds and does the arithmetic;
nothing in this module places, cancels, or changes anything.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

from . import config

log = logging.getLogger("agentic-macro.orders")


class Delta(NamedTuple):
    symbol: str
    current: float           # what the broker holds for this strategy now
    target: float            # what the netted book wants
    holders: tuple = ()      # active worldview ids that want this name

    @property
    def delta(self) -> float:
        return self.target - self.current

    @property
    def side(self) -> str:
        return "BUY" if self.delta > 0 else "SELL"

    @property
    def closes(self) -> bool:
        return self.target == 0 and self.current != 0


def broker_positions(client, store=None, strategy_id: str = None) -> dict:
    """What this strategy holds at the broker.

    Reads the executor's `strategy_positions` view, not the account's — the sleeve must never
    difference against another strategy's book, or it would try to trade its way to owning
    the whole account.

    In PAPER mode there is no executor, so the stored book stands in for the broker: nothing
    else moves these positions, so what the active worldviews add up to is exactly what would
    be held. That keeps the order previews honest rather than showing every position as a
    fresh trade every time."""
    if config.PAPER:
        return dict(store.net_positions()) if store is not None else {}
    sid = strategy_id or config.STRATEGY_ID
    positions = client.positions() or {}
    held = (positions.get("strategy_positions") or {}).get(sid) or {}
    return {s: float(q) for s, q in held.items() if q}


def deltas(client, store, extra=None, exclude=None) -> list:
    """The orders that submitting the current book would generate.

    `extra` adds legs not yet stored (a proposal being considered); `exclude` drops a
    worldview's legs (one being closed). Both are how a change is costed BEFORE it is
    written, so the preview describes the future without creating it.

    A symbol is included only when the delta is non-zero: a name whose target already
    matches the broker generates no order, and listing it would pad the confirmation with
    lines that are not decisions."""
    target = store.net_positions(extra=extra, exclude=exclude)
    current = broker_positions(client, store)

    out = []
    for symbol in sorted(set(target) | set(current)):
        want, have = target.get(symbol, 0.0), current.get(symbol, 0.0)
        if abs(want - have) < 1e-9:
            continue
        holders = tuple(wid for wid, _ in store.holders(symbol)
                        if exclude is None or wid != exclude)
        out.append(Delta(symbol, have, want, holders))
    # biggest trades first — the ones worth reading carefully
    out.sort(key=lambda d: -abs(d.delta))
    return out


def render(rows: list, prices: dict = None) -> str:
    """The order list as it appears in a confirmation. Says what changes and why."""
    if not rows:
        return ("NO ORDERS — the broker already holds this book.\n"
                "  (Submitting anyway is harmless: /targets takes absolute positions.)")

    lines = ["ORDERS TO BE PLACED  (vs the broker right now)"]
    traded = 0.0
    for row in rows:
        price = (prices or {}).get(row.symbol)
        value = f"  = {'$%s' % f'{abs(row.delta) * price:,.0f}'}" if price else ""
        if price:
            traded += abs(row.delta) * price
        note = "  CLOSE" if row.closes else ""
        owners = (f"  [#{', #'.join(str(h) for h in row.holders)}]" if row.holders else "")
        lines.append(f"  {row.side:<4} {row.symbol:<5} {abs(row.delta):>9,.0f}"
                     f"   ({row.current:+,.0f} -> {row.target:+,.0f}){note}{value}{owners}")

    lines.append(f"\n{len(rows)} order(s)" + (f" · {'$%s' % f'{traded:,.0f}'} traded"
                                              if traded else ""))
    return "\n".join(lines)
