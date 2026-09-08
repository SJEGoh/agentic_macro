"""
agentic_macro/store.py — the worldviews, and the legs each one holds.

The sleeve holds several worldviews at once. Each is a thesis you approved, plus the legs
that express it; the book the executor is given is the NET of every active worldview's legs,
summed per symbol. That indirection is the whole design:

  * you can hold contradictory-ish views side by side and still send one coherent book;
  * closing a worldview removes exactly its legs and nothing else, so the unwind is
    computed rather than remembered — no "which of these shares were the oil trade?";
  * the netted book is a pure function of the stored rows, so re-sending it is idempotent
    and self-healing, which is exactly what `/targets` wants.

Quantities are stored, prices are not — beyond the entry price kept for the record. A leg
means "300 shares of TLT", not "$90k of TLT", so re-pricing the book at submit time changes
what it is worth, never what it holds. Sizing drifts only when you approve a new view.

Everything is written inside a transaction and read back through the same connection, so a
crash between "approved" and "submitted" leaves a worldview that is either wholly there or
wholly absent — a half-written view would net into a book nobody approved.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS worldviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis      TEXT    NOT NULL,
    reasoning   TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active',   -- 'active' | 'closed'
    created_at  TEXT    NOT NULL,
    created_by  TEXT    NOT NULL DEFAULT '',
    closed_at   TEXT,
    closed_by   TEXT
);
CREATE TABLE IF NOT EXISTS legs (
    worldview_id INTEGER NOT NULL REFERENCES worldviews(id) ON DELETE CASCADE,
    symbol       TEXT    NOT NULL,
    quantity     REAL    NOT NULL,
    entry_price  REAL    NOT NULL,
    rationale    TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (worldview_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_worldviews_status ON worldviews(status);
"""


class Leg(NamedTuple):
    symbol: str
    quantity: float
    entry_price: float
    rationale: str = ""


class Worldview(NamedTuple):
    id: int
    thesis: str
    reasoning: str
    status: str
    created_at: str
    created_by: str
    legs: tuple

    def gross(self, prices: dict = None) -> float:
        """Gross notional — the sum of |quantity| * price, at current prices when given and
        entry prices otherwise. Gross rather than net because a long/short pair consumes
        capital on both sides; netting them to zero would report a paired trade as free."""
        return sum(abs(l.quantity) * float((prices or {}).get(l.symbol, l.entry_price))
                   for l in self.legs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ writes
    def add(self, thesis: str, legs, reasoning: str = "", created_by: str = "") -> Worldview:
        """Persist an APPROVED worldview and its legs, atomically.

        Called only after you confirm, and before the book is submitted. That order matters:
        a view that reached the broker but is missing here would hold positions the netted
        book does not know about, and the next submission — being authoritative — would
        close them without ever saying so."""
        legs = [Leg(l.symbol.upper(), float(l.quantity), float(l.entry_price),
                    getattr(l, "rationale", "")) for l in legs]
        with self.conn:
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "INSERT INTO worldviews (thesis, reasoning, status, created_at, created_by) "
                "VALUES (?, ?, 'active', ?, ?)",
                (thesis, reasoning, _now(), created_by))
            wid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO legs (worldview_id, symbol, quantity, entry_price, rationale) "
                "VALUES (?, ?, ?, ?, ?)",
                [(wid, l.symbol, l.quantity, l.entry_price, l.rationale) for l in legs])
        return self.get(wid)

    def close_worldview(self, wid: int, closed_by: str = "") -> Worldview:
        """Retire a worldview. Its legs stay on the row for the record but stop counting
        toward the netted book, so the next submission unwinds exactly this view's
        positions and leaves every other view's untouched."""
        view = self.get(wid)
        if view is None:
            raise KeyError(f"no worldview #{wid}")
        if view.status != "active":
            raise ValueError(f"worldview #{wid} is already closed")
        with self.conn:
            self.conn.execute(
                "UPDATE worldviews SET status='closed', closed_at=?, closed_by=? WHERE id=?",
                (_now(), closed_by, wid))
        return self.get(wid)

    # ------------------------------------------------------------------ reads
    def get(self, wid: int) -> Worldview:
        row = self.conn.execute("SELECT * FROM worldviews WHERE id=?", (wid,)).fetchone()
        return self._hydrate(row) if row else None

    def active(self) -> list:
        rows = self.conn.execute(
            "SELECT * FROM worldviews WHERE status='active' ORDER BY id").fetchall()
        return [self._hydrate(r) for r in rows]

    def all(self, limit: int = 50) -> list:
        rows = self.conn.execute(
            "SELECT * FROM worldviews ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._hydrate(r) for r in rows]

    def _hydrate(self, row) -> Worldview:
        legs = self.conn.execute(
            "SELECT symbol, quantity, entry_price, rationale FROM legs "
            "WHERE worldview_id=? ORDER BY symbol", (row["id"],)).fetchall()
        return Worldview(
            id=row["id"], thesis=row["thesis"], reasoning=row["reasoning"],
            status=row["status"], created_at=row["created_at"],
            created_by=row["created_by"],
            legs=tuple(Leg(l["symbol"], l["quantity"], l["entry_price"], l["rationale"])
                       for l in legs))

    # ------------------------------------------------------------------ the netted book
    def net_positions(self, extra: list = None, exclude: int = None) -> dict:
        """{symbol: total quantity} across every ACTIVE worldview.

        `extra` adds legs not yet stored and `exclude` drops one worldview's, which together
        are how a change is costed against the book it would produce WITHOUT being written
        first. That matters: a preview that had to mutate the store to show you what it would
        do would leave rows behind whenever you declined it, and the next submission would
        trade them.

        Symbols that net to zero are dropped: a leg long in one view and short in another is
        genuinely no position, and sending it as `quantity=0` would only add a line to the
        book saying so."""
        totals = {}
        for view in self.active():
            if exclude is not None and view.id == exclude:
                continue
            for leg in view.legs:
                totals[leg.symbol] = totals.get(leg.symbol, 0.0) + leg.quantity
        for leg in (extra or []):
            symbol = leg.symbol.upper()
            totals[symbol] = totals.get(symbol, 0.0) + float(leg.quantity)
        return {s: q for s, q in totals.items() if abs(q) > 1e-9}

    def holders(self, symbol: str) -> list:
        """Which active worldviews hold a symbol, and how much. Answers the question you
        actually ask when looking at a position: *why* do I own this?"""
        symbol = symbol.upper()
        return [(v.id, leg.quantity) for v in self.active()
                for leg in v.legs if leg.symbol == symbol]
