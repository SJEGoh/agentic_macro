"""
agentic_macro/strategy.py — the sleeve's book, and how it reaches the executor.

The book submitted to `/targets` is the NET of every ACTIVE worldview's legs, summed per
symbol and priced fresh. That is the whole contract between this repo and the executor, and
it has three consequences worth stating plainly, because each one is a property you get for
free only as long as the netting stays the single source of truth:

  * `/targets` is AUTHORITATIVE — every name omitted from the book is closed. That is
    exactly what you want here: a worldview you close should have its legs disappear, and
    they do, without anyone having to remember which shares belonged to it. It is also why
    the book must always be built from ALL active worldviews and never from just the one
    that changed. Submitting a single view's legs would flatten every other view's.

  * submission is IDEMPOTENT. The book is a pure function of the stored rows and the current
    price, so re-sending it cannot double a position. `/sync` leans on this to self-heal
    drift, and a retry after a timeout is safe.

  * a worldview is written to the store BEFORE the book is submitted. If the submission
    then fails, the sleeve holds a view whose legs are not at the broker — visible, and
    fixed by `/sync`. The other order would be worse: positions at the broker that no
    worldview claims, which the next submission would close without explanation.
"""
from __future__ import annotations

import logging

from . import config, prices, universe
from .executor.remote_strategy import RemoteStrategy, StrategyError
from .store import Store

log = logging.getLogger("agentic-macro.strategy")


class MacroSleeve(RemoteStrategy):
    """The discretionary macro sleeve: whatever the active worldviews add up to."""

    strategy_id = config.STRATEGY_ID
    #: The book can hold futures, but it is usually mixed, and the executor rejects a pooled
    #: book containing equities while the equity market is shut. Skipping cleanly here turns
    #: that into an exit 0 "not now" rather than a 409 someone has to go and read.
    require_market_open = True

    def __init__(self, store: Store = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or Store()
        self.last_prices: dict = {}

    def generate_book(self, capital: float) -> list:
        """The netted book across every active worldview, priced now.

        Prices are fetched at submit time rather than reused from approval: `expected_price`
        is what the executor values the book with, so it has to describe the market the
        order is going into, not the one the view was written in. Quantities are untouched —
        those are what you approved."""
        targets = self.store.net_positions()
        if not targets:
            log.info("no active worldviews — submitting an empty book (closes everything)")
            return []

        self.last_prices = prices.fetch(targets.keys())
        book = []
        for symbol, quantity in sorted(targets.items()):
            inst = universe.resolve(symbol)
            # A futures leg MUST carry its multiplier: without it the executor's netting
            # values the contract at its bare price, understating notional by up to 1,000x
            # and letting the order through a cap it should have breached.
            book.append(self.intent(
                symbol, quantity, self.last_prices[symbol],
                sec_type=inst.sec_type, exchange=inst.exchange,
                asset_class="future" if inst.sec_type == "FUT" else "equity",
                multiplier=inst.multiplier if inst.sec_type == "FUT" else None))

        gross = sum(abs(i["target_quantity"]) * i["expected_price"]
                    * (i["instrument"].get("multiplier") or 1.0) for i in book)
        cap = capital * config.MAX_GROSS_WEIGHT
        if gross > cap:
            # Local check before the executor's. Its cap is the one that protects the
            # account; this one exists so the refusal names the worldviews responsible
            # instead of arriving as an opaque server-side rejection.
            raise StrategyError(
                f"the netted book is ${gross:,.0f} gross against a ${cap:,.0f} limit "
                f"({config.MAX_GROSS_WEIGHT:.0%} of ${capital:,.0f}). Close a worldview "
                f"before adding another — /views shows what is on.")

        log.info("%d active worldviews -> %d names, $%s gross",
                 len(self.store.active()), len(book), f"{gross:,.0f}")
        return book

    def describe(self, book: list) -> str:
        views = self.store.active()
        if not book:
            return f"{self.strategy_id}: flat — no active worldviews"
        return (f"{self.strategy_id}: {len(views)} worldview(s) netting to {len(book)} "
                f"names — " + ", ".join(
                    f"{i['instrument']['symbol']}:{i['target_quantity']:g}" for i in book))

    def journal_detail(self, book: list) -> str:
        """The theses behind the book. This is the entry you will actually want in two
        months, when the position is still on and the reason is not obvious."""
        lines = []
        for view in self.store.active():
            legs = ", ".join(f"{l.symbol} {l.quantity:+g}" for l in view.legs)
            lines.append(f"#{view.id} [{view.created_at[:10]}] {view.thesis}\n    {legs}")
        return "\n".join(lines)


def submit(store: Store = None, dry_run: bool = False) -> int:
    """Push the current netted book. Returns the RemoteStrategy exit code — 0 submitted or
    deliberately skipped, 1 refused, 2 unreachable (the orders did NOT go in)."""
    return MacroSleeve.cli(argv=["--dry-run"] if dry_run else [], store=store)
