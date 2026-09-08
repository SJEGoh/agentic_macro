"""
agentic_macro/prices.py — the one place a price comes from.

`expected_price` is the single most load-bearing number in a book. The executor values each
leg with it to apply the allocation cap, and measures fill slippage against it, so a stale,
missing or zero price mis-sizes the order AND the limit meant to contain it. Everything in
this module therefore fails closed: a symbol without a fresh, positive, finite price raises
rather than returning a default, and the caller has no way to ask for "whatever you've got".

Prices are fetched fresh at PROPOSAL time (to size the legs) and again at SUBMIT time (to
value the book the executor receives). Quantities are frozen in between — see `store.py` —
which is what makes an approval mean a fixed number of shares rather than a fixed dollar
amount that drifts while you read it.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from . import config

log = logging.getLogger("agentic-macro.prices")


class PriceUnavailable(RuntimeError):
    """No trustworthy price for a symbol. Never substitute a guess: a book priced with a
    placeholder is sized wrong in a way nothing downstream can detect."""


def _check(symbol: str, price) -> float:
    """The same fail-closed rule the executor's RiskManager and RemoteStrategy.validate()
    both apply, enforced at the source so a bad price never gets into a proposal."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        raise PriceUnavailable(f"{symbol}: price {price!r} is not a number")
    if not math.isfinite(value) or value <= 0:
        raise PriceUnavailable(
            f"{symbol}: price {value!r} is not positive and finite — the allocation cap "
            "and the slippage check are both computed from it")
    return value


def fetch(symbols) -> dict:
    """{symbol: last close} for every symbol, or raise.

    All-or-nothing on purpose. A partially priced book would be submitted as an
    authoritative whole book with some legs missing, and `/targets` reads a missing name as
    "close it" — so a half-priced fetch would quietly flatten the legs it could not price.
    """
    # Strip BEFORE testing for emptiness: a whitespace-only entry is truthy, and letting one
    # through would send yfinance an empty ticker and get back a confusing failure instead
    # of the "nothing to price" this is.
    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if not symbols:
        return {}

    import yfinance as yf                      # imported here so --dry-run needs no network

    # `period` is generous because a Monday morning fetch must still see Friday's close;
    # staleness is then judged from the bar's own date, not from how much we asked for.
    data = yf.download(symbols, period="10d", progress=False, auto_adjust=False,
                       group_by="column", threads=True)
    if data is None or data.empty:
        raise PriceUnavailable(f"no price data returned for {', '.join(symbols)}")

    closes = data["Close"]
    if len(symbols) == 1 and closes.ndim == 1:  # yfinance drops the column level for one name
        closes = closes.to_frame(symbols[0])

    cutoff = datetime.now(timezone.utc) - timedelta(days=config.MAX_PRICE_AGE_DAYS)
    out, problems = {}, []
    for symbol in symbols:
        if symbol not in closes.columns:
            problems.append(f"{symbol}: no column in the price response")
            continue
        series = closes[symbol].dropna()
        if series.empty:
            problems.append(f"{symbol}: no closing prices in the window")
            continue

        stamp = series.index[-1]
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        if stamp.to_pydatetime() < cutoff:
            problems.append(f"{symbol}: last close is {stamp.date()}, older than "
                            f"{config.MAX_PRICE_AGE_DAYS:g} days")
            continue
        try:
            out[symbol] = _check(symbol, series.iloc[-1])
        except PriceUnavailable as e:
            problems.append(str(e))

    if problems:
        raise PriceUnavailable("could not price this book:\n  " + "\n  ".join(problems))
    return out


def drift(proposed: dict, current: dict) -> dict:
    """{symbol: fractional change} for every symbol priced in both. Used to refuse an
    approval whose legs have moved out from under the notional you were shown."""
    return {s: (current[s] - p) / p
            for s, p in proposed.items() if s in current and p}
