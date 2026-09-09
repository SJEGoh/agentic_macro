"""
agentic_macro/prices.py — the one place a price or a volatility comes from.

`expected_price` is the single most load-bearing number in a book. The executor values each
leg with it to apply the allocation cap, and measures fill slippage against it, so a stale,
missing or zero price mis-sizes the order AND the limit meant to contain it. Everything in
this module therefore fails closed: a symbol without a fresh, positive, finite price raises
rather than returning a default, and the caller has no way to ask for "whatever you've got".

Two providers, and which one is not a preference
------------------------------------------------
Massive is primary for equities and ETFs. Futures fall through to yfinance because this
account is **not entitled to Massive's futures data** — `list_futures_products` answers
"You are not entitled to this data", so there is nothing to fall back FROM. yfinance also
covers an equity Massive fails on, which is the ordinary backup case.

A provider that answers with a bad number is worse than one that does not answer, so the
per-symbol result is validated before it counts as an answer; a symbol that fails validation
falls through to the backup exactly like a symbol that was missing.

Everything is keyed by the symbol the EXECUTOR uses ("ZN"), never the feed's ("ZN=F"). That
translation lives here and nowhere else.

Bars, not quotes
----------------
Both the price and the volatility come from the same daily bars, fetched once. Inverse-vol
sizing needs a return series, and pulling it separately would double the API calls and let
the price and the vol drift out of sync with each other.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os

from . import config, universe

log = logging.getLogger("agentic-macro.prices")

#: Trading days of history pulled. Enough for a 60-day vol with room for holidays.
HISTORY_DAYS = int(os.environ.get("AGENTIC_HISTORY_DAYS", "120"))
#: Lookback for realised volatility, in observations.
VOL_LOOKBACK = int(os.environ.get("AGENTIC_VOL_LOOKBACK", "60"))
#: Fewer returns than this and the vol estimate is noise pretending to be a risk number.
VOL_MIN_OBS = int(os.environ.get("AGENTIC_VOL_MIN_OBS", "30"))


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


# --------------------------------------------------------------------------- providers
def _massive_bars(feed_symbols: list, days: int) -> dict:
    """{feed symbol: [(date, close)]} from Massive. Never raises — a provider that cannot
    answer returns what it has, and the caller falls through for the rest."""
    out = {}
    try:
        from massive import RESTClient
    except ImportError:
        return out
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return out

    client = RESTClient(key)
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days * 1.6) + 10)   # calendar days for trading days
    for sym in feed_symbols:
        try:
            bars = [(dt.datetime.fromtimestamp(a.timestamp / 1000, dt.timezone.utc).date(),
                     a.close)
                    for a in client.list_aggs(sym, 1, "day", start.isoformat(),
                                              end.isoformat(), limit=50_000)]
            if bars:
                out[sym] = sorted(bars)
        except Exception as e:
            log.debug("massive has no bars for %s: %s", sym, e)
    return out


def _yfinance_bars(feed_symbols: list, days: int) -> dict:
    """{feed symbol: [(date, close)]} from yfinance. The backup, and the ONLY source for
    futures on this account."""
    out = {}
    if not feed_symbols:
        return out
    try:
        import yfinance as yf
    except ImportError:
        return out
    try:
        data = yf.download(feed_symbols, period=f"{int(days * 1.6) + 10}d", progress=False,
                           auto_adjust=False, group_by="column", threads=True)
    except Exception as e:
        log.warning("yfinance download failed: %s", e)
        return out
    if data is None or data.empty:
        return out

    closes = data["Close"]
    if len(feed_symbols) == 1 and closes.ndim == 1:      # yfinance drops the level for one
        closes = closes.to_frame(feed_symbols[0])
    for sym in feed_symbols:
        if sym not in closes.columns:
            continue
        series = closes[sym].dropna()
        if not series.empty:
            out[sym] = [(i.date() if hasattr(i, "date") else i, float(v))
                        for i, v in series.items()]
    return out


def bars(symbols, days: int = HISTORY_DAYS) -> dict:
    """{executor symbol: [(date, close)]} for every symbol, or raise.

    All-or-nothing on purpose. A partially priced book would be submitted as an
    authoritative whole book with some legs missing, and `/targets` reads a missing name as
    "close it" — so a half-priced fetch would quietly flatten the legs it could not price.
    """
    wanted = []
    for s in symbols:
        if s and s.strip():
            wanted.append(universe.resolve(s))
    if not wanted:
        return {}

    # Futures skip Massive entirely — this account has no entitlement, so asking is just
    # latency and a misleading log line.
    equities = [i for i in wanted if i.sec_type != "FUT"]
    got = _massive_bars([i.feed for i in equities], days) if equities else {}

    missing = [i for i in wanted if not got.get(i.feed)]
    if missing:
        got.update(_yfinance_bars([i.feed for i in missing], days))

    cutoff = dt.date.today() - dt.timedelta(days=config.MAX_PRICE_AGE_DAYS)
    out, problems = {}, []
    for inst in wanted:
        series = got.get(inst.feed)
        if not series:
            problems.append(f"{inst.symbol}: no price data from either provider")
            continue
        last_date, last_close = series[-1]
        if last_date < cutoff:
            problems.append(f"{inst.symbol}: last close is {last_date}, older than "
                            f"{config.MAX_PRICE_AGE_DAYS:g} days")
            continue
        try:
            _check(inst.symbol, last_close)
        except PriceUnavailable as e:
            problems.append(str(e))
            continue
        out[inst.symbol] = series

    if problems:
        raise PriceUnavailable("could not price this book:\n  " + "\n  ".join(problems))
    return out


def fetch(symbols, days: int = HISTORY_DAYS) -> dict:
    """{executor symbol: last close}."""
    return {s: series[-1][1] for s, series in bars(symbols, days).items()}


def realized_vol(symbols, lookback: int = VOL_LOOKBACK, series: dict = None) -> dict:
    """{executor symbol: annualised realised volatility} from daily log returns.

    This is the risk unit an inverse-vol structure is weighted by, so it fails closed like a
    price does: too few observations raises rather than returning a small number. A vol
    estimated from six days would quietly hand the largest position to whichever leg happened
    to be quiet that week — the exact opposite of what inverse-vol sizing is for.

    Pass `series` to reuse bars already fetched, so the vol and the price describe the same
    pull rather than two calls that can disagree."""
    series = series if series is not None else bars(symbols)
    out, problems = {}, []
    for symbol, rows in series.items():
        closes = [c for _, c in rows][-(lookback + 1):]
        rets = [math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i] > 0 and closes[i - 1] > 0]
        if len(rets) < VOL_MIN_OBS:
            problems.append(f"{symbol}: only {len(rets)} usable returns, need {VOL_MIN_OBS}")
            continue
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(var) * math.sqrt(252)
        if not math.isfinite(sigma) or sigma <= 0:
            problems.append(f"{symbol}: volatility came out {sigma!r}")
            continue
        out[symbol] = sigma
    if problems:
        raise PriceUnavailable("could not measure volatility:\n  " + "\n  ".join(problems))
    return out


def drift(proposed: dict, current: dict) -> dict:
    """{symbol: fractional change} for every symbol priced in both. Used to refuse an
    approval whose legs have moved out from under the notional you were shown."""
    return {s: (current[s] - p) / p
            for s, p in proposed.items() if s in current and p}
