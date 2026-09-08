"""
agentic_macro/universe.py — the closed list of instruments a worldview may express itself in.

This list is the reason a hallucinated ticker cannot reach the broker. The model is given
these symbols and nothing else, and `resolve()` rejects anything outside them BEFORE a
proposal is ever shown to you — so the failure mode is "the model named something not on
the list, and we said no", not "we sent an order for a ticker that does not exist".

Keep it to instruments that are liquid enough that a market order at the top of the book is
not itself the trade. Every entry is an ETF with continuous US listing, so `expected_price`
from a daily close is a fair estimate of the fill, which is the assumption the executor's
allocation cap and slippage alert both rest on.

The `note` is not documentation — it is the only thing the model knows about the instrument
beyond its ticker, so it should say what EXPOSURE the name gives, in the language a macro
view is written in ("long duration", "the dollar", "US banks"), not what the fund holds.

`duration` and `beta` are what make a risk-weighted structure possible
--------------------------------------------------------------------
A curve trade weighted by equal dollars is not a curve trade. SHY carries ~1.8 years of
duration and TLT ~16.5; long $1m of one against short $1m of the other is a large outright
short in duration that happens to have a steepener attached. To express the curve you
weight the legs so their DV01s match, which needs a duration number per instrument — hence
these fields, and hence `playbooks.py` being able to say `weighting="dv01_neutral"` and
have it mean something.

These numbers are APPROXIMATE and they drift: effective duration moves with yields and with
the fund's own roll, and equity betas are regime-dependent. They are here to get a hedge
ratio into the right neighbourhood — 2:1 instead of 9:1 — not to be precise. A structure
whose profitability depends on the third significant figure of a beta is not a structure
this sleeve should be putting on. Refresh them when they are visibly wrong; nothing breaks
quietly if they are a little stale, because the weighting they produce is shown to you in
the approval message before anything is sent.
"""
from __future__ import annotations

from typing import NamedTuple, Optional


class Instrument(NamedTuple):
    symbol: str
    bucket: str
    note: str
    #: Effective rate duration in years — the DV01 proxy. Set for anything whose dominant
    #: risk is interest rates; None where duration is not the right risk axis.
    duration: Optional[float] = None
    #: Beta to SPY, for equity-risk-neutral pairs. None where the name is not equity beta.
    beta: Optional[float] = None


#: The tradeable universe. Adding a name here is the ONLY way to make it tradeable.
UNIVERSE: tuple = (
    # ---- rates & duration ------------------------------------------------------------
    Instrument("SHY",  "rates",  "1-3y Treasuries — the front end; cash-like, low duration",
               duration=1.8),
    Instrument("IEF",  "rates",  "7-10y Treasuries — the belly of the curve",
               duration=7.3),
    Instrument("TLT",  "rates",  "20y+ Treasuries — long duration, the cleanest rates expression",
               duration=16.5),
    Instrument("TIP",  "rates",  "TIPS — real yields; long TIP vs long IEF is a breakeven-inflation view",
               duration=6.5),

    # ---- US equity beta --------------------------------------------------------------
    Instrument("SPY",  "equity", "S&P 500 — broad US large-cap beta", beta=1.00),
    Instrument("QQQ",  "equity", "Nasdaq 100 — US large-cap growth and tech; long-duration equity",
               beta=1.15),
    Instrument("IWM",  "equity", "Russell 2000 — US small caps; domestic, rate- and credit-sensitive",
               beta=1.15),
    Instrument("DIA",  "equity", "Dow 30 — US large-cap value tilt", beta=0.92),

    # ---- US sectors ------------------------------------------------------------------
    Instrument("XLE",  "sector", "US energy equities — oil and gas producers", beta=0.90),
    Instrument("XLF",  "sector", "US financials — banks, brokers, insurers; steepener-sensitive",
               beta=1.05),
    Instrument("XLK",  "sector", "US technology", beta=1.25),
    Instrument("XLV",  "sector", "US healthcare — defensive", beta=0.70),
    Instrument("XLI",  "sector", "US industrials — cyclical, capex- and PMI-sensitive", beta=1.05),
    Instrument("XLU",  "sector", "US utilities — defensive, bond-proxy, long-duration equity",
               beta=0.55),
    Instrument("XLP",  "sector", "US consumer staples — defensive", beta=0.55),
    Instrument("XLY",  "sector", "US consumer discretionary — cyclical, consumer-spending-sensitive",
               beta=1.20),
    Instrument("XLB",  "sector", "US materials — commodity- and China-cycle-sensitive", beta=1.05),
    Instrument("XLRE", "sector", "US real estate — REITs; highly rate-sensitive", beta=1.00),

    # ---- regions ---------------------------------------------------------------------
    Instrument("EFA",  "region", "developed markets ex-US — Europe, Australasia, Far East",
               beta=0.95),
    Instrument("EEM",  "region", "emerging markets broad — dollar- and China-sensitive", beta=1.00),
    Instrument("FXI",  "region", "China large-cap", beta=1.00),
    Instrument("EWJ",  "region", "Japan equities", beta=0.75),
    Instrument("EWG",  "region", "Germany equities — the European cyclical proxy", beta=1.10),
    Instrument("EWZ",  "region", "Brazil equities — commodity- and EM-risk-sensitive", beta=1.20),
    Instrument("INDA", "region", "India equities", beta=0.70),

    # ---- credit ----------------------------------------------------------------------
    Instrument("LQD",  "credit", "US investment-grade credit — IG spread PLUS ~8y of duration",
               duration=8.4),
    Instrument("HYG",  "credit", "US high yield — the cleanest listed credit-risk expression; "
                                 "little rate duration, lots of spread duration",
               duration=3.2, beta=0.45),
    Instrument("EMB",  "credit", "USD emerging-market sovereign debt", duration=7.0),

    # ---- commodities -----------------------------------------------------------------
    Instrument("GLD",  "commodity", "gold — real rates, debasement and haven demand"),
    Instrument("SLV",  "commodity", "silver — gold with an industrial beta"),
    Instrument("USO",  "commodity", "WTI crude oil (futures-backed; carries roll cost)"),
    Instrument("UNG",  "commodity", "US natural gas (futures-backed; heavy roll cost, very volatile)"),
    Instrument("DBA",  "commodity", "agricultural commodities basket"),
    Instrument("DBC",  "commodity", "broad commodity basket — energy-weighted"),

    # ---- FX --------------------------------------------------------------------------
    # Currency ETFs, not spot FX. The executor builds a STOCK contract for anything whose
    # sec_type is not "FUT" (CentralExecutor.place_order), so a CASH/IDEALPRO leg would be
    # sent as a stock named "EUR" rather than a currency pair — these proxies are the FX
    # exposure that actually reaches the broker correctly today. Each is quoted in dollars
    # and sized in shares like any other name here, so the allocation cap works unchanged.
    Instrument("UUP",  "fx",     "long US dollar vs a developed-market basket (DXY-like)"),
    Instrument("UDN",  "fx",     "short US dollar vs the same basket — the clean way to be "
                                 "dollar-bearish without picking a single counter-currency"),
    Instrument("FXE",  "fx",     "long euro vs the dollar"),
    Instrument("FXY",  "fx",     "long yen vs the dollar — carry-unwind and haven sensitive"),
    Instrument("FXB",  "fx",     "long sterling vs the dollar"),
    Instrument("FXF",  "fx",     "long Swiss franc vs the dollar — haven, SNB-policy sensitive"),
    Instrument("FXA",  "fx",     "long Australian dollar — a China-growth and commodity proxy"),
    Instrument("FXC",  "fx",     "long Canadian dollar — crude-sensitive"),
    Instrument("CEW",  "fx",     "long a basket of emerging-market currencies vs the dollar"),

    # ---- other -----------------------------------------------------------------------
    Instrument("BITO", "other",  "bitcoin futures — a liquidity and risk-appetite proxy"),
    Instrument("VXX",  "other",  "VIX short-term futures — LONG ONLY as a hedge; bleeds hard in "
                                 "calm markets and is never a buy-and-hold"),
)

BY_SYMBOL: dict = {i.symbol: i for i in UNIVERSE}

#: Names whose structure makes a resting long position decay regardless of the view being
#: right. The model is told this explicitly; it is the single most common way a correct
#: macro call still loses money in ETF form.
DECAY_PRONE: frozenset = frozenset({"VXX", "UNG", "USO"})

#: Names that trade, but thinly enough that a large market order moves the print. Sizing a
#: real position in these off a daily close overstates how good the fill will be, so the
#: model is told to keep them small or reach for a liquid substitute (UUP/UDN over the
#: single-currency funds, DBC over DBA). Not a ban — a thin name is often the only honest
#: way to express a specific view, and saying "small" is better than silently sizing big.
THIN: frozenset = frozenset({"UDN", "FXB", "FXF", "FXA", "FXC", "CEW", "DBA", "INDA", "EWG"})


class UnknownSymbol(ValueError):
    """A symbol that is not on the list. Never let one of these through — the whole point
    of a closed universe is that this is caught before a human is asked to approve it."""


def resolve(symbol: str) -> Instrument:
    """Look a symbol up, or raise. Case- and whitespace-insensitive, because the model
    writes `spy` as readily as `SPY` and that is not a reason to reject a good view."""
    key = (symbol or "").strip().upper()
    if key not in BY_SYMBOL:
        raise UnknownSymbol(
            f"{symbol!r} is not in the tradeable universe. "
            f"Add it to agentic_macro/universe.py if it belongs there.")
    return BY_SYMBOL[key]


def symbols() -> list:
    return [i.symbol for i in UNIVERSE]


def risk_unit(symbol: str, weighting: str) -> float:
    """The per-dollar risk of one instrument on the axis a structure is weighted along.

    This is the number that turns "long the front, short the long end" into a hedge ratio.
    Returns 1.0 for equal-notional weighting, the duration for a DV01-matched structure,
    and the beta for an equity-market-neutral one.

    Raises rather than defaulting to 1.0 when the axis is undefined for the instrument: a
    missing duration silently treated as 1.0 would size a steepener as if TLT and SHY were
    the same risk, which is the exact error this function exists to prevent."""
    instrument = resolve(symbol)
    if weighting in ("equal_notional", "directional"):
        return 1.0
    if weighting == "dv01_neutral":
        if not instrument.duration:
            raise ValueError(
                f"{instrument.symbol} has no duration, so it cannot carry a leg of a "
                f"DV01-weighted structure — use equal_notional, or add a duration to "
                f"universe.py if this name really is a rates instrument")
        return instrument.duration
    if weighting == "beta_neutral":
        if not instrument.beta:
            raise ValueError(
                f"{instrument.symbol} has no beta, so it cannot carry a leg of a "
                f"beta-weighted structure — use equal_notional instead")
        return instrument.beta
    raise ValueError(f"unknown weighting scheme {weighting!r}")


def catalogue() -> str:
    """The universe as the model sees it: grouped, one line each, tickers exact.

    Grouping is not cosmetic — it is what lets the model reach for a bucket it has not
    thought of ("this is really an FX view") instead of expressing everything in SPY."""
    lines, current = [], None
    for i in UNIVERSE:
        if i.bucket != current:
            current = i.bucket
            lines.append(f"\n[{current.upper()}]")
        facts = []
        if i.duration:
            facts.append(f"duration {i.duration:g}y")
        if i.beta:
            facts.append(f"beta {i.beta:g}")
        if i.symbol in DECAY_PRONE:
            facts.append("** structurally decays if held **")
        if i.symbol in THIN:
            facts.append("** thin — keep small **")
        tail = f"  ({'; '.join(facts)})" if facts else ""
        lines.append(f"  {i.symbol:<5} {i.note}{tail}")
    return "\n".join(lines).strip()
