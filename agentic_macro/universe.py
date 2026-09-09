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
    #: "STK" or "FUT". The executor dispatches on exactly this: place_order builds a futures
    #: contract for "FUT" and a STOCK contract for everything else.
    sec_type: str = "STK"
    #: Contract multiplier. One unit is worth price * multiplier — 1.0 for shares, 1,000 for
    #: a 10-year note future. Getting this wrong understates notional by the multiplier and
    #: sails straight past the allocation cap, which is why every FUT entry must carry one
    #: and `RemoteStrategy.validate()` refuses a futures leg without it.
    multiplier: float = 1.0
    #: IB routing destination.
    exchange: str = "SMART"
    #: Ticker the PRICE FEED knows this by, when it differs from the symbol the broker uses
    #: (yfinance calls the 10-year note future "ZN=F"; IB calls it "ZN").
    price_symbol: Optional[str] = None
    #: The ETF that gives the same exposure, used when one contract costs more than this
    #: leg's share of the budget. Substituting it keeps the underlying, the direction, the
    #: structure and the leg's place in the hedge ratio — only the fee and roll profile
    #: change. That is categorically different from DROPPING the leg, which would change the
    #: structure into a different trade.
    etf_equivalent: Optional[str] = None

    @property
    def feed(self) -> str:
        """The symbol to ask a data provider for."""
        return self.price_symbol or self.symbol

    def notional(self, price: float) -> float:
        """Dollar value of `quantity=1`. The multiplier is the whole point."""
        return float(price) * self.multiplier


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

    # ---- RATES FUTURES (CBOT) --------------------------------------------------------
    # The real instrument for a curve trade: an ETF proxy carries the fund's duration, a
    # note future carries the cheapest-to-deliver's. `duration` here is the CTD-based
    # effective duration of the FUTURES PRICE, so DV01-dollars stay price * multiplier *
    # duration and the existing hedge-ratio maths needs no special case.
    #
    # Note the size. One ZT is ~$205k of notional and one ZN ~$107k, so a DV01-matched
    # ZT/ZB steepener runs to seven figures. Below roughly $1m of sleeve these are simply
    # unaffordable and SHY/IEF/TLT remain the correct expression — the sizing code refuses
    # rather than quietly building a different trade.
    Instrument("ZT", "rates_fut", "2-year T-Note future — the front end, in the real instrument",
               duration=1.9, sec_type="FUT", multiplier=2000, exchange="CBOT", price_symbol="ZT=F", etf_equivalent="SHY"),
    Instrument("ZF", "rates_fut", "5-year T-Note future — the belly",
               duration=4.4, sec_type="FUT", multiplier=1000, exchange="CBOT", price_symbol="ZF=F", etf_equivalent="IEF"),
    Instrument("ZN", "rates_fut", "10-year T-Note future — the benchmark duration contract",
               duration=6.5, sec_type="FUT", multiplier=1000, exchange="CBOT", price_symbol="ZN=F", etf_equivalent="IEF"),
    Instrument("ZB", "rates_fut", "30-year T-Bond future — the long end",
               duration=15.5, sec_type="FUT", multiplier=1000, exchange="CBOT", price_symbol="ZB=F", etf_equivalent="TLT"),

    # ---- EQUITY INDEX MICRO FUTURES (CME) --------------------------------------------
    # Micros, not e-minis: one ES is ~$385k and would exceed the whole sleeve, while one
    # MES is ~$38k and is actually sizeable against it.
    #
    # Every micro below prices off its FULL-SIZE sibling (MES -> ES=F, MCL -> CL=F, ...).
    # That is not a substitution: a micro and its parent track the same underlying at the
    # same quoted price and differ only in contract size, which the multiplier already
    # carries. The parent's series is simply far better populated — yfinance has 167 daily
    # bars for CL=F and exactly ONE for MCL=F, and an inverse-vol weight computed from one
    # bar is not a risk number.
    Instrument("MES", "index_fut", "Micro E-mini S&P 500 — index beta with leverage and no borrow",
               beta=1.00, sec_type="FUT", multiplier=5, exchange="CME", price_symbol="ES=F", etf_equivalent="SPY"),
    Instrument("MNQ", "index_fut", "Micro E-mini Nasdaq 100 — growth beta",
               beta=1.15, sec_type="FUT", multiplier=2, exchange="CME", price_symbol="NQ=F", etf_equivalent="QQQ"),
    Instrument("M2K", "index_fut", "Micro E-mini Russell 2000 — small-cap beta",
               beta=1.15, sec_type="FUT", multiplier=5, exchange="CME", price_symbol="RTY=F", etf_equivalent="IWM"),

    # ---- COMMODITY MICRO FUTURES -----------------------------------------------------
    # These hold the commodity directly rather than a futures-backed ETF, so the roll is
    # yours to manage and is not silently charged to you as decay (cf. USO, UNG).
    Instrument("MCL", "commodity_fut", "Micro WTI crude future — oil without USO's roll drag",
               sec_type="FUT", multiplier=100, exchange="NYMEX", price_symbol="CL=F", etf_equivalent="USO"),
    Instrument("MGC", "commodity_fut", "Micro gold future — gold without GLD's fee",
               sec_type="FUT", multiplier=10, exchange="COMEX", price_symbol="GC=F", etf_equivalent="GLD"),
    Instrument("SIL", "commodity_fut", "Micro silver future",
               sec_type="FUT", multiplier=1000, exchange="COMEX", price_symbol="SI=F", etf_equivalent="SLV"),

    # ---- FX FUTURES (CME) ------------------------------------------------------------
    # Actual currency exposure rather than an ETF wrapper — and the executor CAN route
    # these, unlike spot FX, which its place_order would build as a stock.
    Instrument("6E", "fx_fut", "EUR/USD future — 125,000 euro per contract",
               sec_type="FUT", multiplier=125000, exchange="CME", price_symbol="6E=F", etf_equivalent="FXE"),
    Instrument("6J", "fx_fut", "JPY/USD future — 12.5m yen per contract; the funding currency",
               sec_type="FUT", multiplier=12500000, exchange="CME", price_symbol="6J=F", etf_equivalent="FXY"),
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


def with_fallbacks(symbols) -> list:
    """`symbols` plus the ETF equivalent of every futures name among them.

    The caller prices and measures volatility for this wider set, because a substitution
    decided during sizing must not then need a network call to find out what the replacement
    costs — a price fetched mid-sizing is a price fetched at a different moment from the
    rest of the book."""
    out = []
    for s in symbols:
        i = resolve(s)
        out.append(i.symbol)
        if i.etf_equivalent:
            out.append(i.etf_equivalent)
    return sorted(set(out))


def qty(quantity: float) -> str:
    """Render a size without hiding a fraction. `f"{555.5556:,.0f}"` prints 556, which is
    both wrong and invisible — the whole point of fractional sizing is that the number is
    not a whole one."""
    q = float(quantity)
    if q == int(q):
        return f"{int(q):,}"
    return f"{q:,.4f}".rstrip("0").rstrip(".")


def symbols() -> list:
    return [i.symbol for i in UNIVERSE]


def risk_unit(symbol: str, weighting: str, vols: dict = None) -> float:
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
    if weighting == "inverse_vol":
        sigma = (vols or {}).get(instrument.symbol)
        if not sigma or not (sigma > 0):
            raise ValueError(
                f"{instrument.symbol}: no realised volatility, so it cannot carry a leg of "
                f"an inverse-vol structure. Sizing it as if vol were 1.0 would give the "
                f"most volatile name the largest position, which is the opposite of the "
                f"intent")
        return float(sigma)
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


def catalogue(prices: dict = None) -> str:
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
        if i.sec_type == "FUT":
            px = (prices or {}).get(i.symbol)
            # The single most useful fact about a futures leg is what ONE contract costs.
            # Without it the model proposes ZT into a $250k sleeve, where one contract is
            # 82% of the whole thing.
            facts.append(f"1 contract = ${i.notional(px):,.0f}" if px
                         else f"multiplier {i.multiplier:,.0f}")
        if i.symbol in DECAY_PRONE:
            facts.append("** structurally decays if held **")
        if i.symbol in THIN:
            facts.append("** thin — keep small **")
        tail = f"  ({'; '.join(facts)})" if facts else ""
        lines.append(f"  {i.symbol:<5} {i.note}{tail}")
    return "\n".join(lines).strip()
