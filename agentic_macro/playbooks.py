"""
agentic_macro/playbooks.py — the named structures a macro view gets expressed in.

A worldview like "the Fed starts cutting into a soft landing" does not map to an instrument.
It maps to a STRUCTURE — a bull steepener — and the structure maps to instruments, a
direction per leg, and a weighting rule. This module is that middle layer, written down.

Why a module and not a vector database
--------------------------------------
The obvious design is to embed a library of strategies and retrieve the relevant ones. At
this size that is strictly worse. The whole library below is a few thousand tokens: it fits
in the prompt, it caches, and the model sees ALL of it on every call. Retrieval could only
subtract — its characteristic failure is returning four plausible neighbours and not the
right one, and a missing playbook does not announce itself, it just produces a vaguer trade
that still looks reasonable on approval.

The seam is here for when that stops being true. `select()` is the only thing the proposer
calls; it returns playbooks ranked for a thesis and currently returns all of them. Swapping
in an embedding index means reimplementing that one function, and the ranking it already
does (`score()`) is the thing you would measure a retriever against. Reach for the index
when the library outgrows the prompt — several hundred entries, or entries long enough to
be documents rather than recipes — not before.

On weighting
------------
`weighting` is the field that makes these real rather than decorative. A steepener is not
"long SHY, short TLT" — it is long SHY against short TLT *sized so the two legs have the
same DV01*. Equal dollars would make it a large outright duration short with a steepener
attached, which is a different trade with a different P&L and a different way of being
wrong. `universe.risk_unit()` supplies the per-instrument number, and refuses rather than
defaulting when an instrument has no risk on that axis.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class PlaybookLeg(NamedTuple):
    role: str            # what this leg is FOR, in the language of the structure
    symbols: tuple       # acceptable instruments, best first
    direction: str       # "long" | "short"


class Playbook(NamedTuple):
    name: str
    aliases: tuple
    summary: str
    legs: tuple
    weighting: str       # dv01_neutral | beta_neutral | equal_notional | directional
    works_when: str
    fails_when: str

    def render(self) -> str:
        legs = "; ".join(f"{l.direction} {'/'.join(l.symbols)} ({l.role})" for l in self.legs)
        return (f"{self.name} [{self.weighting}]\n"
                f"    {self.summary}\n"
                f"    legs:   {legs}\n"
                f"    works:  {self.works_when}\n"
                f"    fails:  {self.fails_when}")


PLAYBOOKS: tuple = (
    # ---------------------------------------------------------------- curve
    Playbook(
        name="steepener",
        aliases=("bull steepener", "bear steepener", "curve steepener", "steepening",
                 "2s10s", "5s30s", "curve widens", "disinversion", "un-invert"),
        summary="Long the front end against short the long end, DV01-matched, so the P&L is "
                "the SLOPE and not the level of rates. A BULL steepener (front rallies "
                "hardest, driven by cuts) and a BEAR steepener (long end sells off hardest, "
                "driven by term premium, inflation or supply) are the same structure — the "
                "difference is the driver. If you also want the level, tilt the front leg "
                "heavier rather than abandoning the hedge ratio.",
        legs=(PlaybookLeg("front end, rallies on cuts", ("SHY",), "long"),
              PlaybookLeg("long end, carries the term premium", ("TLT",), "short")),
        weighting="dv01_neutral",
        works_when="an easing cycle begins, or term premium rebuilds on fiscal supply",
        fails_when="the curve moves in parallel — a DV01-neutral steepener makes nothing; "
                   "and the short TLT leg pays away carry while you wait",
    ),
    Playbook(
        name="flattener",
        aliases=("bull flattener", "bear flattener", "curve flattener", "flattening",
                 "curve inverts", "inversion"),
        summary="The mirror of the steepener: long the long end against short the front, "
                "DV01-matched. BULL flattener = the long end rallies on growth fear or a "
                "flight to quality; BEAR flattener = the front sells off because the central "
                "bank is hiking into it.",
        legs=(PlaybookLeg("long end", ("TLT",), "long"),
              PlaybookLeg("front end", ("SHY",), "short")),
        weighting="dv01_neutral",
        works_when="hiking into a slowdown, or a growth scare that bids duration",
        fails_when="the front end reprices to cuts faster than the long end rallies",
    ),
    Playbook(
        name="long_duration",
        aliases=("long duration", "rates rally", "yields fall", "buy bonds", "receive rates",
                 "flight to quality", "bond bull"),
        summary="Outright long duration. The direct expression of 'yields fall'. TLT for the "
                "most convexity per dollar, IEF for the belly when the view is about cuts "
                "rather than the whole curve.",
        legs=(PlaybookLeg("duration", ("TLT", "IEF"), "long"),),
        weighting="directional",
        works_when="growth disappoints, inflation falls, or the bid for havens returns",
        fails_when="inflation resurprises, or supply/term premium pushes long yields up "
                   "even as the front end is cut",
    ),
    Playbook(
        name="short_duration",
        aliases=("short duration", "yields rise", "rates selloff", "short bonds",
                 "pay rates", "bond bear", "higher for longer"),
        summary="Outright short duration — 'yields rise'. Note the carry: shorting TLT pays "
                "the coupon away, so this needs a view on timing, not only direction.",
        legs=(PlaybookLeg("duration", ("TLT", "IEF"), "short"),),
        weighting="directional",
        works_when="inflation resurprises, term premium rebuilds, or supply overwhelms demand",
        fails_when="a growth scare bids duration regardless of the inflation picture",
    ),
    Playbook(
        name="curve_butterfly",
        aliases=("butterfly", "fly", "belly", "curvature", "2s5s10s", "5s10s30s",
                 "2s10s30s", "barbell", "bullet", "barbell vs bullet", "buy the belly",
                 "sell the belly", "hump"),
        summary="Three legs: the belly against both wings, DV01-weighted so the structure is "
                "neutral to BOTH the level and the slope of the curve, leaving only "
                "CURVATURE. Long the belly (long IEF, short SHY and TLT) says the middle of "
                "the curve outperforms both ends — the classic expression of a market "
                "repricing the pace of a cycle without changing where it ends up. Reverse "
                "every leg to sell the fly. Weight the two wings equally: that is what makes "
                "it a 50/50 fly and keeps a slope bet out of it.",
        legs=(PlaybookLeg("the belly", ("IEF",), "long"),
              PlaybookLeg("front wing", ("SHY",), "short"),
              PlaybookLeg("back wing", ("TLT",), "short")),
        weighting="dv01_neutral",
        works_when="the market reprices the PATH of policy — the timing or pace of a cycle — "
                   "without changing the terminal rate or the level of long yields",
        fails_when="the curve moves in parallel or simply steepens; a fly is the lowest-beta "
                   "of the curve trades, so it needs size to matter and dies of carry and "
                   "spread if held through nothing happening",
    ),
    Playbook(
        name="credit_butterfly",
        aliases=("credit fly", "quality butterfly", "credit curvature",
                 "quality barbell", "middle of the credit stack"),
        summary="The credit-quality version of a fly: the middle of the quality stack "
                "against both ends. Long IG against short Treasuries and short high yield "
                "says the middle of the stack outperforms — spreads compress at the top but "
                "the tail does not rally. Rarer than the rates fly and worth naming only "
                "when the view really is about the SHAPE of the quality curve.",
        legs=(PlaybookLeg("the belly of the quality stack", ("LQD",), "long"),
              PlaybookLeg("risk-free wing", ("IEF",), "short"),
              PlaybookLeg("distressed wing", ("HYG",), "short")),
        weighting="inverse_vol",
        works_when="quality is bid but the tail stays cheap — a late-cycle discrimination trade",
        fails_when="a broad risk-on rally lifts high yield hardest, or a default cycle takes "
                   "IG down with everything else",
    ),
    Playbook(
        name="breakeven_widener",
        aliases=("breakeven", "inflation expectations", "breakevens widen", "TIPS",
                 "real yields", "inflation protection"),
        summary="Long TIPS against short nominals of similar duration: isolates BREAKEVEN "
                "inflation and strips out the level of real rates. Long TIP outright is a "
                "view on inflation AND on duration, which is usually not what is meant.",
        legs=(PlaybookLeg("inflation-linked", ("TIP",), "long"),
              PlaybookLeg("nominal, same part of the curve", ("IEF",), "short")),
        weighting="dv01_neutral",
        works_when="inflation expectations rise — energy shocks, tariffs, fiscal expansion",
        fails_when="a demand shock takes breakevens down with growth",
    ),

    # ---------------------------------------------------------------- credit
    Playbook(
        name="credit_compression",
        aliases=("credit compression", "spreads tighten", "risk on credit", "high yield rally",
                 "carry trade", "reach for yield"),
        summary="Long high yield against short investment grade. HYG carries the spread risk "
                "and little duration; LQD is mostly duration. Pairing them isolates the "
                "CREDIT cycle instead of accidentally trading rates.",
        legs=(PlaybookLeg("spread risk", ("HYG",), "long"),
              PlaybookLeg("duration-heavy IG", ("LQD",), "short")),
        weighting="inverse_vol",
        works_when="growth holds up, defaults stay low, and the hunt for yield is on",
        fails_when="a genuine default cycle starts — HY gaps and does not trade on the way down",
    ),
    Playbook(
        name="credit_decompression",
        aliases=("credit decompression", "spreads widen", "credit stress", "default cycle",
                 "risk off credit", "high yield selloff"),
        summary="Short high yield against long investment grade or Treasuries. The cleanest "
                "listed expression of a credit cycle turning, and it usually leads equities.",
        legs=(PlaybookLeg("spread risk", ("HYG",), "short"),
              PlaybookLeg("quality", ("LQD", "IEF"), "long")),
        weighting="inverse_vol",
        works_when="funding tightens, defaults rise, or a recession is being priced",
        fails_when="central banks backstop credit — the reversal is violent",
    ),

    # ---------------------------------------------------------------- equity structure
    Playbook(
        name="defensive_rotation",
        aliases=("risk off", "defensive", "recession", "hard landing", "slowdown",
                 "growth scare", "quality rotation"),
        summary="Long defensives against short cyclicals, beta-matched so the pair expresses "
                "ROTATION rather than a short of the index. Beta-matching matters: staples "
                "run ~0.55 beta and discretionary ~1.20, so equal dollars is a large net "
                "short of the market wearing a rotation costume.",
        legs=(PlaybookLeg("defensives", ("XLP", "XLU", "XLV"), "long"),
              PlaybookLeg("cyclicals", ("XLY", "XLI"), "short")),
        weighting="beta_neutral",
        works_when="growth decelerates but the index does not crash",
        fails_when="a melt-up — defensives underperform badly in a liquidity-driven rally",
    ),
    Playbook(
        name="cyclical_rotation",
        aliases=("risk on", "cyclicals", "reopening", "growth accelerating", "PMI rising",
                 "recovery", "soft landing"),
        summary="The mirror: long cyclicals against short defensives, beta-matched. Expresses "
                "an improving growth impulse without paying for index beta.",
        legs=(PlaybookLeg("cyclicals", ("XLI", "XLB", "XLY"), "long"),
              PlaybookLeg("defensives", ("XLP", "XLU"), "short")),
        weighting="beta_neutral",
        works_when="PMIs turn up, capex recovers, the consumer holds",
        fails_when="the acceleration is inflationary and the central bank leans against it",
    ),
    Playbook(
        name="banks_on_the_curve",
        aliases=("banks", "financials", "steepener equity", "net interest margin",
                 "bank profitability", "XLF"),
        summary="Long financials against short the index, beta-matched. Banks earn the spread "
                "between funding and lending, so this is the EQUITY expression of a steeper "
                "curve — and it often works when the rates version is being paid negative carry.",
        legs=(PlaybookLeg("banks", ("XLF",), "long"),
              PlaybookLeg("index beta", ("SPY",), "short")),
        weighting="beta_neutral",
        works_when="the curve steepens with the economy intact",
        fails_when="the steepening is a credit event — banks lead the way down",
    ),
    Playbook(
        name="growth_vs_value",
        aliases=("growth over value", "value over growth", "tech leadership", "long duration equity",
                 "rotation out of tech", "QQQ vs"),
        summary="Long growth against short value (or reversed), beta-matched. Growth is "
                "long-duration equity: it re-rates on falling real yields and de-rates on "
                "rising ones. Treat this as a duration trade wearing an equity ticker.",
        legs=(PlaybookLeg("growth", ("QQQ",), "long"),
              PlaybookLeg("value", ("DIA", "IWM"), "short")),
        weighting="beta_neutral",
        works_when="real yields fall and scarce growth is bid",
        fails_when="real yields rise, or the leadership broadens out",
    ),
    Playbook(
        name="small_vs_large",
        aliases=("small caps", "russell", "small vs large", "domestic economy",
                 "broadening out", "IWM"),
        summary="Long small caps against short large, beta-matched. Small caps are the "
                "domestic, floating-rate-indebted, credit-sensitive part of the market — this "
                "is really a bet on easier funding and a resilient US consumer.",
        legs=(PlaybookLeg("small", ("IWM",), "long"),
              PlaybookLeg("large", ("SPY",), "short")),
        weighting="beta_neutral",
        works_when="cuts arrive with growth intact and funding costs fall",
        fails_when="a slowdown — small caps carry the credit risk first",
    ),
    Playbook(
        name="rate_sensitive_short",
        aliases=("bond proxies", "utilities short", "REITs", "rates hurt equities",
                 "XLRE", "duration equity short"),
        summary="Short the bond-proxy sectors against long the index, beta-matched. Utilities "
                "and REITs are the equity market's longest-duration assets and de-rate first "
                "when the discount rate rises.",
        legs=(PlaybookLeg("bond proxies", ("XLU", "XLRE"), "short"),
              PlaybookLeg("index beta", ("SPY",), "long")),
        weighting="beta_neutral",
        works_when="real yields rise while growth holds",
        fails_when="a growth scare — these are also defensives and get bid",
    ),

    # ---------------------------------------------------------------- regional
    Playbook(
        name="exus_outperformance",
        aliases=("europe over us", "international", "ex-us", "rest of world", "EFA",
                 "us underperforms", "valuation gap"),
        summary="Long developed ex-US against short the US, beta-matched. Usually a dollar "
                "trade in disguise — EFA is unhedged, so a falling dollar is a large part of "
                "the return. Say which of the two you actually mean.",
        legs=(PlaybookLeg("ex-US developed", ("EFA", "EWG", "EWJ"), "long"),
              PlaybookLeg("US", ("SPY",), "short")),
        weighting="beta_neutral",
        works_when="the dollar falls and the US growth premium narrows",
        fails_when="US exceptionalism reasserts, or a risk-off bids the dollar",
    ),
    Playbook(
        name="em_outperformance",
        aliases=("emerging markets", "EM rally", "EEM", "em over dm", "china recovery",
                 "em outperforms"),
        summary="Long EM against short the US, beta-matched — or expressed in EM currencies "
                "(CEW) when the view is really about the dollar. EM is a levered dollar and "
                "China-impulse trade more than it is an equity view.",
        legs=(PlaybookLeg("EM", ("EEM", "FXI"), "long"),
              PlaybookLeg("US", ("SPY",), "short")),
        weighting="beta_neutral",
        works_when="the dollar weakens, China stimulates, and global trade recovers",
        fails_when="the dollar rallies — EM underperforms on both the equity and the FX leg",
    ),
    Playbook(
        name="china_impulse",
        aliases=("china stimulus", "china reopening", "china slowdown", "FXI",
                 "commodity demand", "property"),
        summary="China's credit impulse transmits to commodities and commodity exporters "
                "before it shows in Chinese equities. Expressing it through XLB/DBC/EWZ/FXA "
                "is often cleaner than through FXI, which carries policy and governance risk "
                "the view is not about.",
        legs=(PlaybookLeg("china equity", ("FXI",), "long"),
              PlaybookLeg("the commodity transmission", ("XLB", "DBC", "EWZ"), "long")),
        weighting="inverse_vol",
        works_when="credit is eased and property stabilises",
        fails_when="stimulus is announced but not transmitted — a decade of false starts",
    ),

    # ---------------------------------------------------------------- FX
    Playbook(
        name="dollar_strength",
        aliases=("strong dollar", "dollar rally", "DXY up", "king dollar", "UUP",
                 "dollar smile", "usd higher"),
        summary="Long the dollar basket. The dollar smiles: it rallies both when the US "
                "outgrows everyone and when the world is in crisis, and falls in the boring "
                "middle. Say which limb of the smile you are on — they need different hedges.",
        legs=(PlaybookLeg("the dollar", ("UUP",), "long"),),
        weighting="directional",
        works_when="US growth or rate differentials widen, or a global risk-off hits",
        fails_when="the Fed cuts into a soft landing and the rest of the world catches up",
    ),
    Playbook(
        name="dollar_weakness",
        aliases=("weak dollar", "dollar decline", "DXY down", "debasement", "UDN",
                 "dollar bear", "usd lower"),
        summary="Short the dollar via UDN, or express it in what a weak dollar lifts — gold, "
                "EM, commodities. UDN is the pure version; the others pay you twice when "
                "right and hurt twice when wrong.",
        legs=(PlaybookLeg("short the dollar", ("UDN",), "long"),
              PlaybookLeg("what it lifts", ("GLD", "EEM", "DBC"), "long")),
        weighting="inverse_vol",
        works_when="the Fed cuts faster than peers, or the twin deficits get repriced",
        fails_when="a risk-off — the dollar is still the funding currency of last resort",
    ),
    Playbook(
        name="carry_unwind",
        aliases=("carry unwind", "yen strength", "risk off fx", "deleveraging",
                 "yen carry", "FXY", "volatility spike"),
        summary="Long yen (and optionally long vol) against short high-carry risk assets. The "
                "yen is the world's funding currency, so it spikes precisely when levered "
                "positions are being closed — the classic non-linear risk-off hedge.",
        legs=(PlaybookLeg("funding currency", ("FXY",), "long"),
              PlaybookLeg("carry assets", ("EEM", "HYG"), "short")),
        weighting="inverse_vol",
        works_when="a volatility shock forces deleveraging",
        fails_when="calm persists — you pay the carry you are shorting, every day",
    ),

    # ---------------------------------------------------------------- inflation & commodities
    Playbook(
        name="reflation",
        aliases=("reflation", "inflation rising", "commodity boom", "nominal growth",
                 "fiscal expansion", "inflation trade"),
        summary="Long real assets against short nominal bonds. The canonical inflation trade: "
                "commodities and energy equities go up, duration goes down, and the two legs "
                "reinforce rather than hedge each other.",
        legs=(PlaybookLeg("real assets", ("DBC", "XLE", "XLB"), "long"),
              PlaybookLeg("nominal duration", ("TLT",), "short")),
        weighting="inverse_vol",
        works_when="demand runs hot, supply is tight, or fiscal policy is loose",
        fails_when="the inflation is supply-driven and kills demand — see stagflation",
    ),
    Playbook(
        name="stagflation",
        aliases=("stagflation", "supply shock", "oil shock", "tariffs", "cost push",
                 "inflation without growth", "70s"),
        summary="The hard one: inflation up, growth down, so bonds and equities fall TOGETHER "
                "and the 60/40 hedge stops working. Long the shock (energy, commodities, "
                "gold), short what it taxes (the consumer, duration).",
        legs=(PlaybookLeg("the shock", ("DBC", "XLE", "GLD"), "long"),
              PlaybookLeg("what it taxes", ("XLY", "TLT"), "short")),
        weighting="inverse_vol",
        works_when="a genuine supply constraint — energy, tariffs, war",
        fails_when="the shock passes through quickly and demand destruction wins",
    ),
    Playbook(
        name="gold_debasement",
        aliases=("gold", "debasement", "real rates fall", "central bank buying", "GLD",
                 "store of value", "fiscal dominance"),
        summary="Long gold. Gold is a short position in real yields plus a call on monetary "
                "credibility. If the view is about REAL rates specifically, TIP is the more "
                "direct instrument and gold is the levered, noisier version of it.",
        legs=(PlaybookLeg("gold", ("GLD", "SLV"), "long"),),
        weighting="directional",
        works_when="real yields fall, deficits widen, or reserve managers diversify",
        fails_when="real yields rise sharply — gold has no cash flow to defend it",
    ),
    Playbook(
        name="energy_supply_shock",
        aliases=("oil spike", "energy crisis", "OPEC", "supply disruption", "crude",
                 "geopolitical risk", "war premium"),
        summary="Long energy. Prefer XLE over USO for anything held more than a few weeks: "
                "USO holds futures and pays the roll, so in contango a correct oil view still "
                "loses money. XLE owns the cash flows instead.",
        legs=(PlaybookLeg("energy equities", ("XLE",), "long"),
              PlaybookLeg("crude, only with a near-term catalyst", ("USO",), "long")),
        weighting="inverse_vol",
        works_when="supply is disrupted with demand intact",
        fails_when="OPEC spare capacity absorbs it, or the price kills demand",
    ),

    # ---------------------------------------------------------------- regime
    Playbook(
        name="hard_landing",
        aliases=("recession", "hard landing", "credit crunch", "unemployment rising",
                 "growth collapse", "crisis"),
        summary="The full risk-off basket: long duration, long defensives, short credit and "
                "small caps. Note these legs are all the same bet — this is a concentrated "
                "position, not a diversified one, and it should be sized as one.",
        legs=(PlaybookLeg("duration", ("TLT",), "long"),
              PlaybookLeg("defensives", ("XLP", "XLV"), "long"),
              PlaybookLeg("credit and cyclical risk", ("HYG", "IWM"), "short")),
        weighting="inverse_vol",
        works_when="labour cracks and credit tightens together",
        fails_when="the landing is soft — every leg loses at once",
    ),
    Playbook(
        name="soft_landing",
        aliases=("soft landing", "goldilocks", "immaculate disinflation", "no landing",
                 "cuts without recession"),
        summary="Inflation falls without growth breaking: cuts arrive into an intact economy. "
                "Long the credit- and rate-sensitive laggards, fund it by shorting the haven "
                "premium that is no longer needed.",
        legs=(PlaybookLeg("rate-sensitive laggards", ("IWM", "XLY", "HYG"), "long"),
              PlaybookLeg("the haven premium", ("GLD", "XLP"), "short")),
        weighting="inverse_vol",
        works_when="disinflation continues and the labour market cools without cracking",
        fails_when="either half breaks — inflation resurprises, or growth rolls over",
    ),
    Playbook(
        name="tail_hedge",
        aliases=("hedge", "tail risk", "protection", "insurance", "VXX", "crash protection",
                 "volatility"),
        summary="Long volatility as insurance. VXX holds front VIX futures, which are usually "
                "in contango, so it bleeds continuously — this is a PREMIUM you pay, never a "
                "position you hold. Size it as an insurance cost with an explicit expiry in "
                "mind, and never as an expression of a view you are patient about.",
        legs=(PlaybookLeg("volatility", ("VXX",), "long"),),
        weighting="directional",
        works_when="a discrete, dated catalyst could gap the market",
        fails_when="nothing happens — which is most of the time, and it costs you every day",
    ),
)

BY_NAME: dict = {p.name: p for p in PLAYBOOKS}

#: How a structure's legs are balanced against each other.
#:
#:   dv01_neutral   duration-matched — curve trades, where the axis is the slope
#:   beta_neutral   beta-matched — equity pairs, where the axis is the sector not the market
#:   inverse_vol    risk-matched on realised volatility. The DEFAULT for multi-asset
#:                  baskets: DBC, GLD and TLT sized by equal dollars is really a bet on
#:                  whichever leg happens to be most volatile, because equal dollars in a
#:                  50%-vol commodity and a 10%-vol bond is five times the risk in one leg
#:   equal_notional dollar-matched. Available, but rarely what is meant
#:   directional    a single-leg outright, where there is nothing to balance against
WEIGHTINGS: frozenset = frozenset(
    {"dv01_neutral", "beta_neutral", "inverse_vol", "equal_notional", "directional"})

#: Weightings that deliberately cancel the dominant factor. A DV01-neutral steepener is
#: built to be neutral to the LEVEL of rates, and a beta-neutral pair to the level of the
#: market — which is exactly what a CPI print or a payrolls number mostly moves. Excellent
#: for a view held over weeks; the wrong instrument for a forty-minute repricing.
NEUTRALISING = frozenset({"dv01_neutral", "beta_neutral"})


def directional_names() -> list:
    """Structures whose P&L comes from the level rather than from a spread."""
    return [p.name for p in PLAYBOOKS if p.weighting not in NEUTRALISING]


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def score(thesis: str, playbook: Playbook) -> float:
    """How well a playbook matches a thesis, by alias and name overlap.

    Crude on purpose. It ranks the catalogue so the likely structures are read first; it
    does NOT filter, because the cost of ranking a good playbook fourth is nothing and the
    cost of hiding it is a worse trade. If this is ever replaced by an embedding index, this
    function is the baseline that index has to beat."""
    words = _tokens(thesis)
    if not words:
        return 0.0
    best = 0.0
    for phrase in (playbook.name.replace("_", " "),) + playbook.aliases:
        terms = _tokens(phrase)
        if terms:
            best = max(best, len(terms & words) / len(terms))
    return best


def select(thesis: str, limit: int = None) -> list:
    """The playbooks the proposer shows the model, best match first.

    Returns everything by default: the library fits in the prompt and caches, so there is no
    reason to withhold any of it, and a structure the model never sees is one it cannot
    choose. `limit` exists for when the library grows past that — and is the point where a
    real retriever would replace this ordering."""
    ranked = sorted(PLAYBOOKS, key=lambda p: (-score(thesis, p), p.name))
    return ranked[:limit] if limit else ranked


def catalogue(thesis: str = "", limit: int = None) -> str:
    """The playbook library as the model sees it."""
    return "\n\n".join(p.render() for p in select(thesis, limit))
