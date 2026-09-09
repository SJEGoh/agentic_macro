"""
agentic_macro/proposer.py — turn a worldview into a proposed structure, then into legs.

The division of labour here is the safety property, not a style choice:

    the model chooses  ->  the structure, the instruments, the direction, the conviction
    the code chooses   ->  every number that determines size

The model never sees a quantity and never emits a price. It names a structure and a relative
weight per leg; `size()` turns those into share counts using a price from `prices.py`, a
budget derived from the executor's own allocation, and a hedge ratio from the instrument
risk data in `universe.py`. So the classic failure — a confidently hallucinated price
producing a confidently wrong position size — cannot happen: there is no path from model
output to notional that does not pass through a real price.

The other half of the sizing is the hedge ratio, and it is the half that is easy to get
silently wrong. A steepener sized by equal dollars is a duration short. `_risk_split()`
below is what makes "long the front against the long end" mean the thing it says.
"""
from __future__ import annotations

import json
import logging
import os
from typing import NamedTuple

from . import config, playbooks, universe

log = logging.getLogger("agentic-macro.proposer")


class ProposalError(RuntimeError):
    """The model could not produce a usable proposal. Never fall back to a default book —
    "I don't know" must reach you as a message, not as a trade."""


class ProposedLeg(NamedTuple):
    symbol: str
    quantity: float          # signed, whole units — computed here, never by the model
    entry_price: float
    notional: float          # signed dollars, before rounding to whole units
    rationale: str
    #: Contract multiplier, carried on the leg rather than looked up later. A futures
    #: notional computed without it is understated by the multiplier — 1,000x on a note
    #: future — which sails straight past the allocation cap without tripping anything.
    multiplier: float = 1.0

    @property
    def unit_value(self) -> float:
        """What ONE unit costs: price for a share, price * multiplier for a contract."""
        return self.entry_price * self.multiplier

    @property
    def risk(self) -> float:
        return abs(self.quantity) * self.unit_value


class Proposal(NamedTuple):
    thesis: str
    structure: str
    weighting: str
    reasoning: str
    legs: tuple
    conflicts: str
    confidence: str
    budget: float
    hedge_note: str = ""     # how the hedge ratio was derived, in words
    dropped: tuple = ()      # legs the budget could not buy a single share of
    substituted: tuple = ()  # futures legs replaced by their ETF equivalent, and why
    context: tuple = ()      # the news the model was shown
    context_note: str = ""   # why there was none, when there was none

    def gross(self) -> float:
        return sum(l.risk for l in self.legs)


# --------------------------------------------------------------------------- the schema
# Hand-written rather than generated from a model class: `output_config.format` requires
# `additionalProperties: false` and an explicit `required` on every object, and a schema
# that is wrong in those details fails at request time rather than at review time.
def gemini_schema(schema: dict) -> dict:
    """Strip what Gemini's response schema will not accept.

    Gemini takes an OpenAPI 3.0 subset, not full JSON Schema, and `additionalProperties` is
    not in it — sending it is a hard 400 ("Unknown name additional_properties"), verified
    against the live API rather than assumed. The key is kept in SCHEMA because it documents
    the intent (no extra fields) and because a stricter provider would enforce it; this
    removes it on the way out.

    Losing it costs nothing here: every field the code reads is named explicitly, and an
    unexpected extra key in the response is ignored rather than acted on."""
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items()
                if k != "additionalProperties"}
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema


SCHEMA = {
    "type": "object",
    "properties": {
        "expressible": {
            "type": "boolean",
            "description": "false if this view cannot be honestly expressed in the universe",
        },
        "structure": {
            "type": "string",
            "enum": [p.name for p in playbooks.PLAYBOOKS] + ["custom"],
            "description": "the playbook this uses, or 'custom' when none fits",
        },
        "weighting": {
            "type": "string",
            "enum": sorted(playbooks.WEIGHTINGS),
            "description": "how legs are risk-matched. Ignored when a named playbook is "
                           "chosen — that playbook's own weighting is used.",
        },
        "reasoning": {
            "type": "string",
            "description": "the causal chain from the view to this structure, and what "
                           "would have to be true for it to be wrong",
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "conflicts": {
            "type": "string",
            "description": "how this interacts with the worldviews already held — "
                           "duplication, offsetting, or concentration. Empty if none.",
        },
        "legs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": universe.symbols()},
                    "direction": {"type": "string", "enum": ["long", "short"]},
                    "weight": {
                        "type": "number",
                        "description": "relative weight WITHIN this leg's side of the trade, "
                                       "> 0. The long/short hedge ratio is computed from "
                                       "instrument risk — do not try to encode it here.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "one line: why this instrument, this direction",
                    },
                },
                "required": ["symbol", "direction", "weight", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["expressible", "structure", "weighting", "reasoning", "confidence",
                 "conflicts", "legs"],
    "additionalProperties": False,
}


def contract_prices() -> dict:
    """Last price for each FUTURES instrument, for the "1 contract = $X" line in the
    catalogue. Best effort on purpose, and this is the one price path where that is right:
    these numbers are shown to the model so it can tell what fits the sleeve, and they size
    nothing. A failure here costs the model a hint; a failure in `size()` would cost a
    mis-sized order, which is why that one raises."""
    from . import prices as _prices
    futures = [i.symbol for i in universe.UNIVERSE if i.sec_type == "FUT"]
    try:
        return _prices.fetch(futures)
    except Exception as e:
        log.warning("no futures prices for the prompt (contract sizes will show as "
                    "multipliers): %s", e)
        return {}


EVENT_GUIDANCE = """

THIS IS AN EVENT TRADE, NOT A SLEEVE POSITION.

A scheduled release reprices markets in minutes and the position is held for well under an
hour. That changes which structures are appropriate, and the change is not a preference:

  * A DV01-neutral or beta-neutral structure is CONSTRUCTED to cancel the dominant factor —
    the level of rates, the level of the market. That is exactly what a CPI print or a
    payrolls number moves. Those structures are excellent for a view held over weeks and are
    the wrong instrument here, so they are not on the menu below.
  * Take a DIRECTION. Say which way the release moves the most exposed instrument, and hold
    that. A spread trade whose two legs both move with the surprise nets to nothing.
  * Prefer the single most exposed instrument over a basket. Diversifying across a
    forty-minute window dilutes the only thing being bet on.
  * Size for a move of a few tenths of a percent at most. If the release is genuinely
    unlikely to move your instrument, say the view is not expressible rather than
    manufacturing a trade for the sake of one."""


def schema_for(mode: str = "sleeve") -> dict:
    """The response schema. In event mode the structure enum drops the neutralising
    playbooks, so the model cannot pick a trade built to ignore what the event moves."""
    if mode != "event":
        return SCHEMA
    import copy
    s = copy.deepcopy(SCHEMA)
    s["properties"]["structure"]["enum"] = playbooks.directional_names() + ["custom"]
    s["properties"]["weighting"]["enum"] = sorted(
        playbooks.WEIGHTINGS - playbooks.NEUTRALISING)
    return s


def system_prompt(thesis: str = "", prices: dict = None, mode: str = "sleeve") -> str:
    """Built per call so the playbook catalogue can be ranked for the thesis. The universe
    and the instructions are byte-identical every time and sit first, so the expensive part
    of this prompt caches even though the ranking below it moves."""
    prices = prices if prices is not None else contract_prices()
    return f"""\
You are the analyst for a discretionary macro sleeve. You turn one stated worldview into
the most direct STRUCTURE available in a fixed universe of ETFs and futures.

You do not size positions and you do not quote prices. You choose a structure, instruments,
directions, and relative weights within each side of the trade; the system converts those
into share and contract counts using live prices, the sleeve's capital, and the instrument
risk data below. Never state a dollar amount, a share count, or a price level.

TRADEABLE UNIVERSE — you may name NOTHING outside this list:

{universe.catalogue(prices)}

On the futures. They give you the real instrument rather than a proxy: a note future carries
the cheapest-to-deliver's duration instead of a fund's, and a commodity future holds the
commodity instead of paying a roll to an ETF that holds it for you. But ONE CONTRACT IS
INDIVISIBLE and large — the size is printed above where a price was available. A structure
whose smallest leg costs more than its share of the budget cannot be built at all, so on a
small sleeve the ETF proxies (SHY, IEF, TLT, GLD, USO) are frequently the only expression
that fits. Reach for a future when the contract size works against the capital shown below,
and for the ETF when it does not.

PLAYBOOKS — the named structures, best matches for this view first:

{playbooks.catalogue(thesis) if mode != "event" else chr(10).join(
    p.render() for p in playbooks.select(thesis)
    if p.weighting not in playbooks.NEUTRALISING)}
{EVENT_GUIDANCE if mode == "event" else ""}

How to work:

1. Name the structure first. Most macro views are an instance of one of the playbooks
   above; pick that one and say so. Use "custom" only when nothing fits, and then say what
   the structure is in your reasoning.

2. Find the causal chain. Move from the view to the variable it implies (real yields, the
   curve, the dollar, the credit cycle, the China impulse) to the structure most exposed to
   that variable. State the chain. If it needs more than two steps, the trade is indirect
   and you should say so.

3. Express relative views as pairs. "Europe outperforms the US" is long EFA AND short SPY.
   An outright long leaves you owning beta you never had a view about, which is the most
   common way one of these trades is right and still loses money.

4. Do NOT try to encode the hedge ratio in the weights, and never compensate for an
   instrument's duration or beta yourself. The system computes the ratio from the duration
   and beta data above. Your weights only split each SIDE among its own instruments, and in
   a risk-weighted structure they are shares of that side's RISK, not of its dollars.

   A steepener is "long SHY weight 1, short TLT weight 1" — the system works out that this
   means many times more dollars in SHY than in TLT. A butterfly is "long IEF weight 1,
   short SHY weight 1, short TLT weight 1" — equal weights on the two wings is what makes
   it a 50/50 fly, and the system converts that into the very different dollar amounts the
   two wings need. Write the structure the way it is described in the playbook, and let the
   arithmetic happen downstream.

   Most multi-asset baskets are weighted `inverse_vol`: the legs are balanced on realised
   volatility, so equal weights mean equal RISK rather than equal dollars. Do not try to
   hand-correct for a volatile leg by giving it a smaller weight — that double-counts.
   Weight by conviction and let the volatility scaling do its job.

5. Say when it is already priced. If the view is consensus, say so and lower your
   confidence. You are not obliged to find a trade.

6. Respect the structural warnings. Names flagged as decaying (VXX, UNG, USO) lose money
   while you wait even when the view is right — use them only with a clear near-term
   catalyst, and say why the timing works. Names flagged thin should be small or replaced.

7. At most {config.MAX_LEGS} legs. A view needing more is several views; propose the
   cleanest and say what you left out.

8. Read the worldviews already held. Say plainly if this duplicates one (concentration you
   may not intend), offsets one (you would pay spread to hold nothing), or is a sharper
   version of one that should replace it.

If the view cannot be honestly expressed here — a single stock, a non-listed market, an
option structure, or a timing ETFs cannot capture — set expressible to false, leave legs
empty, and explain what would express it. That is a useful answer. Inventing an approximate
trade for a view the universe cannot hold is not.

Be concise and concrete. This is read on a phone before real money moves.
"""


def _context(views: list, allocation: float, capital_used: float) -> str:
    """What the model is told about the book it would be joining. Without this it proposes
    each view in isolation and quietly stacks three versions of the same duration trade."""
    if not views:
        held = "  None — this would be the first worldview in the sleeve."
    else:
        lines = []
        for v in views:
            legs = ", ".join(
                f"{'long' if l.quantity > 0 else 'short'} {l.symbol}" for l in v.legs)
            lines.append(f"  #{v.id} [{v.created_at[:10]}] {v.thesis}\n"
                         f"      holds: {legs or 'nothing'}")
        held = "\n".join(lines)
    room = max(0.0, allocation * config.MAX_GROSS_WEIGHT - capital_used)
    return (f"WORLDVIEWS CURRENTLY HELD\n{held}\n\n"
            f"SLEEVE CAPITAL\n"
            f"  allocation      ${allocation:,.0f}\n"
            f"  already at work ${capital_used:,.0f} gross\n"
            f"  room remaining  ${room:,.0f}\n")


def macro_context(thesis: str, as_of: float = None) -> tuple:
    """(prompt block, hits, note). Never raises.

    Memory is enrichment: the sleeve proposed structures for weeks without it. But a failure
    must be VISIBLE, because a proposal that silently lost its context is indistinguishable
    from one that never had any — and the difference matters when you are deciding whether
    the model knew about last week's CPI print."""
    if not config.MEMORY_ENABLED:
        return "", [], ""
    try:
        from . import memory
        block, hits = memory.context_block(thesis, as_of=as_of)
        if not hits:
            return "", [], "no recent news in the store (run `ingest`)"
        return block, hits, ""
    except Exception as e:
        log.warning("no macro context: %s", e)
        return "", [], f"context unavailable ({str(e)[:90]})"


def propose(thesis: str, views: list, allocation: float, capital_used: float,
            prices: dict = None, as_of: float = None, mode: str = "sleeve") -> dict:
    """Ask the model for a structure and its legs. Returns the raw validated dict; `size()`
    turns it into a Proposal. Split in two so the model call can be tested without pricing
    and the sizing can be tested without the network.

    `as_of` restricts retrieved context to what was published before that moment, which is
    what makes a backtest a backtest rather than a demonstration of hindsight.

    The model call goes through `providers.complete_json`, which is the seam to patch in a
    test and the switch between hosted and local models."""
    from . import providers

    block, hits, note = macro_context(thesis, as_of=as_of)
    user = (f"{_context(views, allocation, capital_used)}\n"
            + (f"{block}\n\n" if block else "")
            + f"NEW WORLDVIEW TO EXPRESS\n  {thesis.strip()}\n")

    try:
        out = providers.complete_json(system_prompt(thesis, prices, mode), user,
                                      schema_for(mode))
    except providers.ProviderError as e:
        raise ProposalError(str(e))
    if not isinstance(out, dict):
        raise ProposalError(f"the model returned {type(out).__name__}, not an object")

    # Carried on the result so the approval message can show what the model was told, and
    # say plainly when it was told nothing.
    out["_context"] = [{"published": h["published"],
                        "title": h["meta"].get("title") or h["text"][:110]} for h in hits]
    out["_context_note"] = note
    return out


#: Reasons the model stopped that are NOT an answer. Each arrives as a normal 200 with a
#: body that may well parse, so treating any of them as a proposal would place orders off a
#: response the model never finished — or deliberately declined to make.
_REFUSED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _check_finished(response) -> None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise ProposalError(f"the model returned nothing"
                            + (f" — the prompt was blocked ({blocked})" if blocked else ""))

    reason = getattr(candidates[0], "finish_reason", None)
    name = getattr(reason, "name", None) or str(reason or "")
    if name in ("STOP", "FINISH_REASON_UNSPECIFIED", "None", ""):
        return
    if name == "MAX_TOKENS":
        raise ProposalError("the model ran out of tokens mid-proposal — raise "
                            "AGENTIC_MAX_TOKENS, or shorten the worldview")
    if name in _REFUSED:
        raise ProposalError(f"the model declined to answer this one ({name})")
    raise ProposalError(f"the model stopped without finishing ({name})")


def _allocate(longs: list, shorts: list, weighting: str, budget: float,
              vols: dict = None) -> tuple:
    """Turn per-leg weights into signed dollar notionals. This is the hedge ratio, and it is
    the whole reason a structure is more than a list of tickers.

    For an unweighted structure the model's weights are DOLLAR shares and the two sides
    split the budget evenly. For a risk-weighted one they are RISK shares, and the sizing
    happens in risk space:

        risk_i    = w_i (normalised within its side) x R,  R the risk each side carries
        dollars_i = risk_i / risk_unit_i

    with R chosen so both sides carry the same total risk and the gross adds to `budget`.
    Working in risk space rather than dollar space is what makes multi-leg structures come
    out right without special-casing any of them:

      * a STEEPENER — long SHY (1.8y duration) against short TLT (16.5y) — lands at ~9.2:1
        dollars long:short. Sized by equal dollars it would instead be a large outright
        duration SHORT that merely contains a steepener, with a P&L driven by the level of
        rates rather than the slope the view was about;

      * a BUTTERFLY — long IEF against short SHY and TLT, wings weighted 1 and 1 — comes out
        DV01-neutral overall AND balanced between the wings (a 50/50 fly), so it expresses
        curvature alone. Equal dollars across the two wings would leave ~9x more DV01 in the
        TLT wing than the SHY wing, which is a slope bet wearing a butterfly's name.

    Returns ({symbol: signed notional}, note). A one-sided structure has nothing to
    neutralise against and simply takes the whole budget."""
    def split(side):
        total = sum(w for _, w in side)
        return [(sym, w / total) for sym, w in side]

    if not longs or not shorts:
        # One-sided, but still multi-leg: the legs are weighted against each OTHER by the
        # same risk axis. A basket of long DBC, XLE and GLD split by equal dollars is
        # dominated by whichever is most volatile, which is not what "equal conviction"
        # was meant to express.
        side, sign = (longs, 1.0) if longs else (shorts, -1.0)
        if weighting in ("equal_notional", "directional") or len(side) == 1:
            return {sym: sign * budget * w for sym, w in split(side)}, ""
        units = {sym: w / universe.risk_unit(sym, weighting, vols) for sym, w in split(side)}
        total = sum(units.values())
        axis = {"dv01_neutral": "duration", "beta_neutral": "beta",
                "inverse_vol": "volatility"}[weighting]
        return ({sym: sign * budget * u / total for sym, u in units.items()},
                f"{weighting}: legs balanced on {axis}")

    if weighting in ("equal_notional", "directional"):
        out = {}
        for members, sign in ((longs, 1.0), (shorts, -1.0)):
            for sym, w in split(members):
                out[sym] = sign * (budget / 2.0) * w
        return out, ""

    # Dollars per unit of risk carried by each side. `risk_unit` raises rather than
    # defaulting to 1.0 when an instrument has no risk on this axis — a missing duration
    # silently treated as one year would size a steepener as if SHY and TLT were the same.
    def dollars_per_risk(side):
        return sum(w / universe.risk_unit(sym, weighting, vols) for sym, w in split(side))

    k_long, k_short = dollars_per_risk(longs), dollars_per_risk(shorts)
    if k_long <= 0 or k_short <= 0:
        raise ProposalError(f"cannot risk-match this structure: a side carries no "
                            f"{weighting.split('_')[0]} risk")

    risk = budget / (k_long + k_short)          # the risk EACH side carries
    out = {}
    for members, sign in ((longs, 1.0), (shorts, -1.0)):
        for sym, w in split(members):
            out[sym] = sign * risk * w / universe.risk_unit(sym, weighting, vols)

    axis = {"dv01_neutral": "duration", "beta_neutral": "beta",
            "inverse_vol": "volatility"}[weighting]
    gross_long = sum(v for v in out.values() if v > 0)
    gross_short = -sum(v for v in out.values() if v < 0)
    note = (f"{weighting}: each side carries {risk:,.0f} of {axis}-dollars "
            f"-> {gross_long / gross_short:.2f}:1 dollars long:short")
    return out, note


def size(thesis: str, raw: dict, prices: dict, budget: float,
         vols: dict = None) -> Proposal:
    """Turn a structure and its weights into whole-share quantities.

    ETFs are sized fractionally, so a leg lands on its intended notional instead of being
    truncated down to a whole share — which on an expensive name like GLD threw away a
    meaningful slice of the position and quietly unbalanced every hedge ratio it was part of.

    FUTURES ARE ALWAYS WHOLE CONTRACTS. There is no such thing as 0.4 of a contract, and an
    order for one is rejected at the broker rather than rounded, so those still truncate
    toward zero: rounding a contract UP spends more of the sleeve than you approved, and on
    a $107,000 note future that is not a rounding error, it is a position.

    A leg the budget cannot afford at all is DROPPED and reported — silently shrinking it to
    zero would show you a hedge that is not actually there."""
    structure = raw.get("structure", "custom")

    # The playbook's weighting wins over the model's. The library is the written-down
    # version of how these structures are risk-matched; if the model recalls it differently
    # that is the failure the library exists to catch, not a preference to defer to.
    weighting = raw.get("weighting", "equal_notional")
    book = playbooks.BY_NAME.get(structure)
    override = ""
    if book and book.weighting != weighting:
        override = (f"weighting from the {structure} playbook ({book.weighting}) "
                    f"overrode the model's ({weighting})")
        weighting = book.weighting
    if weighting not in playbooks.WEIGHTINGS:
        raise ProposalError(f"unknown weighting {weighting!r}")

    if not raw.get("expressible", False):
        return Proposal(thesis=thesis, structure=structure, weighting=weighting,
                        reasoning=raw.get("reasoning", ""), legs=(),
                        conflicts=raw.get("conflicts", ""),
                        confidence=raw.get("confidence", "low"), budget=budget,
                        context=tuple(raw.get("_context") or ()),
                        context_note=raw.get("_context_note", ""))

    entries = raw.get("legs") or []
    if len(entries) > config.MAX_LEGS:
        raise ProposalError(f"the model proposed {len(entries)} legs, more than the "
                            f"{config.MAX_LEGS} allowed")
    if not entries:
        raise ProposalError("the model called the view expressible but proposed no legs")

    # Resolve and validate EVERY leg before sizing any of them: a book that is half-sized
    # and then rejected for an unknown ticker has already spent the budget it was checking.
    longs, shorts, by_symbol = [], [], {}
    for entry in entries:
        instrument = universe.resolve(entry["symbol"])       # raises on anything off-list
        weight = float(entry.get("weight", 0.0))
        if weight <= 0:
            raise ProposalError(f"{instrument.symbol}: weight must be positive, got {weight}")
        if instrument.symbol in by_symbol:
            raise ProposalError(f"{instrument.symbol} appears twice in one structure")
        by_symbol[instrument.symbol] = entry
        (longs if entry["direction"] == "long" else shorts).append(
            (instrument.symbol, weight))

    # Allocate, then check every FUTURES leg can afford at least one contract at the share
    # it was given. A leg that cannot is SUBSTITUTED for its ETF equivalent and the whole
    # structure is re-allocated — the ETF has a different volatility and duration, so
    # swapping after the fact would leave the hedge ratio computed for an instrument no
    # longer in the trade. Re-allocating is the only way the ratio stays true.
    #
    # Substitution is not the same concession as dropping a leg. It keeps the underlying,
    # the direction and the leg's place in the structure; only the fee and roll profile
    # change. Dropping would turn a pair into an outright.
    swaps = {}
    for _ in range(len(longs) + len(shorts) + 1):        # each pass fixes >=1 leg
        notionals, hedge_note = _allocate(longs, shorts, weighting, budget, vols)
        unaffordable = {}
        for symbol, notional in notionals.items():
            inst = universe.resolve(symbol)
            if inst.sec_type != "FUT":
                continue
            unit = inst.notional(prices[symbol])
            if abs(notional) < unit and inst.etf_equivalent:
                unaffordable[symbol] = (inst, unit, abs(notional))
        if not unaffordable:
            break
        for symbol, (inst, unit, got) in unaffordable.items():
            swaps[symbol] = inst.etf_equivalent
            entry = by_symbol.pop(symbol)
            # The rationale was written about the FUTURE. Left alone it reads absurdly on
            # the replacement — "micro WTI, avoiding USO's roll drag" attached to a USO leg.
            entry = dict(entry, rationale=(entry.get("rationale", "") +
                                           f" [written for {symbol}]").strip())
            by_symbol[inst.etf_equivalent] = entry
            longs = [(inst.etf_equivalent if s == symbol else s, w) for s, w in longs]
            shorts = [(inst.etf_equivalent if s == symbol else s, w) for s, w in shorts]
    else:
        notionals, hedge_note = _allocate(longs, shorts, weighting, budget, vols)

    substituted = tuple(
        f"{f} -> {e} (one {f} contract is "
        f"${universe.resolve(f).notional(prices[f]):,.0f}, more than this leg's share)"
        for f, e in swaps.items())
    if override:
        hedge_note = f"{override}; {hedge_note}" if hedge_note else override

    legs, dropped = [], []
    for symbol, notional in notionals.items():
        instrument = universe.resolve(symbol)
        price = prices[symbol]                   # a KeyError here is a bug, not bad input
        unit = instrument.notional(price)        # price * multiplier — the cost of ONE unit
        if instrument.sec_type == "FUT" or not config.FRACTIONAL:
            quantity = float(int(notional / unit))    # truncates toward zero, both signs
        else:
            # Truncate at the fractional precision rather than round. round() can go UP,
            # which would spend fractionally more of the sleeve than was approved and quietly
            # break the invariant that gross never exceeds the budget. The precision given up
            # is ~$0.02 on a $400 share; the invariant is worth more than that.
            scale = 10 ** config.FRACTIONAL_DP
            quantity = int(notional / unit * scale) / scale
        if quantity == 0:
            what = "contract" if instrument.sec_type == "FUT" else "share"
            dropped.append(f"{symbol} (${abs(notional):,.0f} does not buy one {what} "
                           f"at ${unit:,.0f})")
            continue
        legs.append(ProposedLeg(symbol=symbol, quantity=quantity, entry_price=price,
                                notional=notional, multiplier=instrument.multiplier,
                                rationale=by_symbol[symbol].get("rationale", "")))

    # A dropped leg in a MULTI-LEG structure is not a smaller version of the trade — it is a
    # different one. Losing a wing turns a butterfly into a steepener; losing the long side
    # of a pair leaves a naked outright pointing the other way.
    #
    # This once checked only dv01_neutral and beta_neutral, and an inverse_vol proposal walked
    # straight through it: long MGC + MCL against short UUP put so few dollars on the volatile
    # commodity side that neither contract was affordable, both legs were dropped, and what
    # survived was a naked short-dollar position under a name that said "long gold and crude".
    # The axis was never the point — ANY structure minus a leg is a different structure.
    if dropped and len(entries) > 1:
        surviving = ", ".join(f"{l.symbol} {'long' if l.quantity > 0 else 'short'}"
                              for l in legs) or "nothing"
        raise ProposalError(
            f"this {len(entries)}-leg structure cannot be built at a ${budget:,.0f} budget "
            f"— {'; '.join(dropped)}. That would leave {surviving}, which is a different "
            f"trade, not a smaller one. Raise the size, or use ETF legs instead of futures.")

    if not legs:
        why = f" — {'; '.join(dropped)}" if dropped else ""
        raise ProposalError(
            f"the whole structure rounded to nothing at a ${budget:,.0f} budget{why}. "
            f"Raise the size, or use a smaller-denomination instrument.")

    legs.sort(key=lambda l: (-l.risk, l.symbol))
    return Proposal(thesis=thesis, structure=structure, weighting=weighting,
                    reasoning=raw.get("reasoning", ""), legs=tuple(legs),
                    conflicts=raw.get("conflicts", ""),
                    confidence=raw.get("confidence", "medium"), budget=budget,
                    hedge_note=hedge_note, dropped=tuple(dropped),
                    substituted=substituted,
                    context=tuple(raw.get("_context") or ()),
                    context_note=raw.get("_context_note", ""))


#: How much of the sleeve a view gets before the remaining-capital clamp. Conviction scales
#: it because "high confidence" should mean a bigger position, not just a more emphatic
#: paragraph — but the ceiling is MAX_WORLDVIEW_WEIGHT either way, so no single view can
#: take the sleeve over regardless of how sure the model sounds.
CONVICTION = {"low": 0.4, "medium": 0.7, "high": 1.0}


def default_budget(allocation: float, confidence: str) -> float:
    return allocation * config.MAX_WORLDVIEW_WEIGHT * CONVICTION.get(confidence, 0.7)
