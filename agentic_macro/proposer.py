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
    quantity: float          # signed, whole shares — computed here, never by the model
    entry_price: float
    notional: float          # signed dollars, before rounding to whole shares
    rationale: str

    @property
    def risk(self) -> float:
        return abs(self.quantity) * self.entry_price


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


def system_prompt(thesis: str = "") -> str:
    """Built per call so the playbook catalogue can be ranked for the thesis. The universe
    and the instructions are byte-identical every time and sit first, so the expensive part
    of this prompt caches even though the ranking below it moves."""
    return f"""\
You are the analyst for a discretionary macro sleeve. You turn one stated worldview into
the most direct STRUCTURE available in a fixed universe of ETFs.

You do not size positions and you do not quote prices. You choose a structure, instruments,
directions, and relative weights within each side of the trade; the system converts those
into share counts using live prices, the sleeve's capital, and the instrument risk data
below. Never state a dollar amount, a share count, or a price level.

TRADEABLE UNIVERSE — you may name NOTHING outside this list:

{universe.catalogue()}

PLAYBOOKS — the named structures, best matches for this view first:

{playbooks.catalogue(thesis)}

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


def propose(thesis: str, views: list, allocation: float, capital_used: float,
            client=None) -> dict:
    """Ask the model for a structure and its legs. Returns the raw validated dict; `size()`
    turns it into a Proposal. Split in two so the model call can be tested without pricing
    and the sizing can be tested without the network."""
    from google.genai import types

    if client is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ProposalError("GEMINI_API_KEY is not set — the proposer cannot run")
        client = genai.Client(api_key=key)

    user = (f"{_context(views, allocation, capital_used)}\n"
            f"NEW WORLDVIEW TO EXPRESS\n  {thesis.strip()}\n")
    request = types.GenerateContentConfig(
        system_instruction=system_prompt(thesis),
        response_mime_type="application/json",
        response_schema=gemini_schema(SCHEMA),
        thinking_config=types.ThinkingConfig(thinking_level=config.THINKING_LEVEL),
        max_output_tokens=config.MAX_TOKENS,
    )

    response, tried = None, []
    for model in [config.MODEL] + config.FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model, contents=user, config=request)
            if model != config.MODEL:
                log.warning("proposed with fallback model %s", model)
            break
        except Exception as e:
            # Only availability failures are worth another model. A 400 means the request
            # itself is malformed and every model will say the same thing, so it must not be
            # buried under a retry loop that ends in a vague "all models failed".
            code = getattr(e, "code", None)
            if code not in (429, 500, 503, 404):
                raise ProposalError(f"{model} rejected the request: {e}")
            tried.append(f"{model} ({code})")
            log.warning("%s unavailable (%s), trying the next model", model, code)
    if response is None:
        raise ProposalError("no model was available to propose with: " + ", ".join(tried))

    _check_finished(response)
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ProposalError(f"could not read the model's proposal as JSON: {e}")


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


def _allocate(longs: list, shorts: list, weighting: str, budget: float) -> tuple:
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
        side, sign = (longs, 1.0) if longs else (shorts, -1.0)
        return {sym: sign * budget * w for sym, w in split(side)}, ""

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
        return sum(w / universe.risk_unit(sym, weighting) for sym, w in split(side))

    k_long, k_short = dollars_per_risk(longs), dollars_per_risk(shorts)
    if k_long <= 0 or k_short <= 0:
        raise ProposalError(f"cannot risk-match this structure: a side carries no "
                            f"{weighting.split('_')[0]} risk")

    risk = budget / (k_long + k_short)          # the risk EACH side carries
    out = {}
    for members, sign in ((longs, 1.0), (shorts, -1.0)):
        for sym, w in split(members):
            out[sym] = sign * risk * w / universe.risk_unit(sym, weighting)

    axis = "duration" if weighting == "dv01_neutral" else "beta"
    gross_long = sum(v for v in out.values() if v > 0)
    gross_short = -sum(v for v in out.values() if v < 0)
    note = (f"{weighting}: each side carries {risk:,.0f} of {axis}-dollars "
            f"-> {gross_long / gross_short:.2f}:1 dollars long:short")
    return out, note


def size(thesis: str, raw: dict, prices: dict, budget: float) -> Proposal:
    """Turn a structure and its weights into whole-share quantities.

    Rounding toward zero is deliberate: rounding a leg UP spends more of the sleeve than you
    approved, and six such roundings compound into a book meaningfully larger than the one
    on screen. A leg the budget cannot buy a single share of is DROPPED and reported —
    silently shrinking it to zero would show you a hedge that is not actually there."""
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
                        confidence=raw.get("confidence", "low"), budget=budget)

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

    notionals, hedge_note = _allocate(longs, shorts, weighting, budget)
    if override:
        hedge_note = f"{override}; {hedge_note}" if hedge_note else override

    legs, dropped = [], []
    for symbol, notional in notionals.items():
        price = prices[symbol]                      # a KeyError here is a bug, not bad input
        quantity = float(int(notional / price))     # int() truncates toward zero, both signs
        if quantity == 0:
            dropped.append(f"{symbol} (${abs(notional):,.0f} does not buy one share "
                           f"at ${price:,.2f})")
            continue
        legs.append(ProposedLeg(symbol=symbol, quantity=quantity, entry_price=price,
                                notional=notional,
                                rationale=by_symbol[symbol].get("rationale", "")))

    # A dropped leg in a risk-matched structure is not a smaller version of the trade — it
    # is a different one. Losing a wing turns a butterfly into a steepener, and losing the
    # short leg of a pair leaves an outright. Refuse rather than quietly reshaping it.
    if dropped and weighting in ("dv01_neutral", "beta_neutral"):
        raise ProposalError(
            f"a {weighting} structure cannot be built at a ${budget:,.0f} budget — "
            f"{'; '.join(dropped)}. The remaining legs would be a different trade, "
            f"not a smaller one. Raise the size.")

    if not legs:
        raise ProposalError(
            f"the whole structure rounded to nothing at a ${budget:,.0f} budget — "
            f"raise the size, or the sleeve's allocation")

    legs.sort(key=lambda l: (-abs(l.quantity * l.entry_price), l.symbol))
    return Proposal(thesis=thesis, structure=structure, weighting=weighting,
                    reasoning=raw.get("reasoning", ""), legs=tuple(legs),
                    conflicts=raw.get("conflicts", ""),
                    confidence=raw.get("confidence", "medium"), budget=budget,
                    hedge_note=hedge_note, dropped=tuple(dropped))


#: How much of the sleeve a view gets before the remaining-capital clamp. Conviction scales
#: it because "high confidence" should mean a bigger position, not just a more emphatic
#: paragraph — but the ceiling is MAX_WORLDVIEW_WEIGHT either way, so no single view can
#: take the sleeve over regardless of how sure the model sounds.
CONVICTION = {"low": 0.4, "medium": 0.7, "high": 1.0}


def default_budget(allocation: float, confidence: str) -> float:
    return allocation * config.MAX_WORLDVIEW_WEIGHT * CONVICTION.get(confidence, 0.7)
