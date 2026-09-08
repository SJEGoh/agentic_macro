"""tests/test_sizing.py — the hedge ratio is the trade.

Every failure pinned here is one that produces a book which LOOKS right on approval and is
a different trade than the one named:

  * a steepener sized by equal dollars is a large outright duration SHORT. It has a
    steepener in it, but its P&L is driven by the level of rates, which is the thing the
    view was explicitly neutral on;
  * a butterfly whose wings are equal DOLLARS carries ~9x more DV01 in the TLT wing than
    the SHY wing — a slope bet wearing a butterfly's name;
  * an instrument with no duration silently treated as 1.0 sizes a curve trade as if SHY
    and TLT were the same risk;
  * rounding a leg UP spends more of the sleeve than was approved, every time, in the same
    direction — six legs of that is a materially bigger book than the one on screen;
  * dropping an unaffordable leg from a RISK-MATCHED structure does not shrink the trade,
    it changes it: a butterfly missing a wing is a steepener.

None of these raise on their own. They just quietly trade something else.
"""
import pytest

from agentic_macro import proposer, universe

PRICES = {"SHY": 82.0, "IEF": 95.0, "TLT": 88.0, "SPY": 600.0, "XLF": 50.0,
          "XLU": 70.0, "EFA": 80.0, "GLD": 250.0}
BUDGET = 100_000.0


def raw(structure, weighting, legs, **kw):
    return {"expressible": True, "structure": structure, "weighting": weighting,
            "reasoning": "", "confidence": "medium", "conflicts": "",
            "legs": [{"symbol": s, "direction": d, "weight": w, "rationale": ""}
                     for s, d, w in legs], **kw}


def dv01(proposal):
    """Signed duration-dollars — the number a curve trade is supposed to be neutral on."""
    return sum(l.quantity * l.entry_price * (universe.resolve(l.symbol).duration or 0.0)
               for l in proposal.legs)


def notional(proposal, symbol):
    return next(l.quantity * l.entry_price for l in proposal.legs if l.symbol == symbol)


# --------------------------------------------------------------------------- steepener
def test_steepener_is_dv01_neutral_not_equal_dollar():
    """The core claim: long SHY / short TLT is sized so the DV01s match, not the dollars.

    SHY carries 1.8y and TLT 16.5y, so the correct ratio is ~9.2:1 in dollars. Equal
    dollars would leave a huge net short in duration — the silent failure this pins."""
    p = proposer.size("cuts coming", raw("steepener", "dv01_neutral",
                                         [("SHY", "long", 1), ("TLT", "short", 1)]),
                      PRICES, BUDGET)
    ratio = notional(p, "SHY") / -notional(p, "TLT")
    assert 8.5 < ratio < 10.0, f"expected ~9.2:1 dollars, got {ratio:.2f}:1"
    # Residual DV01 is share rounding only, not a directional bet.
    assert abs(dv01(p)) < 0.02 * BUDGET


def test_steepener_gross_respects_the_budget():
    p = proposer.size("x", raw("steepener", "dv01_neutral",
                               [("SHY", "long", 1), ("TLT", "short", 1)]), PRICES, BUDGET)
    assert p.gross() <= BUDGET


# --------------------------------------------------------------------------- butterfly
def test_butterfly_wings_are_balanced_against_each_other():
    """A 50/50 fly needs EQUAL DV01 in each wing. Equal dollars would put ~9x more risk in
    the TLT wing, which is a slope bet, not curvature."""
    p = proposer.size("belly outperforms",
                      raw("curve_butterfly", "dv01_neutral",
                          [("IEF", "long", 1), ("SHY", "short", 1), ("TLT", "short", 1)]),
                      PRICES, BUDGET)
    wing_shy = abs(notional(p, "SHY")) * universe.resolve("SHY").duration
    wing_tlt = abs(notional(p, "TLT")) * universe.resolve("TLT").duration
    assert wing_shy == pytest.approx(wing_tlt, rel=0.02), \
        f"wings unbalanced: SHY {wing_shy:,.0f} vs TLT {wing_tlt:,.0f} DV01-dollars"


def test_butterfly_is_level_neutral():
    """Belly DV01 must equal the sum of the wings, or the fly carries a duration view."""
    p = proposer.size("belly outperforms",
                      raw("curve_butterfly", "dv01_neutral",
                          [("IEF", "long", 1), ("SHY", "short", 1), ("TLT", "short", 1)]),
                      PRICES, BUDGET)
    assert abs(dv01(p)) < 0.02 * BUDGET, f"fly is not level-neutral: {dv01(p):,.0f} DV01-$"
    assert len(p.legs) == 3


def test_butterfly_dollars_differ_wildly_between_wings():
    """The wings are balanced in RISK, which necessarily means very different dollars.
    A test that saw similar dollars would mean the risk weighting had stopped working."""
    p = proposer.size("x", raw("curve_butterfly", "dv01_neutral",
                               [("IEF", "long", 1), ("SHY", "short", 1), ("TLT", "short", 1)]),
                      PRICES, BUDGET)
    assert abs(notional(p, "SHY")) > 5 * abs(notional(p, "TLT"))


# --------------------------------------------------------------------------- beta
def test_beta_neutral_pair_matches_beta_not_dollars():
    p = proposer.size("banks outperform",
                      raw("banks_on_the_curve", "beta_neutral",
                          [("XLF", "long", 1), ("SPY", "short", 1)]), PRICES, BUDGET)
    long_beta = notional(p, "XLF") * universe.resolve("XLF").beta
    short_beta = notional(p, "SPY") * universe.resolve("SPY").beta
    assert long_beta + short_beta == pytest.approx(0, abs=0.02 * BUDGET)


# --------------------------------------------------------------------------- fail closed
def test_risk_unit_refuses_an_instrument_with_no_duration():
    """Defaulting a missing duration to 1.0 is the silent version of this: GLD would be
    sized as if it were a one-year bond and the curve trade would be nonsense."""
    with pytest.raises(ValueError, match="no duration"):
        universe.risk_unit("GLD", "dv01_neutral")
    assert universe.risk_unit("GLD", "equal_notional") == 1.0


def test_playbook_weighting_overrides_the_model():
    """The library is the written-down version of how a structure is risk-matched. If the
    model recalls it differently, the library wins and says so — that disagreement is the
    failure the library exists to catch."""
    p = proposer.size("x", raw("steepener", "equal_notional",
                               [("SHY", "long", 1), ("TLT", "short", 1)]), PRICES, BUDGET)
    assert p.weighting == "dv01_neutral"
    assert "overrode" in p.hedge_note
    assert notional(p, "SHY") / -notional(p, "TLT") > 5


def test_off_universe_symbol_is_rejected():
    with pytest.raises(universe.UnknownSymbol):
        proposer.size("x", raw("custom", "equal_notional", [("NVDA", "long", 1)]),
                      {"NVDA": 100.0}, BUDGET)


def test_duplicate_symbol_in_one_structure_is_rejected():
    with pytest.raises(proposer.ProposalError, match="twice"):
        proposer.size("x", raw("custom", "equal_notional",
                               [("SPY", "long", 1), ("SPY", "short", 1)]), PRICES, BUDGET)


def test_too_many_legs_is_rejected():
    legs = [(s, "long", 1) for s in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLU", "GLD")]
    with pytest.raises(proposer.ProposalError, match="more than"):
        proposer.size("x", raw("custom", "equal_notional", legs), PRICES, BUDGET)


def test_non_positive_weight_is_rejected():
    with pytest.raises(proposer.ProposalError, match="weight must be positive"):
        proposer.size("x", raw("custom", "equal_notional", [("SPY", "long", 0)]),
                      PRICES, BUDGET)


# --------------------------------------------------------------------------- rounding
def test_rounding_never_spends_more_than_the_budget():
    """Truncation toward zero, on BOTH signs. Rounding up would overspend the sleeve in the
    same direction every time, and six legs of that is a book bigger than the one shown."""
    for budget in (10_000.0, 33_333.0, 97_531.0, 250_000.0):
        p = proposer.size("x", raw("custom", "equal_notional",
                                   [("SPY", "long", 1), ("TLT", "short", 1),
                                    ("GLD", "long", 1)]), PRICES, budget)
        assert p.gross() <= budget, f"overspent at budget {budget}"
        for leg in p.legs:
            assert leg.quantity == int(leg.quantity), "fractional shares are not tradeable"
            assert abs(leg.quantity * leg.entry_price) <= abs(leg.notional) + 1e-6


def test_short_leg_rounds_toward_zero_too():
    """int() truncates toward zero for negatives as well; a floor() here would round a
    short leg AWAY from zero and silently oversize every short in the sleeve."""
    p = proposer.size("x", raw("custom", "equal_notional", [("TLT", "short", 1)]),
                      PRICES, 10_000.0)
    leg = p.legs[0]
    assert leg.quantity == -113          # 10000/88 = 113.6 -> 113, not 114
    assert abs(leg.quantity * leg.entry_price) <= 10_000.0


# --------------------------------------------------------------------------- degradation
def test_unaffordable_leg_in_a_risk_matched_structure_refuses():
    """A butterfly that cannot afford a wing is a steepener, not a small butterfly.
    Silently dropping the leg would place a structurally different trade under the name
    that was approved."""
    with pytest.raises(proposer.ProposalError, match="different trade"):
        proposer.size("x", raw("curve_butterfly", "dv01_neutral",
                               [("IEF", "long", 1), ("SHY", "short", 1), ("TLT", "short", 1)]),
                      PRICES, 500.0)


def test_inexpressible_view_yields_no_legs_rather_than_a_guess():
    p = proposer.size("something untradeable",
                      {"expressible": False, "structure": "custom",
                       "weighting": "equal_notional", "reasoning": "needs options",
                       "confidence": "low", "conflicts": "", "legs": []},
                      PRICES, BUDGET)
    assert p.legs == ()
    assert "options" in p.reasoning


def test_one_sided_structure_uses_the_whole_budget():
    p = proposer.size("gold up", raw("gold_debasement", "directional",
                                     [("GLD", "long", 1)]), PRICES, BUDGET)
    assert p.gross() == pytest.approx(BUDGET, rel=0.01)
