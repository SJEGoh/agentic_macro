"""tests/test_futures_and_vol.py — the multiplier, and the volatility.

Two new ways to be silently wrong, both of which produce a book that looks fine:

  * a futures notional computed without the multiplier is understated by up to 1,000x. The
    order is not rejected — it is ACCEPTED at a size the allocation cap should have stopped,
    and the cap is computed from the same understated number, so nothing downstream can
    notice. This is the exact failure the executor's own netting warns about;
  * inverse-vol sizing that falls back to a default when volatility is missing hands the
    LARGEST position to whichever leg it knows least about. Sizing a 50%-vol commodity and a
    2%-vol front-end bond as if both were 1.0 is not a smaller error than getting the
    direction wrong.

Everything here uses fixed prices and vols; nothing touches a network.
"""
import pytest

from agentic_macro import proposer, universe

PRICES = {"SHY": 82.0, "IEF": 95.0, "TLT": 88.0, "SPY": 600.0, "GLD": 250.0,
          "USO": 72.5, "SLV": 30.0, "QQQ": 500.0, "IWM": 220.0, "FXE": 100.0,
          "FXY": 60.0, "UUP": 27.98,
          "ZT": 102.56, "ZF": 105.52, "ZN": 107.44, "ZB": 108.75,
          "MES": 7697.0, "M2K": 2969.9, "MCL": 92.74, "MGC": 4444.0, "6J": 0.0065}
VOLS = {"SHY": 0.018, "IEF": 0.055, "TLT": 0.096, "SPY": 0.121, "GLD": 0.15,
        "USO": 0.42, "SLV": 0.30, "QQQ": 0.16, "IWM": 0.19, "FXE": 0.06,
        "FXY": 0.09, "UUP": 0.05,
        "ZT": 0.013, "ZF": 0.030, "ZN": 0.041, "ZB": 0.089,
        "MES": 0.119, "M2K": 0.138, "MCL": 0.511, "MGC": 0.249, "6J": 0.098}
BUDGET = 1_000_000.0


def raw(structure, weighting, legs):
    return {"expressible": True, "structure": structure, "weighting": weighting,
            "reasoning": "", "confidence": "medium", "conflicts": "",
            "legs": [{"symbol": s, "direction": d, "weight": w, "rationale": ""}
                     for s, d, w in legs]}


def leg(p, symbol):
    return next(l for l in p.legs if l.symbol == symbol)


# --------------------------------------------------------------------------- multiplier
def test_every_futures_instrument_carries_a_multiplier_and_an_exchange():
    """A futures leg without a multiplier is refused by RemoteStrategy.validate() and
    valued at None by the executor's netting. Catching it here means it can never reach
    either."""
    for i in universe.UNIVERSE:
        if i.sec_type == "FUT":
            assert i.multiplier and i.multiplier > 1, f"{i.symbol} has no multiplier"
            assert i.exchange != "SMART", f"{i.symbol} needs a real exchange, not SMART"
            assert i.price_symbol, f"{i.symbol} needs a feed symbol"


def test_a_futures_position_is_valued_with_its_multiplier():
    """One ZN contract is ~$107,000, not ~$107. Everything about the cap depends on this."""
    p = proposer.size("duration", raw("long_duration", "directional", [("ZN", "long", 1)]),
                      PRICES, 500_000.0, VOLS)
    zn = leg(p, "ZN")
    assert zn.multiplier == 1000
    assert zn.unit_value == pytest.approx(107_440, rel=0.01)
    assert zn.quantity == 4                      # 500k / 107.44k = 4.65 -> 4 contracts
    assert p.gross() == pytest.approx(4 * 107_440, rel=0.01)


def test_gross_would_be_understated_1000x_without_the_multiplier():
    """States the bug explicitly so a regression is unmistakable rather than merely wrong."""
    p = proposer.size("x", raw("long_duration", "directional", [("ZN", "long", 1)]),
                      PRICES, 500_000.0, VOLS)
    naive = sum(abs(l.quantity) * l.entry_price for l in p.legs)   # the tempting mistake
    assert p.gross() == pytest.approx(naive * 1000, rel=0.01)
    assert p.gross() > 400_000 and naive < 500


def test_a_budget_too_small_for_one_contract_falls_back_to_the_etf():
    """One ZN is $107,440, so a $50k budget cannot hold a single contract. Rather than
    refusing, the leg becomes IEF — same exposure, same direction, same place in the
    structure; only the fee and roll profile change."""
    p = proposer.size("x", raw("long_duration", "directional", [("ZN", "long", 1)]),
                      PRICES, 50_000.0, VOLS)
    assert [l.symbol for l in p.legs] == ["IEF"]
    assert p.substituted and "ZN -> IEF" in p.substituted[0]
    assert "107,4" in p.substituted[0], "the swap must say what the contract cost"
    assert p.gross() == pytest.approx(50_000.0, rel=1e-3)


def test_a_future_that_fits_is_kept():
    """Substitution is a fallback, not a preference — the real instrument wins when it fits."""
    p = proposer.size("x", raw("energy_supply_shock", "directional", [("MCL", "long", 1)]),
                      PRICES, 87_500.0, VOLS)
    assert [l.symbol for l in p.legs] == ["MCL"]
    assert p.substituted == ()


def test_a_futures_curve_trade_is_dv01_matched_on_contract_notional():
    """ZT and ZB, DV01-matched. The multiplier differs between them (2,000 vs 1,000) as
    well as the duration, so getting this right requires both."""
    p = proposer.size("steepener", raw("steepener", "dv01_neutral",
                                       [("ZT", "long", 1), ("ZB", "short", 1)]),
                      PRICES, 20_000_000.0, VOLS)
    dv01 = sum(l.quantity * l.unit_value * (universe.resolve(l.symbol).duration or 0)
               for l in p.legs)
    gross = p.gross()
    assert abs(dv01) < 0.02 * gross, f"futures steepener is not DV01-neutral: {dv01:,.0f}"
    assert leg(p, "ZT").quantity > 0 and leg(p, "ZB").quantity < 0


# --------------------------------------------------------------------------- inverse vol
def test_inverse_vol_gives_the_quieter_leg_more_dollars():
    """The defining property. MCL at 51% vol and MGC at 25% should land near 1:2 in
    dollars, not 1:1."""
    p = proposer.size("commodities", raw("reflation", "inverse_vol",
                                         [("MCL", "long", 1), ("MGC", "long", 1)]),
                      PRICES, 2_000_000.0, VOLS)
    ratio = leg(p, "MGC").risk / leg(p, "MCL").risk
    assert 1.7 < ratio < 2.4, f"expected ~2:1 gold:crude by dollars, got {ratio:.2f}"


def test_inverse_vol_equalises_risk_not_dollars():
    p = proposer.size("x", raw("reflation", "inverse_vol",
                               [("MCL", "long", 1), ("MGC", "long", 1)]),
                      PRICES, 2_000_000.0, VOLS)
    risks = [l.risk * VOLS[l.symbol] for l in p.legs]
    assert max(risks) / min(risks) < 1.15, "vol-adjusted risk is not equalised"


def test_inverse_vol_balances_the_two_sides_of_a_pair():
    p = proposer.size("x", raw("credit_compression", "inverse_vol",
                               [("SPY", "long", 1), ("TLT", "short", 1)]),
                      PRICES, 1_000_000.0, VOLS)
    long_risk = leg(p, "SPY").risk * VOLS["SPY"]
    short_risk = leg(p, "TLT").risk * VOLS["TLT"]
    assert long_risk == pytest.approx(short_risk, rel=0.05)


def test_missing_volatility_refuses_rather_than_defaulting():
    """Defaulting to 1.0 would give the unknown leg the SMALLEST weight, or with a default
    of zero the largest — either way a number nobody chose, driving a real position."""
    with pytest.raises(ValueError, match="no realised volatility"):
        universe.risk_unit("SPY", "inverse_vol", {})
    with pytest.raises(ValueError, match="no realised volatility"):
        universe.risk_unit("SPY", "inverse_vol", {"SPY": 0.0})


def test_a_one_sided_basket_is_still_vol_balanced():
    """A long-only basket has no opposing side, but its legs still have to be balanced
    against each other or the most volatile name dominates the whole trade."""
    p = proposer.size("x", raw("stagflation", "inverse_vol",
                               [("MCL", "long", 1), ("MGC", "long", 1), ("SPY", "long", 1)]),
                      PRICES, 3_000_000.0, VOLS)
    risks = [l.risk * VOLS[l.symbol] for l in p.legs]
    assert max(risks) / min(risks) < 1.2
    assert leg(p, "SPY").risk > leg(p, "MCL").risk      # SPY is far quieter than crude


def test_equal_notional_still_means_equal_dollars():
    """The old behaviour has to remain available and unchanged, or every playbook that
    deliberately wants dollar-matching silently becomes something else."""
    p = proposer.size("x", raw("custom", "equal_notional",
                               [("MCL", "long", 1), ("MGC", "long", 1)]),
                      PRICES, 2_000_000.0, VOLS)
    assert leg(p, "MCL").risk == pytest.approx(leg(p, "MGC").risk, rel=0.05)


def test_dv01_and_beta_are_unaffected_by_vols_being_present():
    """A curve trade is weighted on duration even when volatility is available — passing
    vols must not silently change the axis a structure is matched on."""
    p = proposer.size("x", raw("steepener", "dv01_neutral",
                               [("SHY", "long", 1), ("TLT", "short", 1)]),
                      PRICES, 1_000_000.0, VOLS)
    ratio = leg(p, "SHY").risk / leg(p, "TLT").risk
    assert 8.5 < ratio < 10.0, "dv01 weighting changed when vols were supplied"


# --------------------------------------------------------------------------- fractional
def test_fractional_applies_to_etfs_and_never_to_futures():
    """The one asymmetry that matters. IB rejects an order for 0.4 contracts rather than
    rounding it, so the fractional flag must not reach a futures leg — while an ETF leg
    sized to a whole share throws away up to a share's worth of the position."""
    p = proposer.size("x", raw("reflation", "inverse_vol",
                               [("GLD", "long", 1), ("MGC", "long", 1)]),
                      PRICES | {"GLD": 404.17}, 900_000.0, VOLS)
    gld, mgc = leg(p, "GLD"), leg(p, "MGC")
    assert mgc.quantity == int(mgc.quantity), "a futures leg came out fractional"
    assert gld.quantity != int(gld.quantity), "an ETF leg was truncated to a whole share"


def test_a_fractional_quantity_still_never_exceeds_its_budget():
    for budget in (7_331.0, 44_444.0, 250_001.0):
        p = proposer.size("x", raw("gold_debasement", "directional", [("GLD", "long", 1)]),
                          PRICES | {"GLD": 404.17}, budget, VOLS)
        assert p.gross() <= budget


def test_quantities_render_without_hiding_the_fraction():
    """A %.0f would print 556 for a 555.5556-share leg — wrong, and invisibly so."""
    assert universe.qty(555.5556) == "555.5556"
    assert universe.qty(40.0) == "40"
    assert universe.qty(-113.25) == "-113.25"


# --------------------------------------------------------------------------- /universe
@pytest.fixture
def offline(monkeypatch):
    """/universe quotes live contract sizes; the tests must not depend on a price feed to
    check that it lists what it should."""
    from agentic_macro import prices
    monkeypatch.setattr(prices, "fetch", lambda syms, **kw: {s: PRICES[s] for s in syms
                                                            if s in PRICES})


def test_universe_command_summarises_every_bucket(offline):
    from agentic_macro import bot
    text = bot.cmd_universe([], {})
    assert str(len(universe.UNIVERSE)) in text
    for bucket in {i.bucket for i in universe.UNIVERSE}:
        assert bucket in text


def test_universe_command_details_one_instrument(offline):
    from agentic_macro import bot
    text = bot.cmd_universe(["ZN"], {})
    assert "multiplier" in text and "1,000" in text and "CBOT" in text
    assert "6.5y" in text


def test_universe_command_surfaces_the_decay_warning(offline):
    """The single most useful thing to know before holding VXX."""
    from agentic_macro import bot
    assert "DECAYS" in bot.cmd_universe(["VXX"], {})


def test_universe_command_handles_an_unknown_query(offline):
    from agentic_macro import bot
    text = bot.cmd_universe(["NVDA"], {})
    assert "no instrument or bucket" in text and "buckets:" in text


def test_the_naked_short_case_now_keeps_every_leg():
    """The live bug: an inverse_vol trio — long MGC + MCL against short UUP — put so few
    dollars on the volatile commodity side that neither contract was affordable. Both long
    legs were dropped and what survived was a NAKED SHORT DOLLAR position under a proposal
    that read "long gold and crude". Substitution keeps all three legs and the trade."""
    p = proposer.size("dollar down", raw("dollar_weakness", "inverse_vol",
                                         [("UUP", "short", 1), ("MGC", "long", 1),
                                          ("MCL", "long", 1)]),
                      PRICES, 61_250.0, VOLS)
    assert len(p.legs) == 3, "a leg went missing again"
    assert {l.symbol for l in p.legs} == {"UUP", "GLD", "USO"}
    assert leg(p, "UUP").quantity < 0
    assert leg(p, "GLD").quantity > 0 and leg(p, "USO").quantity > 0
    assert len(p.substituted) == 2


def test_substitution_keeps_a_curve_trade_dv01_neutral():
    """The substituted ETF has a DIFFERENT duration from the future it replaces, so the
    structure must be RE-allocated after the swap. Swapping after sizing would leave the
    hedge ratio computed for an instrument no longer in the trade."""
    p = proposer.size("steepener", raw("steepener", "dv01_neutral",
                                       [("ZT", "long", 1), ("ZB", "short", 1)]),
                      PRICES, 87_500.0, VOLS)
    assert {l.symbol for l in p.legs} == {"SHY", "TLT"}
    dv01 = sum(l.quantity * l.unit_value * (universe.resolve(l.symbol).duration or 0)
               for l in p.legs)
    assert abs(dv01) < 0.02 * p.gross(), f"not neutral after substitution: {dv01:,.0f}"


def test_a_leg_is_still_dropped_and_refused_when_substitution_cannot_help(monkeypatch):
    """Substitution rescues futures. With fractional sizing off, an ETF leg can still round
    to nothing, and then the multi-leg guard must still refuse rather than reshape."""
    from agentic_macro import config as _c
    monkeypatch.setattr(_c, "FRACTIONAL", False)
    with pytest.raises(proposer.ProposalError, match="different trade|rounded to nothing"):
        proposer.size("x", raw("steepener", "dv01_neutral",
                               [("SHY", "long", 1), ("TLT", "short", 1)]),
                      PRICES, 120.0, VOLS)



