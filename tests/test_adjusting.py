"""tests/test_adjusting.py — changing a position you already hold.

Four operations, and the distinction between them is the point:

  * /resize scales EVERY leg by one factor, which is exactly what leaves a hedge ratio
    untouched — DV01 balance, beta match and vol weighting are all ratios between legs;
  * /adjust changes one leg, which necessarily breaks whatever the structure was balanced
    on. That is a legitimate manual override, so it is allowed and the resulting imbalance
    is SHOWN rather than refused;
  * /switch replaces a view, closing the old and opening the new in ONE netted book — two
    submissions would flatten the shared names and buy them straight back, paying spread
    twice for nothing;
  * /cancel discards a pending proposal, so an abandoned one cannot be confirmed by accident
    later.

The silent failures pinned here: a resize that quietly drops a leg it cannot round, and an
adjust that breaks a hedge without saying so.
"""
import pytest

from agentic_macro import bot, config, orders, prices as _prices, universe
from agentic_macro.store import Leg, Store

MARKS = {"SHY": 82.0, "TLT": 88.0, "IEF": 95.0, "SPY": 600.0, "GLD": 250.0, "ZN": 107.44}


@pytest.fixture
def sleeve(tmp_path, monkeypatch):
    s = Store(tmp_path / "adj.db")
    monkeypatch.setattr(bot, "store", s)
    monkeypatch.setattr(config, "PAPER", True)
    monkeypatch.setattr(config, "allowed_users", lambda: {42})
    monkeypatch.setattr(_prices, "fetch",
                        lambda syms, **kw: {x: MARKS[x] for x in syms if x in MARKS})
    monkeypatch.setattr(orders, "broker_positions",
                        lambda client, store=None, strategy_id=None: dict(s.net_positions()))
    yield s
    s.close()


def ctx():
    return {"user": "sjegoh", "user_id": 42, "thread": None}


def steepener(store):
    return store.add("cuts coming", [Leg("SHY", 1000, 82.0, "front"),
                                     Leg("TLT", -100, 88.0, "back")])


# --------------------------------------------------------------------------- resize
def test_resize_scales_every_leg_by_the_same_factor(sleeve):
    v = steepener(sleeve)
    bot.cmd_resize([str(v.id), "x1.5"], ctx())
    legs = {l.symbol: l.quantity for l in sleeve.get(v.id).legs}
    assert legs["SHY"] == pytest.approx(1500)
    assert legs["TLT"] == pytest.approx(-150)


def test_resize_preserves_the_hedge_ratio(sleeve):
    """The reason a resize is safe and an adjust is not."""
    v = steepener(sleeve)
    before = v.legs[0].quantity / v.legs[1].quantity
    bot.cmd_resize([str(v.id), "-40%"], ctx())
    after = sleeve.get(v.id).legs[0].quantity / sleeve.get(v.id).legs[1].quantity
    assert before == pytest.approx(after, rel=1e-6)


@pytest.mark.parametrize("spec,expected", [
    ("x2", 2000), ("+50%", 1500), ("-30%", 700),
])
def test_resize_accepts_factors_and_percentages(sleeve, spec, expected):
    v = steepener(sleeve)
    bot.cmd_resize([str(v.id), spec], ctx())
    assert sleeve.get(v.id).legs[0].quantity == pytest.approx(expected)


def test_resize_to_an_absolute_gross(sleeve):
    v = steepener(sleeve)
    target = v.gross(MARKS) / 2
    bot.cmd_resize([str(v.id), f"{target:.0f}"], ctx())
    assert sleeve.get(v.id).gross(MARKS) == pytest.approx(target, rel=1e-3)


def test_resize_refuses_when_it_would_zero_a_leg(sleeve):
    """Scaling a futures leg to nothing turns a pair into an outright. That is the same
    'different trade, not a smaller one' rule the proposer applies."""
    v = sleeve.add("duration", [Leg("ZN", 1, 107.44, "", 1000.0),
                                Leg("SHY", 1000, 82.0, "")])
    with pytest.raises(ValueError, match="different structure"):
        bot.preview_resize([str(v.id), "x0.4"], ctx())


def test_resize_preview_shows_the_orders_and_sends_nothing(sleeve):
    v = steepener(sleeve)
    text = bot.preview_resize([str(v.id), "x1.5"], ctx())
    assert "ORDERS TO BE PLACED" in text and "hedge ratio is unchanged" in text
    assert sleeve.get(v.id).legs[0].quantity == 1000, "the preview mutated the store"


# --------------------------------------------------------------------------- adjust
def test_adjust_sets_one_leg(sleeve):
    v = steepener(sleeve)
    bot.cmd_adjust([str(v.id), "TLT", "-250"], ctx())
    assert {l.symbol: l.quantity for l in sleeve.get(v.id).legs}["TLT"] == -250


def test_adjust_is_absolute_by_default_even_for_a_short_leg(sleeve):
    """"-250" on a leg already short 100 reads equally as "set to short 250" and "reduce by
    250" — two different positions. Signed-absolute is the reading that matches how legs are
    displayed, and anything relative has to say so explicitly."""
    v = steepener(sleeve)
    bot.cmd_adjust([str(v.id), "TLT", "-250"], ctx())
    assert {l.symbol: l.quantity for l in sleeve.get(v.id).legs}["TLT"] == -250


def test_adjust_nudges_only_with_an_explicit_operator(sleeve):
    v = steepener(sleeve)
    bot.cmd_adjust([str(v.id), "SHY", "+=500"], ctx())
    assert {l.symbol: l.quantity for l in sleeve.get(v.id).legs}["SHY"] == 1500
    bot.cmd_adjust([str(v.id), "SHY", "-=200"], ctx())
    assert {l.symbol: l.quantity for l in sleeve.get(v.id).legs}["SHY"] == 1300


def test_adjust_zero_removes_a_leg(sleeve):
    v = steepener(sleeve)
    bot.cmd_adjust([str(v.id), "TLT", "0"], ctx())
    assert [l.symbol for l in sleeve.get(v.id).legs] == ["SHY"]


def test_adjust_can_add_a_leg_the_view_did_not_hold(sleeve):
    v = steepener(sleeve)
    bot.cmd_adjust([str(v.id), "GLD", "40"], ctx())
    assert "GLD" in {l.symbol for l in sleeve.get(v.id).legs}


def test_adjust_preview_shows_the_hedge_it_is_breaking(sleeve):
    """A manual change is allowed to break the balance — but never silently."""
    v = steepener(sleeve)
    text = bot.preview_adjust([str(v.id), "TLT", "0"], ctx())
    assert "BALANCE" in text and "before" in text and "after" in text
    assert "does not preserve the hedge ratio" in text


def test_adjust_refuses_to_empty_a_worldview(sleeve):
    v = sleeve.add("one leg", [Leg("SHY", 1000, 82.0, "")])
    with pytest.raises(ValueError, match="empty the worldview"):
        bot.preview_adjust([str(v.id), "SHY", "0"], ctx())


# --------------------------------------------------------------------------- cancel
def test_cancel_discards_a_pending_proposal(sleeve):
    class P:
        legs, thesis, reasoning = (), "x", ""
    token = bot._stash(P(), 42)
    assert "discarded" in bot.cmd_cancel([token], ctx())
    assert bot._take(token, 42)[0] is None


def test_cancel_with_no_token_clears_everything_of_yours(sleeve):
    class P:
        legs, thesis, reasoning = (), "x", ""
    bot._stash(P(), 42); bot._stash(P(), 42); bot._stage(P(), 42)
    assert "3 pending" in bot.cmd_cancel([], ctx())
    assert "nothing pending" in bot.cmd_cancel([], ctx())


def test_cancel_leaves_another_users_tokens_alone(sleeve):
    class P:
        legs, thesis, reasoning = (), "x", ""
    mine, theirs = bot._stash(P(), 42), bot._stash(P(), 99)
    bot.cmd_cancel([], ctx())
    assert bot._take(theirs, 99)[0] is not None, "cancelled someone else's proposal"


def test_cancel_ignores_another_bots_token(sleeve):
    """Same rule as /confirm: a bare four-hex token belongs to the executor's control bot."""
    assert bot.cmd_cancel(["a3f9"], ctx()) is None


def test_the_balance_line_does_not_hide_an_off_axis_leg(sleeve):
    """Adding gold to a curve trade leaves net duration-$ unchanged, because GLD has no
    duration. Reporting only duration would print an identical line before and after and
    read as "nothing changed", when a whole new exposure was added."""
    v = steepener(sleeve)
    text = bot.preview_adjust([str(v.id), "GLD", "40"], ctx())
    after = text.split("after")[1].split("\n")[0]
    assert "outside both axes: GLD" in after
    assert "gross" in after
    before_gross = float(text.split("before")[1].split("gross $")[1].split(" ")[0].replace(",", ""))
    after_gross = float(after.split("gross $")[1].split(" ")[0].replace(",", ""))
    assert after_gross > before_gross, "the position grew but the line did not say so"
