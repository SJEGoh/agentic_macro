"""tests/test_flow.py — worldview in, netted book out.

The pieces are tested separately elsewhere; this pins the seams between them, where the
failures are the quiet kind:

  * the approval message must show the SIGNED direction and the real share counts. It is the
    only thing a human reads before real money moves, so a rendering that dropped a short
    sign or showed the pre-rounding notional would be approving a trade nobody saw;
  * what is stored must be what was proposed. If `store.add` reshaped or re-sized the legs,
    the book would diverge from the approval on the very first submission;
  * a second worldview must ADD to the netted book rather than replace it — the sleeve holds
    many views at once, and `/targets` closes anything the book omits.
"""
import json

import pytest

from agentic_macro import proposer, universe
from agentic_macro.store import Leg, Store
from agentic_macro.bot import _render

PRICES = {"SHY": 82.0, "TLT": 88.0, "XLE": 90.0}

STEEPENER = {"expressible": True, "structure": "steepener", "weighting": "dv01_neutral",
             "reasoning": "cuts front-load the front end", "confidence": "high",
             "conflicts": "",
             "legs": [{"symbol": "SHY", "direction": "long", "weight": 1,
                       "rationale": "front end"},
                      {"symbol": "TLT", "direction": "short", "weight": 1,
                       "rationale": "long end"}]}

ENERGY = {"expressible": True, "structure": "energy_supply_shock", "weighting": "directional",
          "reasoning": "supply is tight", "confidence": "medium", "conflicts": "",
          "legs": [{"symbol": "XLE", "direction": "long", "weight": 1,
                    "rationale": "producers"}]}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "flow.db")
    yield s
    s.close()


def test_a_worldview_becomes_a_stored_view_and_a_netted_book(store):
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    view = store.add(proposal.thesis, proposal.legs, reasoning=proposal.reasoning,
                     created_by="test")

    # what was stored is exactly what was proposed
    stored = {l.symbol: l.quantity for l in view.legs}
    assert stored == {l.symbol: l.quantity for l in proposal.legs}
    assert store.net_positions() == stored
    assert stored["SHY"] > 0 and stored["TLT"] < 0


def test_a_second_worldview_adds_to_the_book_rather_than_replacing_it(store):
    store.add("cuts coming", proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0).legs)
    store.add("oil tight", proposer.size("oil tight", ENERGY, PRICES, 50_000.0).legs)

    book = store.net_positions()
    assert set(book) == {"SHY", "TLT", "XLE"}, "an earlier view's legs went missing"
    assert book["XLE"] == pytest.approx(50_000.0 / 90.0, rel=1e-3)


def test_closing_one_of_two_views_leaves_the_other_in_the_book(store):
    a = store.add("cuts coming", proposer.size("c", STEEPENER, PRICES, 100_000.0).legs)
    store.add("oil tight", proposer.size("o", ENERGY, PRICES, 50_000.0).legs)
    store.close_worldview(a.id)
    assert set(store.net_positions()) == {"XLE"}


def test_the_approval_message_shows_direction_size_and_the_hedge_ratio():
    """This text is the entire basis on which a human approves real orders."""
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    text = _render(proposal, allocation=250_000.0, capital_used=0.0, token="ab12")

    assert "LONG  SHY" in text and "SHORT TLT" in text
    for leg in proposal.legs:
        # the rendered size must be the ACTUAL size, fraction and all — a %.0f here would
        # print 1,100 for a 1,099.85-share leg, which is both wrong and invisible
        assert universe.qty(abs(leg.quantity)) in text
    assert "dv01_neutral" in text                 # the weighting is not hidden
    assert "dollars long:short" in text           # the ratio is stated, not implied
    assert "/confirm ab12" in text
    assert "confidence high" in text


def test_an_inexpressible_view_renders_as_a_refusal_with_no_token():
    raw = {"expressible": False, "structure": "custom", "weighting": "equal_notional",
           "reasoning": "this needs options", "confidence": "low", "conflicts": "",
           "legs": []}
    text = _render(proposer.size("something odd", raw, {}, 100_000.0),
                   allocation=250_000.0, capital_used=0.0, token="")
    assert "NOT EXPRESSIBLE" in text
    assert "options" in text
    assert "/confirm" not in text, "a refusal must not offer anything to approve"


def test_the_message_warns_about_conflicts_with_what_is_held():
    raw = dict(STEEPENER, conflicts="#2 already holds long SHY — this doubles the front end")
    text = _render(proposer.size("cuts", raw, PRICES, 100_000.0), 250_000.0, 40_000.0, "x1")
    assert "AGAINST WHAT YOU ALREADY HOLD" in text
    assert "doubles the front end" in text


# --------------------------------------------------------------------------- the two gates
class FakeExecutor:
    def __init__(self, held=None):
        self._held = held or {}

    def positions(self):
        from agentic_macro import config as _c
        return {"strategy_positions": {_c.STRATEGY_ID: dict(self._held)}}


@pytest.fixture
def paper(tmp_path, monkeypatch):
    """A sleeve on paper, wired to a fresh store, with prices stubbed."""
    from agentic_macro import bot, config, orders, prices as _prices
    s = Store(tmp_path / "gates.db")
    monkeypatch.setattr(bot, "store", s)
    monkeypatch.setattr(config, "PAPER", True)
    monkeypatch.setattr(config, "allowed_users", lambda: {42})
    monkeypatch.setattr(_prices, "fetch", lambda syms: {x: PRICES.get(x, 100.0) for x in syms})
    monkeypatch.setattr(orders, "broker_positions",
                        lambda client, store=None, strategy_id=None: dict(s.net_positions()))
    yield s
    s.close()


def _ctx():
    return {"user": "sjegoh", "user_id": 42, "thread": None}


def test_confirm_shows_orders_and_sends_nothing(paper):
    """The whole point of the split: /confirm must not place anything."""
    from agentic_macro import bot
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    token = bot._stash(proposal, 42)

    reply = bot.cmd_confirm([token], _ctx())

    assert "ORDERS TO BE PLACED" in reply
    assert "/place " in reply
    assert paper.active() == [], "the worldview was stored before it was placed"
    assert paper.net_positions() == {}, "positions appeared before /place"


def test_place_is_what_actually_commits(paper):
    from agentic_macro import bot
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    confirm_reply = bot.cmd_confirm([bot._stash(proposal, 42)], _ctx())
    place_token = confirm_reply.split("/place ")[1].split()[0]

    reply = bot.cmd_place([place_token], _ctx())

    assert "is on" in reply
    assert len(paper.active()) == 1
    assert set(paper.net_positions()) == {"SHY", "TLT"}


def test_a_confirm_token_cannot_place_orders(paper):
    """Tokens are scoped to the step they were issued for, so a /confirm token replayed at
    the final gate cannot skip the order review."""
    from agentic_macro import bot
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    token = bot._stash(proposal, 42)

    assert "nothing staged" in bot.cmd_place([token], _ctx())
    assert paper.active() == []


def test_a_place_token_is_single_use(paper):
    from agentic_macro import bot
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    reply = bot.cmd_confirm([bot._stash(proposal, 42)], _ctx())
    token = reply.split("/place ")[1].split()[0]

    bot.cmd_place([token], _ctx())
    assert "nothing staged" in bot.cmd_place([token], _ctx())
    assert len(paper.active()) == 1, "the worldview was placed twice"


def test_the_order_list_reflects_netting_not_the_legs(paper):
    """A view whose legs read 'long XLE 555' is a much smaller order when another view
    already holds most of it. This is what /confirm now shows and /worldview cannot."""
    from agentic_macro import bot
    paper.add("older", [Leg("XLE", 500, 90.0, "")])
    proposal = proposer.size("oil tight", ENERGY, PRICES, 50_000.0)   # 555 XLE
    reply = bot.cmd_confirm([bot._stash(proposal, 42)], _ctx())

    assert "BUY" in reply and "XLE" in reply
    assert "BUY" in reply and "XLE" in reply
    assert "+500 ->" in reply, reply          # netted against the view already holding 500


def test_paper_mode_is_stated_on_every_gate(paper):
    from agentic_macro import bot
    proposal = proposer.size("cuts coming", STEEPENER, PRICES, 100_000.0)
    confirm_reply = bot.cmd_confirm([bot._stash(proposal, 42)], _ctx())
    assert "PAPER MODE" in confirm_reply

    token = confirm_reply.split("/place ")[1].split()[0]
    assert "PAPER MODE" in bot.cmd_place([token], _ctx())
    assert "PAPER MODE" in bot.cmd_help([], _ctx())
