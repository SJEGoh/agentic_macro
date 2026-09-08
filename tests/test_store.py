"""tests/test_store.py — netting many worldviews into one authoritative book.

The book sent to `/targets` is absolute: any name it omits is CLOSED. Everything worth
pinning here follows from that, and all of it fails silently rather than loudly:

  * closing one worldview must unwind exactly ITS legs. If two views both hold TLT and
    closing one flattened the symbol, the surviving view would be silently de-risked and
    nothing would say so;
  * a symbol that nets to zero must be dropped rather than sent as quantity 0 — both close
    the position, but only one keeps the book honest about what is held;
  * a view must be written whole or not at all. A half-written view nets into a book nobody
    approved, and the next submission would make it real;
  * a closed view must stop counting immediately — if it lingered in the netting, its
    positions would never come off.
"""
import pytest

from agentic_macro.store import Leg, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def legs(*pairs):
    return [Leg(sym, qty, 100.0, "") for sym, qty in pairs]


def test_two_views_holding_the_same_name_are_summed(store):
    store.add("rates fall", legs(("TLT", 200)))
    store.add("duration bid", legs(("TLT", 150), ("SPY", -50)))
    assert store.net_positions() == {"TLT": 350.0, "SPY": -50.0}


def test_closing_one_view_leaves_the_others_position_intact(store):
    """The failure this pins: closing #1 flattening TLT entirely, silently removing the
    exposure #2 still wants."""
    a = store.add("rates fall", legs(("TLT", 200)))
    store.add("duration bid", legs(("TLT", 150)))
    store.close_worldview(a.id)
    assert store.net_positions() == {"TLT": 150.0}


def test_closing_the_last_holder_removes_the_name_entirely(store):
    """Omission is what closes a position, so the symbol must disappear from the book."""
    a = store.add("rates fall", legs(("TLT", 200)))
    store.close_worldview(a.id)
    assert store.net_positions() == {}


def test_offsetting_views_net_to_nothing_and_are_dropped(store):
    store.add("long duration", legs(("TLT", 200)))
    store.add("short duration", legs(("TLT", -200)))
    assert "TLT" not in store.net_positions()


def test_a_closed_view_stops_counting_at_once(store):
    a = store.add("x", legs(("SPY", 100)))
    assert store.active() and store.net_positions()
    store.close_worldview(a.id)
    assert store.active() == []
    assert store.net_positions() == {}
    assert store.get(a.id).status == "closed"


def test_closing_twice_is_refused(store):
    a = store.add("x", legs(("SPY", 100)))
    store.close_worldview(a.id)
    with pytest.raises(ValueError, match="already closed"):
        store.close_worldview(a.id)


def test_closing_an_unknown_view_is_refused(store):
    with pytest.raises(KeyError):
        store.close_worldview(999)


def test_extra_legs_are_costed_without_being_written(store):
    """How a proposal is checked against the book it would join. It must not persist —
    a rejected proposal that left rows behind would be traded on the next /sync."""
    store.add("held", legs(("SPY", 100)))
    combined = store.net_positions(extra=legs(("SPY", 50), ("GLD", 10)))
    assert combined == {"SPY": 150.0, "GLD": 10.0}
    assert store.net_positions() == {"SPY": 100.0}


def test_holders_says_which_views_own_a_name(store):
    a = store.add("one", legs(("TLT", 200)))
    b = store.add("two", legs(("TLT", -50)))
    assert sorted(store.holders("TLT")) == sorted([(a.id, 200.0), (b.id, -50.0)])


def test_a_view_survives_reopening_the_database(tmp_path):
    """Positions outlive the process. A view lost on restart would leave holdings at the
    broker that the next netted book closes without explanation."""
    path = tmp_path / "persist.db"
    first = Store(path)
    view = first.add("durable", legs(("GLD", 40)), reasoning="because", created_by="me")
    first.close()

    second = Store(path)
    reloaded = second.get(view.id)
    assert reloaded.thesis == "durable"
    assert reloaded.reasoning == "because"
    assert reloaded.legs[0].symbol == "GLD"
    assert second.net_positions() == {"GLD": 40.0}
    second.close()


def test_gross_counts_both_sides_of_a_pair(store):
    """A long/short pair consumes capital on both legs. Netting them to zero would report a
    fully invested pair trade as free and let the sleeve take on unlimited risk."""
    view = store.add("pair", [Leg("SPY", 100, 600.0, ""), Leg("TLT", -100, 88.0, "")])
    assert view.gross() == pytest.approx(100 * 600.0 + 100 * 88.0)


def test_gross_uses_current_marks_when_given(store):
    view = store.add("x", [Leg("SPY", 10, 600.0, "")])
    assert view.gross({"SPY": 700.0}) == pytest.approx(7_000.0)
    assert view.gross() == pytest.approx(6_000.0)
