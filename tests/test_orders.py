"""tests/test_orders.py — the last gate: what the broker is actually told to do.

Approving a worldview is not approving an order, and conflating the two is the failure this
module exists to prevent. The legs say what a view wants to HOLD; the order is the delta
against what the broker holds now, netted with every other active view. Everything pinned
here is a case where those two numbers differ and the difference is invisible:

  * a second view wanting the same name makes the order SMALLER than the legs — approving
    the legs would have you expecting to trade 1,099 shares when 300 will be sent;
  * closing a view that shares a name with another must sell only its own share, not flatten
    the symbol out from under the view that still wants it;
  * a book the broker already matches must produce NO orders, not a re-send of every
    position — otherwise /sync looks like it is trading when it is doing nothing;
  * a preview must not write anything. A previewed-then-declined change that left rows in
    the store would be traded by the next /sync.
"""
import pytest

from agentic_macro import config, orders
from agentic_macro.store import Leg, Store


class FakeClient:
    """Stands in for the executor's /positions."""

    def __init__(self, held=None):
        self._held = held or {}

    def positions(self):
        return {"strategy_positions": {config.STRATEGY_ID: dict(self._held)}}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "orders.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def live(monkeypatch):
    """These tests are about the executor path; paper mode has its own section below."""
    monkeypatch.setattr(config, "PAPER", False)


def legs(*pairs):
    return [Leg(sym, qty, 100.0, "") for sym, qty in pairs]


def by_symbol(rows):
    return {r.symbol: r for r in rows}


# --------------------------------------------------------------------------- deltas
def test_a_new_view_against_a_flat_broker_orders_the_whole_leg(store):
    """The simple case, so the interesting ones below have a baseline to differ from."""
    client = FakeClient({})
    rows = by_symbol(orders.deltas(client, store, extra=legs(("SHY", 1099), ("TLT", -111))))
    assert rows["SHY"].delta == 1099 and rows["SHY"].side == "BUY"
    assert rows["TLT"].delta == -111 and rows["TLT"].side == "SELL"


def test_an_existing_position_makes_the_order_smaller_than_the_legs(store):
    """The headline case. The view wants 1,099 SHY; the broker already holds 800 because
    another worldview bought it, so the ORDER is 299 — not the 1,099 on the proposal."""
    store.add("older view", legs(("SHY", 800)))
    client = FakeClient({"SHY": 800})

    rows = by_symbol(orders.deltas(client, store, extra=legs(("SHY", 299))))
    assert rows["SHY"].target == 1099
    assert rows["SHY"].current == 800
    assert rows["SHY"].delta == 299
    assert rows["SHY"].side == "BUY"


def test_a_view_can_net_to_a_sell_despite_being_long(store):
    """A long leg that reduces an existing larger long is a SELL. Reading the legs alone
    would have you expecting to buy."""
    store.add("big long", legs(("TLT", 500)))
    client = FakeClient({"TLT": 500})

    rows = by_symbol(orders.deltas(client, store, extra=legs(("TLT", -300))))
    assert rows["TLT"].side == "SELL"
    assert rows["TLT"].delta == -300
    assert rows["TLT"].target == 200


def test_a_name_the_broker_already_matches_produces_no_order(store):
    store.add("held", legs(("SHY", 500)))
    client = FakeClient({"SHY": 500})
    assert orders.deltas(client, store) == []


def test_drift_at_the_broker_shows_up_as_an_order(store):
    """A partial fill left the broker short of the book. /sync exists to heal exactly this,
    and the preview has to show it rather than reporting "no orders"."""
    store.add("held", legs(("SHY", 500)))
    client = FakeClient({"SHY": 460})

    rows = by_symbol(orders.deltas(client, store))
    assert rows["SHY"].delta == 40


# --------------------------------------------------------------------------- closing
def test_closing_a_view_sells_only_its_own_share(store):
    """#1 and #2 both hold TLT. Closing #1 must sell 200, leaving #2's 150 — not flatten
    the symbol and silently de-risk a view nobody closed."""
    a = store.add("one", legs(("TLT", 200)))
    store.add("two", legs(("TLT", 150)))
    client = FakeClient({"TLT": 350})

    rows = by_symbol(orders.deltas(client, store, exclude=a.id))
    assert rows["TLT"].delta == -200
    assert rows["TLT"].target == 150
    assert not rows["TLT"].closes


def test_closing_the_last_holder_closes_the_position(store):
    a = store.add("only", legs(("XLE", 400)))
    client = FakeClient({"XLE": 400})

    rows = by_symbol(orders.deltas(client, store, exclude=a.id))
    assert rows["XLE"].target == 0
    assert rows["XLE"].closes


def test_a_preview_writes_nothing(store):
    """A previewed-then-declined change must leave no trace, or the next /sync trades it."""
    store.add("held", legs(("SHY", 100)))
    client = FakeClient({"SHY": 100})
    before = store.net_positions()

    orders.deltas(client, store, extra=legs(("TLT", -50)))
    orders.deltas(client, store, exclude=1)

    assert store.net_positions() == before
    assert len(store.active()) == 1


def test_deltas_only_read_this_strategys_positions(store):
    """Differencing against the whole account would have the sleeve trying to trade its way
    to owning every other strategy's book."""
    store.add("mine", legs(("SPY", 100)))
    client = FakeClient({"SPY": 100})
    client.positions = lambda: {
        "current_positions": {"SPY": 9999, "AAPL": 500},
        "strategy_positions": {config.STRATEGY_ID: {"SPY": 100}, "other_strat": {"AAPL": 500}},
    }
    assert orders.deltas(client, store) == []


# --------------------------------------------------------------------------- rendering
def test_the_render_names_sides_sizes_and_the_resulting_position(store):
    store.add("one", legs(("TLT", 200)))
    client = FakeClient({})
    text = orders.render(orders.deltas(client, store), {"TLT": 88.0})
    assert "BUY" in text and "TLT" in text and "200" in text
    assert "0 -> +200" in text.replace("+0", "0")
    assert "$17,600" in text          # 200 * 88


def test_an_empty_order_list_says_so_rather_than_rendering_nothing():
    text = orders.render([])
    assert "NO ORDERS" in text


# --------------------------------------------------------------------------- paper mode
def test_paper_mode_treats_the_stored_book_as_the_broker(store, monkeypatch):
    """With no executor, the stored book IS what would be held. Without this the preview
    would show every existing position as a fresh trade on every single command."""
    monkeypatch.setattr(config, "PAPER", True)
    store.add("held", legs(("SHY", 500)))

    client = FakeClient({"SHY": 99999})      # must be ignored entirely
    assert orders.deltas(client, store) == []

    rows = by_symbol(orders.deltas(client, store, extra=legs(("SHY", 100))))
    assert rows["SHY"].current == 500 and rows["SHY"].delta == 100


def test_paper_mode_never_calls_the_executor(store, monkeypatch):
    monkeypatch.setattr(config, "PAPER", True)

    class Exploding:
        def positions(self):
            raise AssertionError("the executor was contacted in paper mode")

    store.add("held", legs(("SHY", 500)))
    orders.deltas(Exploding(), store, extra=legs(("TLT", -10)))
