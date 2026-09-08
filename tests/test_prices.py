"""tests/test_prices.py — expected_price fails closed, or it mis-sizes the order.

The executor values every leg with `expected_price` to apply the allocation cap, and
measures fill slippage against it. So a bad price does not produce a rejected order — it
produces an ACCEPTED one of the wrong size, with the limit meant to contain it computed
from the same wrong number. Nothing downstream can detect that.

These pin the cases where returning a plausible default would be worse than raising.
"""
import math

import pytest

from agentic_macro import prices


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), None, "", "abc"])
def test_every_unusable_price_raises_rather_than_defaulting(bad):
    with pytest.raises(prices.PriceUnavailable):
        prices._check("TLT", bad)


def test_a_good_price_passes_through_as_a_float():
    assert prices._check("TLT", "88.25") == 88.25
    assert prices._check("TLT", 88) == 88.0


def test_drift_is_signed_and_fractional():
    moved = prices.drift({"TLT": 100.0, "SPY": 200.0}, {"TLT": 102.0, "SPY": 196.0})
    assert moved["TLT"] == pytest.approx(0.02)
    assert moved["SPY"] == pytest.approx(-0.02)


def test_drift_ignores_symbols_missing_from_either_side():
    """A symbol that could not be re-priced must not silently read as zero drift — it is
    simply absent, and the caller's own all-or-nothing fetch is what catches it."""
    assert prices.drift({"TLT": 100.0, "GONE": 50.0}, {"TLT": 100.0}) == {"TLT": 0.0}


def test_no_symbols_needs_no_network():
    assert prices.fetch([]) == {}
    assert prices.fetch([None, "", "  "]) == {}
