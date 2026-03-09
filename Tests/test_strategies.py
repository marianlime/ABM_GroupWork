"""
test_strategies.py
------------------
Unit tests for all strategies in strategies.py.

Run with:  python Tests/test_strategies.py -v
           (from the crutchachos/ root directory)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))

from strategies import (
    zero_intelligence,
    signal_following,
    utility_maximiser,
    contrarian,
    STRATEGIES,
)

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_KEYS = {"Price", "Quantity", "Buy", "Sell", "Hold"}


def assert_valid_order(test_case, order, label=""):
    """Every order dict must have the right keys, non-negative values,
    and a one-hot action encoding (exactly one of Buy/Sell/Hold == 1.0)."""
    test_case.assertEqual(
        set(order.keys()), VALID_KEYS,
        f"{label} unexpected keys: {set(order.keys())}"
    )
    test_case.assertGreaterEqual(order["Price"],    0.0, f"{label} Price must be non-negative")
    test_case.assertGreaterEqual(order["Quantity"], 0.0, f"{label} Quantity must be non-negative")
    one_hot_sum = order["Buy"] + order["Sell"] + order["Hold"]
    test_case.assertAlmostEqual(
        one_hot_sum, 1.0, places=9,
        msg=f"{label} Buy+Sell+Hold must sum to 1, got {one_hot_sum}"
    )


def is_hold(order):  return order["Hold"] == 1.0
def is_buy(order):   return order["Buy"]  == 1.0 and order["Hold"] == 0.0
def is_sell(order):  return order["Sell"] == 1.0 and order["Hold"] == 0.0


# ---------------------------------------------------------------------------
# STRATEGIES registry
# ---------------------------------------------------------------------------

class TestStrategiesRegistry(unittest.TestCase):

    def test_all_strategies_present(self):
        for name in ["zi", "signal_following", "utility_maximiser", "contrarian"]:
            self.assertIn(name, STRATEGIES, f"'{name}' missing from STRATEGIES")

    def test_all_strategies_callable(self):
        for name, fn in STRATEGIES.items():
            self.assertTrue(callable(fn), f"{name} is not callable")


# ---------------------------------------------------------------------------
# zero_intelligence
# ---------------------------------------------------------------------------

class TestZeroIntelligence(unittest.TestCase):

    def test_hold_when_no_cash_no_shares(self):
        order = zero_intelligence(signal=1.0, cash=0.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_buy_only_when_no_shares(self):
        order = zero_intelligence(signal=1.0, cash=1000.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_buy(order) or is_hold(order))

    def test_sell_only_when_no_cash(self):
        order = zero_intelligence(signal=1.0, cash=0.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_sell(order) or is_hold(order))

    def test_price_always_positive(self):
        for _ in range(50):
            order = zero_intelligence(signal=1.0, cash=500.0, shares=5.0, value=100.0)
            assert_valid_order(self, order)
            if not is_hold(order):
                self.assertGreaterEqual(order["Price"], 0.01)

    def test_buy_quantity_within_budget(self):
        cash = 1000.0
        for _ in range(100):
            order = zero_intelligence(signal=1.0, cash=cash, shares=10.0, value=100.0)
            assert_valid_order(self, order)
            if is_buy(order):
                self.assertLessEqual(
                    order["Quantity"] * order["Price"], cash + 1e-6,
                    "Buy order exceeds cash budget"
                )

    def test_sell_quantity_within_shares(self):
        shares = 5.0
        for _ in range(100):
            order = zero_intelligence(signal=1.0, cash=0.0, shares=shares, value=100.0)
            assert_valid_order(self, order)
            if is_sell(order):
                self.assertLessEqual(
                    order["Quantity"], shares + 1e-6,
                    "Sell order exceeds shares held"
                )

    def test_price_roughly_centred_on_value(self):
        """Over many draws the mean price should be within 10% of value."""
        prices = []
        for _ in range(500):
            order = zero_intelligence(signal=1.0, cash=1e9, shares=1e6, value=100.0)
            if not is_hold(order):
                prices.append(order["Price"])
        if prices:
            mean_price = np.mean(prices)
            self.assertLess(
                abs(mean_price - 100.0) / 100.0, 0.10,
                f"Mean ZI price {mean_price:.2f} too far from value 100.0"
            )

    def test_returns_all_keys(self):
        order = zero_intelligence(signal=1.0, cash=500.0, shares=5.0, value=100.0)
        assert_valid_order(self, order)


# ---------------------------------------------------------------------------
# signal_following
# ---------------------------------------------------------------------------

class TestSignalFollowing(unittest.TestCase):

    def test_buys_when_signal_above_one(self):
        order = signal_following(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_buy(order))

    def test_sells_when_signal_below_one(self):
        order = signal_following(signal=0.9, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_sell(order))

    def test_hold_when_no_cash_to_buy(self):
        order = signal_following(signal=1.1, cash=0.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_shares_to_sell(self):
        order = signal_following(signal=0.9, cash=1000.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_signal_clip_caps_buy_price(self):
        order = signal_following(signal=10.0, cash=1000.0, shares=10.0, value=100.0, signal_clip=0.25)
        assert_valid_order(self, order)
        self.assertLessEqual(order["Price"], 100.0 * 1.25 + 1e-6)

    def test_signal_clip_caps_sell_price(self):
        order = signal_following(signal=0.01, cash=1000.0, shares=10.0, value=100.0, signal_clip=0.25)
        assert_valid_order(self, order)
        self.assertGreaterEqual(order["Price"], 100.0 * 0.75 - 1e-6)

    def test_buy_quantity_within_budget(self):
        cash = 1000.0
        order = signal_following(signal=1.2, cash=cash, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        if is_buy(order):
            self.assertLessEqual(order["Quantity"] * order["Price"], cash + 1e-6)

    def test_sell_quantity_within_shares(self):
        shares = 10.0
        order = signal_following(signal=0.8, cash=1000.0, shares=shares, value=100.0)
        assert_valid_order(self, order)
        if is_sell(order):
            self.assertLessEqual(order["Quantity"], shares + 1e-6)

    def test_min_qty_fraction_respected(self):
        min_frac = 0.05
        order = signal_following(signal=1.1, cash=1000.0, shares=10.0, value=100.0,
                                  min_qty_fraction=min_frac)
        assert_valid_order(self, order)
        if is_buy(order):
            max_qty = 1000.0 / order["Price"]
            self.assertGreaterEqual(order["Quantity"], min_frac * max_qty - 1e-9)

    def test_higher_conviction_higher_fraction(self):
        order_low  = signal_following(signal=1.05, cash=1000.0, shares=10.0, value=100.0)
        order_high = signal_following(signal=1.20, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order_low)
        assert_valid_order(self, order_high)
        if is_buy(order_low) and is_buy(order_high):
            frac_low  = order_low["Quantity"]  / (1000.0 / order_low["Price"])
            frac_high = order_high["Quantity"] / (1000.0 / order_high["Price"])
            self.assertGreaterEqual(frac_high, frac_low - 1e-9)


# ---------------------------------------------------------------------------
# utility_maximiser
# ---------------------------------------------------------------------------

class TestUtilityMaximiser(unittest.TestCase):

    def test_buys_when_signal_above_one(self):
        order = utility_maximiser(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_buy(order))

    def test_sells_when_signal_below_one(self):
        order = utility_maximiser(signal=0.9, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_sell(order))

    def test_hold_when_signal_exactly_one(self):
        order = utility_maximiser(signal=1.0, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_cash(self):
        order = utility_maximiser(signal=1.1, cash=0.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_shares(self):
        order = utility_maximiser(signal=0.9, cash=1000.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_higher_risk_aversion_smaller_quantity(self):
        order_low_ra  = utility_maximiser(signal=1.15, cash=1000.0, shares=10.0,
                                           value=100.0, risk_aversion=1.0)
        order_high_ra = utility_maximiser(signal=1.15, cash=1000.0, shares=10.0,
                                           value=100.0, risk_aversion=5.0)
        assert_valid_order(self, order_low_ra)
        assert_valid_order(self, order_high_ra)
        if is_buy(order_low_ra) and is_buy(order_high_ra):
            self.assertLessEqual(order_high_ra["Quantity"], order_low_ra["Quantity"] + 1e-9)

    def test_buy_quantity_within_budget(self):
        cash = 1000.0
        order = utility_maximiser(signal=1.2, cash=cash, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        if is_buy(order):
            self.assertLessEqual(order["Quantity"] * order["Price"], cash + 1e-6)

    def test_sell_quantity_within_shares(self):
        shares = 10.0
        order = utility_maximiser(signal=0.8, cash=1000.0, shares=shares, value=100.0)
        assert_valid_order(self, order)
        if is_sell(order):
            self.assertLessEqual(order["Quantity"], shares + 1e-6)

    def test_signal_clip_applied(self):
        order = utility_maximiser(signal=100.0, cash=1000.0, shares=10.0, value=100.0, signal_clip=0.25)
        assert_valid_order(self, order)
        self.assertLessEqual(order["Price"], 100.0 * 1.25 + 1e-6)

    def test_returns_all_keys(self):
        order = utility_maximiser(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)


# ---------------------------------------------------------------------------
# contrarian
# ---------------------------------------------------------------------------

class TestContrarian(unittest.TestCase):

    def test_sells_when_signal_above_one(self):
        order = contrarian(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_sell(order))

    def test_buys_when_signal_below_one(self):
        order = contrarian(signal=0.9, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_buy(order))

    def test_opposite_direction_to_signal_following(self):
        for sig in [0.8, 0.95, 1.05, 1.2]:
            sf = signal_following(signal=sig, cash=1000.0, shares=10.0, value=100.0)
            ct = contrarian(signal=sig,       cash=1000.0, shares=10.0, value=100.0)
            assert_valid_order(self, sf)
            assert_valid_order(self, ct)
            if is_buy(sf):
                self.assertTrue(is_sell(ct), f"signal={sig}: sf buys but contrarian doesn't sell")
            elif is_sell(sf):
                self.assertTrue(is_buy(ct),  f"signal={sig}: sf sells but contrarian doesn't buy")

    def test_hold_when_no_shares_to_sell(self):
        order = contrarian(signal=1.1, cash=1000.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_cash_to_buy(self):
        order = contrarian(signal=0.9, cash=0.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_sell_quantity_within_shares(self):
        shares = 10.0
        order = contrarian(signal=1.2, cash=1000.0, shares=shares, value=100.0)
        assert_valid_order(self, order)
        if is_sell(order):
            self.assertLessEqual(order["Quantity"], shares + 1e-6)

    def test_buy_quantity_within_budget(self):
        cash = 1000.0
        order = contrarian(signal=0.8, cash=cash, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        if is_buy(order):
            self.assertLessEqual(order["Quantity"] * order["Price"], cash + 1e-6)

    def test_signal_clip_applied(self):
        order = contrarian(signal=0.01, cash=1000.0, shares=10.0, value=100.0, signal_clip=0.25)
        assert_valid_order(self, order)
        self.assertGreaterEqual(order["Price"], 0.01)

    def test_returns_all_keys(self):
        order = contrarian(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
