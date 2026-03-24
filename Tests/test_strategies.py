"""
test_strategies.py
------------------
Unit tests for strategies in sim/strategies.py.

Run with:  python Tests/test_strategies.py -v
           (from the project root directory)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))

from strategies import zero_intelligence, parameterised_informed, STRATEGIES

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_KEYS = {"Price", "Quantity", "Buy", "Sell", "Hold"}


def assert_valid_order(test_case, order, label=""):
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

    def test_parameterised_informed_present(self):
        self.assertIn("parameterised_informed", STRATEGIES)

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
                self.assertLessEqual(order["Quantity"] * order["Price"], cash + 1e-6)

    def test_sell_quantity_within_shares(self):
        shares = 5.0
        for _ in range(100):
            order = zero_intelligence(signal=1.0, cash=0.0, shares=shares, value=100.0)
            assert_valid_order(self, order)
            if is_sell(order):
                self.assertLessEqual(order["Quantity"], shares + 1e-6)

    def test_returns_all_keys(self):
        order = zero_intelligence(signal=1.0, cash=500.0, shares=5.0, value=100.0)
        assert_valid_order(self, order)


# ---------------------------------------------------------------------------
# parameterised_informed
# ---------------------------------------------------------------------------

class TestParameterisedInformed(unittest.TestCase):

    def test_buys_when_signal_above_one(self):
        order = parameterised_informed(signal=1.1, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_buy(order))

    def test_sells_when_signal_below_one(self):
        order = parameterised_informed(signal=0.9, cash=1000.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_sell(order))

    def test_hold_within_threshold(self):
        order = parameterised_informed(signal=1.01, cash=1000.0, shares=10.0, value=100.0,
                                       threshold=0.05, info_param=1.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_cash_to_buy(self):
        order = parameterised_informed(signal=1.1, cash=0.0, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_hold_when_no_shares_to_sell(self):
        order = parameterised_informed(signal=0.9, cash=1000.0, shares=0.0, value=100.0)
        assert_valid_order(self, order)
        self.assertTrue(is_hold(order))

    def test_buy_quantity_within_budget(self):
        cash = 1000.0
        order = parameterised_informed(signal=1.2, cash=cash, shares=10.0, value=100.0)
        assert_valid_order(self, order)
        if is_buy(order):
            self.assertLessEqual(order["Quantity"] * order["Price"], cash + 1e-6)

    def test_sell_quantity_within_shares(self):
        shares = 10.0
        order = parameterised_informed(signal=0.8, cash=1000.0, shares=shares, value=100.0)
        assert_valid_order(self, order)
        if is_sell(order):
            self.assertLessEqual(order["Quantity"], shares + 1e-6)

    def test_higher_aggression_higher_fraction(self):
        order_lo = parameterised_informed(signal=1.2, cash=1000.0, shares=10.0, value=100.0,
                                          aggression=0.5)
        order_hi = parameterised_informed(signal=1.2, cash=1000.0, shares=10.0, value=100.0,
                                          aggression=3.0)
        assert_valid_order(self, order_lo)
        assert_valid_order(self, order_hi)
        if is_buy(order_lo) and is_buy(order_hi):
            self.assertGreaterEqual(order_hi["Quantity"], order_lo["Quantity"] - 1e-9)

    def test_signal_clip_applied(self):
        order = parameterised_informed(signal=10.0, cash=1000.0, shares=10.0, value=100.0,
                                       signal_clip=0.25)
        assert_valid_order(self, order)
        self.assertLessEqual(order["Price"], 100.0 * 1.25 + 1e-6)

    def test_returns_all_keys(self):
        order = parameterised_informed(signal=1.1, cash=500.0, shares=5.0, value=100.0)
        assert_valid_order(self, order)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
