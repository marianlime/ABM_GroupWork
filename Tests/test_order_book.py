# Run via pytest in terminal

import pytest
from sim.market import clear_market, allocate_trades

def _side_fills(trades, action):
    """Return {agent_id: quantity} for one side of the trades list."""
    return {t['agent_id']: t['quantity'] for t in trades if t['action'] == action}

# Basic clearing

class TestBasicClearing:

    ORDERS = [
        {'agent_id': 1, 'price': 100, 'quantity': 10, 'action': 'buy'},
        {'agent_id': 2, 'price':  99, 'quantity': 15, 'action': 'buy'},
        {'agent_id': 3, 'price': 101, 'quantity':  8, 'action': 'sell'},
        {'agent_id': 4, 'price': 100, 'quantity': 12, 'action': 'sell'},
    ]

    def test_clearing_price(self):
        price, _, _ = clear_market(self.ORDERS, previous_price=99.5)
        assert price == 100.0

    def test_trades_non_empty(self):
        _, _, trades = clear_market(self.ORDERS, previous_price=99.5)
        assert len(trades) > 0

    def test_both_sides_present(self):
        _, _, trades = clear_market(self.ORDERS, previous_price=99.5)
        sides = {t['action'] for t in trades}
        assert 'buy' in sides and 'sell' in sides


# No trade happens when buyers price is below the sellers price

class TestNoMatch:

    ORDERS = [
        {'agent_id': 1, 'price':  90, 'quantity': 10, 'action': 'buy'},
        {'agent_id': 2, 'price': 100, 'quantity': 10, 'action': 'sell'},
    ]

    def test_price_is_none(self):
        price, _, _ = clear_market(self.ORDERS)
        assert price is None

    def test_trades_empty(self):
        _, _, trades = clear_market(self.ORDERS)
        assert trades == []


# One buyer and one seller at the same price - both get fully filled

class TestExactMatch:

    ORDERS = [
        {'agent_id': 'A', 'price': 50, 'quantity': 5, 'action': 'buy'},
        {'agent_id': 'B', 'price': 50, 'quantity': 5, 'action': 'sell'},
    ]

    def test_clearing_price(self):
        price, _, _ = clear_market(self.ORDERS)
        assert price == 50.0

    def test_buyer_fully_filled(self):
        _, _, trades = clear_market(self.ORDERS)
        assert _side_fills(trades, 'buy')['A'] == 5

    def test_seller_fully_filled(self):
        _, _, trades = clear_market(self.ORDERS)
        assert _side_fills(trades, 'sell')['B'] == 5


# No orders / nothing to trade

class TestEdgeCases:

    def test_empty_orders(self):
        price, _, trades = clear_market([])
        assert price is None and trades == []

    def test_buys_only(self):
        orders = [{'agent_id': 1, 'price': 100, 'quantity': 5, 'action': 'buy'}]
        price, _, _ = clear_market(orders)
        assert price is None

    def test_sells_only(self):
        orders = [{'agent_id': 1, 'price': 100, 'quantity': 5, 'action': 'sell'}]
        price, _, _ = clear_market(orders)
        assert price is None


# Tie-breaking rules

class TestTieBreaking:
    """
    Prices 100 and 101 both yield the same volume and imbalance.
    previous_price should select the closer candidate.
    With no previous_price, midpoint (100.5) is chosen.
    """

    ORDERS = [
        {'agent_id': 1, 'price': 101, 'quantity': 10, 'action': 'buy'},
        {'agent_id': 2, 'price': 100, 'quantity': 10, 'action': 'sell'},
    ]

    def test_closest_to_previous_high(self):
        price, _, _ = clear_market(self.ORDERS, previous_price=101)
        assert price == 101.0

    def test_closest_to_previous_low(self):
        price, _, _ = clear_market(self.ORDERS, previous_price=100)
        assert price == 100.0

    def test_midpoint_when_no_previous(self):
        price, _, _ = clear_market(self.ORDERS, previous_price=None)
        assert price == 100.5


# Pro-rata rationing via allocate_trades

class TestAllocateTrades:
    """
    Supply = 15; two buyers compete with quantities 20 and 10.
    The larger buyer must receive a proportionally larger fill, and
    total buys must not exceed available supply.
    """

    BUYS = [
        {'agent_id': 'big',   'price': 100, 'quantity': 20, 'action': 'buy'},
        {'agent_id': 'small', 'price': 100, 'quantity': 10, 'action': 'buy'},
    ]
    SELLS = [
        {'agent_id': 'seller', 'price': 100, 'quantity': 15, 'action': 'sell'},
    ]

    def test_big_gets_more_than_small(self):
        trades = allocate_trades(self.BUYS, self.SELLS, price=100, total_volume=15)
        buys = _side_fills(trades, 'buy')
        assert buys['big'] > buys['small']

    def test_total_buys_within_supply(self):
        trades = allocate_trades(self.BUYS, self.SELLS, price=100, total_volume=15)
        total_bought = sum(t['quantity'] for t in trades if t['action'] == 'buy')
        assert total_bought <= 15
