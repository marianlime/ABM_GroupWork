import pytest
from sim.trader import Trader


def _is_valid(order):
    """Check order dict has correct keys and one-hot encoding."""
    needed = {"Price", "Quantity", "Buy", "Sell", "Hold"}
    return needed.issubset(order.keys()) and abs(order["Buy"] + order["Sell"] + order["Hold"] - 1.0) < 1e-9


def test_defaults():
    t = Trader(agent_id=1, cash=1000, shares=10)
    assert t.trader_type == "zi" and t.info_param == 0.1


def test_custom_type():
    t = Trader(agent_id=2, cash=500, shares=5, trader_type="parameterised_informed")
    assert t.trader_type == "parameterised_informed"


def test_strategy_params_default_empty():
    t = Trader(agent_id=3, cash=100, shares=1)
    assert t.strategy_params == {}


def test_informed_buys_on_high_signal():
    t = Trader(agent_id=1, cash=1000, shares=10, trader_type="parameterised_informed",
               strategy_params={"direction_bias": 1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.0})
    order = t.place_order(signal=1.1, value=100)
    assert order is not None and _is_valid(order) and order["Buy"] == 1.0


def test_informed_sells_on_low_signal():
    t = Trader(agent_id=2, cash=1000, shares=10, trader_type="parameterised_informed",
               strategy_params={"direction_bias": 1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.0})
    order = t.place_order(signal=0.9, value=100)
    assert order is not None and _is_valid(order) and order["Sell"] == 1.0


def test_contrarian_bias_reverses_direction():
    t = Trader(agent_id=3, cash=1000, shares=10, trader_type="parameterised_informed",
               strategy_params={"direction_bias": -1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.0})
    order = t.place_order(signal=1.1, value=100)
    assert order is not None and _is_valid(order) and order["Sell"] == 1.0


def test_order_has_agent_id():
    t = Trader(agent_id=42, cash=1000, shares=10, trader_type="parameterised_informed",
               strategy_params={"direction_bias": 1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.0})
    order = t.place_order(signal=1.1, value=100)
    assert order["agent_id"] == 42


def test_hold_when_no_resources():
    t = Trader(agent_id=5, cash=0, shares=0, trader_type="parameterised_informed",
               strategy_params={"direction_bias": 1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.0})
    order = t.place_order(signal=1.1, value=100)
    assert order is not None and order["Hold"] == 1.0


def test_threshold_causes_hold():
    t = Trader(agent_id=6, cash=1000, shares=10, trader_type="parameterised_informed",
               strategy_params={"direction_bias": 1.0, "aggression": 1.0, "patience": 1.0, "threshold": 0.10})
    order = t.place_order(signal=1.05, value=100)
    assert order is not None and order["Hold"] == 1.0


def test_zi_fallback():
    t = Trader(agent_id=9, cash=1000, shares=10, trader_type="zi")
    order = t.place_order(signal=1.0, value=100)
    assert order is not None and _is_valid(order)
