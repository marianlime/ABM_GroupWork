"""
Baseline Trading Strategies for Double Auction ABM
Usage:
    trader = Trader(uaid=1, cash=5000.0, shares=50.0, info_param=0.1, trader_type="signal_following")
    order = trader.place_order(signal=1.08, value=100.0)
"""

import random
import math

# NOISE TRADER CLASS

class NoiseTrader:
    """A noise trader with no information — trades randomly within budget.
        UAID   - unique agent ID (int)
        Cash   - available cash (float)
        Shares - number of shares held (float)
    """

    # We initialize the NoiseTrader with its unique ID, cash, and shares. 
    # The place_order method generates a random order based on the current stock price (value) and 
    # ensures that it respects the trader's budget constraints.
    # The price of the order is randomly chosen within a specified range around the current value, and 
    # the quantity is determined by how much the trader can afford to buy or sell.
    def __init__(self, uaid: int, cash: float, shares: float):
        self.uaid = uaid
        self.cash = cash
        self.shares = shares

    # we can reuse the same logic as the zero_intelligence strategy function, 
    # but we put it in a method here for direct use by NoiseTrader instances. 
    # This keeps the NoiseTrader class self-contained and allows us to easily adjust parameters like price range if needed.
    def place_order(self, value: float, price_range: float = 0.2) -> dict: # Added price_range parameter for flexibility 
        """Generate a random order within budget constraints.

        Arguments:
            value: current underlying stock price (S_t)
            price_range: how far above/below value the price can be (default +/-20%)

        Returns:
            dict with keys: Price, Quantity, Buy, Sell, Hold, ID
        """
        can_buy = self.cash > 0 and value > 0
        can_sell = self.shares > 0

        # If agent can't trade at all, hold
        if not can_buy and not can_sell:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0, "ID": self.uaid}

        # Random direction from feasible options
        if can_buy and can_sell:
            action = random.choice(["buy", "sell"])
        elif can_buy:
            action = "buy"
        else:
            action = "sell"

        # Random price within range
        low = value * (1 - price_range)
        high = value * (1 + price_range)
        price = round(random.uniform(max(low, 0.01), high), 2)

        # Random quantity within budget
        if action == "buy":
            max_qty = self.cash / price if price > 0 else 0
            if max_qty < 1:
                return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0, "ID": self.uaid}
            quantity = round(random.uniform(1, max_qty), 2)
        else:
            quantity = round(random.uniform(1, self.shares), 2) if self.shares >= 1 else self.shares

        return {"Price": price, "Quantity": quantity, "Buy": 1.0 if action == "buy" else 0.0, "Sell": 1.0 if action == "sell" else 0.0, "Hold": 0.0, "ID": self.uaid}


# STRATEGY FUNCTIONS

# Each function takes: Signal, Cash, Shares, Value
# Each returns: dict with Price, Quantity, Buy, Sell, Hold
# The zero_intelligence strategy ignores the signal and generates random orders within the trader's budget constraints.
# It first checks if the trader can buy or sell based on their cash and shares. If they can't do either, it returns a hold order.
# If they can do both, it randomly chooses to buy or sell. 
# The price is randomly set within a range around the current value, 
# and the quantity is determined by how much the trader can afford to buy or sell.
def zero_intelligence(signal: float, cash: float, shares: float, value: float) -> dict:
    """ZI-C strategy: random orders ignoring the signal.

    Arguments:
        signal: private signal (ignored by ZI)
        cash:   agent's available cash
        shares: agent's current shares
        value:  current underlying stock price (S_t)

    Returns:
        dict with Price, Quantity, Buy, Sell, Hold
    """
    # Check what actions are possible given budget and holdings
    can_buy = cash > 0 and value > 0
    can_sell = shares > 0

    # If agent can't trade at all, hold
    if not can_buy and not can_sell:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Random direction from feasible options
    if can_buy and can_sell:
        action = random.choice(["buy", "sell"])
    elif can_buy:
        action = "buy"
    else:
        action = "sell"

    # Random price within +/-20% of current value
    price_range = 0.2
    low = value * (1 - price_range)
    high = value * (1 + price_range)
    price = round(random.uniform(max(low, 0.01), high), 2)

    # Random quantity within budget
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        if max_qty < 1:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
        quantity = round(random.uniform(1, max_qty), 2)
    else:
        quantity = round(random.uniform(1, shares), 2) if shares >= 1 else shares

    return {"Price": price, "Quantity": quantity, "Buy": 1.0 if action == "buy" else 0.0, "Sell": 1.0 if action == "sell" else 0.0, "Hold": 0.0,}


# signal_following strategy uses the private signal to determine the direction, price, and quantity of the order. 
# The price is set as the current value multiplied by the signal, which represents the trader's
# estimate of the future worth of the stock. The quantity is scaled by how far the signal is from 1.0 (the strength of the signal)
# and an aggression factor that can be tuned to make the trader more or less responsive to the signal.
def signal_following(signal: float, cash: float, shares: float, value: float, aggression: float = 1.0) -> dict:
    """Signal-Following strategy: uses private signal to set price and quantity.

    Direction: signal > 1.0 -> buy (expects price rise), otherwise -> sell
    Price:     value x signal (estimate of future worth)
    Quantity:  scaled by conviction (how far signal is from 1.0)

    Arguments:
        signal:     private signal (multiplier around 1.0)
        cash:       agent's available cash
        shares:     agent's current shares
        value:      current underlying stock price (S_t)
        aggression: multiplier on position size (default 1.0)

    Returns:
        dict with Price, Quantity, Buy, Sell, Hold
    """
    if signal > 1.0:
        action = "buy"
    else:
        action = "sell"

    # Check feasibility
    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Price = estimate of next fundamental
    price = round(value * signal, 2)
    price = max(price, 0.01)

    # Quantity scaled by strength of conviction and aggression
    conviction = abs(signal - 1.0)
    fraction = min(conviction * aggression * 10, 1.0)

    # Determine max quantity based on action and budget, then scale by fraction
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        quantity = max(1.0, round(max_qty * fraction, 2))
        quantity = min(quantity, max_qty)
    else:
        max_qty = shares
        quantity = max(1.0, round(max_qty * fraction, 2))
        quantity = min(quantity, max_qty)

    # If quantity is less than 1 after scaling, treat as hold
    if quantity < 1:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    return {"Price": price, "Quantity": quantity, "Buy": 1.0 if action == "buy" else 0.0, "Sell": 1.0 if action == "sell" else 0.0, "Hold": 0.0,}


# STRATEGY REGISTRY

# Maps strategy names to functions so the Trader
# class can look them up by trader_type string

STRATEGIES = {"zi": zero_intelligence, "signal_following": signal_following,}


# TRADER CLASS

class Trader:
    """A trader with private information and a configurable strategy.

    Inputs:
        UAID                  - unique agent ID (int)
        Cash                  - available cash (float)
        Shares                - number of shares held (float)
        Information Parameter - signal noise / sigma_noise (float)
        Strategy Probabilities - array of weights for mixed strategies
        Trader Type           - "zi", "signal_following", "mixed", etc.
    """

    def __init__(self,uaid: int,cash: float,shares: float,info_param: float = 0.1,strategy_probs: list = None,trader_type: str = "zi",):
        
        self.uaid = uaid
        self.cash = cash
        self.shares = shares
        self.info_param = info_param  # sigma_noise — lower = more precise
        self.strategy_probs = strategy_probs or [1.0]
        self.trader_type = trader_type

    def place_order(self, signal: float, value: float) -> dict:
        """Generate an order using the trader's assigned strategy.

        Arguments:
            signal: private signal about next price shock (float)
            value:  current underlying stock price S_t (float)

        Returns:
            dict with Price, Quantity, Buy, Sell, Hold, ID
        """
        if self.trader_type == "mixed":
            # Pick a strategy randomly based on probability weights
            strategy_names = list(STRATEGIES.keys())
            chosen = random.choices(strategy_names, weights=self.strategy_probs, k=1)[0]
            strategy_fn = STRATEGIES[chosen]
        else:
            strategy_fn = STRATEGIES.get(self.trader_type, zero_intelligence)

        order = strategy_fn(signal, self.cash, self.shares, value)
        order["ID"] = self.uaid
        return order

# VALIDATION HELPER

def validate_order(order: dict, cash: float, shares: float) -> bool:
    """Check that an order dict respects budget and shorting constraints."""
    if order["Hold"] == 1.0:
        return True  # holding is always valid
    if order["Price"] <= 0 or order["Quantity"] < 1:
        return False
    if order["Buy"] == 1.0 and order["Price"] * order["Quantity"] > cash:
        return False
    if order["Sell"] == 1.0 and order["Quantity"] > shares:
        return False
    return True

# TESTS

def run_tests():
    print("Running tests...\n")

    # --- Test ZI strategy function ---
    for i in range(100):
        order = zero_intelligence(signal=1.0, cash=10000.0, shares=100.0, value=100.0)
        assert validate_order(order, 10000.0, 100.0), f"ZI invalid order on iteration {i}: {order}"
        if order["Hold"] != 1.0:
            assert 80.0 <= order["Price"] <= 120.0, f"ZI price out of range: {order['Price']}"
    print("  ✓ ZI: 100 orders all valid and within price range")

    # --- Test ZI with no cash (can only sell) ---
    for _ in range(50):
        order = zero_intelligence(signal=1.0, cash=0.0, shares=10.0, value=100.0)
        if order["Hold"] != 1.0:
            assert order["Sell"] == 1.0, "Broke agent should only sell"
            assert order["Quantity"] <= 10.0
    print("  ✓ ZI: broke agent only sells, respects share limit")

    # --- Test ZI with nothing ---
    order = zero_intelligence(signal=1.0, cash=0.0, shares=0.0, value=100.0)
    assert order["Hold"] == 1.0, "Empty agent should hold"
    print("  ✓ ZI: empty agent holds")

    # --- Test SF: bullish signal -> buy ---
    order = signal_following(signal=1.10, cash=10000.0, shares=100.0, value=100.0)
    assert order["Buy"] == 1.0, f"Bullish signal should buy, got {order}"
    assert abs(order["Price"] - 110.0) < 0.01, f"Price should be ~110, got {order['Price']}"
    assert validate_order(order, 10000.0, 100.0)
    print("  ✓ SF: bullish signal → buy at ~110")

    # --- Test SF: bearish signal -> sell ---
    order = signal_following(signal=0.90, cash=10000.0, shares=100.0, value=100.0)
    assert order["Sell"] == 1.0, f"Bearish signal should sell, got {order}"
    assert abs(order["Price"] - 90.0) < 0.01, f"Price should be ~90, got {order['Price']}"
    print("  ✓ SF: bearish signal → sell at ~90")

    # --- Test SF: strong signal -> bigger position ---
    order_strong = signal_following(signal=1.15, cash=10000.0, shares=100.0, value=100.0)
    order_weak = signal_following(signal=1.02, cash=10000.0, shares=100.0, value=100.0)
    assert order_strong["Quantity"] > order_weak["Quantity"], \
        f"Strong signal should give bigger qty ({order_strong['Quantity']}) than weak ({order_weak['Quantity']})"
    print("  ✓ SF: strong signal → larger position than weak signal")

    # --- Test Trader class ---
    trader = Trader(uaid=1, cash=5000.0, shares=50.0, trader_type="signal_following")
    order = trader.place_order(signal=1.08, value=100.0)
    assert order["ID"] == 1, "Order should carry trader's UAID"
    assert order["Buy"] == 1.0, "Bullish signal should produce buy"
    assert validate_order(order, 5000.0, 50.0)
    print("  ✓ Trader: SF trader produces valid buy order with correct UAID")

    # --- Test Noise Trader class ---
    nt = NoiseTrader(uaid=99, cash=3000.0, shares=30.0)
    order = nt.place_order(value=100.0)
    assert order["ID"] == 99
    assert validate_order(order, 3000.0, 30.0)
    print("  ✓ NoiseTrader: produces valid order with correct UAID")

    # --- Test mixed trader ---
    mixed = Trader(uaid=2, cash=5000.0, shares=50.0, trader_type="mixed",
                   strategy_probs=[0.5, 0.5])
    for _ in range(50):
        order = mixed.place_order(signal=1.05, value=100.0)
        assert validate_order(order, 5000.0, 50.0)
        assert order["ID"] == 2
    print("  ✓ Trader: mixed trader produces valid orders over 50 runs")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_tests()