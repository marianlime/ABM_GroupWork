from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import math

## Data Classes and Interface

@dataclass
class LimitOrder:
    price: float
    quantity: int
    side: str  # 'buy' or 'sell'

@dataclass
class AgentState:
    shares: float
    cash: float

@dataclass
class Signal:
    value: float
    noise: float

class TradingStrategy(ABC):

    @abstractmethod
    def generate_order(self, state: AgentState, signal: Signal, current_price: float) -> LimitOrder | None:
        pass

## Constant Validation Helper

def validate_order(order: LimitOrder, state: AgentState) -> bool:
    if order.price <= 0 or order.quantity < 0:
        return False
    if order.side == 'buy' and order.price * order.quantity > state.cash:
        return False
    if order.side == 'sell' and order.quantity > state.shares:
        return False
    return True

## Zero Intelligence Agent

class ZeroIntelligenceAgent(TradingStrategy):

    def __init__(self, price_range: float = 0.2):
        self.price_range = price_range

    def generate_order(self, state: AgentState, signal: Signal, current_price: float,) ->LimitOrder | None:
        can_buy = state.cash > 0 and current_price > 0
        can_sell = state.shares > 0
        if not can_buy and not can_sell:
            return None
    
        if can_buy and can_sell:
            direction = random.choice(["buy", "sell"])
        elif can_buy:
            direction = "buy"
        else:
           direction = "sell"

        low = current_price * (1 - self.price_range)
        high = current_price * (1 + self.price_range)
        price = round(random.uniform(max(low, 0.01), high), 2)

        if direction == "buy":
            max_quantity = int(state.cash // price) if price > 0 else 0
            if max_quantity < 1:
                return None
            quantity = random.randint(1, max_quantity)
        else:
            quantity = random.randint(1, state.shares)

        order = LimitOrder(price=price, quantity=quantity, side=direction)
        assert validate_order(order, state), f"ZI Agent generated invalid order: {order}" 
        return order
    
## Signal Following Agent

class SignalFollowing(TradingStrategy):
    def __init__(self, aggression: float = 1.0):
        self.aggression = aggression

    def generate_order(self, state: AgentState, signal: Signal, current_price: float,) -> LimitOrder | None:
        if signal.value > 1.0:
            direction = "buy"
        else:
            direction = "sell"
        
        if direction == "buy" and state.cash <= 0:
            return None
        if direction == "sell" and state.shares <= 0:
            return None
        
        price = round(current_price * signal.value, 2)
        price = max(price, 0.01)  # Floor at 1 cent

        confidence = 1.0 / (1.0 + signal.noise)
        fraction = confidence * self.aggression
        fraction = min(max(fraction, 0.0), 1.0)

        if direction == "buy":
            max_quantity = int(state.cash // price) if price > 0 else 0
            quantity = max(1, int(max_quantity * fraction))
            quantity = min(quantity, max_quantity)
        else:
            max_quantity = state.shares
            quantity = max(1, int(max_quantity * fraction))
            quantity = min(quantity, max_quantity)

        if quantity < 1:
            return None
        
        order = LimitOrder(price=price, quantity=quantity, side=direction)
        assert validate_order(order, state), "Signal Following Agent generated invalid order"
        return order
    
## Basic Testing

def run_agents():
    print("Running tests...\n")
    state = AgentState(shares=100, cash=10000)
    current_price = 100.0

# Test Zero Intelligence Agent

    zi = ZeroIntelligenceAgent(price_range=0.2)
    dummy_signal = Signal(value=1.0, noise=0.5)

    for _ in range(100):
        order = zi.generate_order(state, dummy_signal, current_price)
        assert order is None or validate_order(order, state), "ZI Agent generated invalid order"
        assert validate_order(order,state), f"ZI invalid order on iteration {_}: {order}"
        assert 80.0 <= order.price <= 120.0, f"ZI order price out of range: {order.price}"
        
    print("Zero Intelligence Agent tests passed. 100 orders generated with valid prices and quantities.")

# Test ZI with no cash (can only sell)

    broke_state = AgentState(shares=10, cash=0.0)
        
    for _ in range(50):
        order = zi.generate_order(broke_state, dummy_signal, current_price)
        assert order is not None
        assert order.side == "sell", "Broke agent should only sell orders"
        assert order.quantity <= 10

    print("ZI: broke agent only sells, respects share limits")

# Test ZI with no shares/nothing (can return none)

    empty_state = AgentState(shares=0, cash=0.0)
    order = zi.generate_order(empty_state, dummy_signal, current_price)
    assert order is None, "Empty agent should return None when no cash or shares"

    print("ZI: empty agent returns None as expected")

# Test Signal Following Agent: bullish signal -> buy orders

    sf = SignalFollowing(aggression=1.0)
    bullish_signal = Signal(value=1.10, noise=0.1)
    order = sf.generate_order(state, bullish_signal, current_price)
    assert order is not None
    assert order.side == "buy", f"Bullish signal should generate buy orderm got {order.side}"
    assert abs(order.price - 110.0) < 0.01, f"Buy order price should be around 110, got {order.price}"
    assert validate_order(order, state)

    print("Sf: bullish signal generates valid buy order at expected price ~110")

# Test Signal Following Agent: bearish signal -> sell orders

    bearish_signal = Signal(value=0.90, noise=0.1)
    order = sf.generate_order(state, bearish_signal, current_price)
    assert order is not None
    assert order.side == "sell", f"Bearish signal should generate sell order, got {order.side}"
    assert abs(order.price - 90.0) < 0.01, f"Sell order price should be around 90, got {order.price}"
        
    print("Sf: bearish signal generates valid sell order at expected price ~90")

# Test Signal Following Agent: noisy signal -> small quantity

    noisy_signal = Signal(value=1.05, noise=10.0)  # High noise should reduce confidence
    noisy_clean = Signal(value=1.05, noise=0.1)  # Low noise should increase confidence
    order_noisy = sf.generate_order(state, noisy_signal, current_price)
    order_clean = sf.generate_order(state, noisy_clean, current_price)
    assert order_noisy.quantity < order_clean.quantity, f"Noisy signal should generate smaller quantity than clean signal, got {order_noisy.quantity} vs {order_clean.quantity}"
        
    print("Sf: noisy signal generates smaller quantity than clean signal as expected")
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_agents()



