import random
from .strategies import STRATEGIES, zero_intelligence

# ----------------------------
# Trader
# ----------------------------

class Trader:
    def __init__(self, uaid: int, cash: float, shares: float,
                 info_param: float = 0.1, strategy_probs: list = None,
                 trader_type: str = "zi"):
        self.uaid = uaid
        self.cash = cash
        self.shares = shares
        self.info_param = info_param
        self.strategy_probs = strategy_probs or [1.0]
        self.trader_type = trader_type

    def place_order(self, signal: float, value: float) -> dict:
        if self.trader_type == "mixed":
            strategy_names = list(STRATEGIES.keys())
            chosen = random.choices(strategy_names, weights=self.strategy_probs, k=1)[0]
            strategy_fn = STRATEGIES[chosen]
        else:
            strategy_fn = STRATEGIES.get(self.trader_type, zero_intelligence)

        order = strategy_fn(signal, self.cash, self.shares, value)
        order["ID"] = self.uaid
        return order