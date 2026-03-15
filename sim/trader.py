import random
from .strategies import STRATEGIES, zero_intelligence

# ----------------------------
# Trader
# ----------------------------

#UAID used for actually no reason, composite will be [run_ULID, agent_id] which will be unique for every agent making uaid use redundent (https://media3.giphy.com/media/v1.Y2lkPTZjMDliOTUyYnZjeGd5N3BrbzhoNmdod2pwYWdpaWl5OTZwbWRzdzExNnVvNzE0bCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Ij5kcfI6YwcPCN26U2/giphy.gif) so I changed it

class Trader:
    def __init__(self, agent_id: int, cash: float, shares: float,
                 info_param: float = 0.1, strategy_probs: list = None,
                 trader_type: str = "zi"):
        self.agent_id = agent_id
        self.cash = cash
        self.shares = shares
        self.info_param = info_param
        self.strategy_probs = strategy_probs or [1.0]
        self.trader_type = trader_type

    def place_order(self, signal: float, value: float) -> dict | None:
        if self.trader_type == "mixed":
            strategy_names = list(STRATEGIES.keys())
            chosen = random.choices(strategy_names, weights=self.strategy_probs, k=1)[0]
            strategy_fn = STRATEGIES[chosen]
        else:
            strategy_fn = STRATEGIES.get(self.trader_type, zero_intelligence)

        order = strategy_fn(signal, self.cash, self.shares, value)

        if order is None:
            return None

        order["agent_id"] = self.agent_id
        return order
