from .strategies import STRATEGIES, zero_intelligence


class Trader:
    def __init__(self, agent_id: int, cash: float, shares: float,
                 info_param: float = 0.1, trader_type: str = "zi",
                 strategy_params: dict = None):
        self.agent_id = agent_id
        self.cash = cash
        self.shares = shares
        self.info_param = info_param
        self.trader_type = trader_type
        self.strategy_params = strategy_params or {}

    def place_order(self, signal: float, value: float) -> dict | None:
        strategy_fn = STRATEGIES.get(self.trader_type, zero_intelligence)
        order = strategy_fn(signal, self.cash, self.shares, value, **self.strategy_params)

        if order is None:
            return None

        order["agent_id"] = self.agent_id
        return order
