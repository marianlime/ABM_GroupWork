from .noise_signal import assign_noise_parameter_set, signal_generator
from .trader import Trader
from .market import clear_market

# ----------------------------
# Game
# ----------------------------

class game:
    def __init__(self, n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
                 total_initial_shares, cash_to_share_ratio, run_id, stock_path, shock_path,
                 noise_param_dist_type="evenly_spaced",
                 signal_noise_distribution="lognormal"):
        self.current_round = 0
        self.agents = {}
        self.volatility = volatility
        self.n_strategic_agents = n_strategic_agents
        self.n_zi_agents = n_zi_agents
        self.n_agents = n_strategic_agents + n_zi_agents
        self.S0 = S0
        self.drift = drift
        self.stock_path = stock_path
        self.shock_path = shock_path
        self.total_initial_shares = total_initial_shares
        self.total_initial_cash = total_initial_shares * S0 * cash_to_share_ratio
        self.run_id = run_id
        self.noise_param_dist_type = noise_param_dist_type
        self.signal_noise_distribution = signal_noise_distribution

        self.order_history = {}
        self.price_history = {}
        self.noise_parameter_set = assign_noise_parameter_set(
            self.n_agents,
            dist_type=self.noise_param_dist_type
        )
        for i in range(n_strategic_agents):
            self.agents[i] = Trader(
                uaid=i,
                cash=self.total_initial_cash / self.n_agents,
                shares=self.total_initial_shares / self.n_agents,
                info_param=float(self.noise_parameter_set[i]),
                strategy_probs=None,
                trader_type="signal_following"
            )

        for i in range(n_zi_agents):
            aid = i + n_strategic_agents
            self.agents[aid] = Trader(
                uaid=aid,
                cash=self.total_initial_cash / self.n_agents,
                shares=self.total_initial_shares / self.n_agents,
                info_param=0.0,  # unused by zi strategy
                strategy_probs=None,
                trader_type="zi"
            )

    def gather_orders_and_clear(self, current_round):
        order_list = []

        for agent_id, player in self.agents.items():
            S_next = self.stock_path[current_round + 1]
            value = self.stock_path[current_round]

            # signal: for informed traders use info_param; for zi it doesn't matter
            if player.trader_type == "signal_following":
                raw_signal = signal_generator(
                    player.info_param,
                    S_next=S_next,
                    noise_distribution=self.signal_noise_distribution
                )                # convert to multiplier around 1.0 so signal_following behaves as intended
                signal = raw_signal / max(value, 1e-12)
            else:
                signal = 1.0

            strat_order = player.place_order(signal=signal, value=value)

            if strat_order.get("Hold", 0.0) == 1.0:
                continue

            side = "buy" if strat_order.get("Buy", 0.0) == 1.0 else "sell"
            order_list.append({
                "agent_id": strat_order["ID"],
                "side": side,
                "price": float(strat_order["Price"]),
                "quantity": float(strat_order["Quantity"])
            })

        prev_price = self.price_history.get(current_round - 1, None)
        best_price, total_volume, trades = clear_market(order_list, previous_price=prev_price)

        self.order_history[current_round] = order_list
        self.price_history[current_round] = best_price

        return best_price, total_volume, trades

    def update_portfolio(self, trade):
        agent = self.agents[trade['agent_id']]
        qty = trade['quantity']
        px = trade['price']

        if trade['side'] == 'buy':
            agent.shares += qty
            agent.cash -= qty * px
        elif trade['side'] == 'sell':
            agent.shares -= qty
            agent.cash += qty * px
        else:
            raise ValueError("Unknown trade side")

    def liquidate_assets(self, agent_id):
        return float(self.stock_path[-1]) * float(self.agents[agent_id].shares) + float(self.agents[agent_id].cash)