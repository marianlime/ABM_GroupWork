from .noise_signal import assign_noise_parameter_set, signal_generator
from .trader import Trader
from .market import clear_market
import numpy as np


class Game:
    def __init__(
        self,
        DB_PATH,
        population_spec: list[dict],
        # --- Run / Market ---
        n_rounds: int,
        total_initial_shares: float,
        total_initial_cash: float,
        cash_to_share_ratio: float,
        run_id: str,
        market_mechanism: str,
        pricing_rule: str,
        rationing_rule: str,
        tie_break_rule: str,
        transaction_cost_rate: float,
        # --- Signal / Noise ---
        noise_parameter_distribution_type: str,
        distribution_data: dict,
        signal_generator_noise_distribution: str,
        bias: float,
        # --- Fundamentals ---
        fundamental_source: str,
        # --- GBM Configuration ---
        S0: float | None = None,
        volatility: float | None = None,
        drift: float | None = None,
        # --- Historical Configuration ---
        ticker: str | None = None,
        interval: str | None = None,
        start_date: str | None = None,
        price_col: str | None = None,
        auto_adjust: bool | None = None,
        # --- stock_path ---
        fundamental_path=None,
        seed=None
    ):
        self.DB_PATH = DB_PATH

        self.current_round = 0
        self.agents = {}

        self.population_spec = population_spec
        self.n_agents = len(population_spec)

        if self.n_agents == 0:
            raise ValueError("population_spec must contain at least one agent")

        self.n_zi_agents = sum(1 for agent in population_spec if agent["trader_type"] == "zi")
        self.n_signal_following_agents = sum(1 for agent in population_spec if agent["trader_type"] == "signal_following")
        self.n_utility_maximiser_agents = sum(1 for agent in population_spec if agent["trader_type"] == "utility_maximiser")
        self.n_contrarian_agents = sum(1 for agent in population_spec if agent["trader_type"] == "contrarian")
        self.n_adapt_sig_agents = sum(1 for agent in population_spec if agent["trader_type"] == "adapt_sig")

        self.fundamental_source = fundamental_source
        self.n_rounds = n_rounds
        self.fundamental_path = fundamental_path
        self.total_initial_shares = total_initial_shares
        self.total_initial_cash = total_initial_cash

        # fundamental source = GBM
        self.S0 = S0
        self.volatility = volatility
        self.drift = drift
        self.run_id = run_id

        # fundamental source = Historical
        self.ticker = ticker
        self.interval = interval
        self.start_date = start_date
        self.price_col = price_col
        self.auto_adjust = auto_adjust

        self.market_mechanism = market_mechanism
        self.pricing_rule = pricing_rule
        self.rationing_rule = rationing_rule
        self.tie_break_rule = tie_break_rule
        self.transaction_cost_rate = transaction_cost_rate

        self.noise_parameter_distribution_type = noise_parameter_distribution_type
        self.distribution_data = distribution_data
        self.signal_noise_distribution = signal_generator_noise_distribution
        self.bias = bias

        self.market_round_records = []
        self.agent_round_records = []

        if self.S0 is not None:
            self.S0_effective = float(self.S0)
        elif self.fundamental_path is not None and len(self.fundamental_path) > 0:
            self.S0_effective = float(self.fundamental_path[0])
        else:
            self.S0_effective = 1.0

        self.order_history = {}
        self.price_history = {}

        self.noise_parameter_set = assign_noise_parameter_set(
            n_agents=self.n_agents,
            noise_parameter_distribution_type=self.noise_parameter_distribution_type,
            distribution_data=self.distribution_data
        )

        cash_per_agent = self.total_initial_cash / self.n_agents
        shares_per_agent = self.total_initial_shares / self.n_agents

        valid_trader_types = {
            "zi",
            "signal_following",
            "utility_maximiser",
            "contrarian",
            "adapt_sig",
        }

        for agent_id, agent_spec in enumerate(self.population_spec, start=1):
            trader_type = agent_spec["trader_type"]

            if trader_type not in valid_trader_types:
                raise ValueError(f"Unknown trader_type in population_spec: {trader_type}")

            self.agents[agent_id] = Trader(
                agent_id=agent_id,
                cash=cash_per_agent,
                shares=shares_per_agent,
                info_param=float(self.noise_parameter_set[agent_id - 1]),
                strategy_probs=None,
                trader_type=trader_type
            )

    def gather_orders_and_clear(self, current_round):

        agent_round_records = {}

        if self.fundamental_path is None:
            raise ValueError("fundamental_path is None")
        if current_round + 1 >= len(self.fundamental_path):
            raise IndexError(f"fundamental_path is too short for current_round ({current_round}).")

        order_list = []

        S_next = float(self.fundamental_path[current_round + 1])
        value = float(self.fundamental_path[current_round])

        for agent_id, player in self.agents.items():

            cash_start = float(player.cash)
            inventory_start = float(player.shares)

            if player.trader_type == "zi":
                signal = 1.0
                signal_error = 0.0
            else:
                raw_signal = signal_generator(
                    noise_parameter=player.info_param,
                    S_next=S_next,
                    bias=self.bias,
                    signal_generator_noise_distribution=self.signal_noise_distribution
                )
                signal = raw_signal / max(value, 1e-12)
                true_ratio = S_next / max(value, 1e-12)
                signal_error = float(signal - true_ratio)

            strat_order = player.place_order(signal=signal, value=value)

            if strat_order is None or strat_order.get("Hold", 0.0) == 1.0:
                action = "hold"
                limit_price = None
                order_qty = 0.0
            else:
                action = "buy" if strat_order.get("Buy", 0.0) == 1.0 else "sell"
                limit_price = float(strat_order["Price"])
                order_qty = float(strat_order["Quantity"])

            aggressiveness = float(abs(signal - 1.0))

            agent_round_records[agent_id] = {
                "run_id": self.run_id,
                "round_number": current_round,
                "agent_id": agent_id,
                "signal": float(signal),
                "signal_error": float(signal_error),
                "action": action,
                "limit_price": limit_price,
                "order_qty": float(order_qty),
                "aggressiveness": aggressiveness,
                "executed_qty": 0.0,
                "executed_price_avg": None,
                "fill_ratio": 0.0,
                "is_filled": False,
                "is_partial": False,
                "cash_start": cash_start,
                "inventory_start": inventory_start,
                "cash_end": cash_start,
                "inventory_end": inventory_start,
            }

            if strat_order is not None and strat_order.get("Hold", 0.0) != 1.0:
                order_list.append({
                    "agent_id": strat_order["agent_id"],
                    "action": action,
                    "price": float(strat_order["Price"]),
                    "quantity": float(strat_order["Quantity"])
                })

        buy_orders = [o for o in order_list if o["action"] == "buy"]
        sell_orders = [o for o in order_list if o["action"] == "sell"]

        best_bid = max((o["price"] for o in buy_orders), default=None)
        best_ask = min((o["price"] for o in sell_orders), default=None)

        n_active_buyers = len(buy_orders)
        n_active_sellers = len(sell_orders)
        n_active_total = len(order_list)

        bid_depth_total = float(sum(o["quantity"] for o in buy_orders)) if buy_orders else 0.0
        ask_depth_total = float(sum(o["quantity"] for o in sell_orders)) if sell_orders else 0.0

        price_levels_bid = len(set(o["price"] for o in buy_orders))
        price_levels_ask = len(set(o["price"] for o in sell_orders))

        prev_price = self.price_history.get(current_round - 1, None)
        best_price, total_volume, trades = clear_market(order_list, previous_price=prev_price)

        exec_summary = {}

        for trade in trades:
            aid = trade["agent_id"]
            qty = float(trade["quantity"])
            px = float(trade["price"])
            action = trade["action"]

            if aid not in exec_summary:
                exec_summary[aid] = {
                    "executed_qty": 0.0,
                    "executed_notional": 0.0,
                    "cash_delta": 0.0,
                    "inventory_delta": 0.0
                }

            exec_summary[aid]["executed_qty"] += qty
            exec_summary[aid]["executed_notional"] += qty * px

            trade_value = qty * px
            fee = trade_value * self.transaction_cost_rate

            if action == "buy":
                exec_summary[aid]["cash_delta"] -= trade_value
                exec_summary[aid]["cash_delta"] -= fee
                exec_summary[aid]["inventory_delta"] += qty

            elif action == "sell":
                exec_summary[aid]["cash_delta"] += trade_value
                exec_summary[aid]["cash_delta"] -= fee
                exec_summary[aid]["inventory_delta"] -= qty

        if best_price is not None:
            demand_at_p = float(sum(o["quantity"] for o in buy_orders if o["price"] >= best_price))
            supply_at_p = float(sum(o["quantity"] for o in sell_orders if o["price"] <= best_price))
        else:
            demand_at_p = 0.0
            supply_at_p = 0.0

        self.order_history[current_round] = order_list
        self.price_history[current_round] = best_price

        self.market_round_records.append({
            "run_id": self.run_id,
            "round_number": current_round,
            "p_t": float(best_price) if best_price is not None else None,
            "best_bid": float(best_bid) if best_bid is not None else None,
            "best_ask": float(best_ask) if best_ask is not None else None,
            "volume": float(total_volume),
            "n_trades": len(trades),
            "demand_at_p": demand_at_p,
            "supply_at_p": supply_at_p,
            "n_active_buyers": n_active_buyers,
            "n_active_sellers": n_active_sellers,
            "n_active_total": n_active_total,
            "bid_depth_total": bid_depth_total,
            "ask_depth_total": ask_depth_total,
            "price_levels_bid": price_levels_bid,
            "price_levels_ask": price_levels_ask
        })

        for aid, rec in agent_round_records.items():
            executed_qty = exec_summary.get(aid, {}).get("executed_qty", 0.0)
            executed_notional = exec_summary.get(aid, {}).get("executed_notional", 0.0)
            cash_delta = exec_summary.get(aid, {}).get("cash_delta", 0.0)
            inventory_delta = exec_summary.get(aid, {}).get("inventory_delta", 0.0)

            executed_price_avg = executed_notional / executed_qty if executed_qty > 0 else None

            order_qty = rec["order_qty"]
            fill_ratio = (executed_qty / order_qty) if order_qty > 0 else 0.0

            rec["executed_qty"] = float(executed_qty)
            rec["executed_price_avg"] = float(executed_price_avg) if executed_price_avg is not None else None
            rec["fill_ratio"] = float(fill_ratio)
            rec["is_filled"] = bool(order_qty > 0 and np.isclose(executed_qty, order_qty))
            rec["is_partial"] = bool(executed_qty > 0 and executed_qty < order_qty)
            rec["cash_end"] = float(rec["cash_start"] + cash_delta)
            rec["inventory_end"] = float(rec["inventory_start"] + inventory_delta)

            self.agent_round_records.append(rec)

        return best_price, total_volume, trades

    def update_portfolio(self, trade):
        agent = self.agents[trade["agent_id"]]

        qty = float(trade["quantity"])
        px = float(trade["price"])
        action = trade["action"]

        trade_value = qty * px
        fee = trade_value * self.transaction_cost_rate

        if action == "buy":
            agent.shares += qty
            agent.cash -= qty * px
            agent.cash -= fee
        elif action == "sell":
            agent.shares -= qty
            agent.cash += qty * px
            agent.cash -= fee
        else:
            raise ValueError("Unknown trade action")

    def liquidate_assets(self, agent_id):
        return float(self.fundamental_path[-1]) * float(self.agents[agent_id].shares) + float(self.agents[agent_id].cash)