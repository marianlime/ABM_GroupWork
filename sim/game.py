from collections import Counter, defaultdict
from .noise_signal import assign_noise_parameter_set
from .trader import Trader
from .market import clear_market
import hashlib
import numpy as np


def _noise_seed_from_run_id(run_id: str) -> int:
    """Derive a stable integer seed from the run ULID so noise parameter
    assignment is reproducible for a given run."""
    return int(hashlib.sha256(f"noise_{run_id}".encode()).hexdigest()[:16], 16)


def _zi_seed_from_run_id(run_id: str) -> int:
    """Derive a separate stable seed for the ZI-agent RNG, distinct from the
    noise-parameter seed so the two random streams are independent."""
    return int(hashlib.sha256(f"zi_{run_id}".encode()).hexdigest()[:16], 16)


class Game:
    def __init__(
        self,
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
        self.current_round = 0
        self.agents = {}

        self.population_spec = population_spec
        self.n_agents = len(population_spec)

        if self.n_agents == 0:
            raise ValueError("population_spec must contain at least one agent")

        _type_counts = Counter(agent["trader_type"] for agent in population_spec)
        self.n_zi_agents = _type_counts.get("zi", 0)
        self.n_signal_following_agents = _type_counts.get("signal_following", 0)
        self.n_utility_maximiser_agents = _type_counts.get("utility_maximiser", 0)
        self.n_contrarian_agents = _type_counts.get("contrarian", 0)
        self.n_adapt_sig_agents = _type_counts.get("adapt_sig", 0)
        self.n_threshold_signal_agents = _type_counts.get("threshold_signal", 0)
        self.n_inventory_aware_utility_agents = _type_counts.get("inventory_aware_utility", 0)
        self.n_patient_signal_agents = _type_counts.get("patient_signal", 0)

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

        noise_seed = _noise_seed_from_run_id(self.run_id) if self.run_id else None
        self.noise_parameter_set = assign_noise_parameter_set(
            n_agents=self.n_agents,
            noise_parameter_distribution_type=self.noise_parameter_distribution_type,
            distribution_data=self.distribution_data,
            seed=noise_seed
        )

        cash_per_agent = self.total_initial_cash / self.n_agents
        shares_per_agent = self.total_initial_shares / self.n_agents

        valid_trader_types = {
            "zi",
            "signal_following",
            "utility_maximiser",
            "contrarian",
            "adapt_sig",
            "threshold_signal",
            "inventory_aware_utility",
            "patient_signal",
        }

        for agent_id, agent_spec in enumerate(self.population_spec, start=1):
            trader_type = agent_spec["trader_type"]

            if trader_type not in valid_trader_types:
                raise ValueError(f"Unknown trader_type in population_spec: {trader_type}")

            # Use evolved info_param if present, otherwise sample from the distribution.
            if "info_param" in agent_spec:
                info_param = float(agent_spec["info_param"])
            else:
                info_param = float(self.noise_parameter_set[agent_id - 1])

            self.agents[agent_id] = Trader(
                agent_id=agent_id,
                cash=cash_per_agent,
                shares=shares_per_agent,
                info_param=info_param,
                strategy_probs=None,
                trader_type=trader_type
            )

        # Pre-cache informed agent IDs and their noise parameters so signal
        # generation can be vectorised across all informed agents in one numpy
        # call per round instead of N individual calls.
        self._informed_agent_ids = [
            aid for aid, agent in self.agents.items() if agent.trader_type != "zi"
        ]
        self._informed_noise_params = np.array(
            [self.agents[aid].info_param for aid in self._informed_agent_ids],
            dtype=float
        )

        # Pre-cache ZI agent IDs for the same reason — batch order generation.
        self._zi_agent_ids = [
            aid for aid, agent in self.agents.items() if agent.trader_type == "zi"
        ]

        # Dedicated, reproducible RNG for ZI order generation — seeded from the
        # run ULID so results are repeatable, and isolated from the noise-parameter
        # RNG so the two streams don't interfere with each other.
        self._zi_rng = np.random.default_rng(
            _zi_seed_from_run_id(self.run_id) if self.run_id else None
        )

        # Initial per-agent endowment — used to cap ZI order sizes so that agents
        # who have accumulated wealth don't place proportionally larger orders.
        self._zi_initial_cash = cash_per_agent
        self._zi_initial_shares = shares_per_agent

        # Probability that a ZI agent abstains (holds) in any given round.
        self._zi_hold_prob = 0.1

    def _batch_zi_orders(self, prev_price) -> dict:
        """
        Generate Zero-Intelligence orders for all ZI agents in one vectorised batch.

        Fixes vs. original implementation
        ----------------------------------
        1. Price anchor  : uses the previous clearing price (public market info) rather
                           than the fundamental value, so ZI agents carry no implicit
                           fundamental knowledge.
        2. Fixed price std: spread is fixed at S0_effective * 0.2 so it does not widen
                           as the fundamental drifts — ZI noise is level-independent.
        3. Hold probability: each agent independently abstains with probability
                           _zi_hold_prob, giving ZI agents a genuine hold option.
        4. Quantity cap  : order sizes are capped at the initial per-agent endowment
                           (cash_per_agent / shares_per_agent) so accumulated wealth
                           does not amplify ZI order flow over generations.
        5. Seeded RNG    : uses self._zi_rng (np.random.Generator seeded from run_id)
                           so ZI draws are fully reproducible and isolated from the
                           noise-parameter RNG stream.

        Returns {agent_id: order_dict | None} where None means hold.
        """
        if not self._zi_agent_ids:
            return {}

        n = len(self._zi_agent_ids)
        zi_cash   = np.array([self.agents[aid].cash   for aid in self._zi_agent_ids])
        zi_shares = np.array([self.agents[aid].shares for aid in self._zi_agent_ids])

        # --- Price generation ---------------------------------------------------
        # Anchor on the previous clearing price (observable, public) rather than
        # the fundamental value (private/model information ZI agents shouldn't have).
        # Fall back to S0_effective when no prior clearing price exists.
        price_ref = float(prev_price) if prev_price is not None else self.S0_effective

        # Fixed std relative to the initial price level — does not scale with the
        # current fundamental, preventing spread blow-up at high price levels.
        price_std = self.S0_effective * 0.2
        prices = np.maximum(
            np.round(self._zi_rng.normal(price_ref, price_std, n), 2),
            0.01
        )

        # --- Hold probability ---------------------------------------------------
        # Each agent independently abstains with probability _zi_hold_prob.
        active = self._zi_rng.random(n) >= self._zi_hold_prob

        # --- Feasibility --------------------------------------------------------
        can_buy  = active & (zi_cash > 0) & (price_ref > 0)
        can_sell = active & (zi_shares > 0)

        # 50/50 random action for agents that can do both
        rand = self._zi_rng.random(n) < 0.5
        action_buy  = (can_buy & ~can_sell) | (can_buy & can_sell & rand)
        action_sell = (~can_buy & can_sell) | (can_buy & can_sell & ~rand)

        # --- Quantities ---------------------------------------------------------
        # Cap at the initial per-agent endowment so wealthy ZI agents don't
        # dominate order flow just because they've accumulated cash/shares.
        eps = 1e-6

        max_affordable  = np.where(prices > 0, zi_cash / prices, 0.0)
        initial_buy_cap = self._zi_initial_cash / np.maximum(prices, eps)
        max_buy_qty     = np.minimum(max_affordable, initial_buy_cap)
        action_buy      = action_buy & (max_buy_qty > 0)

        u_buy   = self._zi_rng.random(n)
        buy_qty = np.where(
            max_buy_qty > eps,
            np.round(eps + u_buy * (max_buy_qty - eps), 6),
            0.0
        )
        action_buy = action_buy & (buy_qty > 0)

        max_sell_qty = np.minimum(zi_shares, self._zi_initial_shares)
        action_sell  = action_sell & (max_sell_qty > 0)

        u_sell   = self._zi_rng.random(n)
        sell_qty = np.where(
            max_sell_qty > eps,
            np.round(eps + u_sell * (max_sell_qty - eps), 6),
            0.0
        )
        action_sell = action_sell & (sell_qty > 0)

        # --- Build result dict --------------------------------------------------
        orders = {}
        for i, aid in enumerate(self._zi_agent_ids):
            if action_buy[i]:
                orders[aid] = {
                    "Price": float(prices[i]), "Quantity": float(buy_qty[i]),
                    "Buy": 1.0, "Sell": 0.0, "Hold": 0.0, "agent_id": aid,
                }
            elif action_sell[i]:
                orders[aid] = {
                    "Price": float(prices[i]), "Quantity": float(sell_qty[i]),
                    "Buy": 0.0, "Sell": 1.0, "Hold": 0.0, "agent_id": aid,
                }
            else:
                orders[aid] = None  # hold
        return orders

    def gather_orders_and_clear(self, current_round):

        agent_round_records = {}

        if self.fundamental_path is None:
            raise ValueError("fundamental_path is None")
        if current_round + 1 >= len(self.fundamental_path):
            raise IndexError(f"fundamental_path is too short for current_round ({current_round}).")

        order_list = []

        S_next = float(self.fundamental_path[current_round + 1])
        value = float(self.fundamental_path[current_round])
        _value_safe = max(value, 1e-12)
        _true_ratio = S_next / _value_safe

        # Compute prev_price here so _batch_zi_orders can use it as a price anchor.
        prev_price = self.price_history.get(current_round - 1, None)

        # Vectorised signal generation — one numpy call for all informed agents.
        if self._informed_agent_ids:
            if self.signal_noise_distribution == 'lognormal':
                _multipliers = np.exp(
                    np.random.normal(self.bias, self._informed_noise_params)
                )
            elif self.signal_noise_distribution == 'uniform':
                _lows = np.maximum(1.0 - self._informed_noise_params, 1e-6)
                _highs = 1.0 + self._informed_noise_params
                _u = np.random.uniform(0, 1, len(self._informed_noise_params))
                _multipliers = _lows + _u * (_highs - _lows)
            else:
                raise ValueError(f"Unknown noise_distribution: {self.signal_noise_distribution}")
            _informed_signals = dict(
                zip(self._informed_agent_ids, (S_next * _multipliers) / _value_safe)
            )
        else:
            _informed_signals = {}

        # Vectorised ZI order generation — uses prev_price as price anchor.
        _zi_orders = self._batch_zi_orders(prev_price)

        for agent_id, player in self.agents.items():

            cash_start = float(player.cash)
            inventory_start = float(player.shares)

            if player.trader_type == "zi":
                signal = 1.0
                signal_error = 0.0
                strat_order = _zi_orders[agent_id]
            else:
                signal = float(_informed_signals[agent_id])
                signal_error = float(signal - _true_ratio)
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

        best_price, total_volume, trades = clear_market(order_list, previous_price=prev_price)

        exec_summary = defaultdict(lambda: {
            "executed_qty": 0.0,
            "executed_notional": 0.0,
            "cash_delta": 0.0,
            "inventory_delta": 0.0,
        })

        for trade in trades:
            aid = trade["agent_id"]
            qty = float(trade["quantity"])
            px = float(trade["price"])
            action = trade["action"]

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
            s = exec_summary[aid]
            executed_qty = s["executed_qty"]
            executed_notional = s["executed_notional"]
            cash_delta = s["cash_delta"]
            inventory_delta = s["inventory_delta"]

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