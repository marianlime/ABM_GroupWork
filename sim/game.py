"""
Core Game class that manages agent initialisation, per-round order collection,
market clearing, and portfolio accounting for one generation of the simulation.
"""

from collections import Counter, defaultdict
from .noise_signal import assign_info_param_set
from .trader import Trader
from .market import clear_market, TRANSACTION_COST_RATE
import hashlib
import numpy as np


def _pair_trade_executions(trades, experiment_id, generation_id, round_number):
    buy_trades = [
        {
            "agent_id": int(trade["agent_id"]),
            "quantity": float(trade["quantity"]),
            "price": float(trade["price"]),
        }
        for trade in trades
        if trade["action"] == "buy" and float(trade["quantity"]) > 0
    ]
    sell_trades = [
        {
            "agent_id": int(trade["agent_id"]),
            "quantity": float(trade["quantity"]),
            "price": float(trade["price"]),
        }
        for trade in trades
        if trade["action"] == "sell" and float(trade["quantity"]) > 0
    ]

    paired_records = []
    buy_index = 0
    sell_index = 0
    trade_id = 1

    while buy_index < len(buy_trades) and sell_index < len(sell_trades):
        buy_trade = buy_trades[buy_index]
        sell_trade = sell_trades[sell_index]
        matched_qty = min(buy_trade["quantity"], sell_trade["quantity"])
        if matched_qty <= 0:
            break

        trade_price = buy_trade["price"] if buy_trade["price"] == sell_trade["price"] else (
            buy_trade["price"] + sell_trade["price"]
        ) / 2.0
        paired_records.append(
            {
                "experiment_id": experiment_id,
                "generation_id": int(generation_id),
                "round_number": int(round_number),
                "trade_id": int(trade_id),
                "buyer_agent_id": int(buy_trade["agent_id"]),
                "seller_agent_id": int(sell_trade["agent_id"]),
                "price": float(trade_price),
                "quantity": float(matched_qty),
                "notional": float(matched_qty * trade_price),
            }
        )
        trade_id += 1

        buy_trade["quantity"] -= matched_qty
        sell_trade["quantity"] -= matched_qty
        if buy_trade["quantity"] <= 1e-12:
            buy_index += 1
        if sell_trade["quantity"] <= 1e-12:
            sell_index += 1

    return paired_records


def _stable_seed(namespace: str, experiment_id: str, generation_id: int) -> int:
    """Derive a stable, reproducible integer seed namespaced by purpose.

    Each distinct namespace (e.g. 'noise', 'zi', 'signal') produces an
    independent RNG stream so the three random processes don't interfere.
    """
    return int(hashlib.sha256(f"{namespace}_{experiment_id}_{generation_id}".encode()).hexdigest()[:16], 16)


class Game:
    """Encapsulates one generation of the market: agents, fundamentals, round state, and history."""

    def __init__(
        self,
        population_spec: list[dict],
        # --- Generation / Market ---
        n_rounds: int,
        total_initial_shares: float,
        total_initial_cash: float,
        experiment_id: str,
        generation_id: int,
        # --- Signal / Noise ---
        info_param_distribution_type: str,
        distribution_data: dict,
        signal_generator_noise_distribution: str,
        # --- Fundamentals ---
        S0: float | None = None,
        gbm_volatility: float = 0.2,
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
        self.strategy_type_counts = dict(_type_counts)

        self.n_rounds = n_rounds
        self.fundamental_path = fundamental_path
        self.total_initial_shares = total_initial_shares
        self.total_initial_cash = total_initial_cash
        self.S0 = S0
        self.gbm_volatility = float(gbm_volatility)
        self.experiment_id = experiment_id
        self.generation_id = int(generation_id)

        self.info_param_distribution_type = info_param_distribution_type
        self.distribution_data = distribution_data
        self.signal_noise_distribution = signal_generator_noise_distribution

        self.market_round_records = []
        self.agent_round_records = []
        self.trade_execution_records = []

        self.S0_effective = float(self.S0) if self.S0 is not None else 1.0

        self.order_history = {}
        self.price_history = {}

        n_informed = sum(1 for a in self.population_spec if a["trader_type"] != "zi")
        noise_seed = _stable_seed("noise", self.experiment_id, self.generation_id)
        self.info_param_set = assign_info_param_set(
            n_agents=n_informed,
            info_param_distribution_type=self.info_param_distribution_type,
            distribution_data=self.distribution_data,
            seed=noise_seed
        )

        cash_per_agent = self.total_initial_cash / self.n_agents
        shares_per_agent = self.total_initial_shares / self.n_agents

        from .evolution import VALID_STRATEGIES  # avoid circular import at module level

        informed_idx = 0
        for agent_id, agent_spec in enumerate(self.population_spec, start=1):
            trader_type = agent_spec["trader_type"]

            if trader_type not in VALID_STRATEGIES:
                raise ValueError(f"Unknown trader_type in population_spec: {trader_type}")

            # Use evolved info_param if present, otherwise sample from the distribution.
            if "info_param" in agent_spec:
                info_param = float(agent_spec["info_param"])
            elif trader_type != "zi":
                info_param = float(self.info_param_set[informed_idx])
                informed_idx += 1
            else:
                info_param = 0.0

            self.agents[agent_id] = Trader(
                agent_id=agent_id,
                cash=cash_per_agent,
                shares=shares_per_agent,
                info_param=info_param,
                trader_type=trader_type,
                strategy_params=agent_spec.get("strategy_params", {}),
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
        # generation ULID so results are repeatable, and isolated from the noise-parameter
        # RNG so the two streams don't interfere with each other.
        self._zi_rng = np.random.default_rng(
            _stable_seed("zi", self.experiment_id, self.generation_id)
        )

        # Dedicated, reproducible RNG for informed-agent signal generation —
        # isolated from both the ZI and noise-parameter streams.
        self._signal_rng = np.random.default_rng(
            _stable_seed("signal", self.experiment_id, self.generation_id)
        )

    def _batch_zi_orders(self, value: float) -> dict:
        """
        Generate Zero-Intelligence orders for all ZI agents in one vectorised batch.

        ZI agents observe the current fundamental value as public information
        (analogous to a company balance sheet) and place random orders dispersed
        around it. Prices are drawn from a lognormal centred on the fundamental,
        with spread set by market volatility. Direction is equally likely buy or
        sell; quantity is a uniform random fraction of the agent's current portfolio.

        Returns {agent_id: order_dict | None} where None means the agent cannot trade.
        """
        if not self._zi_agent_ids:
            return {}

        n         = len(self._zi_agent_ids)
        zi_cash   = np.array([self.agents[aid].cash   for aid in self._zi_agent_ids])
        zi_shares = np.array([self.agents[aid].shares for aid in self._zi_agent_ids])

        # Price: lognormal centred on the current fundamental value
        prices = np.maximum(
            self._zi_rng.lognormal(np.log(max(value, 1e-12)), self.gbm_volatility, n),
            0.01,
        )

        # Direction: random 50/50, with fallback if preferred side is infeasible
        want_buy = self._zi_rng.random(n) < 0.5
        can_buy  = zi_cash  > 0
        can_sell = zi_shares > 0

        do_buy  = (want_buy  & can_buy)  | (~want_buy & ~can_sell & can_buy)
        do_sell = (~want_buy & can_sell) | (want_buy  & ~can_buy  & can_sell)

        # Quantity: uniform random fraction of available resources (symmetric)
        u        = self._zi_rng.random(n)
        buy_qty  = (zi_cash / prices) * u
        sell_qty = zi_shares * u

        orders = {}
        for i, aid in enumerate(self._zi_agent_ids):
            if do_buy[i] and buy_qty[i] > 0:
                orders[aid] = {
                    "Price": float(prices[i]), "Quantity": float(buy_qty[i]),
                    "Buy": 1.0, "Sell": 0.0, "Hold": 0.0, "agent_id": aid,
                }
            elif do_sell[i] and sell_qty[i] > 0:
                orders[aid] = {
                    "Price": float(prices[i]), "Quantity": float(sell_qty[i]),
                    "Buy": 0.0, "Sell": 1.0, "Hold": 0.0, "agent_id": aid,
                }
            else:
                orders[aid] = None
        return orders

    def gather_orders_and_clear(self, current_round):
        """
        Collect orders from all agents, clear the market, and record round outcomes.

        - Generates vectorised signals for informed agents and ZI orders in one batch
        - Appends entries to market_round_records and agent_round_records
        - Returns (best_price, total_volume, trades)
        """

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

        prev_price = self.price_history.get(current_round - 1, None)

        # Vectorised signal generation — one numpy call for all informed agents.
        if self._informed_agent_ids:
            if self.signal_noise_distribution == 'lognormal':
                _multipliers = np.exp(
                    self._signal_rng.normal(0.0, self._informed_noise_params)
                )
            elif self.signal_noise_distribution == 'uniform':
                _lows = np.maximum(1.0 - self._informed_noise_params, 1e-6)
                _highs = 1.0 + self._informed_noise_params
                _u = self._signal_rng.uniform(0, 1, len(self._informed_noise_params))
                _multipliers = _lows + _u * (_highs - _lows)
            else:
                raise ValueError(f"Unknown noise_distribution: {self.signal_noise_distribution}")
            _informed_signals = dict(
                zip(self._informed_agent_ids, (S_next * _multipliers) / _value_safe)
            )
        else:
            _informed_signals = {}

        # Vectorised ZI order generation — anchored on the current fundamental value.
        _zi_orders = self._batch_zi_orders(value)

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
                "experiment_id": self.experiment_id,
                "generation_id": self.generation_id,
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

        buy_orders, sell_orders = [], []
        best_bid, best_ask = None, None
        bid_depth_total, ask_depth_total = 0.0, 0.0
        price_level_bid_set, price_level_ask_set = set(), set()

        for o in order_list:
            px  = o["price"]
            qty = o["quantity"]
            if o["action"] == "buy":
                buy_orders.append(o)
                bid_depth_total += qty
                price_level_bid_set.add(px)
                if best_bid is None or px > best_bid:
                    best_bid = px
            else:
                sell_orders.append(o)
                ask_depth_total += qty
                price_level_ask_set.add(px)
                if best_ask is None or px < best_ask:
                    best_ask = px

        n_active_buyers  = len(buy_orders)
        n_active_sellers = len(sell_orders)
        n_active_total   = len(order_list)
        price_levels_bid = len(price_level_bid_set)
        price_levels_ask = len(price_level_ask_set)
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
            fee = trade_value * TRANSACTION_COST_RATE

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
            # No clearing: record total order imbalance so failed clearings can be analysed.
            demand_at_p = float(sum(o["quantity"] for o in buy_orders))
            supply_at_p = float(sum(o["quantity"] for o in sell_orders))

        self.order_history[current_round] = order_list
        self.price_history[current_round] = best_price

        self.market_round_records.append({
            "experiment_id": self.experiment_id,
            "generation_id": self.generation_id,
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
        self.trade_execution_records.extend(
            _pair_trade_executions(
                trades,
                experiment_id=self.experiment_id,
                generation_id=self.generation_id,
                round_number=current_round,
            )
        )

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
        """Apply a filled trade to the relevant agent's cash and share holdings."""
        agent = self.agents[trade["agent_id"]]

        qty = float(trade["quantity"])
        px = float(trade["price"])
        action = trade["action"]

        trade_value = qty * px
        fee = trade_value * TRANSACTION_COST_RATE

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
        """Return the terminal wealth of an agent by valuing shares at the final fundamental price."""
        return float(self.fundamental_path[-1]) * float(self.agents[agent_id].shares) + float(self.agents[agent_id].cash)
