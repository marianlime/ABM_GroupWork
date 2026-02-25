import numpy as np
import random

# ----------------------------
# Noise + signal helpers
# ----------------------------

def assign_noise_parameter_set(num_agents, dist_type="uniform"):
    if dist_type == "uniform":
        return np.random.uniform(0.0, 0.5, num_agents)
    elif dist_type == "evenly_spaced":
        return np.linspace(0, 1, num_agents)
    elif dist_type == "bimodal":
        group_a = np.random.normal(0.1, 0.05, num_agents // 2)
        group_b = np.random.normal(0.9, 0.05, num_agents // 2)
        return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)
    elif dist_type == "skewed":
        return np.random.lognormal(-1, 0.5, num_agents)

def signal_generator(noise_parameter, S_next, noise_distribution='lognormal'):
    if noise_distribution == 'lognormal':
        return S_next * np.exp(np.random.normal(0, noise_parameter))
    if noise_distribution == 'uniform':
        return 0

# ----------------------------
# Strategy functions
# ----------------------------

def zero_intelligence(signal: float, cash: float, shares: float, value: float) -> dict:
    can_buy = cash > 0 and value > 0
    can_sell = shares > 0

    if not can_buy and not can_sell:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    if can_buy and can_sell:
        action = random.choice(["buy", "sell"])
    elif can_buy:
        action = "buy"
    else:
        action = "sell"

    price_range = 0.2
    low = value * (1 - price_range)
    high = value * (1 + price_range)
    price = round(random.uniform(max(low, 0.01), high), 2)

    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        if max_qty <= 0:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
        eps = 1e-6
        quantity = round(random.uniform(eps, max_qty), 6)
    else:
        quantity = round(random.uniform(1, shares), 2) if shares >= 1 else shares

    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}

def signal_following(signal: float, cash: float, shares: float, value: float, aggression: float = 1.0) -> dict:
    action = "buy" if signal > 1.0 else "sell"

    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    price = max(round(value * signal, 2), 0.01)
    conviction = abs(signal - 1.0)
    fraction = min(conviction * aggression * 10, 1.0)
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        quantity = round(max_qty * fraction, 6)
        quantity = min(quantity, max_qty)
    else:
        max_qty = shares
        quantity = round(max_qty * fraction, 6)
        quantity = min(quantity, max_qty)

    if quantity <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}

STRATEGIES = {"zi": zero_intelligence, "signal_following": signal_following}
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

# ----------------------------
# Market clearing
# ----------------------------

def clear_market(orders, previous_price=None):
    buys = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']

    if not buys or not sells:
        return None, 0, []

    all_prices = sorted(set([o['price'] for o in orders]))

    best_volume = 0
    best_imbalance = float('inf')

    for price in all_prices:
        demand = sum(o['quantity'] for o in buys  if o['price'] >= price)
        supply = sum(o['quantity'] for o in sells if o['price'] <= price)

        if demand == 0 or supply == 0:
            continue

        volume = min(demand, supply)
        imbalance = abs(demand - supply)

        if volume > best_volume:
            best_volume = volume
            best_imbalance = imbalance
        elif volume == best_volume and imbalance < best_imbalance:
            best_imbalance = imbalance

    if best_volume == 0:
        return None, 0, []

    candidates = [
        p for p in all_prices
        if min(
            sum(o['quantity'] for o in buys  if o['price'] >= p),
            sum(o['quantity'] for o in sells if o['price'] <= p)
        ) == best_volume
        and abs(
            sum(o['quantity'] for o in buys  if o['price'] >= p) -
            sum(o['quantity'] for o in sells if o['price'] <= p)
        ) == best_imbalance
    ]

    if len(candidates) == 1:
        best_price = candidates[0]
    elif previous_price is not None:
        best_price = min(candidates, key=lambda p: abs(p - previous_price))
    else:
        best_price = (candidates[0] + candidates[-1]) / 2.0

    trades = allocate_trades(buys, sells, best_price, best_volume)
    return best_price, best_volume, trades


def allocate_trades(buys, sells, price, total_volume):
    trades = []

    eligible_buys = [o for o in buys if o['price'] >= price]
    total_buy_qty = sum(o['quantity'] for o in eligible_buys)

    eligible_sells = [o for o in sells if o['price'] <= price]
    total_sell_qty = sum(o['quantity'] for o in eligible_sells)

    if total_buy_qty <= 0 or total_sell_qty <= 0:
        return []

    for order in eligible_buys:
        fill = total_volume * order['quantity'] / total_buy_qty
        fill = min(fill, order['quantity'])
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': float(fill),
                'price': float(price),
                'side': 'buy'
            })

    for order in eligible_sells:
        fill = total_volume * order['quantity'] / total_sell_qty
        fill = min(fill, order['quantity'])
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': float(fill),
                'price': float(price),
                'side': 'sell'
            })

    return trades

# ----------------------------
# GBM
# ----------------------------

def simulate_gbm(S_0, volatility, drift, n_rounds):
    Z = np.zeros(n_rounds + 1)
    Z[0] = np.log(S_0)
    # include drift minimally (per-step drift term)
    Z[1:] = (drift - 0.5 * volatility**2) + volatility * np.random.standard_normal(n_rounds)
    S_t = np.exp(np.cumsum(Z))
    return {"stock_path": S_t, "shock_path": Z}

# ----------------------------
# Game
# ----------------------------

class game:
    def __init__(self, n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
                 total_initial_shares, cash_to_share_ratio, run_id, stock_path, shock_path):
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
        self.total_initial_cash = total_initial_shares * total_initial_shares * cash_to_share_ratio
        self.run_id = run_id

        self.order_history = {}
        self.price_history = {}
        self.noise_parameter_set = assign_noise_parameter_set(self.n_agents, dist_type="evenly_spaced")

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
                raw_signal = signal_generator(player.info_param, S_next=S_next, noise_distribution='lognormal')
                # convert to multiplier around 1.0 so signal_following behaves as intended
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

# ----------------------------
# Runner
# ----------------------------

def play_game(n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
              total_initial_shares, cash_to_share_ratio, run_id):

    gbm_path = simulate_gbm(S0, volatility, drift, n_rounds)
    stock_path = gbm_path['stock_path']
    shock_path = gbm_path['shock_path']

    current_game = game(n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
                        total_initial_shares, cash_to_share_ratio, run_id, stock_path, shock_path)

    while current_game.current_round < n_rounds:
        best_price, total_volume, trades = current_game.gather_orders_and_clear(current_game.current_round)
        for trade in trades:
            current_game.update_portfolio(trade)
        current_game.current_round += 1

    final_score = []
    for agent_id in current_game.agents:
        final_score.append((agent_id, current_game.liquidate_assets(agent_id)))

    return final_score, current_game

# ----------------------------
# Example params + run
# ----------------------------

n_strategic_agents = 100
n_zi_agents = 100
n_rounds = 1000
S0 = 100
volatility = 0.1
drift = 0
total_initial_shares = 100
cash_to_share_ratio = 1
run_id = 1

final_score, g = play_game(n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
                           total_initial_shares, cash_to_share_ratio, run_id)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_game_results(g, final_score, n_strategic_agents=None, title_prefix=""):
    """
    Post-run diagnostics for the game.

    Inputs:
      g            : game instance (must have agents, stock_path, price_history/order_history)
      final_score  : list of tuples (agent_id, final_wealth)
      n_strategic_agents: optional int, if you want to label types by id cutoff
      title_prefix : optional str for plot titles

    Returns:
      results_df : pandas DataFrame with per-agent outcomes + metadata
    """

    # ----- Build per-agent results table -----
    wealth_map = dict(final_score)

    rows = []
    for aid, agent in g.agents.items():
        # infer type
        if hasattr(agent, "trader_type"):
            ttype = agent.trader_type
        elif n_strategic_agents is not None:
            ttype = "signal_following" if aid < n_strategic_agents else "zi"
        else:
            ttype = "unknown"

        info_param = getattr(agent, "info_param", np.nan)

        rows.append({
            "agent_id": aid,
            "type": ttype,
            "info_param": float(info_param) if info_param is not None else np.nan,
            "cash_final": float(getattr(agent, "cash", np.nan)),
            "shares_final": float(getattr(agent, "shares", np.nan)),
            "wealth_final": float(wealth_map.get(aid, np.nan)),
        })

    df = pd.DataFrame(rows).sort_values("wealth_final", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # ----- Print leaderboard + summaries -----
    print("\n=== Top 10 (by final wealth) ===")
    print(df.loc[:9, ["rank", "agent_id", "type", "info_param", "wealth_final"]].to_string(index=False))

    print("\n=== Bottom 10 (by final wealth) ===")
    print(df.loc[len(df)-10:, ["rank", "agent_id", "type", "info_param", "wealth_final"]].to_string(index=False))

    print("\n=== Summary by type ===")
    print(df.groupby("type")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"]).to_string())

    # ----- Info parameter effect diagnostics (informed only) -----
    informed = df[df["type"] == "signal_following"].copy()
    if len(informed) > 5 and informed["info_param"].notna().any():
        # correlation (note: your info_param is "noise sigma" so higher = worse info)
        corr = informed[["info_param", "wealth_final"]].corr().iloc[0, 1]
        print(f"\n=== Informed only: corr(info_param, wealth_final) = {corr:.4f} ===")
        print("Interpretation: if info_param is noise sigma, you'd expect NEGATIVE correlation (more noise -> worse).")

        # binned means (quintiles)
        try:
            informed["info_bin"] = pd.qcut(informed["info_param"], q=5, duplicates="drop")
            bin_stats = informed.groupby("info_bin")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"])
            print("\n=== Informed only: wealth by info_param quintile ===")
            print(bin_stats.to_string())
        except Exception as e:
            print("\n(Binning info_param failed; likely too many identical values.)", e)
    else:
        print("\n(No sufficient informed/info_param data to analyze info effects.)")

    # ----- Time-series: fundamental vs clearing price -----
    # Fundamental path (your simulated stock path)
    fundamental = np.array(g.stock_path, dtype=float)

    # Clearing price per round (some rounds may be None)
    # price_history keys are typically rounds 0..n_rounds-1
    n_rounds = len(fundamental) - 1
    clearing = []
    for t in range(n_rounds):
        p = g.price_history.get(t, None)
        clearing.append(np.nan if p is None else float(p))
    clearing = np.array(clearing, dtype=float)

    # Plot: fundamental (S_t) and clearing (p_t)
    plt.figure()
    plt.plot(range(n_rounds), fundamental[:n_rounds], label="Fundamental (S_t)")
    plt.plot(range(n_rounds), clearing, label="Clearing price")
    plt.xlabel("Round")
    plt.ylabel("Price")
    plt.title(f"{title_prefix} Fundamental vs Clearing Price".strip())
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot: number of rounds with no clearing
    no_clear = np.isnan(clearing).sum()
    print(f"\nRounds: {n_rounds}, no-clearing rounds: {no_clear} ({no_clear/n_rounds:.1%})")

    # ----- Wealth distribution plots -----
    plt.figure()
    for ttype, sub in df.groupby("type"):
        plt.hist(sub["wealth_final"].values, bins=30, alpha=0.6, label=ttype)
    plt.xlabel("Final wealth")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Wealth distribution by type".strip())
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Wealth vs info_param scatter (informed) -----
    if len(informed) > 5 and informed["info_param"].notna().any():
        plt.figure()
        plt.scatter(informed["info_param"].values, informed["wealth_final"].values)
        plt.xlabel("info_param (noise sigma)")
        plt.ylabel("Final wealth")
        plt.title(f"{title_prefix} Informed: wealth vs info_param".strip())
        plt.tight_layout()
        plt.show()

    return df



# Example usage at end of your script:
# final_score, g = play_game(...)
# results_df = analyze_game_results(g, final_score, n_strategic_agents=n_strategic_agents, title_prefix=f"Run {run_id} |")

analyze_game_results(g, final_score, n_strategic_agents=None, title_prefix="")
print("Example (all results):", final_score)
print("Last clearing price:", g.price_history.get(n_rounds - 1))