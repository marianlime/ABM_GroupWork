"""
market.py
=============
A call-auction order-book clearing engine.

Algorithm summary
-----------------
1. For every candidate price level, compute total demand (buy orders willing
   to pay >= price) and total supply (sell orders willing to accept <= price).
2. Pick the price that maximises traded volume; break ties by minimum
   |demand - supply| imbalance, then by proximity to the previous clearing
   price (or the midpoint of the equilibrium range if no history exists).
3. Allocate fills to eligible orders using *pro-rata* rationing so that
   larger orders receive proportionally larger fills.

Numpy vectorisation
-------------------
All per-price demand/supply calculations are done with numpy broadcasting
instead of nested Python loops.  A single (N_orders × N_prices) boolean
matrix replaces the inner list-comprehension loop, so the hot path runs
entirely inside numpy's C layer.
"""

import numpy as np


def clear_market(orders, previous_price=None):
    """
    Find the single clearing price and allocate trades for a batch of orders.

    Parameters
    ----------
    orders : list of dict
        Each dict must contain 'agent_id', 'price', 'quantity', and 'side'
        ('buy' or 'sell').
    previous_price : float, optional
        Last known clearing price used to break ties (price-continuity rule).

    Returns
    -------
    best_price : float or None
        The clearing price, or None when no trade is possible.
    best_volume : float
        Total volume traded at the clearing price.
    trades : list of dict
        One entry per filled order, each with 'agent_id', 'quantity', 'price', 'side'.
    """
    buys  = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']

    if not buys or not sells:
        return None, 0, []

    all_prices = sorted(set([o['price'] for o in orders]))

    # ── Build numpy arrays once ──────────────────────────────────────────────
    buy_prices   = np.array([o['price']    for o in buys],  dtype=float)
    buy_qtys     = np.array([o['quantity'] for o in buys],  dtype=float)
    sell_prices  = np.array([o['price']    for o in sells], dtype=float)
    sell_qtys    = np.array([o['quantity'] for o in sells], dtype=float)
    price_levels = np.array(all_prices, dtype=float)          # shape (P,)

    # ── Vectorised demand / supply for all price levels simultaneously ───────
    demand = (buy_qtys[:,  None] * (buy_prices[:,  None] >= price_levels)).sum(axis=0)
    supply = (sell_qtys[:, None] * (sell_prices[:, None] <= price_levels)).sum(axis=0)

    # ── Pass 1: find best_volume and best_imbalance ──────────────────────────
    tradeable = (demand > 0) & (supply > 0)

    if not tradeable.any():
        return None, 0, []

    volumes   = np.minimum(demand, supply)
    imbalance = np.abs(demand - supply)

    volumes[~tradeable]   = 0
    imbalance[~tradeable] = np.inf

    best_volume    = float(volumes.max())
    best_imbalance = float(imbalance[volumes == best_volume].min())

    if best_volume == 0:
        return None, 0, []

    # ── Pass 2: collect candidate prices ────────────────────────────────────
    candidate_mask = (volumes == best_volume) & (imbalance == best_imbalance)
    candidates     = price_levels[candidate_mask].tolist()

    if len(candidates) == 1:
        best_price = candidates[0]
    elif previous_price is not None:
        best_price = min(candidates, key=lambda p: abs(p - previous_price))
    else:
        best_price = (candidates[0] + candidates[-1]) / 2.0

    trades = allocate_trades(buys, sells, best_price, best_volume)

    return best_price, best_volume, trades          # ← 3-tuple: matches game.py


def allocate_trades(buys, sells, price, total_volume):
    """
    Distribute total_volume among eligible orders using pro-rata rationing.
    Fills are kept as floats to stay consistent with the rest of the codebase.
    """
    trades = []

    eligible_buys  = [o for o in buys  if o['price'] >= price]
    total_buy_qty  = sum(o['quantity'] for o in eligible_buys)

    eligible_sells = [o for o in sells if o['price'] <= price]
    total_sell_qty = sum(o['quantity'] for o in eligible_sells)

    if total_buy_qty <= 0 or total_sell_qty <= 0:
        return []

    for order in eligible_buys:
        fill = round(total_volume * order['quantity'] / total_buy_qty, 6)  # ← float, not int
        fill = min(fill, order['quantity'])
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': float(fill),
                'price':    float(price),
                'side':     'buy'
            })

    for order in eligible_sells:
        fill = round(total_volume * order['quantity'] / total_sell_qty, 6)  # ← float, not int
        fill = min(fill, order['quantity'])
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': float(fill),
                'price':    float(price),
                'side':     'sell'
            })

    return trades


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orders = [
        {'agent_id': 1, 'price': 100, 'quantity': 10, 'side': 'buy'},
        {'agent_id': 2, 'price':  99, 'quantity': 15, 'side': 'buy'},
        {'agent_id': 3, 'price': 101, 'quantity':  8, 'side': 'sell'},
        {'agent_id': 4, 'price': 100, 'quantity': 12, 'side': 'sell'},
    ]

    price, volume, trades = clear_market(orders, previous_price=99.5)

    print(f"Clearing Price: {price}")
    print(f"Total Volume:   {volume}")
    print(f"\nTrades:")
    for t in trades:
        print(f"  Agent {t['agent_id']}: {t['side'].upper()} {t['quantity']} shares @ {t['price']}")
