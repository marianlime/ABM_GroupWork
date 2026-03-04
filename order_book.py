"""
order_book.py
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
    trades : list of dict
        One entry per filled order, each with 'agent_id', 'quantity', 'side'.
    """
    # Separate buy and sell orders
    buys  = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']

    if not buys or not sells:
        return None, []

    # Get all unique prices
    all_prices = sorted(set([o['price'] for o in orders]))

    # ── Build numpy arrays once (avoids repeated Python-level iteration) ─────
    #
    # buy_prices / buy_qtys   : 1-D arrays aligned with `buys`
    # sell_prices / sell_qtys : 1-D arrays aligned with `sells`
    # price_levels            : 1-D array of every unique price (= all_prices)
    #
    buy_prices   = np.array([o['price']    for o in buys],  dtype=float)
    buy_qtys     = np.array([o['quantity'] for o in buys],  dtype=float)
    sell_prices  = np.array([o['price']    for o in sells], dtype=float)
    sell_qtys    = np.array([o['quantity'] for o in sells], dtype=float)
    price_levels = np.array(all_prices, dtype=float)   # shape (P,)

    # ── Vectorised demand / supply for all price levels simultaneously ───────
    #
    # Broadcasting:
    #   buy_prices[:, None]  shape (N_b, 1) >= price_levels shape (1, P)
    #   produces a boolean mask of shape (N_b, P).
    #   Multiplying by buy_qtys[:, None] and summing over axis 0 gives the
    #   total demand at every price in one numpy operation.
    #
    demand = (buy_qtys[:,  None] * (buy_prices[:,  None] >= price_levels)).sum(axis=0)  # (P,)
    supply = (sell_qtys[:, None] * (sell_prices[:, None] <= price_levels)).sum(axis=0)  # (P,)

    # ── Pass 1: find best_volume and best_imbalance ──────────────────────────
    #
    # Mirrors the original loop logic exactly, now over numpy arrays.
    #
    tradeable = (demand > 0) & (supply > 0)   # price levels with both sides present

    if not tradeable.any():
        return None, []

    volumes   = np.minimum(demand, supply)    # traded volume at each price level
    imbalance = np.abs(demand - supply)       # |demand - supply| at each price level

    # Mask out untradeable levels so they cannot win
    volumes[~tradeable]   = 0
    imbalance[~tradeable] = np.inf

    best_volume    = float(volumes.max())
    best_imbalance = float(imbalance[volumes == best_volume].min())

    if best_volume == 0:
        return None, []

    # ── Pass 2: collect all prices that achieve (best_volume, best_imbalance)
    candidate_mask = (volumes == best_volume) & (imbalance == best_imbalance)
    candidates     = price_levels[candidate_mask].tolist()   # plain Python list

    if len(candidates) == 1:
        best_price = candidates[0]

    elif previous_price is not None:
        # Rule 1: closest to previous clearing price (price continuity)
        best_price = min(candidates, key=lambda p: abs(p - previous_price))

    else:
        # Rule 2: midpoint of equilibrium range (symmetry / no prior info)
        best_price = (candidates[0] + candidates[-1]) / 2.0

    # Allocate trades using pro-rata rationing
    trades = allocate_trades(buys, sells, best_price, best_volume)

    return best_price, trades


def allocate_trades(buys, sells, price, total_volume):
    """
    Distribute total_volume units among eligible orders using pro-rata
    rationing.  Each order's fill is proportional to its size relative to
    the total eligible quantity on its side.

    Parameters
    ----------
    buys / sells   : list of order dicts (filtered to their respective sides)
    price          : confirmed clearing price
    total_volume   : maximum units to be traded (= min(demand, supply))

    Returns
    -------
    trades : list of dict with 'agent_id', 'quantity', 'side'
    """
    trades = []

    # Who can buy at this price?
    eligible_buys  = [o for o in buys  if o['price'] >= price]
    total_buy_qty  = sum(o['quantity'] for o in eligible_buys)

    # Who can sell at this price?
    eligible_sells = [o for o in sells if o['price'] <= price]
    total_sell_qty = sum(o['quantity'] for o in eligible_sells)

    # Pro-rata allocation for buyers
    for order in eligible_buys:
        fill = int(round(total_volume * order['quantity'] / total_buy_qty))
        fill = min(fill, order['quantity'])   # never exceed the order's own size
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': fill,
                'side': 'buy'
            })

    # Pro-rata allocation for sellers
    for order in eligible_sells:
        fill = int(round(total_volume * order['quantity'] / total_sell_qty))
        fill = min(fill, order['quantity'])
        if fill > 0:
            trades.append({
                'agent_id': order['agent_id'],
                'quantity': fill,
                'side': 'sell'
            })

    return trades


# Example usage
if __name__ == "__main__":

    # Create some orders
    orders = [
        {'agent_id': 1, 'price': 100, 'quantity': 10, 'side': 'buy'},
        {'agent_id': 2, 'price':  99, 'quantity': 15, 'side': 'buy'},
        {'agent_id': 3, 'price': 101, 'quantity':  8, 'side': 'sell'},
        {'agent_id': 4, 'price': 100, 'quantity': 12, 'side': 'sell'},
    ]

    # Clear the market
    price, trades = clear_market(orders, previous_price=99.5)

    print(f"Clearing Price: {price}")
    print(f"\nTrades:")
    for t in trades:
        print(f"  Agent {t['agent_id']}: {t['side'].upper()} {t['quantity']} shares")
