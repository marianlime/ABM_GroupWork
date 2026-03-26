import numpy as np
from numba import njit

# ── Fixed market configuration ────────────────────────────────────────────────
# These are not exposed as run parameters — edit here to change market behaviour.
MARKET_MECHANISM     = "call_auction"
PRICING_RULE         = "maximum_volume_minimum_imbalance"
RATIONING_RULE       = "proportional_rationing"
TIE_BREAK_RULE       = "previous_price_proximity"
TRANSACTION_COST_RATE = 0.0   # fraction of trade value charged as fee


def clear_market(orders, previous_price=None):

    buys  = [o for o in orders if o['action'] == 'buy']
    sells = [o for o in orders if o['action'] == 'sell']

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

    eligible_buys  = [o for o in buys  if o['price'] >= price]
    eligible_sells = [o for o in sells if o['price'] <= price]

    if not eligible_buys or not eligible_sells:
        return []

    buy_agent_ids = np.array([o['agent_id'] for o in eligible_buys], dtype=int)
    buy_qtys      = np.array([o['quantity'] for o in eligible_buys], dtype=float)
    sell_agent_ids = np.array([o['agent_id'] for o in eligible_sells], dtype=int)
    sell_qtys      = np.array([o['quantity'] for o in eligible_sells], dtype=float)

    trades = []
    
    buy_fills = _allocate_fills_numba(total_volume, buy_qtys)
    for i in range(len(buy_agent_ids)):
        fill = min(buy_fills[i], buy_qtys[i])
        if fill > 0:
            trades.append({
                'agent_id': int(buy_agent_ids[i]),
                'quantity': float(fill),
                'price':    float(price),
                'action':   'buy'
            })

    sell_fills = _allocate_fills_numba(total_volume, sell_qtys)
    for i in range(len(sell_agent_ids)):
        fill = min(sell_fills[i], sell_qtys[i])
        if fill > 0:
            trades.append({
                'agent_id': int(sell_agent_ids[i]),
                'quantity': float(fill),
                'price':    float(price),
                'action':   'sell'
            })

    return trades

@njit
def _allocate_fills_numba(total_volume, qtys):
    n = len(qtys)
    fills = np.zeros(n)
    total_qty = qtys.sum()
    if total_qty <= 0:
        return fills
    for i in range(n):
        fills[i] = round(total_volume * qtys[i] / total_qty, 6)
    return fills


# ── Example usage ───────────────────────────-────────────────────────────────
if __name__ == "__main__":
    orders = [
        {'agent_id': 1, 'price': 100, 'quantity': 10, 'action': 'buy'},
        {'agent_id': 2, 'price':  99, 'quantity': 15, 'action': 'buy'},
        {'agent_id': 3, 'price': 101, 'quantity':  8, 'action': 'sell'},
        {'agent_id': 4, 'price': 100, 'quantity': 12, 'action': 'sell'},
    ]

    price, volume, trades = clear_market(orders, previous_price=99.5)

    print(f"Clearing Price: {price}")
    print(f"Total Volume:   {volume}")
    print(f"\nTrades:")
    for t in trades:
        print(f"  Agent {t['agent_id']}: {t['action'].upper()} {t['quantity']} shares @ {t['price']}")
