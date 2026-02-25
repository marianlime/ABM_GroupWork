import numpy as np


def clear_market(orders, previous_price=None):
    
    # Separate buy and sell orders
    buys = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']

    if not buys or not sells:
        return None, []

    # Get all unique prices
    all_prices = sorted(set([o['price'] for o in orders]))

    # ── Pass 1: find best volume and minimum imbalance ────────────────────
    best_volume = 0
    best_imbalance = float('inf')

    for price in all_prices:
        demand = sum(o['quantity'] for o in buys  if o['price'] >= price)
        supply = sum(o['quantity'] for o in sells if o['price'] <= price)

        if demand == 0 or supply == 0:
            continue

        volume    = min(demand, supply)
        imbalance = abs(demand - supply)

        if volume > best_volume:
            best_volume    = volume
            best_imbalance = imbalance
        elif volume == best_volume and imbalance < best_imbalance:
            best_imbalance = imbalance

    if best_volume == 0:
        return None, []

    # ── Pass 2: collect all prices that achieve (best_volume, best_imbalance)
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
        # Rule 1: closest to previous clearing price (price continuity)
        best_price = min(candidates, key=lambda p: abs(p - previous_price))

    else:
        # Rule 2: midpoint of equilibrium range (symmetry / no prior info)
        best_price = (candidates[0] + candidates[-1]) / 2.0

    # Allocate trades using pro-rata rationing
    trades = allocate_trades(buys, sells, best_price, best_volume)

    return best_price, best_volume ,trades


def allocate_trades(buys, sells, price, total_volume):
    trades = []
    
    # Who can buy at this price?
    eligible_buys = [o for o in buys if o['price'] >= price]
    total_buy_qty = sum(o['quantity'] for o in eligible_buys)
    
    # Who can sell at this price?
    eligible_sells = [o for o in sells if o['price'] <= price]
    total_sell_qty = sum(o['quantity'] for o in eligible_sells)
    
    # Pro-rata allocation for buyers
    for order in eligible_buys:
        fill = int(round(total_volume * order['quantity'] / total_buy_qty))
        fill = min(fill, order['quantity'])
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
        {'agent_id': 2, 'price': 99, 'quantity': 15, 'side': 'buy'},
        {'agent_id': 3, 'price': 101, 'quantity': 8, 'side': 'sell'},
        {'agent_id': 4, 'price': 100, 'quantity': 12, 'side': 'sell'},
    ]
    
    # Clear the market
    price, trades = clear_market(orders, previous_price=99.5)
    
    print(f"Clearing Price: {price}")
    print(f"\nTrades:")
    for t in trades:
        print(f"  Agent {t['agent_id']}: {t['side'].upper()} {t['quantity']} shares")
