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