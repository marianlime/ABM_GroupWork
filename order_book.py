import numpy as np


def clear_market(orders, previous_price=None):
    # Separate buy and sell orders
    buys = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']
    
    if not buys or not sells:
        return None, []
    
    # Get all unique prices
    all_prices = sorted(set([o['price'] for o in orders]))
    
    # Find best clearing price
    best_price = None
    best_volume = 0
    best_imbalance = float('inf')
    
    for price in all_prices:
        # How much demand at this price? (buyers willing to pay >= price)
        demand = sum(o['quantity'] for o in buys if o['price'] >= price)
        
        # How much supply at this price? (sellers willing to accept <= price)
        supply = sum(o['quantity'] for o in sells if o['price'] <= price)
        
        # Can we trade?
        if demand == 0 or supply == 0:
            continue
        
        volume = min(demand, supply)
        imbalance = abs(demand - supply)
        
        # Better than current best?
        is_better = False
        if volume > best_volume:
            is_better = True
        elif volume == best_volume and imbalance < best_imbalance:
            is_better = True
        elif volume == best_volume and imbalance == best_imbalance:
            # Tie-breaker: closest to previous price
            if previous_price is not None:
                if best_price is None:
                    is_better = True
                elif abs(price - previous_price) < abs(best_price - previous_price):
                    is_better = True
        
        if is_better:
            best_price = price
            best_volume = volume
            best_imbalance = imbalance
    
    if best_price is None:
        return None, []
    
    # Allocate trades using pro-rata rationing
    trades = allocate_trades(buys, sells, best_price, best_volume)
    
    return best_price, trades


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
