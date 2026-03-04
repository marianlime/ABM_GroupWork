import random

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