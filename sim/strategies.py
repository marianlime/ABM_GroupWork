import random
import numpy as np

# ----------------------------
# Strategy functions
# ----------------------------

def zero_intelligence(signal: float, cash: float, shares: float, value: float,
                      gbm_volatility: float = 0.2) -> dict:
    """
    Reference implementation only — not called during simulation.
    ZI orders are generated in bulk by Game._batch_zi_orders for performance.

    gbm_volatility: GBM sigma used as the log-normal spread parameter.
    """
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

    price = max(round(float(np.random.lognormal(np.log(value), gbm_volatility)), 2), 0.01)

    eps = 1e-6
    max_dollar = max(min(cash, shares * price), eps)

    if action == "buy":
        max_qty = max_dollar / price if price > 0 else 0
        if max_qty <= 0:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
        quantity = round(random.uniform(eps / price, max_qty), 6)
    else:
        sell_dollar = random.uniform(eps, max_dollar)
        quantity = round(sell_dollar / price, 6)

    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}


def parameterised_informed(
    signal: float,
    cash: float,
    shares: float,
    value: float,
    qty_aggression: float = 1.0,
    signal_aggression: float = 1.0,
    threshold: float = 0.0,
    signal_clip: float = 0.50,  # evolved parameter
    min_qty_fraction: float = 0.01,
) -> dict:
    """
    Single parameterised strategy for all informed agents.
    Evolution operates on the three continuous parameters.

    qty_aggression     ∈ [0,1]   : scales order size per unit of conviction
    signal_aggression  ∈ [0,1]   : 0 = post at current value, 1 = post at full signal price
    threshold          ∈ [0,1]   : no-trade band — hold when |signal − 1| < threshold
    signal_clip        ∈ [0,1]   : clamps signal to [1 − clip, 1 + clip] before acting;
                                   must exceed threshold for the agent to ever trade
    """
    signal = float(np.clip(signal, 1.0 - signal_clip, 1.0 + signal_clip))
    edge = signal - 1.0

    if abs(edge) < threshold:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    if abs(edge) < 1e-8:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    action = "buy" if edge > 0 else "sell"

    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    patient_level = 1.0 + signal_aggression * edge
    price = max(round(value * patient_level, 2), 0.01)

    conviction = abs(edge)
    fraction = min(conviction * qty_aggression * 10.0, 1.0)

    if action == "buy":
        max_qty = cash / price if price > 0 else 0.0
        if max_qty <= 0:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))
    else:
        max_qty = shares
        if max_qty <= 0:
            return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))

    if quantity <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    return {
        "Price": price,
        "Quantity": quantity,
        "Buy": 1.0 if action == "buy" else 0.0,
        "Sell": 1.0 if action == "sell" else 0.0,
        "Hold": 0.0,
    }


STRATEGIES = {
    "parameterised_informed": parameterised_informed,
}
