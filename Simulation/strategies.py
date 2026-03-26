# ----------------------------
# Strategy functions
# ----------------------------

# Note: ZI ("zi") agents are NOT routed through this module.
# Their orders are generated in bulk by Game._batch_zi_orders, which is
# vectorised, reproducibly seeded, and uses the correct price anchor
# (previous clearing price rather than the fundamental value).

_HOLD = {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}


def parameterised_informed(signal: float, cash: float, shares: float, value: float, qty_aggression: float = 0.5, signal_aggression: float = 0.5, info_param: float = 0.0) -> dict:
    
    """
    Single parameterised strategy for all informed agents.
    Evolution operates on two continuous parameters:

    qty_aggression     ∈ [0,1]   : fraction of available resources deployed, scaled by conviction
    signal_aggression  ∈ [0,1]   : 0 = price at fundamental, 1 = price at full perceived value
    info_param                   : agent's signal noise std (passed from Trader, not evolved
                                   via strategy_params)
    """
    edge = signal - 1.0

    if abs(edge) < 1e-8:
        return _HOLD

    action = "buy" if edge > 0 else "sell"

    if action == "buy" and cash <= 0:
        return _HOLD
    if action == "sell" and shares <= 0:
        return _HOLD

    price    = max(value * (1.0 + signal_aggression * edge), 0.01)
    fraction = qty_aggression

    if action == "buy":
        max_qty = cash / price if price > 0 else 0.0
    else:
        max_qty = shares

    quantity = max_qty * fraction

    if quantity <= 0:
        return _HOLD

    return {
        "Price":    price,
        "Quantity": quantity,
        "Buy":      1.0 if action == "buy" else 0.0,
        "Sell":     1.0 if action == "sell" else 0.0,
        "Hold":     0.0,
    }


STRATEGIES = {
    "parameterised_informed": parameterised_informed,
    # "zi" is intentionally absent — ZI orders bypass place_order entirely
    # and are generated in bulk by Game._batch_zi_orders.
}
