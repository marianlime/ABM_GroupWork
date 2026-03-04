import random
import numpy as np

# ----------------------------
# Strategy functions
# ----------------------------

def zero_intelligence(signal: float, cash: float, shares: float, value: float) -> dict:
    can_buy = cash > 0 and value > 0
    can_sell = shares > 0

    # If neither buying nor selling is possible, return a hold action.
    if not can_buy and not can_sell:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Randomly choose to buy or sell if both are possible, otherwise do the one that's possible.
    if can_buy and can_sell:
        action = random.choice(["buy", "sell"])
    elif can_buy:
        action = "buy"
    else:
        action = "sell"

    # Price drawn from normal distribution centred on current value.
    # std = value * price_range so spread scales with price level.
    # Clipped to a minimum of 0.01 to prevent non-positive prices.
    price_range = 0.2
    price = max(round(float(np.random.normal(value, value * price_range)), 2), 0.01)

    # Quantity is random fraction of max affordable (for buys) or max sellable (for sells),
    # with a small epsilon to ensure non-zero quantity when possible.
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


def signal_following(signal: float, cash: float, shares: float, value: float,
                     aggression: float = 1.0, signal_clip: float = 0.25,
                     min_qty_fraction: float = 0.01) -> dict:
    """
    Signal-following strategy. The agent buys when the signal implies the next
    price is above the current value, and sells when it implies below.

    signal_clip      : clips signal to [1 - signal_clip, 1 + signal_clip] before acting.
                       Without this, high-noise agents (large info_param) receive signals
                       many multiples away from 1.0 (e.g. exp(N(0,1)) can be ~7x or ~0.1x),
                       causing extreme prices and full-conviction orders every round.
    min_qty_fraction : minimum quantity as a fraction of max_qty. Orders below this
                       threshold are treated as hold to avoid near-zero noise orders.
    """
    # Clip signal to prevent extreme orders from high-noise agents.
    signal = float(np.clip(signal, 1.0 - signal_clip, 1.0 + signal_clip))

    # Determine action based on signal: buy if signal > 1.0, sell if signal < 1.0.
    action = "buy" if signal > 1.0 else "sell"

    # If action is buy but no cash, or action is sell but no shares, treat as hold.
    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Price is value scaled by signal, with a minimum of 0.01 to prevent non-positive prices.
    price = max(round(value * signal, 2), 0.01)
    conviction = abs(signal - 1.0)
    fraction = min(conviction * aggression * 10, 1.0)

    # For buys, quantity is fraction of max affordable. For sells, fraction of shares held.
    # Quantity is bounded to [min_qty_fraction * max_qty, max_qty] to prevent noise orders.
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))
    else:
        max_qty = shares
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))

    # If quantity is still zero or negative (e.g. max_qty itself is zero), treat as hold.
    if quantity <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Note: The aggression parameter allows tuning how strongly the agent reacts to the signal.
    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}


def utility_maximiser(signal: float, cash: float, shares: float, value: float,
                      risk_aversion: float = 2.0, signal_clip: float = 0.25,
                      min_qty_fraction: float = 0.01) -> dict:
    """
    CARA (Constant Absolute Risk Aversion) utility-maximising trader.

    Derives optimal trade size analytically from the mean-variance framework:
        optimal_fraction = E[r] / (risk_aversion * Var[r])
    where E[r] = signal - 1.0 (expected fractional return from the signal)
    and Var[r] is proxied by (signal_clip / 2)^2, representing uncertainty
    in the signal given the clipping range.

    This is grounded in Grossman-Stiglitz (1980) and Kyle (1985): informed agents
    trade a quantity proportional to their edge divided by their risk aversion.
    A higher risk_aversion produces smaller, more conservative orders. Unlike
    signal_following, conviction and position size are jointly determined by the
    utility function rather than a fixed aggression scalar.
    """
    # Clip signal to prevent extreme orders.
    signal = float(np.clip(signal, 1.0 - signal_clip, 1.0 + signal_clip))

    expected_return = signal - 1.0

    # Variance proxy: (signal_clip / 2)^2 treats the clipping range as roughly ±2σ.
    # Floored to prevent division by zero if signal_clip is set to 0.
    variance_proxy = max((signal_clip / 2.0) ** 2, 1e-8)

    # CARA optimal fraction of portfolio to allocate. Clipped to [-1, 1].
    optimal_fraction = float(np.clip(
        expected_return / (risk_aversion * variance_proxy), -1.0, 1.0
    ))

    # Determine action based on optimal_fraction: buy if > 0, sell if < 0, hold if = 0.
    if optimal_fraction > 0:
        action = "buy"
    elif optimal_fraction < 0:
        action = "sell"
    else:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # If action is buy but no cash, or action is sell but no shares, treat as hold.
    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Price reflects the agent's expected value of the asset given their signal.
    price = max(round(value * signal, 2), 0.01)

    # Quantity bounded to [min_qty_fraction * max_qty, max_qty].
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        quantity = float(np.clip(
            round(max_qty * abs(optimal_fraction), 6),
            min_qty_fraction * max_qty,
            max_qty
        ))
    else:
        max_qty = shares
        quantity = float(np.clip(
            round(max_qty * abs(optimal_fraction), 6),
            min_qty_fraction * max_qty,
            max_qty
        ))

    # If quantity is zero or negative (e.g. max_qty itself is zero), treat as hold.
    if quantity <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Note: The risk_aversion parameter allows tuning how aggressively the agent trades on the signal.
    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}


def contrarian(signal: float, cash: float, shares: float, value: float,
               aggression: float = 1.0, signal_clip: float = 0.25,
               min_qty_fraction: float = 0.01) -> dict:
    """
    Contrarian (mean-reversion) trader. Acts opposite to the signal:
      - If signal > 1.0 (market expects price to rise): SELL — agent believes it is overbought.
      - If signal < 1.0 (market expects price to fall): BUY  — agent believes it is oversold.

    Represents a fundamentals-anchored belief that prices revert to their mean.
    Contrarians provide stabilising liquidity against signal_following momentum
    and dampen volatility in the clearing price — making them a useful counterpart
    for thesis comparisons of agent heterogeneity effects on market efficiency.

    Conviction and quantity scale identically to signal_following (same aggression
    scalar) so the two strategies are directly comparable in experiments.
    """
    # Clip signal — same bounds as signal_following for a fair comparison.
    signal = float(np.clip(signal, 1.0 - signal_clip, 1.0 + signal_clip))

    # Invert the action relative to signal_following.
    action = "sell" if signal > 1.0 else "buy"

    if action == "buy" and cash <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}
    if action == "sell" and shares <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Contrarian posts limit prices leaning against the signal:
    # buys below value (expects reversion up from an oversold dip),
    # sells above value (expects reversion down from an overbought spike).
    # Using 1/signal inverts the direction while preserving magnitude.
    inverted_signal = float(np.clip(
        1.0 / signal if signal != 0 else 1.0,
        1.0 - signal_clip, 1.0 + signal_clip
    ))
    price = max(round(value * inverted_signal, 2), 0.01)
 
    conviction = abs(signal - 1.0)
    fraction = min(conviction * aggression * 10, 1.0)

    # Quantity bounded to [min_qty_fraction * max_qty, max_qty].
    if action == "buy":
        max_qty = cash / price if price > 0 else 0
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))
    else:
        max_qty = shares
        quantity = float(np.clip(
            round(max_qty * fraction, 6),
            min_qty_fraction * max_qty,
            max_qty
        ))

    # If quantity is zero or negative (e.g. max_qty itself is zero), treat as hold.
    if quantity <= 0:
        return {"Price": 0.0, "Quantity": 0.0, "Buy": 0.0, "Sell": 0.0, "Hold": 1.0}

    # Note: The aggression parameter allows tuning how strongly the agent reacts to the signal.
    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}


def adapted_signal_following(signal: float, cash: float, shares: float, value: float, aggression: float = None):
    """
    NOTE: Your original version had several bugs:
      - used '&' instead of 'and' (bitwise vs boolean)
      - inconsistent Buy/Sell fields and duplicated keys
      - "Quantity: " key has a typo
    This version keeps your apparent intent but returns standard order objects.

    I’m preserving the general structure: choose a price in [low, high] based on signal and
    submit a fixed fraction of cash as qty proxy.
    """
    price_range = 0.2
    low = value * (1 - price_range)
    high = value * (1 + price_range)

    signal_value = signal * value
    cash_slice = cash / 5 if cash > 0 else 0.0

    # Case 1: very bullish -> buy near high (if affordable)
    if (signal_value >= high) and (signal_value >= value):
        price = float(high)
        side = "buy"

    # Case 2: mildly bullish -> buy at signal-implied price (bounded)
    elif (signal_value >= value) and (signal_value < high):
        price = float(max(min(signal_value, high), 0.01))
        side = "buy"

    # Case 3: mildly bearish -> sell at signal-implied price (bounded)
    elif (signal_value < value) and (signal_value > low):
        price = float(max(min(signal_value, high), 0.01))
        side = "sell"

    # Case 4: very bearish -> sell near low
    else:
        price = float(max(low, 0.01))
        side = "sell"

    if side == "buy":
        if cash_slice <= 0:
            return None
        max_qty = cash / price if price > 0 else 0.0
        quantity = min(cash_slice / price if price > 0 else 0.0, max_qty)
    else:
        if shares <= 0:
            return None
        # sell a fraction of shares analogous to cash_slice notion
        # (kept minimal: sell up to 20% of shares)
        quantity = min(0.2 * float(shares), float(shares))

    if quantity <= 0:
        return None

    return {"Price": price, "Quantity": quantity,
            "Buy": 1.0 if action == "buy" else 0.0,
            "Sell": 1.0 if action == "sell" else 0.0,
            "Hold": 0.0}

STRATEGIES = {
    "zi": zero_intelligence,
    "signal_following": signal_following,
    "utility_maximiser": utility_maximiser,
    "contrarian": contrarian,
    "Adapt_sig": adapted_signal_following,
}