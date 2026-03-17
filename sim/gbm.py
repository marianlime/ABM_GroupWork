import numpy as np
import hashlib
from typing import Optional


def rng_from_string(seed):
    if not isinstance(seed, str):
        seed = str(seed)
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    seed = seed % (2**32)
    return np.random.default_rng(seed)


def simulate_gbm(S_0: float, volatility: float, drift: float, n_rounds: int, seed: Optional[str] = None):
    if S_0 <= 0 or not np.isfinite(S_0):
        raise ValueError(f"S_0 must be positive and finite (got {S_0})")
    if volatility < 0 or not np.isfinite(volatility):
        raise ValueError(f"volatility must be non-negative and finite (got {volatility})")
    if not np.isfinite(drift):
        raise ValueError(f"drift must be finite (got {drift})")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer.")
    if seed is None:
        seed = str(np.random.SeedSequence().entropy)

    rng = rng_from_string(seed)
    increments = (drift - 0.5 * volatility**2) + volatility * rng.standard_normal(n_rounds)

    logS = np.empty(n_rounds + 1)
    logS[0] = np.log(S_0)
    logS[1:] = logS[0] + np.cumsum(increments)

    return np.exp(logS)
