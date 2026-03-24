"""
Geometric Brownian Motion price-path simulation with a Numba-accelerated inner
loop and deterministic seeding derived from arbitrary string identifiers.
"""

import numpy as np
import hashlib
from typing import Optional
from numba import njit


def rng_from_string(seed):
    """Convert an arbitrary seed value to a seeded numpy Generator via SHA-256 hashing."""
    if not isinstance(seed, str):
        seed = str(seed)
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    seed = seed % (2**32)
    return np.random.default_rng(seed)



@njit
def simulate_gbm_numba(S_0, n_rounds, increments):
    """Numba-compiled GBM path computation; accumulates log-increments then exponentiates."""
    logS = np.empty(n_rounds + 1)
    logS[0] = np.log(S_0)
    logS[1:] = logS[0] + np.cumsum(increments)
    return np.exp(logS)

def simulate_gbm(S_0: float, volatility: float, drift: float, n_rounds: int, seed: Optional[str] = None):
    """
    Simulate a Geometric Brownian Motion price path of length n_rounds + 1.

    - Validates inputs and raises ValueError for degenerate parameters
    - Derives a reproducible RNG from seed via rng_from_string when provided
    - Returns a numpy array of prices [S_0, S_1, ..., S_n_rounds]
    """
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
    return simulate_gbm_numba(S_0, n_rounds, increments)
