""" 
Geometric Brownian Motion price-path simulation with a Numba-accelerated internal function for log-price path generation. 
Reproducible random number generation via string seeding, enabling consistent price paths across runs with the same seed. #
Used for fundamental value path in the market simulation.
Deterministic seeding derived from arbitrary string identifiers.
"""

import numpy as np
import hashlib
from typing import Optional
from numba import njit


def rng_from_string(seed):

    #Generates reproducible RNG 
    #string seed -> SHA-256 hash -> integer seed for numpy RNG

    if not isinstance(seed, str): #Ensures seed is string
        seed = str(seed) #Converts non-string seed to string 
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) #Converts string to integer.
    seed = seed % (2**32) #Limits integer to 32 bits (numpy's default seed size)
    generated_seed = np.random.default_rng(seed) #Creates and returns a numpy RNG seed.
    return generated_seed

@njit
def simulate_gbm_numba(S_0, n_rounds, increments):

    #Generates GBM price path using JIT-compiled function for performance.

    logS = np.empty(n_rounds + 1) #Pre-allocates the log-price array for efficiency.
    logS[0] = np.log(S_0) #Sets initial log price to log(S_0).
    logS[1:] = logS[0] + np.cumsum(increments) #Computes price_path (by exponentiating log-price) using cumulative sum of increments.
    return np.exp(logS) 

def simulate_gbm(S_0: float, volatility: float, drift: float, n_rounds: int, seed: Optional[str] = None):
    
    # Generates Simulation of GBM price path

    if S_0 <= 0 or not np.isfinite(S_0):
        raise ValueError(f"S_0 must be positive and finite (got {S_0})")
    if volatility < 0 or not np.isfinite(volatility):
        raise ValueError(f"volatility must be non-negative and finite (got {volatility})")
    if not np.isfinite(drift):
        raise ValueError(f"drift must be finite (got {drift})")
    if n_rounds <= 0 or not isinstance(n_rounds, int):
        raise ValueError(f"n_rounds must be a positive integer (got {n_rounds})")
    if seed is None:
        seed = str(np.random.SeedSequence().entropy) #Generates random seed if none provided.

    rng = rng_from_string(seed) #Generates reproducible RNG from string seed.
    increments = (drift - 0.5 * volatility**2) + volatility * rng.standard_normal(n_rounds) #Calculates GBM increments
    return simulate_gbm_numba(S_0, n_rounds, increments) #Simulates GBM price path using Numba-function for performance.
