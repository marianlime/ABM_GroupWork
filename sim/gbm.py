import numpy as np

# ----------------------------
# GBM
# ----------------------------

def simulate_gbm(S0, volatility, drift, n_rounds):
    logS = np.empty(n_rounds+1)
    logS[0] = np.log(S0)
    increments = (drift - 0.5 * volatility**2) + volatility * np.random.randn(n_rounds)
    logS[1:] = logS[0] + np.cumsum(increments)
    S = np.exp(logS)
    return {"stock_path": S, "shock_path": increments}